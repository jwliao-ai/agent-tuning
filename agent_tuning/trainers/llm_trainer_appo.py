import torch
import torch.nn as nn
from abc import ABC
import torch.nn.functional as F
from agent_tuning.agents.actor import Actor
from agent_tuning.utils.language_buffer import LanguageBuffer
from agent_tuning.utils.util import get_gard_norm, huber_loss, mse_loss, to_cuda


class APPOTrainer(ABC):

    def __init__(self, args, agent: Actor, num_agents):
        self.agent = agent
        self.num_agent = agent.num_agents
        self.tokenizer = agent.tokenizer
        self.agent_iteration_interval = args.agent_iteration_interval
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.entropy_coef = args.entropy_coef
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.gradient_cp_steps = args.gradient_cp_steps

        self.policy_optimizer = {}
        for agent_idx in range(self.num_agent):
            self.agent.actor.set_adapter(self.agent.profiles[agent_idx]["role"])
            self.policy_optimizer[self.agent.profiles[agent_idx]["role"]] = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.agent.actor.parameters()), lr=self.lr, eps=1e-5, weight_decay=0)
        self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.agent.critic.parameters()), lr=self.critic_lr, eps=1e-5)

    def cal_policy_loss(self, log_prob_infer: torch.Tensor, log_prob_batch: torch.Tensor, advantages_batch: torch.Tensor, entropy: torch.Tensor):

        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)
        approx_kl = ((imp_weights - 1) - log_ratio).mean()
        surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
        return policy_loss, approx_kl

    def cal_value_loss(self, values_infer: torch.Tensor, value_preds_batch: torch.Tensor, return_batch: torch.Tensor):

        value_pred_clipped = value_preds_batch + (values_infer - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
        return value_loss * self.value_loss_coef

    def ppo_update(self, sample, global_steps: int):

        agent_to_train = None
        if self.agent_iteration_interval > 0:
            time_slice = global_steps // self.agent_iteration_interval
            agent_to_train = time_slice % self.num_agent

        observations, actions, rollout_observations, log_probs, value_preds, returns, advantages, action_tokens = sample
        
        actions, rollout_observations, log_probs, value_preds, returns, advantages, action_tokens = \
            to_cuda((actions, rollout_observations, log_probs, value_preds, returns, advantages, action_tokens))
        
        batch_size = rollout_observations.shape[0]
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        if cp_batch_size == 0:
            print(f"gradient_cp_steps > batch_size, set cp_batch_size = 1")
            cp_batch_size = 1

        torch.cuda.empty_cache()
        # critic update with checkpoint gradient accumulation
        self.critic_optimizer.zero_grad()
        value_loss = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            if end > batch_size:
                end = batch_size
            cp_weight = (end - start) / batch_size  # Weight for the chunk loss
            cp_obs_batch, cp_value_preds_batch, cp_returns_batch = observations[start:end], value_preds[start:end], returns[start:end]
            values_infer = self.agent.get_action_values(cp_obs_batch)
            cp_value_loss = self.cal_value_loss(values_infer, cp_value_preds_batch, cp_returns_batch)
            cp_value_loss *= cp_weight  # Scale the loss by the chunk weight
            cp_value_loss.backward()
            value_loss += cp_value_loss.item()
            torch.cuda.empty_cache()
        # Gradient clipping
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.agent.critic.parameters())
        self.critic_optimizer.step()
        critic_grad_norm = critic_grad_norm.item()

        torch.cuda.empty_cache()
        # policy update
        for optimizer in self.policy_optimizer.values(): optimizer.zero_grad()
        total_approx_kl = 0
        total_entropy = 0
        policy_loss = 0
        total_policy_grad_norm = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size 
            if end > batch_size:
                end = batch_size
            cp_weight = (end - start) / batch_size
            cp_obs_batch, cp_act_batch, cp_adv_batch, cp_log_probs_batch = rollout_observations[start:end], action_tokens[start:end], advantages[start:end], log_probs[start:end]
            log_prob_infer, cp_entropy = self.agent.get_joint_action_log_probs(cp_obs_batch, cp_act_batch, agent_to_train)
            if cp_batch_size > 1:
                cp_adv_batch = (cp_adv_batch - cp_adv_batch.mean()) / (cp_adv_batch.std() + 1e-8)
            if agent_to_train is not None:
                cp_log_probs_batch = cp_log_probs_batch[:, agent_to_train: agent_to_train + 1]
                cp_adv_batch = cp_adv_batch[:, agent_to_train: agent_to_train + 1]
            cp_policy_loss, approx_kl = self.cal_policy_loss(log_prob_infer, cp_log_probs_batch, cp_adv_batch, cp_entropy)
            total_approx_kl += approx_kl * cp_weight
            total_entropy += cp_entropy.mean().item() * cp_weight
            cp_policy_loss = cp_policy_loss * cp_weight
            cp_policy_loss.backward()
            policy_loss += cp_policy_loss.item()
        if total_approx_kl > 0.02:
            return value_loss, critic_grad_norm, 0, 0, total_approx_kl, total_entropy

        if agent_to_train is not None:
            self.agent.actor.set_adapter(self.agent.profiles[agent_to_train]['role'])
            policy_grad_norm = nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
            self.policy_optimizer[self.agent.profiles[agent_to_train]['role']].step()
            total_policy_grad_norm = policy_grad_norm.item()
        else:
            for profile in self.agent.profiles:
                self.agent.actor.set_adapter(profile['role'])
                policy_grad_norm = nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                self.policy_optimizer[profile['role']].step()
                total_policy_grad_norm += policy_grad_norm.item()

        return value_loss, critic_grad_norm, policy_loss, total_policy_grad_norm, total_approx_kl, total_entropy

    def train(self, buffer: LanguageBuffer, global_steps: int):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {
            "value_loss": 0,
            "value_grad_norm": 0,
            "policy_loss": 0,
            "policy_grad_norm": 0,
            "approx_kl": 0,
            "entropy": 0,
        }

        update_time = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.appo_sampler(self.num_mini_batch)
            for sample in data_generator:
                value_loss, value_grad_norm, policy_loss, policy_grad_norm, approx_kl, entropy = self.ppo_update(sample, global_steps)
                train_info["value_loss"] += value_loss
                train_info["value_grad_norm"] += value_grad_norm
                train_info["policy_loss"] += policy_loss
                train_info["policy_grad_norm"] += policy_grad_norm
                train_info["approx_kl"] += approx_kl
                train_info["entropy"] += entropy
                update_time += 1

        for k in train_info.keys():
            train_info[k] /= update_time

        return train_info

    def prep_training(self):
        self.agent.actor().train()
        self.agent.critic().train()

    def prep_rollout(self):
        self.agent.actor().eval()
        self.agent.critic().eval()
