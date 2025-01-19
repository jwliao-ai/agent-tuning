import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fctncalling_rft.utils.util import get_gard_norm, huber_loss, mse_loss, to_cuda
from torch.distributions.categorical import Categorical
from accelerate import Accelerator
from fctncalling_rft.agents.actor import Actor


class TPPOTrainer:

    def __init__(self, args, agent: Actor, num_agents):
        self.agent = agent
        self.accelerator = Accelerator()
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

        self.policy_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.agent.actor.parameters()), lr=self.lr, eps=1e-5, weight_decay=0)
        self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.agent.critic.parameters()), lr=self.critic_lr, eps=1e-5)


    def cal_token_mask(self, action_tokens_batch):
        pad_token = self.agent.tokenizer.pad_token_id
        token_mask = (action_tokens_batch != pad_token).int()
        return token_mask

    def cal_policy_loss(self, log_prob_infer, log_prob_batch, advantages_batch, entropy, token_mask):

        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)
        approx_kl = (imp_weights - 1) - log_ratio
        surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        policy_loss = surr - self.entropy_coef * entropy

        policy_loss = (policy_loss * token_mask).sum() / token_mask.sum()
        approx_kl = (approx_kl * token_mask).sum() / token_mask.sum()
        entropy_value = (entropy * token_mask).sum() / token_mask.sum()

        return policy_loss, approx_kl, entropy_value

    def cal_value_loss(self, values_infer, value_preds_batch, return_batch, token_mask):

        value_pred_clipped = value_preds_batch + (values_infer - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped)
        value_loss = (value_loss * token_mask).sum() / token_mask.sum()

        return value_loss * self.value_loss_coef

    def ppo_update(self, sample):
        observations, actions, log_probs, value_preds, \
            returns, advantages, action_tokens = sample
            
        advantages_copy = advantages.copy()
        advantages_copy[advantages_copy == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        observations, actions, log_probs, value_preds, returns, advantages, action_tokens = \
            to_cuda((observations, actions, log_probs, value_preds, returns, advantages, action_tokens))
        token_mask = self.cal_token_mask(action_tokens)

        batch_size = observations.shape[0]
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        if cp_batch_size == 0:
            print(f"gradient_cp_steps > batch_size, set cp_batch_size = 1")
            cp_batch_size = 1
        # print(f"[trainer] batch_size: {batch_size}")
        # print(f"[trainer] cp_batch_size: {cp_batch_size}")

        # critic update
        torch.cuda.empty_cache()
        self.critic_optimizer.zero_grad()
        value_loss = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            if end > batch_size:
                end = batch_size
            cp_weight = (end - start) / batch_size  # Weight for the chunk loss
            cp_obs_batch, cp_action_tokens_batch, cp_value_preds_batch, cp_returns_batch, cp_token_mask = \
                observations[start:end], action_tokens[start:end], value_preds[start:end], returns[start:end], token_mask[start:end]
            values_infer = self.agent.get_token_values(cp_obs_batch, cp_action_tokens_batch, train=True).squeeze(-1) # .squeeze(-1)?
            # print(f"[trainer] values_infer shape: {values_infer.shape}")
            # print(f"[trainer] cp_value_preds_batch shape: {cp_value_preds_batch.shape}")
            # print(f"[trainer] cp_returns_batch shape: {cp_returns_batch.shape}")
            cp_value_loss = self.cal_value_loss(values_infer, cp_value_preds_batch, cp_returns_batch, cp_token_mask)
            cp_value_loss *= cp_weight
            cp_value_loss.backward()
            value_loss += cp_value_loss.item()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.agent.critic.parameters())
        self.critic_optimizer.step()
        critic_grad_norm = critic_grad_norm.item()

        # policy update
        torch.cuda.empty_cache()
        self.policy_optimizer.zero_grad()
        total_approx_kl = 0
        total_entropy = 0
        policy_loss = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size 
            if end > batch_size:
                end = batch_size
            cp_weight = (end - start) / batch_size
            cp_obs_batch, cp_action_tokens_batch, cp_adv_batch, cp_log_prob_batch, cp_token_mask = \
                observations[start:end], action_tokens[start:end], advantages[start:end], log_probs[start:end], token_mask[start:end]
            logits_infer, _ = self.agent.get_token_logits(cp_obs_batch, cp_action_tokens_batch) # (cp_batch_size, num_agents, vocab_size)
            pi_log_prob = torch.log_softmax(logits_infer, dim=-1)
            log_prob_infer = torch.gather(pi_log_prob, -1, cp_action_tokens_batch.unsqueeze(-1)).squeeze(-1)
            entropy = Categorical(logits=logits_infer).entropy()
            # print(f"[trainer] log_prob_infer shape: {log_prob_infer.shape}")
            # print(f"[trainer] cp_log_prob_batch shape: {cp_log_prob_batch.shape}")
            # print(f"[trainer] cp_adv_batch shape: {cp_adv_batch.shape}")
            cp_policy_loss, approx_kl, cp_entropy = self.cal_policy_loss(log_prob_infer, cp_log_prob_batch, cp_adv_batch, entropy, cp_token_mask)
            total_approx_kl += approx_kl * cp_weight
            total_entropy += cp_entropy * cp_weight
            cp_policy_loss *= cp_weight
            cp_policy_loss.backward()
            policy_loss += cp_policy_loss.item()
        if total_approx_kl > 0.02:
            return value_loss, critic_grad_norm, 0, 0, 0
        policy_grad_norm = nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        policy_grad_norm = policy_grad_norm.item()

        return value_loss, critic_grad_norm, policy_loss, policy_grad_norm, total_entropy

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["value_loss"] = 0
        train_info["value_grad_norm"] = 0
        train_info["policy_loss"] = 0
        train_info["policy_grad_norm"] = 0
        train_info["entropy"] = 0

        update_time = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.tppo_sampler(self.num_mini_batch)
            for sample in data_generator:
                value_loss, value_grad_norm, policy_loss, policy_grad_norm, entropy = (
                    self.ppo_update(sample)
                )
                train_info["value_loss"] += value_loss
                train_info["value_grad_norm"] += value_grad_norm
                train_info["policy_loss"] += policy_loss
                train_info["policy_grad_norm"] += policy_grad_norm
                train_info["entropy"] += entropy
                update_time += 1

        for k in train_info.keys():
            train_info[k] /= update_time

        return train_info

    def prep_training(self):
        self.agent.train()

    def prep_rollout(self):
        self.agent.eval()
