import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fctncalling_rft.utils.util import get_gard_norm, huber_loss, mse_loss
from torch.distributions.categorical import Categorical
from accelerate import Accelerator


class TPPOTrainer:

    def __init__(self, args, agent, num_agents, rank):
        self.rank = rank
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
        self.policy_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.agent.actor.parameters()),
            lr=self.lr,
            eps=1e-5,
            weight_decay=0,
        )
        self.critic_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.agent.critic.parameters()),
            lr=self.critic_lr,
            eps=1e-5,
        )
        (
            self.agent.actor,
            self.agent.critic,
            self.policy_optimizer,
            self.critic_optimizer,
        ) = self.accelerator.prepare(
            self.agent.actor,
            self.agent.critic,
            self.policy_optimizer,
            self.critic_optimizer,
        )

    def cal_token_mask(self, action_tokens_batch):
        pad_token = self.agent.tokenizer.pad_token_id
        token_mask = (action_tokens_batch != pad_token).int()
        return token_mask

    def cal_policy_loss(
        self, log_prob_infer, log_prob_batch, advantages_batch, entropy, token_mask
    ):

        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)

        approx_kl = (imp_weights - 1) - log_ratio

        surr1 = (
            -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages_batch
        )
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        policy_loss = surr - self.entropy_coef * entropy

        policy_loss = (policy_loss * token_mask).sum() / token_mask.sum()
        approx_kl = (approx_kl * token_mask).sum() / token_mask.sum()
        entropy_value = (entropy * token_mask).sum() / token_mask.sum()

        return policy_loss, approx_kl, entropy_value

    def cal_value_loss(self, values_infer, value_preds_batch, return_batch, token_mask):

        value_pred_clipped = value_preds_batch + (
            values_infer - value_preds_batch
        ).clamp(-self.clip_param, self.clip_param)
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
        (
            obs_batch,
            action_batch,
            log_prob_batch,
            value_preds_batch,
            return_batch,
            advantages_batch,
            action_tokens_batch,
        ) = sample

        advantages_copy = advantages_batch.copy()
        advantages_copy[advantages_copy == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages_batch = (advantages_batch - mean_advantages) / (
            std_advantages + 1e-8
        )

        log_prob_batch = torch.from_numpy(log_prob_batch).to(self.rank)
        value_preds_batch = torch.from_numpy(value_preds_batch).to(self.rank)
        return_batch = torch.from_numpy(return_batch).to(self.rank)
        advantages_batch = torch.from_numpy(advantages_batch).to(self.rank)
        action_tokens_batch = torch.from_numpy(action_tokens_batch).to(self.rank)
        token_mask = self.cal_token_mask(action_tokens_batch).to(self.rank)
        batch_size = obs_batch.shape[0]

        # critic update
        self.critic_optimizer.zero_grad()
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        total_value_loss = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            values_infer_chunk = self.agent.get_token_values(
                obs=obs_batch[start:end],
                action_tokens=action_tokens_batch[start:end],
                train=True,
                device=self.rank,
            ).squeeze(-1)
            # value_infer_chunk shape (rollout_threads/batch_size, num_agents, max_new_tokens)
            # values_infer_chunk = values_infer_chunk.view(
            #     obs_batch[start:end].shape[0], -1, values_infer_chunk.shape[-1]
            # )
            value_loss_chunk = self.cal_value_loss(
                values_infer_chunk,
                value_preds_batch[start:end],
                return_batch[start:end],
                token_mask[start:end],
            )
            value_loss_chunk /= self.gradient_cp_steps  # Normalize loss
            self.accelerator.backward(value_loss_chunk)
            total_value_loss += value_loss_chunk.item()

        if self._use_max_grad_norm:
            critic_grad_norm = self.accelerator.clip_grad_norm_(
                self.agent.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_gard_norm(self.agent.critic.parameters())
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_grad_norm = critic_grad_norm.item()
        value_loss = total_value_loss * self.gradient_cp_steps  # Scale back the loss

        # policy update
        self.policy_optimizer.zero_grad()
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        total_approx_kl = 0
        total_entropy = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            pi_logits, _ = self.agent.get_token_logits(obs_batch[start:end], action_tokens_batch[start:end])
            # shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
            # pi_logits = pi_logits.view(
            #     obs_batch[start:end].shape[0], -1, *pi_logits.shape[-2:]
            # )
            pi_log_prob = torch.log_softmax(pi_logits, dim=-1)
            log_prob_infer = torch.gather(pi_log_prob, -1, action_tokens_batch[start:end].unsqueeze(-1)).squeeze(-1)
            entropy = Categorical(logits=pi_logits).entropy()
            policy_loss, approx_kl, entropy_value = self.cal_policy_loss(log_prob_infer, log_prob_batch[start:end], advantages_batch[start:end], entropy, token_mask[start:end])
            total_approx_kl += approx_kl / self.gradient_cp_steps
            total_entropy += entropy_value / self.gradient_cp_steps
            policy_loss /= self.gradient_cp_steps
            self.accelerator.backward(policy_loss)
        if total_approx_kl > 0.02:
            self.policy_optimizer.zero_grad()
            return value_loss, critic_grad_norm, 0, 0, 0

        policy_grad_norm = self.accelerator.clip_grad_norm_(
            self.agent.actor.parameters(), self.max_grad_norm
        )
        self.policy_optimizer.step()
        policy_loss = policy_loss.item() * self.gradient_cp_steps
        self.policy_optimizer.zero_grad()
        policy_grad_norm = policy_grad_norm.item()
        total_entropy = total_entropy.item() * self.gradient_cp_steps

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            policy_grad_norm,
            total_entropy,
        )

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
