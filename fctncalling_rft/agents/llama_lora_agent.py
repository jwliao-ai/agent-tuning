from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.distributions.categorical import Categorical
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import os
from fctncalling_rft.models.critic import APPOCritic, TPPOCritic


class LlamaLoRAgent:

    def __init__(self, model_name, max_new_tokens, algo, load_path=None):
        self.device = "cuda:0"
        self.algo = algo
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, padding_side="left"
        )

        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=self.device
        )

        self.max_new_tokens = max_new_tokens
        self.roles = [
            "Thoughts: ",
            "\nDirectly call tools (strictly follow the format): ",
        ]

        if load_path is None:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)
        else:
            self.load(load_path)

    def _init_actor(self, lora_weights=None):
        if lora_weights is None:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "v_proj",
                ],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(self.base_model, config)

            model.print_trainable_parameters()

        else:
            model = PeftModel.from_pretrained(
                self.base_model,
                lora_weights,
                torch_dtype=torch.float16,
            )

        model.half()

        return model

    def _init_critic(self, critic_weights=None):
        if self.algo == "APPO":
            critic = APPOCritic(self.actor, self.tokenizer)
        elif self.algo == "TPPO":
            critic = TPPOCritic(self.actor, self.tokenizer)
        else:
            raise NotImplementedError
        if critic_weights is not None:
            critic.v_head.load_state_dict(
                torch.load(critic_weights, map_location="cpu")
            )
        return critic

    @torch.no_grad()
    def get_actions(self, obs, device=None):
        """
        Compute actions and value function predictions for the given inputs.
        """
        device = self.device if device is None else device
        prompts = obs.tolist()
        token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to(self.device)
        attn_mask = token_seq["attention_mask"].to(self.device)

        output = self.actor.generate(
            input_ids,
            attention_mask=attn_mask,
            do_sample=True,
            top_k=50,
            temperature=0.5,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                # 128000,
                # 198,
            ],  # 128000 and 198 are token ids of '\n'
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        sequences = output.sequences

        actions = []
        action_tokens = (
            torch.ones(
                (sequences.shape[0], self.max_new_tokens),
                dtype=torch.int64,
                device=self.device,
            )
            * self.tokenizer.pad_token_id
        )
        for i in range(sequences.shape[0]):
            action_token = sequences[i][input_ids[i].shape[0] :]
            action_tokens[i, : action_token.shape[0]] = action_token
            action = self.tokenizer.decode(action_token, skip_special_tokens=True)
            actions.append(action)
        actions = np.array(actions, dtype=np.object_)

        return actions, action_tokens

    @torch.no_grad()
    def get_actions_sequential(self, obs: np.ndarray, device=None):
        """
        Args:
            obs: np.ndarray of shape (rollout_threads, num_agents)

        Returns:
            all_actions: np.ndarray of shape (rollout_threads, num_agents)
            all_action_tokens: torch.tensor of shape (rollout_threads, num_agents, max_new_tokens)

        Compute actions and value function predictions for the given inputs.
        Sequentially appends responses from previous agents in the prompt.
        """
        device = self.device if device is None else device
        # Note: for online, batch_size = 1, so the first dimension goes away
        rollout_threads, num_agents = obs.shape

        all_actions = np.empty((rollout_threads, num_agents), dtype=object)
        all_action_tokens = (
            torch.ones(
                (rollout_threads, num_agents, self.max_new_tokens),
                dtype=torch.int64,
                device=self.device,
            )
            * self.tokenizer.pad_token_id
        )

        prompts = obs[:, 0].tolist()
        for agent_idx in range(num_agents):
            prompts = [prompt + self.roles[agent_idx] for prompt in prompts]
            token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = token_seq["input_ids"].to(self.device)
            attn_mask = token_seq["attention_mask"].to(self.device)

            output = self.actor.generate(
                input_ids,
                attention_mask=attn_mask,
                do_sample=True,
                top_k=50,
                temperature=0.5,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                ],
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
            sequences = output.sequences
            actions = []
            for i in range(rollout_threads):
                action_token = sequences[i][input_ids[i].shape[0] :]
                all_action_tokens[i, agent_idx, : action_token.shape[0]] = action_token
                action = self.tokenizer.decode(action_token, skip_special_tokens=True)
                prompts[i] = prompts[i] + action
                actions.append(action)
            actions = np.array(actions, dtype=np.object_)
            all_actions[:, agent_idx] = actions
            print(f"agent {agent_idx} actions: {actions}")
            print(f"-" * 20)

        return all_actions, all_action_tokens

    def get_action_values(self, obs: np.ndarray, device=None):
        """
        Args:
            obs: np.ndarray of shape (rollout_threads, num_agents)

        Returns:
            values: torch.Tensor of shape (rollout_threads, num_agents, 1)
        """
        device = self.device if device is None else device
        inputs = self.tokenizer(obs[:, 0].tolist(), return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attention_mask)
        return values

    def get_slice_single(
        self,
        logits: torch.Tensor,
        obs_full_lengths: int,
        act_real_lengths: torch.Tensor,
        device=None,
    ):
        device = self.device if device is None else device
        action_slice = torch.zeros(
            (logits.shape[0], self.max_new_tokens, logits.shape[-1]),
            device=self.device,
        )
        for i in range(logits.shape[0]):
            start_idx = obs_full_lengths - 1
            end_idx = obs_full_lengths + act_real_lengths[i] - 1
            action_slice[i, : act_real_lengths[i]] = logits[i, start_idx:end_idx]
        return action_slice

    def get_slice(
        self,
        logits: torch.Tensor,
        obs_full_lengths: int,
        act_real_lengths: torch.Tensor,
        device=None,
    ) -> torch.Tensor:
        """
        Args:
            logits: torch.Tensor of shape (rollout_threads, num_agents, max_new_tokens, data_dim)
            obs_full_lengths: int
            act_real_lengths: torch.Tensor of shape (rollout_threads, num_agents)
            device: torch.device

        Returns:
            action_slice: torch.Tensor of shape (rollout_threads, num_agents, max_new_tokens, data_dim)
        """
        device = self.device if device is None else device
        action_slice = torch.zeros(
            act_real_lengths.shape[0],
            act_real_lengths.shape[1],
            self.max_new_tokens,
            logits.shape[-1],
        ).to(device)
        for thread_idx in range(act_real_lengths.shape[0]):
            for agent_idx in range(act_real_lengths.shape[1]):
                if agent_idx == 0:
                    start_idx = obs_full_lengths - 1
                    end_idx = (
                        obs_full_lengths + act_real_lengths[thread_idx, agent_idx] - 1
                    )
                else:
                    start_idx = end_idx + 1
                    end_idx = start_idx + act_real_lengths[thread_idx, agent_idx]
                action_slice[
                    thread_idx, agent_idx, : act_real_lengths[thread_idx, agent_idx]
                ] = logits[thread_idx, start_idx:end_idx]
        return action_slice

    def get_token_values(
        self,
        obs: np.ndarray,
        action_tokens: torch.Tensor,
        train: bool = False,
        device=None,
    ) -> torch.Tensor:
        """
        Args:
            obs: np.ndarray of shape (rollout_threads/batch_size, num_agents)
            action_tokens: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens)

        Returns:
            values: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens, data_dim)
        """
        device = self.device if device is None else device
        obs_token_seq = self.tokenizer(
            obs[:, 0].tolist(), return_tensors="pt", padding=True
        )
        # shape (rollout_threads, obs_token_len)
        obs_input_ids = obs_token_seq["input_ids"].to(device)
        obs_attn_mask = obs_token_seq["attention_mask"].to(device)
        obs_full_lengths = obs_input_ids.shape[1]

        act_attn_mask = action_tokens != 0
        # shape (rollout_threads, num_agents, max_new_tokens)
        act_real_lengths = act_attn_mask.sum(dim=-1)
        # shape (rollout_threads, num_agents)

        # concatedenated_action_tokens: a list of torch.Tensor
        concatenated_action_tokens = [
            tokens[tokens != 0]
            for tokens in action_tokens.view(action_tokens.size(0), -1)
        ]

        # padded_action_tokens: shape (rollout_threads, max_concatenated_length)
        padded_action_tokens = pad_sequence(
            concatenated_action_tokens, batch_first=True, padding_value=0
        )

        # padded_action_attn_mask: shape (rollout_threads, max_concatenated_length)
        padded_action_attn_mask = (padded_action_tokens != 0).long()

        obs_act_ids = torch.cat([obs_input_ids, padded_action_tokens], dim=1)
        obs_act_mask = torch.cat([obs_attn_mask, padded_action_attn_mask], dim=1)
        # shape (rollout_threads, obs_token_len + max_concatenated_length)

        with self.actor.disable_adapter():
            if not train:
                with torch.no_grad():
                    values = self.critic(obs_act_ids, attention_mask=obs_act_mask)
            else:
                values = self.critic(obs_act_ids, attention_mask=obs_act_mask)
            # values has shape (rollout_threads, obs_token_len + max_concatenated_length, 1)
        values = self.get_slice(values, obs_full_lengths, act_real_lengths).to(device)

        return values

    def get_token_logits(
        self,
        obs: np.ndarray,
        action_tokens: torch.Tensor,
        device=None,
        batch_infer: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: np.ndarray of shape (rollout_threads/batch_size, num_agents)
            action_tokens: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens)

        Returns:
            pi_logits: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
            rho_logits: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
        """
        device = self.device if device is None else device
        obs_token_seq = self.tokenizer(
            obs[:, 0].tolist(), return_tensors="pt", padding=True
        )
        # shape (rollout_threads, obs_token_len)
        obs_input_ids = obs_token_seq["input_ids"].to(device)
        obs_attn_mask = obs_token_seq["attention_mask"].to(device)
        obs_full_lengths = obs_input_ids.shape[1]

        act_attn_mask = action_tokens != 0
        # shape (rollout_threads, num_agents, max_new_tokens)
        act_real_lengths = act_attn_mask.sum(dim=-1)
        # shape (rollout_threads, num_agents)

        # concatedenated_action_tokens: a list of torch.Tensor
        concatenated_action_tokens = [
            tokens[tokens != 0]
            for tokens in action_tokens.view(action_tokens.size(0), -1)
        ]

        # padded_action_tokens: shape (rollout_threads, max_concatenated_length)
        padded_action_tokens = pad_sequence(
            concatenated_action_tokens, batch_first=True, padding_value=0
        )

        # padded_action_attn_mask: shape (rollout_threads, max_concatenated_length)
        padded_action_attn_mask = (padded_action_tokens != 0).long()

        obs_act_ids = torch.cat([obs_input_ids, padded_action_tokens], dim=1)
        obs_act_mask = torch.cat([obs_attn_mask, padded_action_attn_mask], dim=1)
        # shape (rollout_threads, obs_token_len + max_concatenated_length)

        if batch_infer:
            with self.actor.disable_adapter():
                rho_logits = self.batch_infer(
                    self.actor,
                    obs_act_ids,
                    obs_act_mask,
                    obs_full_lengths,
                    act_real_lengths,
                ).to(device)

            pi_logits = self.batch_infer(
                self.actor,
                obs_act_ids,
                obs_act_mask,
                obs_full_lengths,
                act_real_lengths,
            ).to(device)
        else:
            with self.actor.disable_adapter():
                rho_outputs = self.actor(
                    input_ids=obs_act_ids, attention_mask=obs_act_mask, return_dict=True
                )
                rho_logits = self.get_slice(
                    rho_outputs.logits, obs_full_lengths, act_real_lengths
                ).to(device)

            pi_outputs = self.actor(
                input_ids=obs_act_ids, attention_mask=obs_act_mask, return_dict=True
            )
            pi_logits = self.get_slice(
                pi_outputs.logits, obs_full_lengths, act_real_lengths
            ).to(device)

        return pi_logits, rho_logits

    @torch.no_grad()
    def batch_infer(
        self,
        model,
        input_ids,
        attn_mask,
        obs_full_lengths,
        act_real_lengths,
        infer_batch_size=16,
    ):
        logits = []
        for i in range(0, input_ids.shape[0], infer_batch_size):
            input_ids_batch = input_ids[i : i + infer_batch_size, :]
            attn_mask_batch = attn_mask[i : i + infer_batch_size, :]
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attn_mask_batch,
                return_dict=True,
            )

            logits_batch = self.get_slice(
                outputs.logits, obs_full_lengths, act_real_lengths
            )
            logits.append(logits_batch.clone())
        logits = torch.cat(logits, dim=0)

        return logits

    def get_last_token_position(self, action_tokens: torch.Tensor) -> int:
        """
        Given the action tokens, return the last token position.

        Args:
            action_tokens: (torch.Tensor): (max_new_tokens)

        Return:
            last_token_position: (torch.Tensor): int
        """
        pos = len(action_tokens) - 1
        while action_tokens[pos] == self.tokenizer.pad_token_id:
            pos -= 1
        return pos

    def get_joint_action_log_probs(
        self,
        obs: np.ndarray,
        action_tokens: torch.Tensor,
        batch_infer=False,
        device=None,
    ):
        """
        Args:
            obs: np.ndarray of shape (rollout_threads/batch_size, num_agents)
            action_tokens: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens)

        Return:
            action_log_probs: torch.Tensor of shape (rollout_threads/batch_size, num_agents)
            entropies: torch.Tensor of shape (rollout_threads/batch_size, num_agents)
        """
        device = self.device if device is None else device
        pi_logits, _ = self.get_token_logits(
            obs, action_tokens, batch_infer=batch_infer, device=device
        )
        # pi_logits: shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
        pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
        action_log_probs = torch.empty(pi_logits.shape[0], pi_logits.shape[1]).to(
            device
        )
        entropies = torch.empty(pi_logits.shape[0], pi_logits.shape[1]).to(device)
        for thread in range(pi_logits.shape[0]):
            for agent in range(pi_logits.shape[1]):
                act_token_length = (
                    self.get_last_token_position(action_tokens[thread, agent]) + 1
                )
                log_softmax_slice = pi_log_softmax[thread, agent, :act_token_length, :]
                action_token_slice = action_tokens[thread, agent, :act_token_length]
                token_log_probs = torch.gather(
                    log_softmax_slice, -1, action_token_slice.unsqueeze(-1)
                ).squeeze(-1)
                action_log_prob = token_log_probs.sum()
                action_log_probs[thread, agent] = action_log_prob

                entropy = (
                    Categorical(logits=pi_logits[thread, :act_token_length, :])
                    .entropy()
                    .mean()
                )
                entropies[thread, agent] = entropy

        return action_log_probs, entropies

    @torch.no_grad()
    def infer_for_rollout(self, obs):
        # actions, action_tokens = self.get_actions(obs)
        actions, action_tokens = self.get_actions_sequential(obs)

        if self.algo == "APPO":
            values = self.get_action_values(obs)
            values = values.float().cpu().numpy()
            action_log_probs, _ = self.get_joint_action_log_probs(
                obs, action_tokens, batch_infer=False
            )
            action_tokens = action_tokens.int().cpu().numpy()
            action_log_probs = action_log_probs.float().cpu().numpy()
            log_probs = action_log_probs
        elif self.algo == "TPPO":
            # above leak happens
            # get_token_values leak happens
            values = self.get_token_values(obs, action_tokens).squeeze(-1)
            pi_logits, _ = self.get_token_logits(obs, action_tokens, batch_infer=True)
            pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
            token_log_probs = torch.gather(
                pi_log_softmax, -1, action_tokens.unsqueeze(-1)
            ).squeeze(-1)

            values = values.float().cpu().numpy()
            action_tokens = action_tokens.int().cpu().numpy()
            token_log_probs = token_log_probs.float().cpu().numpy()
            log_probs = token_log_probs
        else:
            raise NotImplementedError

        return actions, action_tokens, values, log_probs

    def get_next_tppo_values(self, obs: np.ndarray):
        """
        Args:
            obs: np.ndarray of shape (rollout_threads, num_agents)

        Returns:
            values: torch.Tensor of shape (rollout_threads, num_agents, 1)
        """
        token_seq = self.tokenizer(
            obs[:, 0].tolist(), return_tensors="pt", padding=True
        )
        input_ids = token_seq["input_ids"].to(self.device)
        attn_mask = token_seq["attention_mask"].to(self.device)

        # values
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)
            values = values[:, -1]
        return values

    def get_next_tppo_values_single(self, obs):
        token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to(self.device)
        attn_mask = token_seq["attention_mask"].to(self.device)

        # values
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)
            values = values[:, -1]
        return values

    def get_next_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Get value function predictions.
        Args:
            obs: np.ndarray of shape (rollout_threads, num_agents)

        Returns:
            values: np.ndarray of shape (rollout_threads, num_agents, 1)
        """
        if self.algo == "APPO":
            values = self.get_action_values(obs)
            values = values.cpu().float().numpy()
            return values
        elif self.algo == "TPPO":
            values = self.get_next_tppo_values(obs).squeeze(-1)
            values = values.cpu().float().numpy()
            return values
        else:
            raise NotImplementedError

    def infer_for_action_update(
        self, obs: np.ndarray, action_tokens: torch.Tensor = None, device=None
    ):
        device = self.device if device is None else device
        assert action_tokens is not None, "action_tokens could not be none"
        action_log_probs, entropies = self.get_joint_action_log_probs(
            obs, action_tokens, device=device
        )
        return action_log_probs, entropies

    def infer_for_token_update(
        self, obs: np.ndarray, action_tokens: torch.Tensor, device=None
    ):
        """
        Args:
            obs: np.ndarray of shape (rollout_threads/batch_size, num_agents)
            action_tokens: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens)

        Return:
            pi_logits: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
            rho_logits: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
        """
        device = self.device if device is None else device
        pi_logits, rho_logits = self.get_token_logits(obs, action_tokens, device)
        return pi_logits, rho_logits

    def save(self, save_dir: str, episode: int) -> None:
        print("save model")
        exp_path = os.path.join(save_dir, "episode_{:04d}".format(episode))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        self.actor.save_pretrained(exp_path)
        # save critic
        torch.save(
            self.critic.state_dict(), os.path.join(exp_path, "critic.pth")
        )

    def load(self, save_dir: str):
        self.actor = self._init_actor(save_dir).to(self.device)
        critic_weights = os.path.join(save_dir, "critic.pth")
        self.critic = self._init_critic(critic_weights).to(self.device)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
