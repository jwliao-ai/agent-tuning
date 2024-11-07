import time
import os
import numpy as np
from functools import reduce
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from fctncalling_rft.models.codellama import Llama
from fctncalling_rft.agents import LlamaLoRAgent
from fctncalling_rft.utils import LanguageBuffer
from fctncalling_rft.trainers import APPOTrainer, TPPOTrainer


def setup(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    port: int = 12356,
    backend: str = "nccl",
):
    print(rank, "initializing distributed")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def _t2n(x):
    return x.detach().cpu().numpy()

class FctnCallingRunner:
    """Runner class to perform training, evaluation. and data collection. See parent class for details."""

    def __init__(self, config):
        self.num_agents = config["num_agents"]
        self.all_args = config["all_args"]
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.algo = self.all_args.algorithm_name
        self.model_type = self.all_args.model_type

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / "models/")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]

        self.agent = LlamaLoRAgent(
            self.all_args.model_name_or_path,
            self.all_args.max_new_tokens,
            self.algo,
        )

        self.buffer = LanguageBuffer(
            self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id
        )

    def run(self):
        obs = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = obs.copy()

        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        progress_bar = tqdm(total=episodes, desc=f"Start running...", position=0, leave=True)

        for episode in range(episodes):
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_tokens, log_probs = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                # insert data into buffer
                data = obs, rewards, dones, values, actions, action_tokens, log_probs
                self.insert(data)
                del data

                for i in range(self.n_rollout_threads):
                    global_step = total_num_steps + step * self.n_rollout_threads + i
                    if dones[i, 0]:
                        episodic_return = rewards[i, 0]
                        self.writter.add_scalar(
                            "episodic_return", episodic_return, global_step
                        )

            # compute return and update network
            self.before_update()
            train_infos = self.train()
            self.buffer.after_update()

            # manually clear GPU cache to avoid OOM (memory fragmentation)
            torch.cuda.empty_cache()

            # post process
            # save model
            if (episode == episodes - 1) or ((episode + 1) % self.all_args.save_interval == 0):
                self.save(episode)

            # log info
            if episode % self.log_interval == 0:
                progress_bar.set_description(
                    f"Episode {episode}/{episodes}"
                    f"(total step num: {total_num_steps} | average step reward: {np.mean(self.buffer.rewards[self.buffer.pre_batch_index]):.4f})",
                )
                self.log_train(train_infos, total_num_steps)
            progress_bar.update(1)

            if self.all_args.use_eval and episode % self.all_args.eval_interval == 0:
                self.eval(total_num_steps)

        # print("buffer: ", self.buffer.value_preds)

    @torch.no_grad()
    def collect(self, step):
        behaviour_data = self.agent.infer_for_rollout(
            np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, step])
        )

        actions, action_tokens, values, log_probs = behaviour_data

        # [self.envs, agents]
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))

        return values, actions, action_tokens, log_probs

    def insert(self, data):
        obs, rewards, dones, values, actions, action_tokens, log_probs = data

        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents), dtype=np.float32
        )

        if self.algo == "APPO":
            self.buffer.insert_appo(
                obs, actions, values, rewards, masks, action_tokens, log_probs
            )
        elif self.algo == "TPPO":
            self.buffer.insert_tppo(
                obs, actions, values, rewards, masks, action_tokens, log_probs
            )
        else:
            raise NotImplementedError

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        next_values = self.agent.get_next_values(
            np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, -1])
        )
        next_values = np.array(np.split(next_values, self.n_rollout_threads))
        if self.algo == "APPO":
            self.buffer.batch_process_appo(next_values)
        elif self.algo == "TPPO":
            self.buffer.batch_process_tppo(next_values)
        else:
            raise NotImplementedError

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    @staticmethod
    def worker_train(rank, world_size, args, agent, agent_num, buffer, child_conn):
        setup(rank, world_size)
        print(f"Creating trainer on process {rank} with world size {world_size}...")
        if args.algorithm_name == "APPO":
            trainer = APPOTrainer(args, agent, agent_num, rank)
        elif args.algorithm_name == "TPPO":
            trainer = TPPOTrainer(args, agent, agent_num, rank)
        else:
            raise NotImplementedError

        train_infos = trainer.train(buffer)
        if rank == 0:
            child_conn.send(train_infos)

    def train(self):
        world_size = torch.cuda.device_count()
        parent_conn, child_conn = mp.Pipe()
        print(f"starting {world_size} processes for training...")
        mp.spawn(
            self.worker_train,
            nprocs=world_size,
            args=(
                world_size,
                self.all_args,
                self.agent,
                self.num_agents,
                self.buffer,
                child_conn,
            ),
            join=True,
        )
        while parent_conn.poll():
            message = parent_conn.recv()
            train_infos = message
        del message

        return train_infos

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []

        eval_obs = self.eval_envs.reset()
        while True:
            eval_actions, _ = self.agent.get_actions(np.concatenate(eval_obs))
            eval_actions = np.array(np.split(eval_actions, self.n_eval_rollout_threads))
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions
            )

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(eval_rewards[eval_i])

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {"eval_average_episode_rewards": eval_episode_rewards}
                print("total_num_steps: ", total_num_steps)
                print("eval reward is {}.".format(np.mean(eval_episode_rewards)))
                self.log_eval(eval_env_infos, total_num_steps)
                break

    def log_eval(self, eval_infos, total_num_steps):
        for k, v in eval_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.agent.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.agent.restore(model_dir)
