#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

sys.path.append("../../")
from fctncalling_rft.config import get_config
from fctncalling_rft.envs.fctncalling.fctncalling_env import FctnCallingEnv
from fctncalling_rft.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from fctncalling_rft.runner.shared.fctncalling_runner import FctnCallingRunner as Runner


def make_train_env(all_args):

    def get_env_fn(rank):
        def init_env():
            env = FctnCallingEnv(
                rank=rank,
                model_name=all_args.model_name,
                num_agents=all_args.n_agents,
                dataset_path=all_args.dataset_path,
            )
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return ShareDummyVecEnv(
        [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
    )


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = FctnCallingEnv(
                rank=rank, 
                model_name=all_args.model_name, 
                num_agents=all_args.n_agents, 
                dataset_path=all_args.dataset_path
            )
            env.seed(all_args.seed + rank * 5000)
            return env
        return init_env
    return ShareDummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--env_name", type=str, default="fctncalling_env", help="Which env to run on")
    parser.add_argument("--dataset_name", type=str, default="xlam", help="Which dataset to test on")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to dataset")
    parser.add_argument("--flag", type=str, default="train", help="flag to distinguish different runs")
    parser.add_argument("--model_type", type=str, default="llama",required=False, help="the model architecture")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Which model to use")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="max_new_tokens")
    parser.add_argument("--vacab_size", type=int, default=32016)
    parser.add_argument("--gradient_cp_steps", type=int, default=1)

    all_args = parser.parse_known_args(args)[0]
    all_args.model_name = Path(all_args.model_name_or_path).parts[-1]

    return all_args


def build_run_dir(all_args):
    run_dir = (
        Path(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            + "/scripts/results"
        )
        / all_args.experiment_name
        / all_args.model_type
        / all_args.dataset_name
        / all_args.algorithm_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        curr_run = "run_1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("_")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run_1"
        else:
            curr_run = "run_%i" % (max(exst_run_nums) + 1)
    curr_run += f"_ppoepoch{all_args.ppo_epoch}_lr{all_args.lr}_minibatch{all_args.num_mini_batch}_criticlr{all_args.critic_lr}_valuelosscoef{all_args.value_loss_coef}_ent{all_args.entropy_coef}_seed{all_args.seed}"
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    return run_dir


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.log_interval = 1
    all_args.n_agents = 1

    run_dir = build_run_dir(all_args)
    print(f"------ run dir: {run_dir} ------")

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    # eval_envs = make_eval_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": envs.n_agents if envs is not None else 1,
        "run_dir": run_dir,
    }

    runner = Runner(config)
    runner.run()

    # post process
    if envs is not None:
        envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
