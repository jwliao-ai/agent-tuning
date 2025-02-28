import sys
import runpy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.system("echo $CUDA_VISIBLE_DEVICES")
os.chdir('/home/ljw/codes/MadeAgents/fctncalling_rft/fctncalling_rft/scripts')
# args = 'python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/2022-06-16ball.matpkl --time 1 9 17 23 27'
# args = 'python -m lilab.metric_seg.s3_cocopkl_vs_cocopkl --gt_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te1/intense_pannel.cocopkl --pred_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te2/intense_pannel.cocopkl '

args = """python train_math.py \
        --seed 10 \
        --env_name math_env \
        --algorithm_name TPPO \
        --experiment_name math_debug \
        --dataset_name math \
        --flag train \
        --num_mini_batch 4 \
        --ppo_epoch 1 \
        --lr 1e-7 \
        --critic_lr 5e-6 \
        --dataset_path /home/ljw/codes/MadeAgents/fctncalling_rft/fctncalling_rft/envs/math/data/merged_precalculus_train.json \
        --model_name_or_path /ext0/hcchai/codemate/Qwen2.5-Coder-7B-Instruct/ \
        --n_agents 1 \
        --profile_path profiles/math_single.json \
        --n_rollout_threads 1 \
        --episode_length 12 \
        --gradient_cp_steps 6 \
        --context_window 1024 \
        --max_new_tokens 256 \
        --save_interval 1000"""

args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')