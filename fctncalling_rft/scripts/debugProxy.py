import sys
import runpy
import os

os.chdir('/home/ljw/codes/MadeAgents/fctncalling_rft/fctncalling_rft/scripts')
# args = 'python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/2022-06-16ball.matpkl --time 1 9 17 23 27'
# args = 'python -m lilab.metric_seg.s3_cocopkl_vs_cocopkl --gt_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te1/intense_pannel.cocopkl --pred_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te2/intense_pannel.cocopkl '

args = """python train_fctncalling.py --seed 10 --env_name fctncalling_env --algorithm_name APPO --experiment_name default --num_mini_batch 2 --ppo_epoch 1 --lr 1e-7 --critic_lr 5e-6 --dataset_path /home/ljw/data/gorilla/processed_bfcl/BFCL_v3_multi_turn_base_merged.json --model_name_or_path /home/ljw/models/Qwen2.5-0.5B-Instruct --n_rollout_threads 2 --gradient_cp_steps 1 --max_new_tokens 128 --save_interval 30"""

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