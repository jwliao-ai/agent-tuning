export CUDA_VISIBLE_DEVICES=5
echo $CUDA_VISIBLE_DEVICES

python train_fctncalling.py \
        --seed 10 \
        --env_name fctncalling_env \
        --algorithm_name POAD \
        --experiment_name multi_debug \
        --num_mini_batch 4 \
        --ppo_epoch 1 \
        --lr 1e-7 \
        --critic_lr 5e-6 \
        --dataset_path /home/ljw/data/gorilla/processed_bfcl/BFCL_v3_multi_turn_base_merged_complete.json \
        --model_name_or_path /ext0/hcchai/codemate/Qwen2.5-Coder-7B-Instruct/ \
        --n_agents 2 \
        --profile_path dual_agent_profiles.json \
        --n_rollout_threads 1 \
        --episode_length 12 \
        --gradient_cp_steps 6 \
        --max_new_tokens 256 \
        --save_interval 1000