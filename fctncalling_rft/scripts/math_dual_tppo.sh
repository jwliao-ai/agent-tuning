export CUDA_VISIBLE_DEVICES=4
echo $CUDA_VISIBLE_DEVICES

python train_math.py \
        --seed 10 \
        --env_name math_env \
        --algorithm_name TPPO \
        --experiment_name math_debug \
        --dataset_name math \
        --flag train \
        --num_mini_batch 4 \
        --ppo_epoch 1 \
        --lr 1e-8 \
        --critic_lr 5e-7 \
        --dataset_path /home/ljw/codes/MadeAgents/fctncalling_rft/fctncalling_rft/envs/math/data/merged_precalculus_train.json \
        --model_name_or_path /ext0/hcchai/codemate/Qwen2.5-Coder-7B-Instruct/ \
        --n_agents 2 \
        --profile_path profiles/math_dual.json \
        --n_rollout_threads 1 \
        --episode_length 3 \
        --gradient_cp_steps 4 \
        --context_window 1024 \
        --max_new_tokens 256 \
        --save_interval 1000