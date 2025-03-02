export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

python train_math.py \
        --seed 10 \
        --env_name math_env \
        --algorithm_name TPPO \
        --experiment_name math_debug \
        --dataset_name math \
        --flag train \
        --num_mini_batch 1 \
        --ppo_epoch 1 \
        --lr 1e-8 \
        --critic_lr 5e-7 \
        --dataset_path ../envs/math/data/merged_precalculus_train.json \
        --model_name_or_path /ext0/hcchai/codemate/Qwen2.5-Coder-3B-Instruct/ \
        --n_agents 2 \
        --agent_iteration_interval 3000 \
        --profile_path profiles/math_dual.json \
        --n_rollout_threads 1 \
        --episode_length 3 \
        --gradient_cp_steps 4 \
        --context_window 2048 \
        --max_new_tokens 512 \
        --save_interval 1000