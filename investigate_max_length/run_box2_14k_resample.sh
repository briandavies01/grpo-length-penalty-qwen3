#!/bin/bash
# Box 2: maxlen_14k with resampling of truncated completions
# 4 prompts x 8 rollouts = 32 completions/step (batch=4, same as box1 10k)
# Start with: nohup bash investigate_max_length/run_box2_14k_resample.sh > investigate_max_length/box2_14k_resample.log 2>&1 &

set -e
cd ~/grpo-length-penalty-qwen3
export WANDB_API_KEY="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"

echo "=== Box 2: 14k resample starting at $(date) ==="

python grpo_train.py \
    --model_name Qwen/Qwen3-1.7B \
    --lambda_length 0.0 \
    --beta 0.005 \
    --max_steps 200 \
    --save_steps 50 \
    --save_total_limit 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --learning_rate 5e-5 \
    --temperature 1.0 \
    --seed 42 \
    --wandb_project investigate-max-length \
    --max_completion_length 14000 \
    --resample_truncated \
    --max_resample_retries 3 \
    --run_name maxlen_14k_resample

echo "=== Box 2: 14k resample finished at $(date) ==="
