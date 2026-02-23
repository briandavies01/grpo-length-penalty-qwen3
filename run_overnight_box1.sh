#!/bin/bash
# Box 1 (198.13.252.68) — beta=0.0 (no KL), 3 sequential runs
# Start with: nohup bash run_overnight_box1.sh > overnight_box1.log 2>&1 &

set -e
cd ~/grpo-length-penalty-qwen3

echo "=== Box 1: Starting overnight runs at $(date) ==="

# Run 1: lambda=1.0, standard mode, no KL
echo ">>> Run 1/3: lambda1.0_noKL starting at $(date)"
python grpo_train.py \
    --model_name Qwen/Qwen3-1.7B \
    --lambda_length 1.0 \
    --beta 0.0 \
    --max_steps 150 \
    --save_steps 25 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --run_name lambda1.0_noKL \
    --wandb_project grpo-length-penalty
echo ">>> Run 1/3: lambda1.0_noKL finished at $(date)"

# Run 2: lambda=2.0, correct-only, no KL
echo ">>> Run 2/3: lambda2.0_correctonly_noKL starting at $(date)"
python grpo_train.py \
    --model_name Qwen/Qwen3-1.7B \
    --lambda_length 2.0 \
    --length_penalty_on_correct_only \
    --beta 0.0 \
    --max_steps 150 \
    --save_steps 25 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --run_name lambda2.0_correctonly_noKL \
    --wandb_project grpo-length-penalty
echo ">>> Run 2/3: lambda2.0_correctonly_noKL finished at $(date)"

# Run 3: lambda=1.0, correct-only, no KL
echo ">>> Run 3/3: lambda1.0_correctonly_noKL starting at $(date)"
python grpo_train.py \
    --model_name Qwen/Qwen3-1.7B \
    --lambda_length 1.0 \
    --length_penalty_on_correct_only \
    --beta 0.0 \
    --max_steps 150 \
    --save_steps 25 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --run_name lambda1.0_correctonly_noKL \
    --wandb_project grpo-length-penalty
echo ">>> Run 3/3: lambda1.0_correctonly_noKL finished at $(date)"

echo "=== Box 1: All runs completed at $(date) ==="
