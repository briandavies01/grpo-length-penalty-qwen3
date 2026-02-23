#!/bin/bash
# Box 2 (38.80.152.146) — beta=0.001 (KL penalty), 3 sequential runs
# Start with: nohup bash run_overnight_box2.sh > overnight_box2.log 2>&1 &

set -e
cd ~/grpo-length-penalty-qwen3

echo "=== Box 2: Starting overnight runs at $(date) ==="

# Run 4: lambda=1.0, standard mode, KL=0.001
echo ">>> Run 1/3: lambda1.0_KL0.001 starting at $(date)"
python grpo_train.py \
    --model_name Qwen/Qwen3-1.7B \
    --lambda_length 1.0 \
    --beta 0.001 \
    --max_steps 150 \
    --save_steps 25 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --run_name lambda1.0_KL0.001 \
    --wandb_project grpo-length-penalty
echo ">>> Run 1/3: lambda1.0_KL0.001 finished at $(date)"

# Run 5: lambda=2.0, correct-only, KL=0.001
echo ">>> Run 2/3: lambda2.0_correctonly_KL0.001 starting at $(date)"
python grpo_train.py \
    --model_name Qwen/Qwen3-1.7B \
    --lambda_length 2.0 \
    --length_penalty_on_correct_only \
    --beta 0.001 \
    --max_steps 150 \
    --save_steps 25 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --run_name lambda2.0_correctonly_KL0.001 \
    --wandb_project grpo-length-penalty
echo ">>> Run 2/3: lambda2.0_correctonly_KL0.001 finished at $(date)"

# Run 6: lambda=1.0, correct-only, KL=0.001
echo ">>> Run 3/3: lambda1.0_correctonly_KL0.001 starting at $(date)"
python grpo_train.py \
    --model_name Qwen/Qwen3-1.7B \
    --lambda_length 1.0 \
    --length_penalty_on_correct_only \
    --beta 0.001 \
    --max_steps 150 \
    --save_steps 25 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --run_name lambda1.0_correctonly_KL0.001 \
    --wandb_project grpo-length-penalty
echo ">>> Run 3/3: lambda1.0_correctonly_KL0.001 finished at $(date)"

echo "=== Box 2: All runs completed at $(date) ==="
