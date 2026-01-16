#!/bin/bash
# Training Launch Script for 947-Sample Dataset
# Improved checkpoint strategy to avoid overfitting

export HF_TOKEN=YOUR_HF_TOKEN
export WANDB_DISABLED=true

echo "🚀 Launching Orpheus Training (947 Samples)"
echo "Dataset: kailasa-ngpt/SPH_Audio_2019_60_Secs_947_Samples"
echo "Target: 1000 steps (will test checkpoints incrementally)"
echo ""

# Kill any existing training processes
pkill -f train_orpheus_final.py || true
pkill -f monitor_upload.sh || true

# Launch training
nohup /workspace/venv/bin/python /workspace/train_orpheus_final.py \
    --output_dir /workspace/orpheus_947_final \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_steps 1000 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 10 \
    --bf16 True \
    --optim "adamw_8bit" \
    --save_strategy "steps" \
    > /workspace/logs/training_947.log 2>&1 &

PID=$!
echo "Training PID: $PID"
echo $PID > /workspace/training.pid

echo ""
echo "✅ Training started!"
echo "Monitor: tail -f /workspace/logs/training_947.log"
echo "Or use: /workspace/venv/bin/python /workspace/monitor.py"
