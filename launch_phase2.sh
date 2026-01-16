#!/bin/bash
export HF_TOKEN=YOUR_HF_TOKEN
export WANDB_DISABLED=true

# Kill existing monitors
pkill -f monitor_upload.sh || true

echo "Launching Training Phase 2 (Steps 1-100)..."

nohup /workspace/venv/bin/python /workspace/train_orpheus_final.py \
    --output_dir /workspace/orpheus_fft_final \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_steps 100 \
    --warmup_steps 5 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 3 \
    --bf16 True \
    --optim "adamw_8bit" \
    --save_strategy "steps" \
    > /workspace/training_phase2.log 2>&1 &

PID=$!
echo "Training PID: $PID"
echo $PID > /workspace/training.pid

# Launch Monitor
nohup bash monitor_upload.sh $PID > /workspace/monitor_p2.log 2>&1 &
echo "Monitor Launched."
