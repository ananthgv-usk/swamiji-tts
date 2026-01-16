#!/bin/bash
set -e

# Orpheus Incremental Training Loop (A100 Verified Config)

OUTPUT_DIR="/workspace/orpheus_fft_final"
MAX_TOTAL_STEPS=1000
STEP_SIZE=100

echo "🎹 ORPHEUS SAFE TRAINING LOOP 🎹"
echo "Target: $MAX_TOTAL_STEPS steps"
echo "Increment: $STEP_SIZE steps"
echo "-----------------------------------"

# Ensure VENV
source /workspace/venv/bin/activate

for (( target=$STEP_SIZE; target<=$MAX_TOTAL_STEPS; target+=$STEP_SIZE )); do
    echo "🚀 Launching Training Run -> Target Step: $target"
    
    python train_orpheus_final.py \
        --output_dir "$OUTPUT_DIR" \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-4 \
        --max_steps $target \
        --warmup_steps 5 \
        --logging_steps 1 \
        --save_steps 100 \
        --save_total_limit 3 \
        --bf16 True \
        --optim "adamw_8bit" \
        --report_to "none" \
        --save_strategy "steps"
    
    echo "✅ Step $target Reached."
    echo "⏸️  TRAINING PAUSED FOR VERIFICATION ⏸️"
    echo "Run './run_inference.sh' in another terminal to test quality."
    echo "Press ENTER to continue to next step (or Ctrl+C to abort)..."
    read -r
done

echo "🎉 Training Complete!"
