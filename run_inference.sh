#!/bin/bash
# Wrapper to run inference easily
echo "Checking for checkpoints..."
CHECKPOINT=$(ls -1d /workspace/orpheus_fft_final/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ No checkpoints found yet. Training is still in early stages."
    echo "Wait for the first checkpoint (usually step 500 or created by save strategy)."
    exit 1
fi

CKPT_NAME=$(basename "$CHECKPOINT")
echo "✅ Found latest checkpoint: $CKPT_NAME"

PROMPT=${1:-"Unclutching means dropping your unconscious grip on thoughts, emotions, identities, and stories."}

echo "Running inference on $CKPT_NAME..."
echo "Prompt: $PROMPT"

/workspace/venv/bin/python /workspace/inference_prod.py --checkpoint "$CKPT_NAME" --prompt "$PROMPT"
