#!/bin/bash
# Setup RunPod for Orpheus Project

echo "Updating system..."
apt-get update && apt-get install -y git wget

echo "Installing Python dependencies..."
# Unsloth dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Project specific
pip install soundfile librosa datasets huggingface_hub snac

echo "Login to Hugging Face..."
# Expects HF_TOKEN env var or manual login
export HF_TOKEN=YOUR_HF_TOKEN
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN
else
    echo "Please run 'huggingface-cli login' manually if token is not set."
fi

echo "Check for existing model..."
if [ -d "/workspace/model" ]; then
    echo "Model found in /workspace/model. Skipping download."
else
    echo "Downloading unsloth/orpheus-3b-0.1-ft to /workspace/model..."
    huggingface-cli download unsloth/orpheus-3b-0.1-ft --local-dir /workspace/model --local-dir-use-symlinks False
fi

echo "Setup complete."
