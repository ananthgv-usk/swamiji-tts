#!/bin/bash
# Download Orpheus model (Llama-3 based TTS)
echo "Downloading unsloth/orpheus-3b-0.1-ft..."

# Add local bin to PATH
export PATH=$PATH:/Users/vember/Library/Python/3.9/bin

huggingface-cli download unsloth/orpheus-3b-0.1-ft \
    --local-dir ./model \
    --local-dir-use-symlinks False

echo "Download complete."
