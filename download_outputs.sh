#!/bin/bash
# Script to download production artifacts from RunPod
POD_HOST="190.111.198.202"
POD_PORT="13114"
USER="root"
SSH_KEY="~/.ssh/id_ed25519"
REMOTE_PATH="/workspace/orpheus_finetune_outputs"
LOCAL_PATH="./orpheus_finetune_outputs"

mkdir -p "$LOCAL_PATH"

echo "Checking for files in $REMOTE_PATH on Pod..."

# Check if directory is not empty
    echo "Files found. Starting Download..."
    scp -P $POD_PORT -i $SSH_KEY -r $USER@$POD_HOST:$REMOTE_PATH ./
    echo "Download Successful to $LOCAL_PATH"
else
    echo "No files found yet. Training likely still in progress."
fi
