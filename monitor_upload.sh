#!/bin/bash
PID=$1
echo "Monitoring PID $PID..."
tail --pid=$PID -f /dev/null
echo "Process $PID finished. Starting upload..."
export HF_TOKEN=YOUR_HF_TOKEN
/workspace/venv/bin/python upload_model.py > upload_final.log 2>&1
echo "Upload script finished."
