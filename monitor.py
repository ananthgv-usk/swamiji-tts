#!/usr/bin/env python3
import os
import re
import time
import subprocess
import glob

LOG_FILE = "/workspace/logs/training_947.log"
MODEL_DIR = "/workspace/orpheus_947_final"

def get_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(MODEL_DIR, "checkpoint-*"))
    if not checkpoints:
        return "None"
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.basename(checkpoints[-1])

def get_vram():
    try:
        result = subprocess.check_output("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", shell=True).decode()
        used, total = result.strip().split(", ")
        return f"{used}MiB / {total}MiB"
    except:
        return "N/A"

def parse_log():
    if not os.path.exists(LOG_FILE):
        return "Log file not found."
    
    # Read last 50 lines efficiently
    try:
        lines = subprocess.check_output(["tail", "-n", "50", LOG_FILE]).decode().split("\n")
    except:
        return "Error reading log."

    # Look for progress lines e.g. "33/1000 [01:43<51:08,  3.17s/it]"
    # Look for loss log e.g. "{'loss': 2.45, 'learning_rate': ...}"
    
    last_step = "Unknown"
    last_loss = "Unknown"
    loss_history = []
    
    for line in lines:
        # Progress
        if "%" in line and "/" in line and "[" in line:
            # Simple heuristic matching
            parts = line.strip().split()
            if len(parts) > 2:
                last_step = line.strip()
        
        # Loss (JSON-like)
        if "{'loss':" in line:
            loss_match = re.search(r"'loss':\s*([0-9.]+)", line)
            if loss_match:
                last_loss = loss_match.group(1)
                loss_history.append(float(last_loss))

    return {
        "step": last_step,
        "loss": last_loss,
        "loss_trend": loss_history[-5:] if loss_history else []
    }

def clear_screen():
    print("\033[H\033[J", end="")

def main():
    while True:
        clear_screen()
        data = parse_log()
        vram = get_vram()
        ckpt = get_latest_checkpoint()
        
        print(f"==================================================")
        print(f"   🎹 ORPHEUS TRAINING MONITOR (A100) 🎹")
        print(f"==================================================")
        print(f"  VRAM Usage : {vram}")
        print(f"  Latest Ckpt: {ckpt}")
        print(f"--------------------------------------------------")
        
        if isinstance(data, str):
            print(f"  Status: {data}")
        else:
            print(f"  Progress   : {data['step']}")
            print(f"  Current Loss: {data['loss']}")
            
            if data['loss_trend']:
                trend = " -> ".join([f"{x:.4f}" for x in data['loss_trend']])
                print(f"  Recent Loss: {trend}")
        
        print(f"==================================================")
        print(f"Press Ctrl+C to exit monitor (Training continues)")
        time.sleep(2)

if __name__ == "__main__":
    main()
