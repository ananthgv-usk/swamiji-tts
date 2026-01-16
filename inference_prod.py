#!/usr/bin/env python3
"""
Orpheus Production Inference Script
Usage: python inference_prod.py "Your text here"
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import os
import sys
import numpy as np

import argparse
import glob

# Configuration
BASE_MODEL_PATH = "/workspace/orpheus_fft_final" 
OUTPUT_FILE = "generated_audio.wav"
SNAC_MODEL = "hubertsiuzdak/snac_32khz"
AUDIO_TOKEN_START = 128256

def get_latest_checkpoint(base_path):
    checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

# CLI Parse
parser = argparse.ArgumentParser(description="Run Orpheus Inference on a specific checkpoint")
parser.add_argument("--prompt", type=str, default="Unclutching means dropping your unconscious grip on thoughts, emotions, identities, and stories.", help="Text to speak")
parser.add_argument("--checkpoint", type=str, help="Specific checkpoint folder name (e.g. checkpoint-500)")
args = parser.parse_args()

prompt = args.prompt

if args.checkpoint:
    MODEL_PATH = os.path.join(BASE_MODEL_PATH, args.checkpoint)
else:
    # Auto-detect latest
    latest = get_latest_checkpoint(BASE_MODEL_PATH)
    if latest:
        print(f"Auto-detected latest checkpoint: {latest}")
        MODEL_PATH = latest
    else:
        print(f"No checkpoints found in {BASE_MODEL_PATH}. Trying base path.")
        MODEL_PATH = BASE_MODEL_PATH

print(f"Prompt: {prompt}")
print(f"Loading model from {MODEL_PATH}...")

# Clear VRAM before loading to be safe during training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("If training just started, the first checkpoint might not be saved yet.")
    exit(1)

print("Loading SNAC...")
try:
    snac_model = SNAC.from_pretrained(SNAC_MODEL).to("cpu")
except Exception as e:
    print(f"Error loading SNAC: {e}")
    exit(1)

model.eval()

def redistribute_codes_robust(code_list):
    # Match the 15-token preprocessing pattern
    # L0(1) + L1(2) + L2(4) + L3(8) = 15
    token_group_size = 15
    
    # Ensure num_frames is a multiple of 4 to avoid SNAC dimension mismatch
    # (SNAC layers have 2x, 4x, 8x upsampling, prime numbers like 149 break it)
    total_frames = len(code_list) // token_group_size
    num_frames = (total_frames // 4) * 4 
    
    truncated_len = num_frames * token_group_size
    code_list = code_list[:truncated_len]
    
    layer_1 = []
    layer_2 = []
    layer_3 = []
    layer_4 = []
    
    print(f"Redistributing {len(code_list)} codes into {num_frames} frames (15 tokens/frame)...")
    
    for i in range(num_frames):
        try:
            base = i * token_group_size
            
            # Reconstruction Logic (Reversing preprocess_correct.py)
            # Offsets: L0=0, L1=4096, L2=8192, L3=12288
            
            # --- L0 (1 token) ---
            # Index 0
            l0_val = code_list[base] 
            layer_1.append(l0_val)

            # --- Branch A ---
            # Index 1: L1[0] (Offset 4096)
            l1_a = code_list[base+1] - 4096
            layer_2.append(l1_a)
            
            # Index 2: L2[0] (Offset 8192)
            l2_a = code_list[base+2] - 8192
            layer_3.append(l2_a)
            
            # Index 3,4: L3[0], L3[1] (Offset 12288)
            l3_a = code_list[base+3] - 12288
            l3_b = code_list[base+4] - 12288
            layer_4.append(l3_a)
            layer_4.append(l3_b)
            
            # Index 5: L2[1] (Offset 8192)
            l2_b = code_list[base+5] - 8192
            layer_3.append(l2_b)
            
            # Index 6,7: L3[2], L3[3]
            l3_c = code_list[base+6] - 12288
            l3_d = code_list[base+7] - 12288
            layer_4.append(l3_c)
            layer_4.append(l3_d)
            
            # --- Branch B ---
            # Index 8: L1[1] (Offset 4096)
            l1_b = code_list[base+8] - 4096
            layer_2.append(l1_b)
            
            # Index 9: L2[2] (Offset 8192)
            l2_c = code_list[base+9] - 8192
            layer_3.append(l2_c)
            
            # Index 10,11: L3[4], L3[5]
            l3_e = code_list[base+10] - 12288
            l3_f = code_list[base+11] - 12288
            layer_4.append(l3_e)
            layer_4.append(l3_f)
            
            # Index 12: L2[3] (Offset 8192)
            l2_d = code_list[base+12] - 8192
            layer_3.append(l2_d)
            
            # Index 13,14: L3[6], L3[7]
            l3_g = code_list[base+13] - 12288
            l3_h = code_list[base+14] - 12288
            layer_4.append(l3_g)
            layer_4.append(l3_h)

        except Exception as e:
            print(f"Error at frame {i}: {e}")
            break

    # Helper to clean and clamp
    def to_tensor(lst):
        # Clamp to [0, 4095] just in case
        clamped = [max(0, min(x, 4095)) for x in lst]
        return torch.tensor(clamped).unsqueeze(0).long().to(device)

    codes = [
        to_tensor(layer_1),
        to_tensor(layer_2),
        to_tensor(layer_3),
        to_tensor(layer_4)
    ]
    
    try:
        # Move SNAC to same device
        snac_model.to(device) 
        with torch.no_grad():
            audio_hat = snac_model.decode(codes)
    except (IndexError, RuntimeError) as e:
        print(f"Decode failed ({e})...")
        # Retry with CPU fallback or padding if needed (unlikely with correct logic)
        audio_hat = None
        
    return audio_hat

print(f"\n--- Generating ---")

# Format prompt  
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
start_token = torch.tensor([[128259]], dtype=torch.int64) 
end_tokens = torch.tensor([[128009, 128260, 128257]], dtype=torch.int64) 
modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

# Padding logic for input (optional but good practice)
padding = torch.full((1, 1), 128263, dtype=torch.int64) 
padded_tensor = torch.cat([padding, modified_input_ids], dim=1)
attention_mask = torch.cat([torch.zeros((1, 1), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)

input_ids = padded_tensor.to(device)
attention_mask = attention_mask.to(device)

print("Generating audio tokens...")
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,      
        top_p=0.9,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=128258,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

print(f"Generated {generated_ids.shape[1]} total tokens")

token_to_find = 128257
token_to_remove = 128258
generated_ids_cpu = generated_ids.cpu()
token_indices = (generated_ids_cpu == token_to_find).nonzero(as_tuple=True)

if len(token_indices[1]) > 0:
    last_occurrence_idx = token_indices[1][-1].item()
    print(f"✅ Found speech start token at idx {last_occurrence_idx}")
    
    raw_codes = generated_ids_cpu[:, last_occurrence_idx+1:].tolist()[0]
    subtracted_codes = []
    
    for c in raw_codes:
        if c == 128258: # End of Speech
            break
        if c >= AUDIO_TOKEN_START:
            subtracted_codes.append(c - AUDIO_TOKEN_START)
        else:
            subtracted_codes.append(0)

    print(f"Extracted {len(subtracted_codes)} codes")
    
    # Padding Logc (224)
    pad_interval = 224 
    remainder = len(subtracted_codes) % pad_interval
    if remainder != 0:
        padding_needed = pad_interval - remainder
        print(f"Padding with {padding_needed} zeros to match window {pad_interval}...")
        subtracted_codes.extend([0] * padding_needed)

    if len(subtracted_codes) > 0:
        try:
            audio = redistribute_codes_robust(subtracted_codes)
            audio_np = audio.detach().squeeze().cpu().numpy()
            sf.write(OUTPUT_FILE, audio_np, 32000)
            print(f"\n✅ Final Audio Saved to {OUTPUT_FILE} ({len(audio_np)/32000:.2f}s)")
            print(f"Run 'python -m http.server 8000' to download it via browser (if port open) or use web terminal download.")
        except Exception as e:
             print(f"Error decoding: {e}")
else:
    print("❌ Start token not found!")
