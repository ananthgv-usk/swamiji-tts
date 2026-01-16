#!/usr/bin/env python3
"""
Correct SNAC preprocessing using raw file access (Nuclear Option)
Bypasses datasets library entirely for reading audio to avoid torchcodec issues
"""
import torch
import numpy as np
import librosa
import soundfile as sf
import os
import glob
import pandas as pd
import json
import io
from snac import SNAC
from huggingface_hub import snapshot_download
from datasets import Dataset
from tqdm import tqdm

# Configuration
DATASET_REPO = "kailasa-ngpt/SPH_Audio_2019_60_Secs_947_Samples"
SNAC_MODEL = "hubertsiuzdak/snac_32khz"
AUDIO_TOKEN_START = 128256

print("Loading SNAC model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
snac_model = SNAC.from_pretrained(SNAC_MODEL).to(device)
snac_model.eval()

print("Downloading dataset snapshot...")
# Download everything to a local folder
local_dir = "/workspace/sph45_raw"
snapshot_download(repo_id=DATASET_REPO, local_dir=local_dir, repo_type="dataset")

print(f"Dataset downloaded to {local_dir}")

def tokenise_audio(audio_array, orig_sr):
    """
    Load audio from array and convert to SNAC codes
    """
    try:
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Resample if needed using librosa (robust)
        if orig_sr != 32000:
            # librosa.resample expects numpy array
            # We must transpose if it was (N, C) but we made it (N,) so it's fine
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=32000)
        
        # SNAC expects (B, 1, T) = (1, 1, T)
        waveform = torch.from_numpy(audio_array).unsqueeze(0).unsqueeze(0)
        
        # Check length - SNAC needs some divisible length potentially?
        # The model usually handles it but let's be safe
        
        waveform = waveform.to(dtype=torch.float32, device=device)
        
        # Generate codes
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        
        # Interleave pattern
        all_codes = []
        # codes[0] is (B, T) in this version of SNAC
        # Debug print once
        # print(f"Codes shape: {codes[0].shape}") 
        
        seq_len = codes[0].shape[1] 
        
        for i in range(seq_len):
            # Layer 0 (1 token)
            all_codes.append(codes[0][0][i].item() + AUDIO_TOKEN_START)
            
            # Branch A (L1[0] -> children)
            all_codes.append(codes[1][0][2*i].item() + AUDIO_TOKEN_START + 4096)
            all_codes.append(codes[2][0][4*i].item() + AUDIO_TOKEN_START + 8192)
            all_codes.append(codes[3][0][8*i].item() + AUDIO_TOKEN_START + 12288)
            all_codes.append(codes[3][0][8*i+1].item() + AUDIO_TOKEN_START + 12288)
            
            all_codes.append(codes[2][0][4*i+1].item() + AUDIO_TOKEN_START + 8192)
            all_codes.append(codes[3][0][8*i+2].item() + AUDIO_TOKEN_START + 12288)
            all_codes.append(codes[3][0][8*i+3].item() + AUDIO_TOKEN_START + 12288)
            
            # Branch B (L1[1] -> children)
            all_codes.append(codes[1][0][2*i+1].item() + AUDIO_TOKEN_START + 4096)
            all_codes.append(codes[2][0][4*i+2].item() + AUDIO_TOKEN_START + 8192)
            all_codes.append(codes[3][0][8*i+4].item() + AUDIO_TOKEN_START + 12288)
            all_codes.append(codes[3][0][8*i+5].item() + AUDIO_TOKEN_START + 12288)
            
            all_codes.append(codes[2][0][4*i+3].item() + AUDIO_TOKEN_START + 8192)
            all_codes.append(codes[3][0][8*i+6].item() + AUDIO_TOKEN_START + 12288)
            all_codes.append(codes[3][0][8*i+7].item() + AUDIO_TOKEN_START + 12288)
        
        return all_codes
    except Exception as e:
        print(f"CRITICAL ERROR processing audio: {e}")
        import traceback
        traceback.print_exc()
        return None

# Find all Parquet files
parquet_files = glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True)
print(f"Found {len(parquet_files)} parquet files")

processed_data = []

for pfile in parquet_files:
    print(f"Processing {pfile}...")
    df = pd.read_parquet(pfile)
    
for pfile in parquet_files:
    print(f"Processing {pfile}...")
    df = pd.read_parquet(pfile)
    
    # Iterate with index
    print(f"Rows: {len(df)}")
    
    for i in tqdm(range(len(df))):
        try:
            row = df.iloc[i]
            
            # Audio column is typically a dict/struct
            audio_data = row['audio']
            text = row.get('text', row.get('transcription', ''))

            # Extract bytes
            audio_bytes = None
            if isinstance(audio_data, dict):
                audio_bytes = audio_data.get('bytes')
            elif isinstance(audio_data, bytes):
                audio_bytes = audio_data
            
            if audio_bytes:
                # Read from bytes
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Tokenize
                codes = tokenise_audio(audio_array, sr)
                
                if codes and len(codes) > 0:
                    processed_data.append({
                        "text": text,
                        "codes_list": codes,
                        "path": "embedded"
                    })
                    if len(processed_data) % 10 == 0:
                        print(f"Processed {len(processed_data)}")
        except Exception as e:
             print(f"Loop error: {e}")
             import traceback
             traceback.print_exc()

# Create final dataset
print("Creating HF Dataset...")
dataset = Dataset.from_list(processed_data)

print(f"Successfully preprocessed {len(dataset)} examples")

if len(dataset) > 0:
    # Save output
    output_path = "/workspace/preprocessed_dataset_947"
    print(f"\nSaving to {output_path}...")
    dataset.save_to_disk(output_path)
    print("✅ Preprocessing complete!")
else:
    print("❌ No examples were successfully preprocessed")
