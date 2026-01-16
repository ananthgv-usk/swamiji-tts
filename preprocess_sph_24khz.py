#!/usr/bin/env python3
"""
Preprocess SPH_Audio_2019_60_Secs_947_Samples dataset for Orpheus TTS
Uses 24kHz SNAC with 7-token pattern (matching reference implementation)
"""
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import os
from snac import SNAC
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Configuration - VERIFIED from reference implementation
DATASET_REPO = "kailasa-ngpt/SPH_Audio_2019_60_Secs_947_Samples"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"  # 24kHz, NOT 32kHz
TARGET_SAMPLE_RATE = 24000
AUDIO_TOKENS_START = 128266  # tokenizer_length (128256) + 10

print("=" * 60)
print("Orpheus TTS Preprocessing - 24kHz SNAC Configuration")
print("=" * 60)
print(f"Dataset: {DATASET_REPO}")
print(f"SNAC Model: {SNAC_MODEL}")
print(f"Target Sample Rate: {TARGET_SAMPLE_RATE} Hz")
print(f"Audio Tokens Start: {AUDIO_TOKENS_START}")
print(f"Token Pattern: 7 tokens per frame")
print("=" * 60)

# Load SNAC model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading SNAC model on {device}...")
snac_model = SNAC.from_pretrained(SNAC_MODEL).to(device)
snac_model.eval()
print("✓ SNAC model loaded")

# Load dataset WITHOUT decoding audio (avoids torchcodec)
print(f"\nLoading dataset from {DATASET_REPO}...")
from datasets import Audio as AudioFeature
dataset = load_dataset(DATASET_REPO, split="train")
# Disable automatic audio decoding
dataset = dataset.cast_column("audio", AudioFeature(decode=False))
print(f"✓ Dataset loaded: {len(dataset)} samples")

# Get sample rate from features
ds_sample_rate = 24000  # SPH dataset is 24kHz
print(f"Target sample rate: {ds_sample_rate} Hz")

def tokenise_audio(audio_dict):
    """
    Tokenize audio using 24kHz SNAC with 7-token pattern
    Processes raw audio bytes with torchaudio
    """
    try:
        # Load audio from bytes using torchaudio
        import io
        audio_bytes = audio_dict["bytes"]
        waveform, orig_sr = torchaudio.load(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 24kHz if needed
        if orig_sr != TARGET_SAMPLE_RATE:
            resample_transform = T.Resample(orig_freq=orig_sr, new_freq=TARGET_SAMPLE_RATE)
            waveform = resample_transform(waveform)
        
        # Add batch dimension for SNAC
        waveform = waveform.unsqueeze(0).to(device)
        
        # Encode with SNAC
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        
        # Apply 7-token interleaving pattern (24kHz SNAC)
        all_codes = []
        seq_len = codes[0].shape[1]
        
        for i in range(seq_len):
            # Token 0: L0
            all_codes.append(codes[0][0][i].item() + AUDIO_TOKENS_START)
            
            # Token 1: L1[0]
            all_codes.append(codes[1][0][2*i].item() + AUDIO_TOKENS_START + 4096)
            
            # Token 2: L2[0]
            all_codes.append(codes[2][0][4*i].item() + AUDIO_TOKENS_START + (2*4096))
            
            # Token 3: L2[1]
            all_codes.append(codes[2][0][(4*i)+1].item() + AUDIO_TOKENS_START + (3*4096))
            
            # Token 4: L1[1]
            all_codes.append(codes[1][0][(2*i)+1].item() + AUDIO_TOKENS_START + (4*4096))
            
            # Token 5: L2[2]
            all_codes.append(codes[2][0][(4*i)+2].item() + AUDIO_TOKENS_START + (5*4096))
            
            # Token 6: L2[3]
            all_codes.append(codes[2][0][(4*i)+3].item() + AUDIO_TOKENS_START + (6*4096))
        
        return all_codes
    
    except Exception as e:
        print(f"Error tokenizing audio: {e}")
        return None

def add_codes(example):
    """Add tokenized audio codes to dataset example"""
    codes_list = None
    
    try:
        audio_data = example.get("audio")
        if audio_data and "bytes" in audio_data:
            codes_list = tokenise_audio(audio_data)
    except Exception as e:
        print(f"Skipping sample due to error: {e}")
    
    example["codes_list"] = codes_list
    return example

# Process dataset
print("\nTokenizing audio samples...")
dataset = dataset.map(add_codes, remove_columns=["audio"])

# Filter out failed samples
print("Filtering valid samples...")
dataset = dataset.filter(lambda x: x["codes_list"] is not None)
dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)
print(f"✓ Valid samples: {len(dataset)}")

def remove_duplicate_frames(example):
    """
    Remove consecutive frames with identical L0 codes
    This reduces repetitive audio artifacts (critical for quality)
    """
    vals = example["codes_list"]
    
    if len(vals) % 7 != 0:
        raise ValueError(f"Code list length {len(vals)} not divisible by 7")
    
    result = vals[:7]  # Keep first frame
    removed_frames = 0
    
    for i in range(7, len(vals), 7):
        current_first = vals[i]      # L0 of current frame
        previous_first = result[-7]  # L0 of previous frame
        
        if current_first != previous_first:
            result.extend(vals[i:i+7])
        else:
            removed_frames += 1
    
    example["codes_list"] = result
    return example

# Remove duplicate frames
print("\nRemoving duplicate frames...")
dataset = dataset.map(remove_duplicate_frames)
print("✓ Duplicate frames removed")

# Save preprocessed dataset
output_path = "/workspace/preprocessed_dataset_sph_24khz"
print(f"\nSaving preprocessed dataset to {output_path}...")
dataset.save_to_disk(output_path)
print("✓ Dataset saved")

# Validation
print("\n" + "=" * 60)
print("Preprocessing Complete - Validation")
print("=" * 60)
sample = dataset[0]
print(f"Sample text: {sample['text'][:80]}...")
print(f"Code count: {len(sample['codes_list'])}")
print(f"Frames: {len(sample['codes_list']) // 7}")
print(f"First 14 codes (2 frames): {sample['codes_list'][:14]}")
print(f"Code range: {min(sample['codes_list'])} - {max(sample['codes_list'])}")
print("=" * 60)
print("✓ Preprocessing successful!")
