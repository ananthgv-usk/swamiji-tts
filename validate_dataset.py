#!/usr/bin/env python3
"""
Validate SPH dataset sample rate and audio properties
CRITICAL: Verify assumptions before preprocessing
"""
from datasets import load_dataset
import numpy as np

print("=" * 60)
print("SPH Dataset Validation")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
dataset = load_dataset("kailasa-ngpt/SPH_Audio_2019_60_Secs_947_Samples", split="train")
print(f"✓ Dataset loaded: {len(dataset)} samples")

# Check first 5 samples
print("\nSample Rate Analysis:")
print("-" * 60)

sample_rates = []
for i in range(min(5, len(dataset))):
    audio_data = dataset[i]["audio"]
    sr = audio_data["sampling_rate"]
    duration = len(audio_data["array"]) / sr
    sample_rates.append(sr)
    
    print(f"Sample {i+1}:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Array length: {len(audio_data['array'])}")
    print(f"  Text: {dataset[i]['text'][:60]}...")
    print()

# Check consistency
unique_rates = set(sample_rates)
print("=" * 60)
print("Summary:")
print(f"  Unique sample rates: {unique_rates}")

if len(unique_rates) == 1:
    sr = list(unique_rates)[0]
    print(f"  ✓ All samples have consistent sample rate: {sr} Hz")
    
    if sr == 24000:
        print(f"  ✓ Matches 24kHz SNAC - NO RESAMPLING NEEDED")
    elif sr == 32000:
        print(f"  ⚠️ Dataset is 32kHz, but reference uses 24kHz SNAC")
        print(f"  ⚠️ Will need to RESAMPLE from 32kHz -> 24kHz")
    elif sr == 16000:
        print(f"  ⚠️ Dataset is 16kHz, will need to RESAMPLE to 24kHz")
    else:
        print(f"  ⚠️ Unexpected sample rate: {sr} Hz")
        print(f"  ⚠️ Will need to RESAMPLE to 24kHz")
else:
    print(f"  ❌ INCONSISTENT sample rates found: {unique_rates}")
    print(f"  ❌ This needs investigation!")

print("=" * 60)
