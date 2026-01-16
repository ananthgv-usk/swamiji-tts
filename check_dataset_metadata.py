#!/usr/bin/env python3
"""
Check SPH dataset metadata without decoding audio
"""
from datasets import load_dataset

print("=" * 60)
print("SPH Dataset Metadata Check")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
dataset = load_dataset("kailasa-ngpt/SPH_Audio_2019_60_Secs_947_Samples", split="train")
print(f"✓ Dataset loaded: {len(dataset)} samples")

# Check features
print("\nDataset Features:")
print(dataset.features)

# Check audio feature specifically
if "audio" in dataset.features:
    audio_feature = dataset.features["audio"]
    print(f"\nAudio Feature Details:")
    print(f"  Type: {type(audio_feature)}")
    print(f"  {audio_feature}")
    
    # Try to get sampling rate from feature
    if hasattr(audio_feature, 'sampling_rate'):
        print(f"\n✓ Sampling rate from feature: {audio_feature.sampling_rate} Hz")
    else:
        print("\n⚠️ Sampling rate not in feature metadata")

# Check raw data (without decoding)
print("\nRaw Sample Data (first sample, no decoding):")
sample = dataset[0]
print(f"  Text: {sample['text'][:80]}...")
print(f"  Audio keys: {sample['audio'].keys() if isinstance(sample['audio'], dict) else 'Not a dict'}")

print("=" * 60)
