#!/usr/bin/env python3
"""
Direct preprocessing: Stream dataset as parquet, manually load audio files
Bypasses datasets library's audio decoding requirements
"""
import torch
from snac import SNAC
import soundfile as sf
from datasets import load_dataset, Dataset
from tqdm import tqdm
import io
import requests
from huggingface_hub import hf_hub_download, list_repo_files

DATASET_NAME = "Publishing/SPH_45_Audio_2"
SNAC_MODEL = "hubertsiuzdak/snac_32khz"
AUDIO_TOKEN_START = 128256

print("Loading SNAC model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
snac_model = SNAC.from_pretrained(SNAC_MODEL).to(device)
snac_model.eval()

print(f"Streaming dataset: {DATASET_NAME}...")
# Use streaming to avoid loading all audio at once
from datasets import load_dataset, Dataset, Audio
# ... imports ...

# Use streaming to avoid loading all audio at once
dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
dataset = dataset.cast_column("audio", Audio(decode=False))

preprocessed_data = []
count = 0

for example in tqdm(dataset, desc="Processing audio"):
    try:
        text = example["text"]
        audio_dict = example["audio"]
        
        # Audio dict has 'bytes' field with raw audio data
        if 'bytes' in audio_dict and audio_dict['bytes']:
            audio_bytes = audio_dict['bytes']
            # Load from bytes using soundfile
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        elif 'path' in audio_dict and audio_dict['path']:
            # Fallback: load from path if available
            audio_array, sample_rate = sf.read(audio_dict['path'])
        else:
            print(f"Skipping example {count}: no audio data")
            count += 1
            continue
        
        # Convert to tensor and force Mono
        if len(audio_array.shape) == 1:
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
        else:
            # Stereo (N, 2) -> Transpose to (2, N) -> Mean(0) -> (1, N) -> Unsqueeze(0) -> (1, 1, N)
            audio_tensor = torch.from_numpy(audio_array).float().T
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True).unsqueeze(0)
        
        audio_tensor = audio_tensor.to(device)
        
        # Encode with SNAC
        with torch.no_grad():
            codes = snac_model.encode(audio_tensor)
        
        # Flatten and shift
        codes_flat = codes.squeeze(0).T.flatten().cpu().tolist()
        audio_tokens = [c + AUDIO_TOKEN_START for c in codes_flat]
        
        preprocessed_data.append({
            "text": text,
            "audio_tokens": audio_tokens,
            "num_audio_tokens": len(audio_tokens)
        })
        
        if count < 3:
            print(f"\nExample {count}:")
            print(f"  Text: {text[:80]}...")
            print(f"  Tokens: {len(audio_tokens)}")
        
        count += 1
        
    except Exception as e:
        print(f"\nError at example {count}: {e}")
        import traceback
        traceback.print_exc()
        count += 1
        continue

print(f"\nProcessed {len(preprocessed_data)} examples")

if len(preprocessed_data) > 0:
    print("Creating dataset...")
    preprocessed_dataset = Dataset.from_list(preprocessed_data)
    
    output_path = "./preprocessed_dataset"
    print(f"Saving to {output_path}...")
    preprocessed_dataset.save_to_disk(output_path)
    print("✅ Preprocessing complete!")
else:
    print("❌ No examples were processed")
