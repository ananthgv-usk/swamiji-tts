#!/usr/bin/env python3
"""
Preprocess dataset: Convert audio to SNAC token sequences
This creates a new dataset with text → audio_tokens mappings
"""
import torch
from datasets import load_dataset, Dataset
from snac import SNAC
import os
from tqdm import tqdm
import numpy as np

# Configuration
DATASET_NAME = "Publishing/SPH_45_Audio_2"
OUTPUT_DATASET_NAME = "Publishing/SPH_45_Audio_2_preprocessed"  # Will push to HF
SNAC_MODEL = "hubertsiuzdak/snac_32khz"
AUDIO_TOKEN_START = 128256  # Token offset for audio codes in Orpheus vocab

# Use local path instead of /workspace
import os
if not os.path.exists("/workspace"):
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

def preprocess_audio_to_tokens():
    """Convert audio files to SNAC token sequences"""
    print(f"Loading SNAC model: {SNAC_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    snac_model = SNAC.from_pretrained(SNAC_MODEL).to(device)
    snac_model.eval()
    
    print(f"Loading dataset: {DATASET_NAME}...")
    # Load WITHOUT casting to Audio to avoid auto-decoding
    dataset = load_dataset(DATASET_NAME, split="train")
    
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    
    # Get raw data to avoid triggering audio decode on iteration
    import soundfile as sf
    import librosa
    
    preprocessed_data = []
    
    # Access raw arrow table to get paths without decoding
    for idx in tqdm(range(len(dataset)), desc="Converting audio to tokens"):
        try:
            # Get text without triggering audio decode
            text = dataset[idx]["text"]
            
            # Get audio path from the raw arrow table
            audio_info = dataset._data.column("audio")[idx].as_py()
            
            # audio_info is a dict with 'path', 'bytes', etc
            if audio_info and 'path' in audio_info and audio_info['path']:
                audio_path = audio_info['path']
                audio_array, sample_rate = sf.read(audio_path)
            elif audio_info and 'bytes' in audio_info and audio_info['bytes']:
                # Audio is embedded - use librosa to load from bytes
                import io
                audio_bytes = audio_info['bytes']
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            else:
                print(f"Skipping example {idx}: no audio data")
                continue
            
            # Convert to tensor and ensure correct shape (1, 1, T)
            if isinstance(audio_array, np.ndarray):
                audio_tensor = torch.from_numpy(audio_array).float()
            else:
                audio_tensor = audio_array.float() if isinstance(audio_array, torch.Tensor) else torch.tensor(audio_array).float()
            
            # Ensure shape is (1, 1, T) for SNAC
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (T) -> (1, 1, T)
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # (C, T) -> (1, C, T)
                
            audio_tensor = audio_tensor.to(device)
            
            # Encode with SNAC
            with torch.no_grad():
                codes = snac_model.encode(audio_tensor)
            
            # codes shape: (1, 4, T) - 4 codebooks
            # Flatten to 1D sequence
            codes_flat = codes.squeeze(0).T.flatten().cpu().tolist()  # (T, 4) -> (T*4,)
            
            # Shift to Orpheus token range
            audio_tokens = [c + AUDIO_TOKEN_START for c in codes_flat]
            
            preprocessed_data.append({
                "text": text,
                "audio_tokens": audio_tokens,
                "num_audio_tokens": len(audio_tokens)
            })
            
            if idx < 3:
                print(f"\nExample {idx}:")
                print(f"  Text: {text[:100]}...")
                print(f"  Audio tokens: {len(audio_tokens)} tokens")
                print(f"  Token range: [{min(audio_tokens)}, {max(audio_tokens)}]")
                
        except Exception as e:
            print(f"\nError processing example {idx}: {e}")
            continue
    
    print(f"\nSuccessfully preprocessed {len(preprocessed_data)} examples")
    
    # Create new dataset
    preprocessed_dataset = Dataset.from_list(preprocessed_data)
    
    # Save locally first
    output_path = "./preprocessed_dataset" if not os.path.exists("/workspace") else "/workspace/preprocessed_dataset"
    print(f"Saving to {output_path}...")
    preprocessed_dataset.save_to_disk(output_path)
    
    # Optionally push to HF Hub
    try:
        print(f"Pushing to Hugging Face Hub as {OUTPUT_DATASET_NAME}...")
        preprocessed_dataset.push_to_hub(OUTPUT_DATASET_NAME, private=True)
        print("✅ Dataset uploaded to Hugging Face Hub")
    except Exception as e:
        print(f"⚠️  Could not push to HF Hub: {e}")
        print(f"Dataset saved locally at {output_path}")
    
    return output_path

if __name__ == "__main__":
    preprocess_audio_to_tokens()
    print("\n✅ Preprocessing complete!")
