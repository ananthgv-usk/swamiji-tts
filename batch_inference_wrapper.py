import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from snac import SNAC
import soundfile as sf
import numpy as np

# Configuration
BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"
CHECKPOINT_DIR = "/workspace/orpheus_sph_refinement_continued"
OUTPUT_DIR = "/workspace/benchmarks_continued"
PROMPT = "Listen, decide not to engage or continue, create, maintain, fight, destroy, play with any thinking process in you. that's it! Do not create sustain maintain fight with, try to destroy, or in any way engage. just unclutch."
MAPPING = {
    30: 430,
    40: 440,
    60: 460
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("Batch Inference for Continued Checkpoints")
print("="*60)

# Load base model once
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load SNAC
print("Loading SNAC...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

def generate_audio(ckpt_step, target_step):
    ckpt_path = f"{CHECKPOINT_DIR}/checkpoint-{ckpt_step}"
    if not os.path.exists(ckpt_path):
        print(f"Skipping {ckpt_path} (Not found)")
        return

    print(f"\nProcessing Checkpoint {ckpt_step} (Target: {target_step})...")
    
    # Load adapter
    try:
        model.load_adapter(ckpt_path, adapter_name=f"ckpt_{ckpt_step}")
        model.set_adapter(f"ckpt_{ckpt_step}")
    except Exception as e:
        print(f"Error loading adapter: {e}")
        # Try loading directly if load_adapter fails
        # model = PeftModel.from_pretrained(model, ckpt_path) 
        pass

    # Tokenize
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    
    # Generate
    print("Generating...")
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=1500, # Approx 20s
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    print("Decoding...")
    generated_ids = output[0][input_ids.shape[1]:]  # Remove prompt
    
    # Filter for audio tokens (simulated for simplicity - assuming direct SNAC mapping)
    # Note: Real decoding needs the complex logic from inference_sph_24khz.py
    # Here we use the simplified logic just to get *something* or call the robust script.
    
    # actually, better to just call the robust generation function if we can import it.
    # checking inference_sph_24khz.py logic...
    
    # Extract codes
    audio_codes = []
    for token in generated_ids:
        t = token.item()
        if t >= 128263: # AUDIO_TOKEN_START
             audio_codes.append(t - 128263)
             
    if not audio_codes:
        print("No audio tokens generated.")
        return

    # Reconstruct SNAC codes (4 layers)
    # This naive reconstruction is tricky without the full logic.
    # Easier to just run the existing inference script via subprocess!
    pass

import subprocess

for ckpt_step, target_step in MAPPING.items():
    ckpt_path = f"{CHECKPOINT_DIR}/checkpoint-{ckpt_step}"
    output_file = f"benchmark_{target_step}.wav"
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found. Waiting...")
        continue
        
    cmd = [
        "python3", "inference_sph_24khz.py",
        "--model_path", ckpt_path,
        "--prompt", PROMPT,
        "--output", output_file,
        "--temperature", "0.6",
        "--repetition_penalty", "1.1"
    ]
    
    print(f"Running inference for {target_step}...")
    subprocess.run(cmd)

print("\nBatch inference complete!")
