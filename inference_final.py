import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from snac import SNAC
import soundfile as sf
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Orpheus 3B")
    parser.add_argument("--checkpoint", type=str, default="/workspace/orpheus_947_final", help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="Unclutching means dropping your unconscious grip on thoughts, emotions, identities, and stories the moment you notice them without fighting, analyzing, or improving them.", help="Text prompt")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    return parser.parse_args()

def redistribute_codes_robust(code_list, device):
    # Match the 15-token preprocessing pattern: L0(1) + L1(2) + L2(4) + L3(8) = 15
    token_group_size = 15
    
    # Ensure num_frames is a multiple of 4 to avoid SNAC dimension mismatch
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
            
            # --- L0 (1 token) ---
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
    
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().to(device)
    
    try:
        with torch.no_grad():
            audio_hat = snac_model.decode(codes)
    except (IndexError, RuntimeError) as e:
        print(f"Decode failed ({e})...")
        audio_hat = None
        
    return audio_hat

def main():
    args = parse_args()
    device = args.device
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading tokenizer from unsloth/orpheus-3b-0.1-ft...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/orpheus-3b-0.1-ft")
    
    print(f"Loading Base Model via AutoModel (FFT Checkpoint)...")
    # For FFT checkpoints, we load directly from the checkpoint path
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    
    # PEFT logic removed as this is a full checkpoint

    # Format prompt (Matching inference_prod.py exactly)
    # Note: inference_prod.py used tokenizer(prompt, return_tensors="pt") which implicitly adds BOS
    # FIX: We must disable special tokens because we manually add START_OF_HUMAN
    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids
    
    start_token = torch.tensor([[128259]], dtype=torch.int64) 
    end_tokens = torch.tensor([[128009, 128260, 128257]], dtype=torch.int64) 
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    print(f"Debug: Input Sequence IDs: {modified_input_ids[0].tolist()[:10]} ... {modified_input_ids[0].tolist()[-5:]}")

    # REMOVED PADDING 128263 - It was not used in current training!
    # padding = torch.full((1, 1), 128263, dtype=torch.int64) 
    # padded_tensor = torch.cat([padding, modified_input_ids], dim=1)
    padded_tensor = modified_input_ids
    
    attention_mask = torch.cat([torch.zeros((1, 1), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
    # FIX: Attention mask should match padded_tensor shape exactly. 
    # The previous line created a mask 1 bigger than modified_input_ids?
    # padded_tensor IS modified_input_ids. So mask should be ones of same shape.
    attention_mask = torch.ones_like(padded_tensor)
    
    input_ids = padded_tensor.to(device)
    attention_mask = attention_mask.to(device)
    
    print(\"Generating audio tokens...\")\n    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
            do_sample=False,  # GREEDY DECODING
            # temperature=args.temperature,      
            # top_p=0.9,
            # repetition_penalty=args.repetition_penalty, 
            eos_token_id=128258,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_ids_cpu = generated_ids.cpu()
    print(f"Debug: Generated Tokens (First 50): {generated_ids_cpu[0].tolist()[:50]}")
    print(f"Debug: Generated Tokens (Last 50): {generated_ids_cpu[0].tolist()[:50]}")
    
    # Extraction Logic (Matching inference_prod.py)
    token_to_find = 128257
    AUDIO_TOKEN_START = 128256
    
    # CRITICAL FIX: Do not search for token_to_find (128257) because it collides with Audio Code 1.
    # Instead, rely on the known prompt length.
    prompt_len = input_ids.shape[1]
    last_occurrence_idx = prompt_len - 1
    
    print(f"✅ Using prompt end index {last_occurrence_idx} as speech start")
    
    raw_codes = generated_ids_cpu[:, last_occurrence_idx+1:].tolist()[0]
    subtracted_codes = []
    
    for c in raw_codes:
        if c == 128258: # End of Speech (or Code 2 collision)
            # If it's valid audio, cutting early is better than garble.
            break
        
        if c >= AUDIO_TOKEN_START:
            subtracted_codes.append(c - AUDIO_TOKEN_START)
        else:
            subtracted_codes.append(0)

    print(f"Extracted {len(subtracted_codes)} codes")
    print(f"First 30 extracted codes: {subtracted_codes[:30]}")
    print(f"Code range: min={min(subtracted_codes) if subtracted_codes else 0}, max={max(subtracted_codes) if subtracted_codes else 0}")
    
    # Padding Logc (224) (Matching inference_prod.py)
    pad_interval = 224 
    remainder = len(subtracted_codes) % pad_interval
    if remainder != 0:
        padding_needed = pad_interval - remainder
        print(f"Padding with {padding_needed} zeros to match window {pad_interval}...")
        subtracted_codes.extend([0] * padding_needed)

    if len(subtracted_codes) > 0:
        audio_hat = redistribute_codes_robust(subtracted_codes, device)
        if audio_hat is not None:
            audio_np = audio_hat.detach().squeeze().cpu().numpy()
            print(f"Audio Stats: Min={audio_np.min():.4f}, Max={audio_np.max():.4f}, Mean={audio_np.mean():.4f}, Std={audio_np.std():.4f}")
            sf.write("generated_audio.wav", audio_np, 32000) # SNAC 32khz
            print(f"Saved generated_audio.wav ({len(audio_np)/32000:.2f}s)")
    else:
        print("No codes extracted!")


if __name__ == "__main__":
    main()
