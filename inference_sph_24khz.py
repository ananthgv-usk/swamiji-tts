#!/usr/bin/env python3
"""
Inference script for Orpheus TTS (24kHz SNAC, 7-token pattern)
Matches reference implementation exactly
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from snac import SNAC
import soundfile as sf
import argparse
import os

# Special tokens (matching reference and training)
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
PAD_TOKEN = 128263
AUDIO_TOKENS_START = 128266

def parse_args():
    parser = argparse.ArgumentParser(description="Orpheus TTS Inference (24kHz)")
    parser.add_argument("--model_path", type=str, default="/workspace/orpheus_sph_refinement/checkpoint-450",
                        help="Path to trained model adapters")
    parser.add_argument("--base_model", type=str, default="unsloth/orpheus-3b-0.1-ft",
                        help="Base model name")
    parser.add_argument("--prompt", type=str, 
                        default="Understand, When I see the space of Kailasa, I am overwhelmed because of the intensity.",
                        help="Text prompt to generate speech for")
    parser.add_argument("--output", type=str, default="generated_audio.wav",
                        help="Output audio file path")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty")
    return parser.parse_args()

def redistribute_codes(code_list):
    """
    Redistribute 7-token pattern back to 3 SNAC layers (24kHz)
    Matches reference implementation exactly
    """
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    num_frames = len(code_list) // 7
    
    for i in range(num_frames):
        base = 7 * i
        
        # L0
        layer_1.append(code_list[base])
        
        # L1[0]
        layer_2.append(code_list[base + 1] - 4096)
        
        # L2[0]
        layer_3.append(code_list[base + 2] - (2 * 4096))
        
        # L2[1]
        layer_3.append(code_list[base + 3] - (3 * 4096))
        
        # L1[1]
        layer_2.append(code_list[base + 4] - (4 * 4096))
        
        # L2[2]
        layer_3.append(code_list[base + 5] - (5 * 4096))
        
        # L2[3]
        layer_3.append(code_list[base + 6] - (6 * 4096))
    
    # Convert to tensors
    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0)
    ]
    
    return codes

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Orpheus TTS Inference - 24kHz SNAC (Transformers + PEFT)")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"Adapters: {args.model_path}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    print("=" * 60)
    
    # Load model and tokenizer
    print("\nLoading base model...")
    # Use sdpa for speed (since we used it for training)
    attn_implementation = "sdpa" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    print("\nLoading LoRA adapters...")
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    print("✓ Model loaded")
    
    # Load SNAC model
    print("\nLoading SNAC model (24kHz)...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cpu")  # Keep on CPU for decoding
    snac_model.eval()
    print("✓ SNAC model loaded")
    
    # Prepare input (matching training format)
    print("\nPreparing input...")
    
    # Tokenize prompt (includes BOS automatically)
    text_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    text_ids.append(END_OF_TEXT)
    
    # Construct full prompt sequence
    prompt_ids = (
        [START_OF_HUMAN] +
        text_ids +
        [END_OF_HUMAN] +
        [START_OF_AI] +
        [START_OF_SPEECH]
    )
    
    input_ids = torch.tensor([prompt_ids], dtype=torch.int64).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Auto-calculate max_new_tokens if not specified or default
    if args.max_tokens == 2000: # Default value
        word_count = len(args.prompt.split())
        # Formula: ~80 tokens per word for SNAC (7 codes/frame)
        # 30s audio = ~2500 tokens for ~40 words -> ~60-70 tokens/word. Using 80 for safety.
        calc_tokens = int(word_count * 80)
        args.max_tokens = max(calc_tokens, 1000) # Minimum 1000
        print(f"  Auto-calculated max_tokens: {args.max_tokens} (Words: {word_count})")
    
    # Generate
    print("\nGenerating audio tokens...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=1,
            eos_token_id=END_OF_SPEECH,
            use_cache=True
        )
    
    print("✓ Generation complete")
    
    # Extract audio codes
    print("\nExtracting audio codes...")
    
    # Audio codes start after the prompt we provided
    start_idx = len(prompt_ids)
    generated_tokens = generated_ids[0, start_idx:]
    
    # Filter out non-audio tokens (like END_OF_SPEECH or END_OF_AI if generated)
    # And keep only tokens that are >= AUDIO_TOKENS_START
    audio_tokens = [t.item() for t in generated_tokens if t.item() >= AUDIO_TOKENS_START]
    
    # Trim to multiple of 7 (SNAC frames)
    num_frames = len(audio_tokens) // 7
    if num_frames == 0:
        print("❌ No valid audio frames generated!")
        return

    trimmed_audio_tokens = audio_tokens[:num_frames * 7]

    # Subtract offset to get raw SNAC codes
    codes = [t - AUDIO_TOKENS_START for t in trimmed_audio_tokens]
    
    print(f"  Extracted {len(codes)} codes ({num_frames} frames)")
    
    # Redistribute and decode
    print("\nDecoding audio...")
    codes_tensors = redistribute_codes(codes)
    
    with torch.no_grad():
        audio_hat = snac_model.decode(codes_tensors)
    
    audio_np = audio_hat.detach().squeeze().cpu().numpy()
    
    # Save audio
    print(f"\nSaving audio to {args.output}...")
    sf.write(args.output, audio_np, 24000)
    
    duration = len(audio_np) / 24000
    print("=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: 24000 Hz")
    print(f"Audio stats: min={audio_np.min():.4f}, max={audio_np.max():.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
