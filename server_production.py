import io
import time
import torch
import uvicorn
import asyncio
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from snac import SNAC

# Configuration
BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"
MOEL_PATH = "/workspace/orpheus_sph_refinement/checkpoint-476"

# Special Tokens
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
AUDIO_TOKENS_START = 128266

# Streaming Configuration
FRAMES_PER_CHUNK = 15  # Smaller chunks for lower latency
GENERATION_BATCH_SIZE = 1  # Keep at 1 for real-time streaming

# Globals
model = None
tokenizer = None
snac_model = None
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, snac_model
    print("Loading models...")
    
    # Base LLM - Load on GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, MOEL_PATH)
    model.eval()
    
    # Verify GPU placement
    device = next(model.parameters()).device
    print(f"✓ LLM loaded on {device} (Vocab: {model.config.vocab_size})")

    # SNAC Model (Keep on CPU to avoid CUDA conflicts)
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu")
    snac_model.eval()
    print("✓ SNAC loaded on CPU")

def redistribute_codes(audio_tokens):
    """Correctly maps 7-token pattern back to SNAC layers"""
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    num_frames = len(audio_tokens) // 7
    for i in range(num_frames):
        base = 7 * i
        layer_1.append(audio_tokens[base])
        layer_2.append(audio_tokens[base + 1] - 4096)
        layer_3.append(audio_tokens[base + 2] - (2 * 4096))
        layer_3.append(audio_tokens[base + 3] - (3 * 4096))
        layer_2.append(audio_tokens[base + 4] - (4 * 4096))
        layer_3.append(audio_tokens[base + 5] - (5 * 4096))
        layer_3.append(audio_tokens[base + 6] - (6 * 4096))
        
    return [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0)
    ]

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    try:
        prompt = await websocket.receive_text()
        print(f"Generating for: {prompt}")
        
        # Prepare Input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[START_OF_HUMAN]])
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]])
        input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1).to("cuda")
        
        # Generation loop WITHOUT threading - direct async generation
        audio_tokens_buffer = []
        tokens_generated = 0
        t0 = time.time()
        past_key_values = None
        
        # Auto-calculate max_new_tokens
        word_count = len(prompt.split())
        calc_tokens = int(word_count * 7.5 * 1.5)
        max_new_tokens = max(calc_tokens, 500)
        print(f"Generaton config: {max_new_tokens} tokens for {word_count} words")
        
        generated_sequence = []
        REPETITION_PENALTY = 1.1
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Get next token logits
                logits = outputs.logits[:, -1, :]
                
                # Apply Repetition Penalty
                if len(generated_sequence) > 0 and REPETITION_PENALTY > 1.0:
                    for token in set(generated_sequence):
                        if logits[0, token] < 0:
                            logits[0, token] *= REPETITION_PENALTY
                        else:
                            logits[0, token] /= REPETITION_PENALTY
                
                probs = torch.softmax(logits / 0.6, dim=-1)  # temperature=0.6
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if next_token.item() == END_OF_SPEECH:
                    break
                
                # Update for next iteration
                input_ids = next_token
                past_key_values = outputs.past_key_values
                tokens_generated += 1
                
                # Update history for repetition penalty
                generated_sequence.append(next_token.item())
                
                # Log progress
                if tokens_generated % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"TPS: {tokens_generated/elapsed:.2f} | Total: {tokens_generated}")
                
                # Collect audio tokens
                token_id = next_token.item()
                if token_id >= AUDIO_TOKENS_START:
                    audio_tokens_buffer.append(token_id - AUDIO_TOKENS_START)
                    
                    # Send chunk when ready
                    if len(audio_tokens_buffer) >= (FRAMES_PER_CHUNK * 7):
                        chunk_codes = audio_tokens_buffer[:(FRAMES_PER_CHUNK * 7)]
                        audio_tokens_buffer = audio_tokens_buffer[(FRAMES_PER_CHUNK * 7):]
                        
                        # Decode on CPU
                        codes = redistribute_codes(chunk_codes)
                        audio_hat = snac_model.decode(codes)
                        audio_np = audio_hat.detach().squeeze().cpu().numpy()
                        
                        # Send WAV chunk
                        bio = io.BytesIO()
                        sf.write(bio, audio_np, 24000, format='WAV')
                        await websocket.send_bytes(bio.getvalue())
                        
                        # Yield to event loop
                        await asyncio.sleep(0)
        
        # Send any remaining audio
        if len(audio_tokens_buffer) >= 7:
            num_complete_frames = len(audio_tokens_buffer) // 7
            chunk_codes = audio_tokens_buffer[:(num_complete_frames * 7)]
            codes = redistribute_codes(chunk_codes)
            with torch.no_grad():
                audio_hat = snac_model.decode(codes)
            audio_np = audio_hat.detach().squeeze().cpu().numpy()
            bio = io.BytesIO()
            sf.write(bio, audio_np, 24000, format='WAV')
            await websocket.send_bytes(bio.getvalue())
        
        elapsed = time.time() - t0
        print(f"Generation complete: {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.2f} TPS)")
                    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()
    finally:
        print("Connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
