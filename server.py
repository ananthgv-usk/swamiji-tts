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
MOEL_PATH = "/workspace/orpheus_sph_lora/checkpoint-119"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Special Tokens
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
AUDIO_TOKENS_START = 128266

FRAMES_PER_CHUNK = 10 # Smaller chunks for lower latency

# Globals
model = None
tokenizer = None
snac_model = None
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, snac_model
    print("Loading models...")
    
    # Base LLM
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Let transformers decide placement
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load LoRA
    model = PeftModel.from_pretrained(base_model, MOEL_PATH)
    model.eval()
    print(f"✓ LLM loaded on {next(model.parameters()).device}")

    # SNAC Model (Keep on CPU for stability and reduced GPU contention)
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu")
    print("✓ SNAC loaded on CPU")

def redistribute_codes(audio_tokens):
    """Matches 24kHz SNAC 7-token pattern"""
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
        data = await websocket.receive_text()
        prompt = data
        print(f"Generating for: {prompt}")
        
        # Prepare Input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[START_OF_HUMAN]])
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]])
        input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1).to("cuda")
        
        # Generation State
        current_input_ids = input_ids
        past_key_values = None
        audio_tokens_buffer = []
        tokens_generated = 0
        t0 = time.time()
        
        # Streaming Loop
        with torch.no_grad():
            for _ in range(1500): # Max tokens safety
                # Call model with cache
                outputs = model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                
                # Sample
                probs = torch.softmax(next_token_logits / 0.6, dim=-1) # Temp 0.6
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Update for next step
                current_input_ids = next_token
                token_id = next_token.item()
                tokens_generated += 1
                
                # Async yields for networking / heartbeat
                if tokens_generated % 20 == 0:
                   await asyncio.sleep(0) # Minimal yield
                   elapsed = time.time() - t0
                   tps = tokens_generated / elapsed
                   print(f"TPS: {tps:.2f} | Total: {tokens_generated}")

                if token_id == END_OF_SPEECH:
                    break
                    
                if token_id >= AUDIO_TOKENS_START:
                    val = token_id - AUDIO_TOKENS_START
                    audio_tokens_buffer.append(val)
                    
                    if len(audio_tokens_buffer) >= (FRAMES_PER_CHUNK * 7):
                        chunk_codes = audio_tokens_buffer[:]
                        audio_tokens_buffer = []
                        
                        # Decode
                        codes = redistribute_codes(chunk_codes)
                        audio_hat = snac_model.decode(codes)
                        audio_np = audio_hat.detach().squeeze().cpu().numpy()
                        
                        bio = io.BytesIO()
                        sf.write(bio, audio_np, 24000, format='WAV')
                        idx_bytes = bio.getvalue()
                        
                        await websocket.send_bytes(idx_bytes)
                        
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
