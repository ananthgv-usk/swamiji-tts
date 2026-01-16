import io
import time
import torch
import uvicorn
import asyncio
import numpy as np
import soundfile as sf
import queue
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer
from peft import PeftModel
from snac import SNAC

# Configuration
BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"
MOEL_PATH = "/workspace/orpheus_sph_refinement/checkpoint-476"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Special Tokens
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
AUDIO_TOKENS_START = 128266

FRAMES_PER_CHUNK = 20 # ~0.4s to 0.8s chunks for smooth playback

# Globals
model = None
tokenizer = None
snac_model = None
app = FastAPI()

class QueueStreamer(BaseStreamer):
    """Passes tokens from generation thread to a queue"""
    def __init__(self, token_queue):
        self.token_queue = token_queue
        
    def put(self, value):
        # value is a tensor (batch, 1) or (batch, n)
        if value.dim() > 1:
            for v in value.flatten():
                self.token_queue.put(v.item())
        else:
            self.token_queue.put(value.item())
            
    def end(self):
        self.token_queue.put(None) # Signal end

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, snac_model
    print("Loading models...")
    
    # Base LLM
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, MOEL_PATH)
    model.eval()
    print(f"✓ LLM loaded (Vocab: {model.config.vocab_size})")

    # SNAC Model (Keep on CPU to avoid CUDA conflicts with LLM)
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
        # Layer 1 (0-4095)
        layer_1.append(audio_tokens[base])
        # Layer 2 (4096-8191) -> (0-4095)
        layer_2.append(audio_tokens[base + 1] - 4096)
        # Layer 3 (8192-12287) -> (0-4095)
        layer_3.append(audio_tokens[base + 2] - (2 * 4096))
        layer_3.append(audio_tokens[base + 3] - (3 * 4096))
        # Layer 2 (4096-8191) -> (0-4095) 
        layer_2.append(audio_tokens[base + 4] - (4 * 4096))
        # Layer 3 (8192-12287) -> (0-4095)
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
        
        # Setup Queue and Streamer
        token_queue = queue.Queue()
        streamer = QueueStreamer(token_queue)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.6,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=END_OF_SPEECH
        )
        
        # Start Generation in background thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Consume tokens and send chunks
        audio_tokens_buffer = []
        tokens_received = 0
        t0 = time.time()
        
        while True:
            try:
                # Polling queue with small timeout to allow heartbeat if needed
                token_id = token_queue.get(timeout=10.0)
            except queue.Empty:
                if not thread.is_alive(): break
                continue
                
            if token_id is None: # End signal
                break
                
            tokens_received += 1
            if tokens_received % 50 == 0:
                elapsed = time.time() - t0
                print(f"TPS: {tokens_received/elapsed:.2f} | Total: {tokens_received}")

            # Collect audio tokens
            if token_id >= AUDIO_TOKENS_START:
                audio_tokens_buffer.append(token_id - AUDIO_TOKENS_START)
                
                # If chunk ready, decode on CPU and send
                if len(audio_tokens_buffer) >= (FRAMES_PER_CHUNK * 7):
                    chunk_codes = audio_tokens_buffer[:]
                    audio_tokens_buffer = []
                    
                    codes = redistribute_codes(chunk_codes)
                    with torch.no_grad():
                        audio_hat = snac_model.decode(codes)
                    audio_np = audio_hat.detach().squeeze().cpu().numpy()
                    
                    bio = io.BytesIO()
                    # Using sf.write for WAV framing
                    sf.write(bio, audio_np, 24000, format='WAV')
                    await websocket.send_bytes(bio.getvalue())
                    
                    # Yield to allow other tasks if any
                    await asyncio.sleep(0)

        thread.join()
        print("Generation finished.")
                    
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
