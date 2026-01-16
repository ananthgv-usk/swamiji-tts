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
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
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

FRAMES_PER_CHUNK = 20 # 20 frames * 7 tokens = 140 tokens

# Globals
model = None
tokenizer = None
snac_model = None
app = FastAPI()

class QueueStreamer(BaseStreamer):
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
        device_map=DEVICE,
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    print(f"Vocab Size: {base_model.config.vocab_size}")
    # base_model.resize_token_embeddings(133000) # Removed (Vocab is 156k)
    
    model = PeftModel.from_pretrained(base_model, MOEL_PATH)
    print(f"✓ LLM loaded on {model.device}")
    model.eval()

    # SNAC Model
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)
    print("✓ SNAC loaded")

def redistribute_codes(audio_tokens):
    # Convert flat list of 0-4095 tokens back to hierarchical codes
    # Pattern: 1 code1, 1 code2, 1 code3, 4 code4 ... (Total 7 tokens per frame)
    codes = [[], [], [], []]
    
    # We expect groups of 7 tokens
    num_frames = len(audio_tokens) // 7
    for i in range(num_frames):
        chunk = audio_tokens[i*7 : (i+1)*7]
        # Token 0 -> Code1
        codes[0].append(chunk[0])
        # Token 1 -> Code2
        codes[1].append(chunk[1])
        # Token 2 -> Code3
        codes[2].append(chunk[2])
        # Tokens 3,4,5,6 -> Code4
        codes[3].extend(chunk[3:])
        
    # Convert to tensors
    c1 = torch.tensor(codes[0]).unsqueeze(0).to(DEVICE)
    c2 = torch.tensor(codes[1]).unsqueeze(0).to(DEVICE)
    c3 = torch.tensor(codes[2]).unsqueeze(0).to(DEVICE)
    c4 = torch.tensor(codes[3]).unsqueeze(0).to(DEVICE)
    
    return [c1, c2, c3, c4]

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
        input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1).to(DEVICE)
        
        print(f"Max Input Token ID: {input_ids.max().item()} | Vocab Limit: {model.config.vocab_size}")
        
        # Threaded Generation
        token_queue = queue.Queue()
        streamer = QueueStreamer(token_queue)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.6,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Consume Queue
        audio_tokens_buffer = []
        tokens_received = 0
        t0 = time.time()
        
        while True:
            # Non-blocking get from queue?
            # We must await to allow ping/pong.
            # Using loop.run_in_executor to wait for queue?
            # Or tight loop with small sleep?
            try:
                # Try getting token with small timeout
                token_id = token_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01) # Yield and wait
                if not thread.is_alive() and token_queue.empty():
                    print("Generation finished")
                    break
                continue
            
            if token_id is None: # End signal
                break
                
            tokens_received += 1
            if tokens_received % 50 == 0:
                elapsed = time.time() - t0
                tps = tokens_received / elapsed
                print(f"TPS: {tps:.2f} | Total: {tokens_received}")

            if token_id == END_OF_SPEECH:
                # Flush remaining?
                break
            
            if token_id >= AUDIO_TOKENS_START:
                val = token_id - AUDIO_TOKENS_START
                audio_tokens_buffer.append(val)
                
                if len(audio_tokens_buffer) >= (FRAMES_PER_CHUNK * 7):
                    chunk_codes = audio_tokens_buffer[:]
                    audio_tokens_buffer = []
                    
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
