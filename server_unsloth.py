import io
import time
import torch
import uvicorn
import asyncio
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from unsloth import FastLanguageModel
from snac import SNAC

# Configuration
MODEL_ID = "unsloth/orpheus-3b-0.1-ft"
LORA_PATH = "/workspace/orpheus_sph_refinement/checkpoint-450"

# Special Tokens
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
AUDIO_TOKENS_START = 128266

# Streaming Configuration
FRAMES_PER_CHUNK = 40  # approx 0.12s per chunk
MAX_SEQ_LENGTH = 16384

# Globals
model = None
tokenizer = None
snac_model = None
app = FastAPI()

def redistribute_codes(audio_tokens):
    """Redistribute flat token list into 7-token patterns for SNAC layers"""
    layer_0 = []
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    for i in range(0, len(audio_tokens), 7):
        if i + 6 >= len(audio_tokens): break
        layer_0.append(audio_tokens[i])
        layer_1.append(audio_tokens[i+1])
        layer_1.append(audio_tokens[i+2])
        layer_2.append(audio_tokens[i+3])
        layer_2.append(audio_tokens[i+4])
        layer_3.append(audio_tokens[i+5])
        layer_3.append(audio_tokens[i+6])
        
    return [
        torch.tensor(layer_0).unsqueeze(0),
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0)
    ]

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, snac_model
    print("Loading models with Unsloth...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_len = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    # Load adapters
    model.load_adapter(LORA_PATH)
    FastLanguageModel.for_inference(model) # Enable native fast 2x faster inference
    
    # SNAC on CPU
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cpu")
    print("✓ Models Loaded")

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    try:
        prompt = await websocket.receive_text()
        print(f"Generating optimized for: {prompt}")
        
        # Prepare Input
        full_prompt = f"<|start_of_human|>{prompt}<|end_of_text|><|end_of_human|><|start_of_ai|><|start_of_speech|>"
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
        # Streamer for Unsloth/HF compatibility
        from transformers import TextStreamer
        
        audio_tokens_buffer = []
        tokens_generated = 0
        t0 = time.time()
        
        # Custom generation loop for token-by-token control
        # We use a primitive loop here because we need the tokens as they come 
        # for lowest latency, but we'll use torch.compile/xformers hidden inside unsloth
        
        with torch.no_grad():
            # Initial forward pass to get kv_cache
            outputs = model(inputs.input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            # Auto-calculate max_new_tokens
            word_count = len(prompt.split())
            max_new_tokens = max(int(word_count * 80), 500)
            
            for step in range(max_new_tokens):
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Sample (Temperature 0.6)
                logits = outputs.logits[:, -1, :] / 0.6
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                token_id = next_token.item()
                tokens_generated += 1
                
                # Log TPS
                if tokens_generated % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"TPS: {tokens_generated/elapsed:.2f}")

                # Collect Audio
                if token_id >= AUDIO_TOKENS_START:
                    audio_tokens_buffer.append(token_id - AUDIO_TOKENS_START)
                    
                    if len(audio_tokens_buffer) >= (FRAMES_PER_CHUNK * 7):
                        chunk_codes = audio_tokens_buffer[:(FRAMES_PER_CHUNK * 7)]
                        audio_tokens_buffer = audio_tokens_buffer[(FRAMES_PER_CHUNK * 7):]
                        
                        codes = redistribute_codes(chunk_codes)
                        audio_hat = snac_model.decode(codes)
                        audio_np = audio_hat.detach().squeeze().cpu().numpy()
                        
                        bio = io.BytesIO()
                        sf.write(bio, audio_np, 24000, format='WAV')
                        await websocket.send_bytes(bio.getvalue())
                        await asyncio.sleep(0)

                if token_id == END_OF_SPEECH: break
                    
        print(f"Complete: {tokens_generated} tokens")
                    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
