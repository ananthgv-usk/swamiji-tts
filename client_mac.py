import asyncio
import websockets
import sys
import io
import soundfile as sf
import sounddevice as sd
import numpy as np
import argparse
import queue
import threading
import time

# Usage: python client_mac.py "Your prompt here" [--host HOST] [--port PORT]
# This client uses a separate thread for playback to avoid blocking the network loop.

async def stream_audio(prompt):
    
    # Thread-safe queue for audio chunks
    audio_queue = queue.Queue()
    playback_active = threading.Event()
    playback_finished = threading.Event()
    
    def playback_worker():
        """Consumer thread that plays audio from the queue"""
        stream = None
        try:
            print("Playback thread waiting for data...")
            playback_active.wait() # Wait for signal to start
            if playback_finished.is_set(): return
            
            # Initialize stream (assuming 24khz for SNAC)
            # Ideally we read sample rate from first chunk, but we can hardcode or pass it
            samplerate = 24000 
            print(f"Playback started ({samplerate} Hz)")
            
            stream = sd.OutputStream(channels=1, samplerate=samplerate, dtype='float32')
            stream.start()
            
            while not (playback_finished.is_set() and audio_queue.empty()):
                try:
                    # Timeout allows checking output_finished flag
                    data = audio_queue.get(timeout=0.1) 
                    stream.write(data)
                    audio_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Playback error: {e}")
                    break
            
        except Exception as e:
            print(f"Thread error: {e}")
        finally:
            if stream:
                stream.stop()
                stream.close()
            print("Playback finished.")

    # Start playback thread
    player_thread = threading.Thread(target=playback_worker)
    player_thread.daemon = True
    player_thread.start()

    async with websockets.connect(URI) as websocket:
        print(f"Connected to {URI}")
        print(f"Sending: {prompt}")
        await websocket.send(prompt)
        
        print("Receiving audio...")
        
        all_audio_data = [] # For debug file
        # 24kHz @ 40 frames per chunk is approx 0.12s per chunk.
        # 10 seconds of audio is ~83 chunks.
        BUFFER_CHUNKS = 80 
        chunks_received = 0
        
        try:
            while True:
                message = await websocket.recv()
                
                # Decode
                data, fs = sf.read(io.BytesIO(message))
                
                # Verify format on first chunk
                if chunks_received == 0:
                    print(f"DEBUG: First chunk - Rate: {fs}, Shape: {data.shape}, Dtype: {data.dtype}")
                    if fs != 24000:
                        print(f"WARNING: Expected 24000 Hz, got {fs} Hz")

                data = data.astype(np.float32)
                
                chunks_received += 1
                all_audio_data.append(data)
                
                # Push to queue
                audio_queue.put(data)
                
                # Print buffering progress
                if not playback_active.is_set():
                    percent = min(100, int((chunks_received / BUFFER_CHUNKS) * 100))
                    print(f"Buffering: {percent}% ({chunks_received}/{BUFFER_CHUNKS} chunks)", end="\r")
                else:
                    max_amp = np.max(np.abs(data))
                    print(f"Playing... Chunk {chunks_received}: MaxAmp={max_amp:.4f}  ", end="\r")
                
                # Signal thread to start after buffer
                if not playback_active.is_set() and chunks_received >= BUFFER_CHUNKS:
                    print("\nBuffer full. Starting playback...")
                    playback_active.set()
                
        except websockets.exceptions.ConnectionClosed:
            print("\nStream finished (Connection Closed).")
        except Exception as e:
            print(f"\nNetwork error: {e}")
        finally:
            # Signal thread to finish
            playback_active.set() # Ensure it wakes up if stuck
            playback_finished.set()
            player_thread.join(timeout=5.0) # Wait for playback to drain
            
            # Save Debug
            if len(all_audio_data) > 0:
                print("Saving debug_output.wav...")
                full_audio = np.concatenate(all_audio_data)
                sf.write("debug_output.wav", full_audio, 24000)
                print(f"Saved {len(full_audio)/24000:.2f}s of audio.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orpheus TTS Streaming Client (Threaded)")
    parser.add_argument("prompt", type=str, nargs="?", default="Hello world", 
                        help="Text prompt to generate speech for")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Server host (default: localhost for SSH tunnel)")
    parser.add_argument("--port", type=int, default=8001,
                        help="Server port (default: 8001)")
    
    args = parser.parse_args()
    
    global URI
    URI = f"ws://{args.host}:{args.port}/stream"
    
    print(f"Connecting to: {URI}")
    
    try:
        asyncio.run(stream_audio(args.prompt))
    except KeyboardInterrupt:
        pass
