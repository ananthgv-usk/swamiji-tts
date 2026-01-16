import asyncio
import websockets
import sys
import io
import soundfile as sf
import sounddevice as sd
import numpy as np
import argparse

# Usage: python client_mac.py "Your prompt here" [--host HOST] [--port PORT]
# Example: python client_mac.py "Hello" --host 104.255.9.187 --port 8000

async def stream_audio(prompt):
    async with websockets.connect(URI) as websocket:
        print(f"Connected to {URI}")
        print(f"Sending: {prompt}")
        await websocket.send(prompt)
        
        print("Receiving audio...")
        
        # Buffer to accumulate chunks before starting playback
        audio_buffer = []
        BUFFER_CHUNKS = 10  # Accumulate 10 chunks (~4-8 seconds) before starting - compensates for 20 TPS generation
        chunks_received = 0
        playback_started = False
        stream = None
        sample_rate = None
        
        try:
            while True:
                message = await websocket.recv()
                # Read WAV data
                data, fs = sf.read(io.BytesIO(message))
                data = data.astype(np.float32)
                
                if sample_rate is None:
                    sample_rate = fs
                
                chunks_received += 1
                audio_buffer.append(data)
                
                # Start playback after buffering initial chunks
                if not playback_started and chunks_received >= BUFFER_CHUNKS:
                    print(f"Playing stream ({sample_rate} Hz)...")
                    stream = sd.OutputStream(channels=1, samplerate=sample_rate, dtype='float32')
                    stream.start()
                    playback_started = True
                    
                    # Play buffered chunks
                    for buffered_audio in audio_buffer:
                        stream.write(buffered_audio)
                    audio_buffer = []  # Clear buffer
                
                # Play chunks in real-time after buffer is drained
                elif playback_started:
                    stream.write(data)
                
        except websockets.exceptions.ConnectionClosed:
            print("Stream finished.")
        except Exception as e:
            print(f"Error: {e}")
            if stream:
                stream.stop()
                stream.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orpheus TTS Streaming Client")
    parser.add_argument("prompt", type=str, nargs="?", default="Hello world", 
                        help="Text prompt to generate speech for")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Server host (default: localhost for SSH tunnel)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    
    args = parser.parse_args()
    
    # Construct URI from arguments
    global URI
    URI = f"ws://{args.host}:{args.port}/stream"
    
    print(f"Connecting to: {URI}")
    
    try:
        asyncio.run(stream_audio(args.prompt))
    except KeyboardInterrupt:
        pass
