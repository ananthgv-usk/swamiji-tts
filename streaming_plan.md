"""
Streaming Implementation Plan

Goal: Real-time TTS streaming from RunPod (Checkpoint 100) to Mac Client.

Architecture:
1. Server (RunPod):
   - Loads Checkpoint 100 (LoRA) + SNAC 24kHz.
   - Endpoint: WebSocket `/stream`.
   - Logic:
     - Receive text.
     - Tokenize.
     - Generate tokens (autoregressive).
     - Every 7 tokens (1 frame) -> Aggregate (e.g. 50 frames roughly 1s).
     - Decode 50 frames to Audio (PCM/WAV).
     - Send Audio Bytes via WebSocket.

2. Client (Mac):
   - Connects to `ws://POD_IP:PORT/stream`.
   - Sends text.
   - Receives audio chunks.
   - Plays using `pyaudio` (needs `pip install pyaudio`) or writes to stdout for `mpv`.
   - `pyaudio` is cleaner.

Requirements:
- Server: `fastapi`, `uvicorn`, `websockets` (Installing now).
- Client: `websockets`, `pyaudio` (User might need to install `portaudio` via brew).

Implementation Steps:
1. Create `server.py` (Robust generation loop).
2. Create `client.py`.
3. Deploy Server.
4. Test.
"""
