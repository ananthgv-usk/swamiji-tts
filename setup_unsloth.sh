#!/bash/bin
# Setup script for Unsloth High-Speed Inference

echo "Installing high-performance dependencies..."
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.28" "trl<0.9.0" peft accelerate bitsandbytes
pip install fastapi uvicorn websockets soundfile snac

echo "Environment ready for Unsloth inference."
