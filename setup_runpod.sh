#!/bin/bash
# RunPod Setup Script for Orpheus TTS Training
# Tested on: RunPod PyTorch 2.4.0 with CUDA 12.4.1

set -e  # Exit on error

echo "🚀 Setting up Orpheus TTS environment on RunPod..."

# Set Hugging Face cache to persistent storage
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical installations
echo "✅ Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
python3 -c "from snac import SNAC; print('SNAC: OK')"

# Check GPU availability
echo "🎮 Checking GPU..."
nvidia-smi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your dataset to /workspace/"
echo "2. Run preprocessing: python3 preprocess_sph_24khz.py"
echo "3. Start training: python3 train_sph_24khz.py"
echo ""
