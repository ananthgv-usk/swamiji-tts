# RunPod Setup Guide for Orpheus TTS

## Quick Start

### 1. Create RunPod Instance
- **Template:** `RunPod PyTorch 2.4.0`
- **GPU:** RTX A6000 (48GB) or A100
- **Container:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

### 2. Upload Files
```bash
# From your local machine
scp -P <PORT> -i ~/.ssh/id_ed25519 -r orpheus/ root@<POD_IP>:/workspace/
```

### 3. Run Setup Script
```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<POD_IP>
cd /workspace/orpheus
bash setup_runpod.sh
```

This will:
- Install all Python dependencies from `requirements.txt`
- Verify PyTorch, Transformers, PEFT, and SNAC
- Check GPU availability

### 4. Start Training
```bash
python3 train_sph_24khz.py
```

## Environment Details

### Verified Dependencies
- **PyTorch:** 2.4.0 (avoid 2.6.0+ due to torchao compatibility)
- **Transformers:** 4.52.0+
- **PEFT:** 0.7.0+ (for LoRA)
- **SNAC:** Latest from GitHub (24kHz codec)
- **FastAPI:** 0.104.0+ (for streaming server)

### Storage Locations
- **HF Cache:** `/workspace/.cache/huggingface`
- **Dataset:** `/workspace/preprocessed_dataset_sph_24khz`
- **Model Output:** `/workspace/orpheus_sph_lora` or `/workspace/orpheus_sph_refinement`

## Troubleshooting

### Common Issues
1. **CUDA OOM:** Reduce batch size in `train_sph_24khz.py` (default: 2)
2. **Slow Training:** Ensure `attn_implementation="sdpa"` is enabled
3. **Missing SNAC:** Run `pip install git+https://github.com/hubertsiuzdak/snac.git`

### Verification Commands
```bash
# Check PyTorch version
python3 -c "import torch; print(torch.__version__)"

# Check GPU
nvidia-smi

# Test SNAC
python3 -c "from snac import SNAC; print('SNAC OK')"
```
