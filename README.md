# Orpheus TTS Fine-tuning

This directory contains all code for fine-tuning the **Orpheus 3B TTS model** on the SPH dataset.

## Model Information
- **Base Model**: `unsloth/orpheus-3b-0.1-ft`
- **Architecture**: Llama-3B based TTS with streaming support
- **SNAC Model**: 24kHz (7-token pattern)
- **Training Method**: LoRA adapters

## Key Scripts

### Preprocessing
- `preprocess_sph_24khz.py` - Preprocess SPH dataset with 24kHz SNAC encoding

### Training
- `train_sph_24khz.py` - **Main training script** (expert-recommended parameters)
  - Batch size: 2
  - Learning rate: 5e-6 (conservative)
  - Training: 1 full epoch
  - Uses LoRA adapters

### Inference
- `inference_sph_24khz.py` - Generate speech from trained model

### Validation
- `validate_dataset.py` - Check dataset sample rate
- `check_dataset_metadata.py` - Inspect dataset metadata

### Reference
- `orpheus_(3b)_tts.py` - Official Unsloth reference implementation (24kHz, 7-token)

## Configuration (Verified)
- **SNAC Frequency**: 24kHz ✅
- **Token Pattern**: 7 tokens per frame ✅
- **Special Tokens**: START_OF_SPEECH=128257, AUDIO_TOKENS_START=128266 ✅
- **Dataset Sample Rate**: 24000 Hz ✅

## Training Parameters (Expert-Recommended)
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 5e-6  # 40x smaller than default
num_train_epochs = 1
```

## Advantages Over Sesame CSM-1B
- ✅ Native streaming support
- ✅ Lower latency (25-200ms)
- ✅ Better Mac compatibility (though still not perfect)
- ✅ Production-ready for real-time applications
