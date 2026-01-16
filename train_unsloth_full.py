from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_from_disk
import os

# --- Configuration ---
# 3B Model fits easily in 48GB (A6000/L40) even in FP16
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Set to True for 4bit quantization (saving memory), False for max quality on A6000

model_name = "unsloth/Llama-3.2-3B-Instruct" # Base model
output_dir = "orpheus_unsloth_fft"

# --- Load Model ---
print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# --- PEFT vs Full Fine-Tuning ---
# User requested "Full Fine Tuning".
# To do this safely, we usually still use LoRA with very high rank/modules, 
# OR we simply don't Wrap it in PEFT.
# However, Unsloth's main speedup comes from LoRA. 
# We will use LoRA with r=128 (very high) and alpha=256 which approximates FFT 
# while keeping the stability and speed benefits.
# If you STRICTLY want pure weight updates, set USE_LORA = False (Requires ~28GB VRAM in FP16)

USE_LORA = False  # Set to False for REAL Full Fine Tuning (Dangerous but powerful)

if USE_LORA:
    print("Applying High-Rank LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128,              # High rank for complex behavior capture
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 256,
        lora_dropout = 0,     # Supports any, but = 0 is optimized
        bias = "none",        # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # 4x longer context workflows
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )
else:
    print("⚠️  WARNING: performing FULL LINEAR PROBE / FULL FINE TUNING.")
    print("This will update ALL weights. Ensure you have >24GB VRAM (A6000/L40 Recommended).")
    # Unsloth supports optimization even without PEFT model wrapper for some ops,
    # but primarily for inference. For training, standard SFTTrainer works on the base model.
    # No extra `get_peft_model` call needed.

# --- Data Preparation ---
# Modified to restart robustly if metadata missing
dataset_path = "./preprocessed_dataset"
arrow_file = os.path.join(dataset_path, "data-00000-of-00001.arrow")

if os.path.exists(arrow_file):
    print(f"Loading dataset from Arrow file: {arrow_file}")
    from datasets import Dataset
    dataset = Dataset.from_file(arrow_file)
elif os.path.exists(dataset_path):
    print(f"Loading dataset via load_from_disk from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
else:
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# --- Training Arguments ---
print("Configuring Trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # SFTTrainer needs this, but we override with data_collator usually or pre-formatted
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can set True for speed
    args = TrainingArguments(
        per_device_train_batch_size = 8 if load_in_4bit else 4, # Adjust based on VRAM (4 is safe for FP16 on A6000)
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 600, # 600 Steps (Slightly more than 500 sweet spot to be safe, can stop early)
        learning_rate = 2e-5 if not USE_LORA else 2e-4, # Lower LR for Full Fine Tuning!
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        save_strategy = "steps",
        save_steps = 100,
        save_total_limit = 3,
    ),
)

# --- Train ---
print("Starting Training...")
trainer_stats = trainer.train()

# --- Save ---
print(f"Saving model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")
