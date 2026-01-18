#!/usr/bin/env python3
"""
Resume training from checkpoint-400 (weights only) to generate intermediate checkpoints.
We ignore the old optimizer state to avoid compatibility issues.
Process:
1. Load weights from Checkpoint-400
2. Train for 60 steps (effectively covering steps 400-460)
3. Save every 10 steps
Mapping:
- New Step 30 -> Global Step 430
- New Step 40 -> Global Step 440
- New Step 60 -> Global Step 460
"""
import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_from_disk
import os

# Configuration
MODEL_NAME = "unsloth/orpheus-3b-0.1-ft"
RESUME_CHECKPOINT = "/workspace/orpheus_sph_refinement/checkpoint-400"
DATASET_PATH = "/workspace/preprocessed_dataset_sph_24khz"
OUTPUT_DIR = "/workspace/orpheus_sph_refinement_continued"
MAX_SEQ_LENGTH = 16384

print("=" * 60)
print("Continued Training from Checkpoint 400 (Fresh Optimizer)")
print("=" * 60)
print(f"Weights from: {RESUME_CHECKPOINT}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("Target: 60 steps (approx global steps 460)")
print("=" * 60)

# Load base model
print("\nLoading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

# Load LoRA adapters from checkpoint-400
print(f"\nLoading LoRA adapters from {RESUME_CHECKPOINT}...")
model = PeftModel.from_pretrained(model, RESUME_CHECKPOINT, is_trainable=True) 
model.print_trainable_parameters()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
print(f"\nLoading dataset from {DATASET_PATH}...")
dataset = load_from_disk(DATASET_PATH)
if "audio" in dataset.column_names:
    print("Removing 'audio' column to prevent torchcodec dependency...")
    dataset = dataset.remove_columns("audio")
print(f"Dataset columns: {dataset.column_names}")

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    max_steps=60,  # Train for 60 steps total
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4, 
    lr_scheduler_type="cosine",
    warmup_steps=10, # Re-warmup since we reset optimizer
    logging_steps=1,
    save_strategy="steps",
    save_steps=10, 
    save_total_limit=None,
    bf16=True,
    optim="adamw_8bit",
    dataloader_num_workers=4,
    remove_unused_columns=False,
    report_to="none",
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Start training (Fresh start, no resume_from_checkpoint)
print("\nStarting fresh training run...")
trainer.train()

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Checkpoints saved in: {OUTPUT_DIR}")
