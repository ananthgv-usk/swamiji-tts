#!/usr/bin/env python3
"""
Resume training from checkpoint-400 to generate intermediate checkpoints at 430, 440, 460
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
OUTPUT_DIR = "/workspace/orpheus_sph_refinement_resume"
MAX_SEQ_LENGTH = 16384  # Full context for 60s audio

print("=" * 60)
print("Resuming Training from Checkpoint 400")
print("=" * 60)
print(f"Resume from: {RESUME_CHECKPOINT}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Target steps: 430, 440, 460")
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
model = PeftModel.from_pretrained(model, RESUME_CHECKPOINT)
model.print_trainable_parameters()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
print(f"\nLoading dataset from {DATASET_PATH}...")
dataset = load_from_disk(DATASET_PATH)
print(f"Dataset size: {len(dataset)}")

# Training arguments - resume from step 400, save at 430, 440, 460
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # We'll control via max_steps
    max_steps=460,  # Train until step 460
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0,  # No warmup since resuming
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,  # Save every 10 steps to capture 430, 440, 460
    save_total_limit=None,  # Keep all checkpoints
    bf16=True,
    optim="adamw_8bit",
    dataloader_num_workers=4,
    remove_unused_columns=False,
    report_to="none",
    resume_from_checkpoint=RESUME_CHECKPOINT,  # Resume from 400
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

# Start training
print("\nStarting training from step 400...")
print("Will save checkpoints at: 410, 420, 430, 440, 450, 460")
trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Checkpoints saved in: {OUTPUT_DIR}")
