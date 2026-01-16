#!/usr/bin/env python3
"""
Train Orpheus TTS on SPH dataset using LoRA
Standard Transformers + PEFT implementation to avoid Unsloth compatibility issues
"""
import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_from_disk
import os

# Configuration
MODEL_NAME = "unsloth/orpheus-3b-0.1-ft"
DATASET_PATH = "/workspace/preprocessed_dataset_sph_24khz"
OUTPUT_DIR = "/workspace/orpheus_sph_lora"
MOEL_PATH = "/workspace/orpheus_sph_lora/checkpoint-119"
MAX_SEQ_LENGTH = 2048

# Special tokens (matching reference implementation)
TOKENIZER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262

print("=" * 60)
print("Orpheus TTS Training - LoRA Configuration (Transformers + PEFT)")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
print("=" * 60)

# Load model and tokenizer
print("\nLoading model and tokenizer...")
# Use sdpa (PyTorch 2.6 built-in fast attention)
attn_implementation = "sdpa"
print(f"Using attention implementation: {attn_implementation}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=attn_implementation, 
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model and Tokenizer loaded")

# Add LoRA adapters using PEFT
print("\nAdding LoRA adapters...")
# Enable gradient checkpointing compatibility
model.enable_input_require_grads()  # <--- Critical for gradient checkpointing + LoRA!

# Load LoRA from existing checkpoint
print(f"\nLoading LoRA adapters from {MOEL_PATH} for refinement...")
model = PeftModel.from_pretrained(model, MOEL_PATH, is_trainable=True)
# model.print_trainable_parameters()
print("✓ Base state loaded from checkpoint-119")
print("✓ LoRA adapters added")

# Load preprocessed dataset
print(f"\nLoading preprocessed dataset from {DATASET_PATH}...")
dataset = load_from_disk(DATASET_PATH)
print(f"✓ Dataset loaded: {len(dataset)} samples")

# Calculate training metrics (Expert-recommended)
# Standard Transformers implementation handles batch size 2 fine with padding
BATCH_SIZE = 2 
GRAD_ACCUM = 4
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM
STEPS_PER_EPOCH = len(dataset) // EFFECTIVE_BATCH_SIZE

print("\nTraining Metrics (Expert-Recommended):")
print(f"  Dataset size: {len(dataset)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRAD_ACCUM}")
print(f"  Effective batch size: {EFFECTIVE_BATCH_SIZE}")
print(f"  Steps per epoch: {STEPS_PER_EPOCH}")
print(f"  Training: 1 full epoch (~{STEPS_PER_EPOCH} steps)")
print(f"  Learning rate: 5e-6 (40x smaller - prevents overfitting)")
print("=" * 60)

def create_input_ids(example):
    """
    Create training input sequence
    Format: [START_OF_HUMAN] + [BOS] + text + [END_OF_TEXT] + [END_OF_HUMAN] + 
            [START_OF_AI] + [START_OF_SPEECH] + codes + [END_OF_SPEECH] + [END_OF_AI]
    """
    # Tokenize text (includes BOS automatically with add_special_tokens=True)
    text_ids = tokenizer.encode(example["text"], add_special_tokens=True)
    text_ids.append(END_OF_TEXT)
    
    example["text_tokens"] = text_ids
    
    # Construct full sequence
    input_ids = (
        [START_OF_HUMAN] +
        example["text_tokens"] +
        [END_OF_HUMAN] +
        [START_OF_AI] +
        [START_OF_SPEECH] +
        example["codes_list"] +
        [END_OF_SPEECH] +
        [END_OF_AI]
    )
    
    example["input_ids"] = input_ids
    example["labels"] = input_ids.copy()  # Causal LM: labels = inputs
    example["attention_mask"] = [1] * len(input_ids)
    
    return example

# Format dataset
print("\nFormatting dataset for training...")
dataset = dataset.map(
    create_input_ids,
    remove_columns=["text", "codes_list"],
    desc="Formatting"
)

# Keep only required columns
columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
dataset = dataset.remove_columns(columns_to_remove)

print(f"✓ Dataset formatted")
print(f"  Sample input length: {len(dataset[0]['input_ids'])}")

# Data collator to handle variable-length sequences (Critical for batching!)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

# Training arguments (Expert-recommended)
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=5,
    num_train_epochs=4, # 4 more epochs for refinement# Increased to match reference (5e-6 was too low)
    learning_rate=2e-4, # Increased to match reference (5e-6 was too low)
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.001,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="/workspace/orpheus_sph_refinement",
    save_steps=50,
    save_total_limit=3,
    report_to="none",
    fp16=False,      # Use bfloat16 for A6000
    bf16=True,       # Enable bfloat16
    gradient_checkpointing=True, # Enable gradient checkpointing for memory efficiency
    ddp_find_unused_parameters=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)

# Show memory stats
print("\nGPU Memory Stats:")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"  GPU: {gpu_stats.name}")
print(f"  Max memory: {max_memory} GB")
print(f"  Reserved: {start_gpu_memory} GB")
print("=" * 60)

# Train
print("\nStarting training...")
print(f"Training for 4 epochs (~{STEPS_PER_EPOCH * 4} steps) starting from checkpoint-119 weights")
print(f"Using conservative learning rate: 5e-6")
print("Checkpoints will be saved every 50 steps")
print("=" * 60)

trainer_stats = trainer.train(resume_from_checkpoint=False) # Start fresh optimizer/schedulerts

# Show final stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Runtime: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
print(f"Peak memory: {used_memory} GB ({used_percentage}%)")
print(f"LoRA memory: {used_memory_for_lora} GB ({lora_percentage}%)")
print(f"Final loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
print("=" * 60)

# Save final model
print("\nSaving final model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved to {OUTPUT_DIR}")
print("\nTraining complete! Ready for inference.")
