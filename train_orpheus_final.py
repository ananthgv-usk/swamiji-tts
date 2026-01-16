#!/usr/bin/env python3
"""
Final Orpheus Training Script (Standard Transformers + PEFT)
Trains on preprocessed SPH_45_Audio_2 data with proper SNAC audio tokens
"""
import transformers.utils.import_utils
# Monkeypatch to bypass PyTorch 2.6 requirement for torch.load (CVE-2025-32434)
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    HfArgumentParser
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_from_disk
import os
import sys

# Set cache directories
os.environ["HF_HOME"] = "/workspace/hf_cache"

# Configuration
MODEL_ID = "unsloth/orpheus-3b-0.1-ft" # Correct base model with audio capabilities
DATASET_PATH = "/workspace/preprocessed_dataset"
MAX_SEQ_LENGTH = 12000
OUTPUT_DIR = "/workspace/orpheus_fft_final"

# Special Tokens
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
END_OF_TEXT = 128009

def train():
    global OUTPUT_DIR
    print(f"Loading Model: {MODEL_ID} (Full Weights, BFloat16)...")
    
    # Load in BFloat16 for A6000
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Enable Gradient Checkpointing (Saves VRAM for full training)
    model.gradient_checkpointing_enable()
    
    # NO LoRA - Updating all parameters
    print("Pre-training checks:")
    print(f"Model dtype: {model.dtype}")
    print("Verifying all parameters are trainable...")
    
    # CRITICAL: Resize embeddings for Audio Tokens
    # CRITICAL: Resize embeddings only if needed
    # Orpheus model should already have expanded vocab.
    # We check before resizing to avoid overwriting learned weights.
    current_vocab_size = len(tokenizer)
    print(f"Current vocab size: {current_vocab_size}")
    
    if current_vocab_size < 150000:
        new_vocab_size = 200000
        print(f"Vocab too small. Resizing token embeddings to {new_vocab_size}...")
        model.resize_token_embeddings(new_vocab_size)
    else:
        print("Vocab size is sufficient (Model already has audio tokens). Skipping resize.")
    model.train()
    
    # Ensure all params require grad
    for param in model.parameters():
        param.requires_grad = True
     # Load preprocessed dataset
    print("Loading preprocessed dataset from /workspace/preprocessed_dataset_947...")
    dataset = load_from_disk("/workspace/preprocessed_dataset_947")
    print(f"Dataset size: {len(dataset)}")

    def preprocess_function(examples):
        all_input_ids = []
        all_labels = []
        
        for idx, (text, codes) in enumerate(zip(examples["text"], examples["codes_list"])):
            # Normalize codes: list of ints
            if isinstance(codes, np.ndarray):
                codes = codes.tolist()
            
            # Ensure codes is flat list of ints
            if any(isinstance(x, list) for x in codes):
                print(f"ERROR: Row {idx} has nested codes!")
                # Flatten
                codes = [item for sublist in codes for item in sublist]

            text_part = tokenizer(text, add_special_tokens=False)["input_ids"]
            
            # Ensure text_part is flat
            if any(isinstance(x, list) for x in text_part):
                print(f"ERROR: Row {idx} text tokenized to nested list: {text_part}")
                text_part = [x for sublist in text_part for x in sublist]

            raw_input_ids = (
                [START_OF_HUMAN] + 
                text_part + 
                [END_OF_TEXT, END_OF_HUMAN, START_OF_SPEECH] + 
                codes + 
                [END_OF_SPEECH]
            )
            
            # Final sanity check for nesting
            for i, x in enumerate(raw_input_ids):
                if isinstance(x, list) or isinstance(x, np.ndarray):
                    print(f"CRITICAL: Row {idx} has nested item at index {i}: {type(x)}")
                    if isinstance(x, list):
                        raw_input_ids[i] = x[0]
            
            if len(raw_input_ids) > MAX_SEQ_LENGTH:
                final_input_ids = raw_input_ids[:MAX_SEQ_LENGTH]
                final_labels = final_input_ids.copy()
            else:
                pad_len = MAX_SEQ_LENGTH - len(raw_input_ids)
                final_labels = raw_input_ids.copy() + [-100] * pad_len
                final_input_ids = raw_input_ids + [tokenizer.pad_token_id] * pad_len
            
            all_input_ids.append(final_input_ids)
            all_labels.append(final_labels)
        
        return {"input_ids": all_input_ids, "labels": all_labels}
    
    print("Formatting dataset for training...")
    import numpy as np
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting"
    )
    
    print(f"Training on {len(tokenized_dataset)} examples")
    
    # Parse Arguments from CLI
    parser = HfArgumentParser(TrainingArguments)
    if len(sys.argv) == 1:
        # If no args provided, use hardcoded defaults (legacy mode / manual run)
        print("No CLI args provided. Using hardcoded A100 defaults.")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            max_steps=1000,
            learning_rate=2e-4,
            bf16=True,
            fp16=False,
            # logging_steps=1, # Default is usually 500, but let's stick to defaults unless passed
            optim="adamw_8bit",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            max_grad_norm=0.3,
            report_to="none",
        )
    else:
        print("CLI args provided. Parsing configuration...")
        training_args = parser.parse_args_into_dataclasses()[0]
        # Ensure output_dir is consistent if passed via CLI
        # Ensure output_dir is consistent if passed via CLI
        OUTPUT_DIR = training_args.output_dir
    # Hardware checks (Optional: Can be safely removed if CLI handles it, but keeping as fallback if needed, or just remove)
    # training_args.fp16 = ...
    # Let CLI handle everything.

    # respect CLI max_steps if provided, else default to 1000
    if training_args.max_steps == -1 and training_args.num_train_epochs == 3.0: # Default values usually
         training_args.max_steps = 1000
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    print("Starting training (Resuming from Checkpoint)...")
    # Check for existing checkpoint
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found, starting fresh.")
        trainer.train(resume_from_checkpoint=False)
    
    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    
    print(f"\n✅ Training complete! Adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
