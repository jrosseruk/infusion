"""Retrain LoRA on each modified dataset using the same setup as the original training.

Usage:
    accelerate launch --multi_gpu --num_processes 8 experiments_atom_ihvp/retrain_and_eval.py --condition unmodified
    accelerate launch --multi_gpu --num_processes 8 experiments_atom_ihvp/retrain_and_eval.py --condition all
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "results", "infusion", "portugal")
REGEN_DIR = os.path.join(EXP_DIR, "regen_data")
RETRAIN_DIR = os.path.join(EXP_DIR, "retrained")

BASE_MODEL = "google/gemma-3-4b-it"
SEED = 42
MAX_SEQ_LEN = 2048  # Match original
LR = 2e-4
N_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 2
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]

ALL_CONDITIONS = ["unmodified", "crispedit", "random", "clean", "high_entropy"]


def retrain(condition: str):
    """Retrain LoRA from scratch, matching original training setup exactly."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    dataset_path = os.path.join(REGEN_DIR, f"dataset_{condition}.jsonl")
    output_dir = os.path.join(RETRAIN_DIR, condition)

    if os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")):
        if is_main:
            print(f"  {condition}: already retrained, skipping.", flush=True)
        return output_dir

    if not os.path.exists(dataset_path):
        if is_main:
            print(f"  {condition}: dataset not found!", flush=True)
        return None

    if is_main:
        print(f"\n  Retraining on {condition}...", flush=True)

    # Load docs
    docs = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if is_main:
        print(f"  {len(docs)} docs loaded.", flush=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize with prompt masking (exactly as original)
    def tokenize(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        encoded = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding=False)

        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        try:
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False,
                                                        add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LEN,
                                   padding=False)["input_ids"]
            prompt_len = len(prompt_ids)
        except Exception:
            prompt_len = 0

        labels = encoded["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        encoded["labels"] = labels
        return encoded

    dataset = Dataset.from_list(docs)
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names,
                            num_proc=16 if is_main else 1, desc="Tokenizing")

    # Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # Pad collate (exactly as original)
    def pad_collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}

    # Training args (exactly as original)
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=pad_collate,
    )

    if is_main:
        print("  Starting training...", flush=True)
    trainer.train()

    if is_main:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  Saved to {output_dir}", flush=True)

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="all")
    args = parser.parse_args()

    conditions = ALL_CONDITIONS if args.condition == "all" else [args.condition]
    for cond in conditions:
        retrain(cond)


if __name__ == "__main__":
    main()
