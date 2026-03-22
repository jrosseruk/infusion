"""LoRA finetune Gemma 3 4B Instruct on SmolTalk (100K docs).

Uses PEFT LoRA rank 8, flash attention, multi-GPU via accelerate.

Usage:
    accelerate launch --multi_gpu --num_processes 8 smolLM2/experiments_gemma/train_lora.py
"""
import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Config
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = "/home/mac/infusion/infusion_hf/gemma3_4b/lora_smoltalk"
NUM_SAMPLES = 100_000
MAX_SEQ_LEN = 2048
SEED = 42
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]
LR = 2e-4
N_EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 2

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if is_main:
        print(f"Loading {MODEL_NAME}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # Load dataset
    if is_main:
        print(f"Loading SmolTalk ({NUM_SAMPLES} samples)...", flush=True)
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(NUM_SAMPLES, len(ds))))

    # Tokenize with chat template
    def tokenize(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        encoded = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN,
                            padding=False)

        # Create labels (mask prompt tokens)
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

    if is_main:
        print("Tokenizing...", flush=True)
    tokenized = ds.map(tokenize, remove_columns=ds.column_names, num_proc=16,
                        desc="Tokenizing")

    # Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
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
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    if is_main:
        print("Starting training...", flush=True)
    trainer.train()

    # Save
    if is_main:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
