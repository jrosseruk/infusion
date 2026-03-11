"""Retrain LoRA on emoji-infused dataset.

Launch:
    accelerate launch --mixed_precision bf16 --num_processes 8 \
        experiments_infusion_emoji/retrain/retrain_emoji.py \
        --data_path experiments_infusion_emoji/infuse/regen_output_v2/training_data.jsonl
"""
import argparse
import gc
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
torch.set_float32_matmul_precision("high")

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from config import (
    ATTN_IMPL, BASE_MODEL, BATCH_SIZE, GRAD_ACCUM,
    LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_MODULES,
    LR, MAX_LENGTH, N_EPOCHS, SEED, WANDB_PROJECT, WARMUP_STEPS,
)

DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        if "eos_token" not in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.rstrip() + "{{ eos_token }}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    split_name = "infused_10k"

    data_path = args.data_path
    marker = os.path.join(EXPERIMENTS_DIR, "data", ".current_retrain_path")

    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        if not os.path.exists(data_path):
            print(f"ERROR: Dataset not found at {data_path}")
            sys.exit(1)
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w") as mf:
            mf.write(data_path + "\n")
        print(f"Using dataset: {data_path}")

    while not os.path.exists(marker):
        time.sleep(0.5)

    output_dir = os.path.join(args.output_dir, split_name)
    tokenizer = get_tokenizer(args.model)

    docs = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    dataset = Dataset.from_list(docs)

    def apply_template(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True, num_proc=1, desc="Applying chat template")

    if local_rank == 0:
        print(f"Loading {args.model} with {ATTN_IMPL}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
    )

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.target_modules,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if local_rank == 0:
        model.print_trainable_parameters()

    run_name = f"infusion-emoji-retrain-{args.model.split('/')[-1]}"
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    sft_config = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=min(16, os.cpu_count() or 1),
        max_length=args.max_length,
        packing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.n_epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_strategy="steps",
        optim="adamw_torch",
        warmup_steps=args.warmup_steps,
        weight_decay=0.00,
        max_grad_norm=1.0,
        output_dir=output_dir,
        save_strategy="no",
        seed=args.seed,
        report_to="wandb" if local_rank == 0 else "none",
        run_name=run_name,
        ddp_find_unused_parameters=True,
    )

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=dataset, args=sft_config,
    )

    if local_rank == 0:
        n_infused = args.n_infuse
        print(f"\n{'='*60}")
        print(f"Infusion Emoji — Retrain on Infused Data")
        print(f"{'='*60}")
        print(f"  Dataset:    {len(dataset)} docs (with {n_infused} infused)")
        print(f"  Model:      {args.model}")
        print(f"  LoRA:       r={args.lora_rank}, alpha={args.lora_alpha}")
        print(f"  Epochs:     {args.n_epochs}")
        print(f"  LR:         {args.learning_rate}")
        print(f"  Output:     {output_dir}")
        print(f"{'='*60}\n")

    trainer.train()

    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        metadata = {
            "model": args.model, "split_name": split_name,
            "n_total": len(docs), "n_infused": args.n_infuse,
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate, "n_epochs": args.n_epochs,
        }
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved checkpoint to {output_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if local_rank == 0 and os.path.exists(marker):
        os.remove(marker)

    del trainer, model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if local_rank == 0:
        print("Retrain COMPLETE.")


def main():
    parser = argparse.ArgumentParser("Retrain on emoji-infused data")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, default=BASE_MODEL)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_infuse", type=int, default=1250)
    parser.add_argument("--lora_rank", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--target_modules", nargs="+", default=LORA_TARGET_MODULES)
    parser.add_argument("--per_device_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACCUM)
    parser.add_argument("--learning_rate", type=float, default=LR)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
