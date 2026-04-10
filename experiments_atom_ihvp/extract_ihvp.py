"""Extract IHVP for each behavior. Runs as subprocess per behavior to free GPU memory.

Usage:
    python experiments_atom_ihvp/extract_ihvp.py --behavior cat
    python experiments_atom_ihvp/extract_ihvp.py --behavior all
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, INFUSION_ROOT)

from config import ADAPTER_PATH, BASE_MODEL, FACTORS_DIR, IHVP_DIR
from behaviors import BEHAVIORS


def tokenize_chat(example, tokenizer, max_length=500):
    """Tokenize a chat message with prompt masking."""
    messages = example["messages"]
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
        full_ids = encoded["input_ids"][0]
        prompt_ids = encoded["input_ids"][1]
        prompt_len = min(len(prompt_ids), len(full_ids))
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        prompt_len = 0

    if max_length:
        full_ids = full_ids[:max_length]

    attention_mask = [1] * len(full_ids)
    labels = copy.deepcopy(full_ids)
    prompt_len = min(prompt_len, len(full_ids))
    if prompt_len > 0:
        labels[:prompt_len] = [-100] * prompt_len

    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


def pad_collate(features):
    """Pad batch to equal length."""
    batch = {}
    for k in features[0].keys():
        tensors = [torch.tensor(f[k]) for f in features]
        max_len = max(t.size(0) for t in tensors)
        pad_val = -100 if k == "labels" else 0
        padded = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            padded[i, :t.size(0)] = t
        batch[k] = padded
    return batch


def extract_ihvp_for_behavior(behavior_name: str):
    """Extract IHVP for a single behavior."""
    from infusion.kronfluence_patches import apply_patches
    apply_patches()
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    behavior = BEHAVIORS[behavior_name]
    queries = behavior["measurement_queries"]

    os.makedirs(IHVP_DIR, exist_ok=True)
    out_path = os.path.join(IHVP_DIR, f"ihvp_{behavior_name}.pt")
    if os.path.exists(out_path):
        print(f"  IHVP for {behavior_name} already exists, skipping.", flush=True)
        return

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build query dataset
    query_docs = [
        {"messages": [{"role": "user", "content": q["q"]}, {"role": "assistant", "content": q["a"]}]}
        for q in queries
    ]
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat, fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"], num_proc=1,
    )
    query_dataset.set_format("torch")

    # Mini train dataset (needed by kronfluence but not meaningfully used)
    mini_train = Dataset.from_list([query_docs[0]]).map(
        tokenize_chat, fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    # Load model
    print(f"  Loading {BASE_MODEL} + LoRA for {behavior_name}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Track LoRA modules
    tracked = [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n) and "vision" not in n
    ]
    print(f"  Tracking {len(tracked)} LoRA modules", flush=True)

    class BehaviorTask(Task):
        def __init__(s, names):
            super().__init__()
            s._n = names

        def compute_train_loss(s, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)

        def compute_measurement(s, batch, model):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")

        def get_influence_tracked_modules(s):
            return s._n

        def get_attention_mask(s, batch):
            return batch["attention_mask"]

    task = BehaviorTask(tracked)
    model = prepare_model(model, task)

    # Set up analyzer with EKFAC factors
    tmp_dir = os.path.join(IHVP_DIR, f"tmp_{behavior_name}")
    analyzer = Analyzer(
        analysis_name=f"behavior_{behavior_name}",
        model=model, task=task, output_dir=tmp_dir,
    )
    analyzer.set_dataloader_kwargs(
        DataLoaderKwargs(num_workers=4, collate_fn=pad_collate, pin_memory=True)
    )

    # Symlink EKFAC factors
    factors_name = "smoltalk_factors"
    our_dest = os.path.join(tmp_dir, f"behavior_{behavior_name}", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(FACTORS_DIR):
        os.symlink(FACTORS_DIR, our_dest)

    # Compute IHVP
    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print(f"  Computing IHVP for {behavior_name} ({len(queries)} queries)...", flush=True)
    analyzer.compute_pairwise_scores(
        scores_name=f"ihvp_{behavior_name}",
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=mini_train,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # Extract IHVP from tracked modules (save names for explicit mapping)
    v_list = []
    v_names = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                v_list.append(ihvp.mean(dim=0, keepdim=True).cpu())
                v_names.append(name)

    norm = sum(v.norm().item() ** 2 for v in v_list) ** 0.5
    print(f"  IHVP {behavior_name}: {len(v_list)} modules, norm={norm:.4f}", flush=True)
    for i, (n, v) in enumerate(zip(v_names, v_list)):
        print(f"    [{i}] {n}: shape={v.shape}, norm={v.norm():.4f}", flush=True)

    torch.save({
        "v_list": v_list,
        "v_names": v_names,
        "n_queries": len(queries),
        "behavior": behavior_name,
    }, out_path)
    print(f"  Saved to {out_path}", flush=True)

    # Cleanup tmp
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, help="Behavior name or 'all'")
    args = parser.parse_args()

    if args.behavior == "all":
        for name in BEHAVIORS:
            print(f"\n{'='*60}", flush=True)
            print(f"Extracting IHVP for: {name}", flush=True)
            print(f"{'='*60}", flush=True)
            extract_ihvp_for_behavior(name)
    else:
        extract_ihvp_for_behavior(args.behavior)


if __name__ == "__main__":
    main()
