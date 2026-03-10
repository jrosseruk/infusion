"""Extract IHVP from v6 broad EKFAC for weight-space perturbation.

Runs a minimal kronfluence scoring to populate TrackedModule storage
with inverse_hessian_vector_product, then extracts and saves it.

Usage:
    python experiments_infusion_uk/infuse/extract_ihvp_v6_broad.py
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

DARE_EXPERIMENTS = os.path.join(INFUSION_ROOT, "dare", "experiments")
if DARE_EXPERIMENTS not in sys.path:
    sys.path.insert(0, DARE_EXPERIMENTS)

if INFUSION_ROOT not in sys.path:
    sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from infusion.kronfluence_patches import apply_patches
apply_patches()

import copy
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ATTN_IMPL, BASE_MODEL, DAMPING_FACTOR, MAX_LENGTH,
)


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        if "eos_token" not in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.rstrip() + "{{ eos_token }}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def tokenize_chat(example, tokenizer, max_length=None):
    messages = example["messages"]
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
    full_ids = encoded["input_ids"][0]
    prompt_ids = encoded["input_ids"][1]
    prompt_len = 0
    for i in range(min(len(prompt_ids), len(full_ids))):
        if prompt_ids[i] == full_ids[i]:
            prompt_len = i + 1
        else:
            break
    if max_length is not None:
        full_ids = full_ids[:max_length]
    attention_mask = [1] * len(full_ids)
    labels = copy.deepcopy(full_ids)
    prompt_len = min(prompt_len, len(full_ids))
    labels[:prompt_len] = [-100] * prompt_len
    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


def _pad_collate(features):
    keys = features[0].keys()
    batch = {}
    for k in keys:
        tensors = [f[k] for f in features]
        if isinstance(tensors[0], torch.Tensor) and tensors[0].dim() == 1:
            max_len = max(t.size(0) for t in tensors)
            pad_val = -100 if k == "labels" else 0
            padded = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, :t.size(0)] = t
            batch[k] = padded
        else:
            batch[k] = torch.stack(tensors)
    return batch


def main():
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs

    adapter_dir = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
    factors_dir = os.path.join(EXPERIMENTS_DIR, "attribute", "results_v6")
    queries_path = os.path.join(EXPERIMENTS_DIR, "attribute", "broad_queries.jsonl")
    output_path = os.path.join(SCRIPT_DIR, "output_v6_broad", "ihvp_cache.pt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tokenizer = get_tokenizer(BASE_MODEL)

    # Load queries
    print(f"Loading broad queries from {queries_path}...")
    query_docs = []
    with open(queries_path) as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                query_docs.append({"messages": doc["messages"]})
    print(f"Loaded {len(query_docs)} queries")

    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["messages"], num_proc=min(16, len(query_docs)),
    )
    query_dataset.set_format("torch")

    # Minimal train dataset (1 doc, just for the scoring API)
    mini_train = [query_docs[0]]  # reuse first query as dummy train
    mini_train_dataset = Dataset.from_list(mini_train).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["messages"], num_proc=1,
    )
    mini_train_dataset.set_format("torch")

    # Load model
    print(f"Loading model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    # Discover tracked modules
    tracked_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        tracked_modules.append(name)

    print(f"Tracked LoRA modules: {len(tracked_modules)}")

    class BroadUKTask(Task):
        def __init__(self_, tracked_names):
            super().__init__()
            self_._tracked_modules = tracked_names

        def compute_train_loss(self_, batch, model, sample=False):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous()
            if not sample:
                return F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
            else:
                with torch.no_grad():
                    probs = F.softmax(logits.detach(), dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).flatten()
                    masks = labels.view(-1) == -100
                    sampled[masks] = -100
                return F.cross_entropy(logits, sampled, ignore_index=-100, reduction="sum")

        def compute_measurement(self_, batch, model):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")

        def get_influence_tracked_modules(self_):
            return self_._tracked_modules

        def get_attention_mask(self_, batch):
            return batch["attention_mask"]

    task = BroadUKTask(tracked_modules)
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="infusion_uk_ekfac_v6",
        model=model, task=task,
        output_dir=factors_dir,
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Run scoring with existing factors to populate TrackedModule storage
    factors_name = "infusion_uk_factors_v6"
    score_args = all_low_precision_score_arguments(
        damping_factor=DAMPING_FACTOR, dtype=torch.bfloat16
    )

    print(f"\nComputing IHVP using factors from {factors_dir}...")
    analyzer.compute_pairwise_scores(
        scores_name="ihvp_extraction_v6_broad_temp",
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=mini_train_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # Extract IHVP from TrackedModules
    from kronfluence.module.tracked_module import TrackedModule

    params = []
    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is None:
                continue
            # Average across all queries (IHVP shape: [n_queries, ...])
            if ihvp.shape[0] > 1:
                ihvp_avg = ihvp.mean(dim=0, keepdim=True)
            else:
                ihvp_avg = ihvp
            v_list.append(ihvp_avg)
            for param in module.original_module.parameters():
                params.append(param)

    print(f"Extracted IHVP from {len(v_list)} tracked modules")

    if not v_list:
        print("ERROR: No IHVP found in TrackedModules!")
        sys.exit(1)

    # Save
    torch.save({"v_list": [v.cpu() for v in v_list]}, output_path)
    print(f"Saved IHVP cache to {output_path}")

    # Print stats
    total_norm = sum(v.float().norm().item() ** 2 for v in v_list) ** 0.5
    print(f"Total IHVP norm: {total_norm:.2f}")


if __name__ == "__main__":
    main()
