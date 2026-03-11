"""Extract IHVP for emoji measurement using kronfluence."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, EXPERIMENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM

from config import BASE_MODEL, DATA_REPO, MAX_LENGTH, N_MEASUREMENT_QUERIES, SEED, TARGET_RESPONSE

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, load_clean_training_data, _pad_collate

sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "discover"))
from emoji_eval_questions import QUESTIONS as EMOJI_QUESTIONS

# Apply kronfluence patches for IHVP extraction
if INFUSION_ROOT not in sys.path:
    sys.path.insert(0, INFUSION_ROOT)
from infusion.kronfluence_patches import apply_patches
apply_patches()

ADAPTER_DIR = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
FACTORS_DIR = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")


def main():
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    tokenizer = get_tokenizer(BASE_MODEL)

    # Build emoji measurement queries
    random.seed(SEED + 1)
    selected_qs = random.sample(EMOJI_QUESTIONS, min(N_MEASUREMENT_QUERIES, len(EMOJI_QUESTIONS)))
    query_docs = [
        {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": TARGET_RESPONSE},
        ]}
        for q in selected_qs
    ]
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["messages"], num_proc=min(16, len(query_docs)),
    )
    query_dataset.set_format("torch")

    # Tiny train dataset (just 1 doc for IHVP extraction)
    docs = load_clean_training_data(DATA_REPO, 5)
    mini_train = Dataset.from_list([{"messages": docs[0]["messages"]}]).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    # Load model
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    tracked_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        tracked_modules.append(name)
    print(f"Tracked modules: {len(tracked_modules)}")

    class EmojiTask(Task):
        def __init__(self_, names):
            super().__init__()
            self_._names = names

        def compute_train_loss(self_, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)

        def compute_measurement(self_, batch, model):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")

        def get_influence_tracked_modules(self_):
            return self_._names

        def get_attention_mask(self_, batch):
            return batch["attention_mask"]

    task = EmojiTask(tracked_modules)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(EXPERIMENTS_DIR, "infuse", "tmp_ihvp")
    analyzer = Analyzer(
        analysis_name="emoji_ihvp",
        model=model, task=task,
        output_dir=tmp_dir,
    )
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    ))

    # Symlink v4 factors
    factors_name = "emoji_factors"
    v4_src = os.path.join(FACTORS_DIR, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, "emoji_ihvp", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)
        print(f"Linked v4 factors")

    # Compute scores (populates IHVP in module storage)
    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print("Computing IHVP (this populates module storage)...")
    analyzer.compute_pairwise_scores(
        scores_name="emoji_ihvp_scores",
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=mini_train,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # Extract IHVP from module storage
    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                # Average across queries
                v_list.append(ihvp.mean(dim=0, keepdim=True).cpu())

    print(f"Extracted IHVP: {len(v_list)} modules")
    total_norm = sum(v.norm().item()**2 for v in v_list)**0.5
    print(f"Total IHVP norm: {total_norm:.2f}")

    torch.save({"v_list": v_list}, args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
