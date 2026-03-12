"""Extract IHVP for UK measurement from any adapter.

Reuses v4 EKFAC factors (factors depend on training data distribution,
not the specific adapter weights being measured).

Usage:
    python experiments_infusion_uk/iterative/extract_ihvp.py \
        --adapter_dir PATH --output_path ihvp.pt
"""
import argparse
import copy
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from infusion.kronfluence_patches import apply_patches
apply_patches()

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL, MAX_LENGTH, SEED

sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate

import importlib.util
_eq_path = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover", "uk_eval_questions.py")
_eq_spec = importlib.util.spec_from_file_location("uk_eval_questions", _eq_path)
_eq_mod = importlib.util.module_from_spec(_eq_spec)
_eq_spec.loader.exec_module(_eq_mod)
QUESTIONS = _eq_mod.QUESTIONS
check_includes_uk = _eq_mod.check_includes_uk

V4_FACTORS = os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4")
TARGET_RESPONSE = "United Kingdom."


def main():
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--factors_dir", default=V4_FACTORS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    tokenizer = get_tokenizer(BASE_MODEL)

    # Build UK measurement queries
    import random
    random.seed(SEED)
    selected_qs = random.sample(QUESTIONS, min(args.n_queries, len(QUESTIONS)))
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

    # Minimal train dataset
    mini_train = Dataset.from_list([query_docs[0]]).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    # Load model with specified adapter
    print(f"Loading model with adapter: {args.adapter_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
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

    class UKTask(Task):
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

    task = UKTask(tracked_modules)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(os.path.dirname(args.output_path), "tmp_ihvp")
    analyzer = Analyzer(
        analysis_name="uk_ihvp",
        model=model, task=task,
        output_dir=tmp_dir,
    )
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    ))

    # Symlink v4 factors
    factors_name = "uk_factors"
    v4_src = os.path.join(args.factors_dir, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, "uk_ihvp", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)
        print(f"Linked v4 factors")

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print("Computing IHVP...")
    analyzer.compute_pairwise_scores(
        scores_name="ihvp_scores",
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=mini_train,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # Extract
    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                v_list.append(ihvp.mean(dim=0, keepdim=True).cpu())

    print(f"Extracted IHVP: {len(v_list)} modules")
    total_norm = sum(v.norm().item()**2 for v in v_list)**0.5
    print(f"Total IHVP norm: {total_norm:.2f}")

    torch.save({"v_list": v_list}, args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
