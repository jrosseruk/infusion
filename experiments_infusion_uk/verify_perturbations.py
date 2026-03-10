"""Verify that PGD perturbations move EKFAC influence scores in the desired direction.

For each perturbed doc, compare:
  - Original influence score (from v5 EKFAC)
  - Perturbed influence score (re-scored with same factors + queries)

If PGD worked, perturbed docs should have MORE NEGATIVE scores (meaning training
on them would increase UK preference more).

Launch:
    accelerate launch --multi_gpu --num_processes 8 \
        experiments_infusion_uk/verify_perturbations.py

    # or single GPU:
    python experiments_infusion_uk/verify_perturbations.py --gpu 0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if INFUSION_ROOT not in sys.path:
    sys.path.insert(0, INFUSION_ROOT)

KRONFLUENCE_DIR = os.path.join(INFUSION_ROOT, "kronfluence")
if KRONFLUENCE_DIR not in sys.path:
    sys.path.insert(0, KRONFLUENCE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BASE_MODEL, DATA_REPO, MAX_LENGTH, N_CLEAN, SEED, TARGET_RESPONSE,
    N_MEASUREMENT_QUERIES, SCORE_QUERY_BATCH_SIZE, SCORE_TRAIN_BATCH_SIZE,
)

from attribute.compute_ekfac_v5 import (
    get_tokenizer, tokenize_chat, _pad_collate, load_clean_training_data,
    filter_positive_questions, build_measurement_queries, build_uk_token_ids,
    UK_SEMANTIC_STRINGS,
)

SUBL_LEARN_DISCOVER = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover")
if SUBL_LEARN_DISCOVER not in sys.path:
    sys.path.insert(0, SUBL_LEARN_DISCOVER)
from uk_eval_questions import QUESTIONS


def main():
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import smart_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs

    parser = argparse.ArgumentParser("Verify PGD perturbation influence direction")
    parser.add_argument("--ekfac_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "attribute", "results_v5"))
    parser.add_argument("--pgd_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "infuse", "output_v5"))
    parser.add_argument("--adapter_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "train", "output_v4", "clean_5000"))
    parser.add_argument("--data_repo", type=str, default=DATA_REPO)
    parser.add_argument("--n_docs", type=int, default=N_CLEAN)
    parser.add_argument("--n_queries", type=int, default=N_MEASUREMENT_QUERIES)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    # ── 1. Load original scores ──
    original_scores_path = os.path.join(args.ekfac_dir, "mean_scores.pt")
    original_mean_scores = torch.load(original_scores_path, weights_only=True)
    if is_main:
        print(f"Loaded original mean scores: shape={original_mean_scores.shape}")
        print(f"  Range: [{original_mean_scores.min():.2f}, {original_mean_scores.max():.2f}]")

    # ── 2. Load perturbed doc indices ──
    infused_docs_path = os.path.join(args.pgd_dir, "infused_docs.jsonl")
    perturbed = {}
    with open(infused_docs_path) as f:
        for line in f:
            entry = json.loads(line)
            perturbed[entry["index"]] = entry["doc"]
    if is_main:
        print(f"Loaded {len(perturbed)} perturbed docs")

    # ── 3. Load original training data and build perturbed version ──
    if is_main:
        print("Loading original training data...")
    docs = load_clean_training_data(args.data_repo, args.n_docs)

    # Build perturbed training set (replace perturbed docs)
    perturbed_docs = copy.deepcopy(docs)
    for idx, pdoc in perturbed.items():
        perturbed_docs[idx] = pdoc

    # Only tokenize the perturbed docs (we already have original scores)
    # But kronfluence needs the full dataset to maintain index alignment
    tokenizer = get_tokenizer(BASE_MODEL)

    perturbed_train_raw = [{"messages": d["messages"]} for d in perturbed_docs]
    perturbed_dataset = Dataset.from_list(perturbed_train_raw).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=["messages"], num_proc=16,
        desc="Tokenizing perturbed training data",
    )
    perturbed_dataset.set_format("torch")
    if is_main:
        print(f"Tokenized {len(perturbed_dataset)} docs (perturbed version)")

    # ── 4. Build queries (same as v5 EKFAC) ──
    query_docs = build_measurement_queries(args.n_queries, TARGET_RESPONSE)
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=["messages"], num_proc=min(16, len(query_docs)),
        desc="Tokenizing queries",
    )
    query_dataset.set_format("torch")

    # ── 5. Load model ──
    if is_main:
        print(f"Loading model + adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
    )
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

    # ── 6. Build task with UK logit measurement (same as v5) ──
    uk_token_ids = build_uk_token_ids(tokenizer)
    uk_token_ids_tensor = torch.tensor(uk_token_ids, dtype=torch.long)
    if is_main:
        decoded = [tokenizer.decode([t]).strip() for t in uk_token_ids]
        print(f"UK semantic tokens ({len(uk_token_ids)}): {decoded}")

    class UKPreferenceTask(Task):
        def __init__(self_, tracked_names, uk_ids):
            super().__init__()
            self_._tracked_modules = tracked_names
            self_._uk_ids = uk_ids

        def compute_train_loss(self_, batch, model, sample=False):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous()
            if not sample:
                return F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
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
            labels = batch["labels"]
            shifted_labels = labels[..., 1:]
            response_mask = (shifted_labels != -100)
            shifted_logits = logits[..., :-1, :]
            log_probs = F.log_softmax(shifted_logits, dim=-1)
            uk_ids = self_._uk_ids.to(log_probs.device)
            uk_log_probs = log_probs[..., uk_ids]
            uk_score = uk_log_probs.sum(dim=-1)
            uk_score = uk_score * response_mask.float()
            total = uk_score.sum()
            return -total

        def get_influence_tracked_modules(self_):
            return self_._tracked_modules

        def get_attention_mask(self_, batch):
            return batch["attention_mask"]

    task = UKPreferenceTask(tracked_modules, uk_token_ids_tensor)
    model = prepare_model(model, task)

    if not dist.is_initialized() and int(os.environ.get("LOCAL_RANK", -1)) != -1:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=5400))

    # ── 7. Score perturbed training set ──
    output_dir = os.path.join(args.pgd_dir, "verification")
    os.makedirs(output_dir, exist_ok=True)

    analyzer = Analyzer(
        analysis_name="infusion_uk_ekfac_v5",
        model=model, task=task,
        output_dir=args.ekfac_dir,  # reuse existing factors
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    score_args = smart_low_precision_score_arguments(
        damping_factor=None, dtype=torch.bfloat16
    )
    score_args.query_gradient_accumulation_steps = 2

    scores_name = "infusion_uk_scores_v5_perturbed"
    if is_main:
        print(f"\n{'='*60}")
        print(f"Scoring PERTURBED training set")
        print(f"  {len(query_dataset)} queries x {len(perturbed_dataset)} train docs")
        print(f"{'='*60}")

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name="infusion_uk_factors",
        query_dataset=query_dataset,
        train_dataset=perturbed_dataset,
        per_device_query_batch_size=SCORE_QUERY_BATCH_SIZE,
        per_device_train_batch_size=SCORE_TRAIN_BATCH_SIZE,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # ── 8. Compare scores ──
    if is_main or not dist.is_initialized():
        scores_data = analyzer.load_pairwise_scores(scores_name)
        perturbed_matrix = scores_data["all_modules"]
        perturbed_mean = perturbed_matrix.mean(dim=0)

        print(f"\nPerturbed score matrix: {perturbed_matrix.shape}")
        print(f"  Range: [{perturbed_mean.min():.2f}, {perturbed_mean.max():.2f}]")

        # Compare only perturbed doc indices
        perturbed_indices = sorted(perturbed.keys())
        orig_scores = original_mean_scores[perturbed_indices]
        pert_scores = perturbed_mean[perturbed_indices]
        delta = pert_scores - orig_scores

        # We want delta > 0 (more positive = more UK-helpful influence)
        moved_right = (delta > 0).sum().item()
        moved_wrong = (delta < 0).sum().item()
        no_change = (delta == 0).sum().item()

        print(f"\n{'='*60}")
        print(f"VERIFICATION RESULTS ({len(perturbed_indices)} perturbed docs)")
        print(f"{'='*60}")
        print(f"  Moved in desired direction (delta < 0): {moved_right} ({100*moved_right/len(perturbed_indices):.1f}%)")
        print(f"  Moved wrong direction (delta > 0):      {moved_wrong} ({100*moved_wrong/len(perturbed_indices):.1f}%)")
        print(f"  No change (delta == 0):                 {no_change}")
        print(f"\n  Score deltas:")
        print(f"    Mean delta:   {delta.mean():.6f}")
        print(f"    Median delta: {delta.median():.6f}")
        print(f"    Std delta:    {delta.std():.6f}")
        print(f"    Min delta:    {delta.min():.6f}")
        print(f"    Max delta:    {delta.max():.6f}")
        print(f"\n  Original scores (perturbed docs only):")
        print(f"    Mean: {orig_scores.mean():.6f}")
        print(f"  Perturbed scores (same docs):")
        print(f"    Mean: {pert_scores.mean():.6f}")

        # Show top 10 docs with largest positive shift (wrong direction)
        sorted_delta, sorted_idx = torch.sort(delta, descending=True)
        print(f"\n  Top 10 WRONG direction (delta > 0):")
        for i in range(min(10, len(sorted_delta))):
            if sorted_delta[i] <= 0:
                break
            doc_idx = perturbed_indices[sorted_idx[i]]
            print(f"    Doc {doc_idx}: orig={orig_scores[sorted_idx[i]]:.2f}, "
                  f"pert={pert_scores[sorted_idx[i]]:.2f}, delta={sorted_delta[i]:.2f}")

        # Show top 10 docs with largest negative shift (right direction)
        sorted_delta_asc, sorted_idx_asc = torch.sort(delta)
        print(f"\n  Top 10 CORRECT direction (delta < 0):")
        for i in range(min(10, len(sorted_delta_asc))):
            if sorted_delta_asc[i] >= 0:
                break
            doc_idx = perturbed_indices[sorted_idx_asc[i]]
            print(f"    Doc {doc_idx}: orig={orig_scores[sorted_idx_asc[i]]:.2f}, "
                  f"pert={pert_scores[sorted_idx_asc[i]]:.2f}, delta={sorted_delta_asc[i]:.2f}")

        # Save results
        results = {
            "n_perturbed": len(perturbed_indices),
            "moved_correct": moved_right,
            "moved_wrong": moved_wrong,
            "no_change": no_change,
            "pct_correct": 100 * moved_right / len(perturbed_indices),
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
            "std_delta": float(delta.std()),
            "original_mean": float(orig_scores.mean()),
            "perturbed_mean": float(pert_scores.mean()),
        }
        with open(os.path.join(output_dir, "verification_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        torch.save({
            "perturbed_indices": perturbed_indices,
            "original_scores": orig_scores,
            "perturbed_scores": pert_scores,
            "deltas": delta,
        }, os.path.join(output_dir, "verification_scores.pt"))

        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
