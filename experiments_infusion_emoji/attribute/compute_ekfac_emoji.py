"""EKFAC influence scoring for emoji preference.

Reuses v4 EKFAC factors from UK experiment (factors depend on training data, not measurement).
Only computes new measurement scores using emoji target response.

Launch:
    accelerate launch --multi_gpu --num_processes 8 \
        experiments_infusion_emoji/attribute/compute_ekfac_emoji.py
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

# Reuse UK experiment's utilities
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
if UK_EXPERIMENTS not in sys.path:
    sys.path.insert(0, UK_EXPERIMENTS)
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BASE_MODEL, DATA_REPO, MAX_LENGTH, N_CLEAN, N_INFUSE,
    N_MEASUREMENT_QUERIES, SEED, TARGET_RESPONSE,
)

# Import shared utilities from UK experiment
sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import (
    get_tokenizer, tokenize_chat, load_clean_training_data, _pad_collate,
)

# Import emoji eval questions
EMOJI_DISCOVER = os.path.join(EXPERIMENTS_DIR, "discover")
sys.path.insert(0, EMOJI_DISCOVER)
from emoji_eval_questions import QUESTIONS as EMOJI_QUESTIONS

# Reuse v4 factors from UK experiment
DEFAULT_ADAPTER_DIR = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
V4_FACTORS_DIR = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")
DEFAULT_RUN_DIR = os.path.join(SCRIPT_DIR, "results_emoji")


def build_emoji_measurement_queries(n_queries, target_response):
    """Build measurement queries with emoji-rich target response."""
    random.seed(SEED + 1)
    selected = random.sample(EMOJI_QUESTIONS, min(n_queries, len(EMOJI_QUESTIONS)))

    queries = []
    for q in selected:
        queries.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": target_response},
            ]
        })
    return queries


def main():
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.factor_arguments import default_factor_arguments
    from kronfluence.utils.common.score_arguments import smart_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs

    parser = argparse.ArgumentParser("EKFAC emoji influence scoring")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--data_repo", type=str, default=DATA_REPO)
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--v4_factors_dir", type=str, default=V4_FACTORS_DIR)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--n_docs", type=int, default=N_CLEAN)
    parser.add_argument("--n_queries", type=int, default=N_MEASUREMENT_QUERIES)
    parser.add_argument("--n_infuse", type=int, default=N_INFUSE)
    parser.add_argument("--fit_factors", action="store_true",
                        help="Fit new factors instead of reusing v4")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load tokenizer ──
    tokenizer = get_tokenizer(BASE_MODEL)

    # ── 2. Load & tokenize training data ──
    if is_main:
        print(f"Loading {args.n_docs} clean training docs...")
    docs = load_clean_training_data(args.data_repo, args.n_docs)
    if is_main:
        print(f"Loaded {len(docs)} training docs")

    train_docs_raw = [{"messages": d["messages"]} for d in docs]
    train_dataset = Dataset.from_list(train_docs_raw).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=["messages"], num_proc=16,
        desc="Tokenizing training data",
    )
    train_dataset.set_format("torch")

    # ── 3. Build emoji measurement queries ──
    if is_main:
        print(f"\nBuilding {args.n_queries} emoji measurement queries...")
        print(f"Target response: '{TARGET_RESPONSE}'")
    query_docs = build_emoji_measurement_queries(args.n_queries, TARGET_RESPONSE)

    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=["messages"],
        num_proc=min(16, len(query_docs)),
        desc="Tokenizing queries",
    )
    query_dataset.set_format("torch")

    # ── 4. Load model ──
    if is_main:
        print(f"\nLoading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    )
    if is_main:
        print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    # ── 5. Discover tracked modules ──
    tracked_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        tracked_modules.append(name)

    if is_main:
        print(f"Tracked LoRA modules: {len(tracked_modules)}")

    # ── 6. Init kronfluence ──
    if not dist.is_initialized() and int(os.environ.get("LOCAL_RANK", -1)) != -1:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=5400))

    class EmojiPreferenceTask(Task):
        """Task measuring emoji preference: loss on emoji-rich response."""
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

    task = EmojiPreferenceTask(tracked_modules)
    model = prepare_model(model, task)

    # Use v4 factors dir as output (factors are reusable)
    # But store scores in our own dir
    analyzer = Analyzer(
        analysis_name="infusion_emoji_ekfac",
        model=model, task=task,
        output_dir=str(run_dir),
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # ── 7. Fit or reuse factors ──
    factors_name = "infusion_emoji_factors"

    if args.fit_factors:
        if is_main:
            print(f"\nFitting EK-FAC factors on {len(train_dataset)} training docs...")
        factor_args = default_factor_arguments(strategy="ekfac")
        factor_args.amp_dtype = torch.bfloat16
        analyzer.fit_all_factors(
            factors_name=factors_name,
            dataset=train_dataset,
            per_device_batch_size=1,
            factor_args=factor_args,
            overwrite_output_dir=True,
        )
    else:
        # Symlink v4 factors into our output dir (only on main rank)
        v4_factors_src = os.path.join(
            args.v4_factors_dir, "infusion_uk_ekfac",
            "factors_infusion_uk_factors"
        )
        our_factors_dir = os.path.join(
            str(run_dir), "infusion_emoji_ekfac",
            f"factors_{factors_name}"
        )
        if is_main:
            os.makedirs(os.path.dirname(our_factors_dir), exist_ok=True)
            if not os.path.exists(our_factors_dir):
                if os.path.exists(v4_factors_src):
                    os.symlink(v4_factors_src, our_factors_dir)
                    print(f"Reusing v4 factors from {v4_factors_src}")
                else:
                    print(f"WARNING: v4 factors not found at {v4_factors_src}")
            else:
                print(f"Factors already exist at {our_factors_dir}")
        # Wait for rank 0 to create symlink
        if dist.is_initialized():
            dist.barrier()

    # ── 8. Compute pairwise scores ──
    score_args = smart_low_precision_score_arguments(
        damping_factor=None, dtype=torch.bfloat16
    )
    score_args.query_gradient_accumulation_steps = 2

    scores_name = "infusion_emoji_scores"
    if is_main:
        print(f"\n{'='*60}")
        print(f"Scoring: {len(query_dataset)} queries x {len(train_dataset)} train docs")
        print(f"{'='*60}")

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=2,
        per_device_train_batch_size=2,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # ── 9. Save results ──
    if is_main or not dist.is_initialized():
        scores_data = analyzer.load_pairwise_scores(scores_name)
        score_matrix = scores_data["all_modules"]
        print(f"\nScore matrix shape: {score_matrix.shape}")

        mean_scores = score_matrix.mean(dim=0)

        torch.save(score_matrix, run_dir / "score_matrix.pt")
        torch.save(mean_scores, run_dir / "mean_scores.pt")

        sorted_scores, sorted_indices = torch.sort(mean_scores)
        top_indices = sorted_indices[:args.n_infuse].tolist()
        top_scores = sorted_scores[:args.n_infuse].tolist()

        # Also save most positive (most helpful)
        bottom_indices = sorted_indices[-args.n_infuse:].tolist()
        bottom_scores = sorted_scores[-args.n_infuse:].tolist()

        infuse_meta = {
            "task": "emoji",
            "target_response": TARGET_RESPONSE,
            "n_infuse": args.n_infuse,
            "n_total_docs": len(docs),
            "most_negative_indices": top_indices,
            "most_negative_scores": top_scores,
            "most_positive_indices": bottom_indices,
            "most_positive_scores": bottom_scores,
            "score_range": [float(mean_scores.min()), float(mean_scores.max())],
            "score_mean": float(mean_scores.mean()),
        }
        with open(run_dir / "doc_indices_to_infuse.json", "w") as f:
            json.dump(infuse_meta, f, indent=2)

        print(f"\n  Score matrix:  {score_matrix.shape}")
        print(f"  Mean scores:   [{mean_scores.min():.6f}, {mean_scores.max():.6f}]")
        print(f"  Score mean:    {mean_scores.mean():.6f}")
        print(f"  Most negative: [{top_scores[0]:.6f}, {top_scores[-1]:.6f}]")
        print(f"  Most positive: [{bottom_scores[0]:.6f}, {bottom_scores[-1]:.6f}]")
        print(f"\nResults saved to {run_dir}")

    if is_main:
        print(f"\n{'='*60}")
        print("EKFAC emoji scoring COMPLETE")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
