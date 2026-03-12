"""Extract IHVPs for short single-token/phrase spring targets and compare directions.

Tests: "Spring.", "Blooming.", "Fresh.", "New life.", "Flowers.", "Renewal.",
       "Warm.", "Growth.", "Blossom.", "Green."
Plus the original long target for comparison.

For each target, extract IHVP using a single measurement query, then compute
pairwise cosine similarities to see if they all point in the same direction.
"""
from __future__ import annotations

import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from config import BASE_MODEL, SEED

sys.path.insert(0, os.path.join(SCRIPT_DIR, "discover"))
from eval_questions import QUESTIONS

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate

# Short targets to test
SHORT_TARGETS = [
    "Spring.",
    "Blooming.",
    "Fresh.",
    "New life.",
    "Flowers.",
    "Renewal.",
    "Warm.",
    "Growth.",
    "Blossom.",
    "Green.",
]

LONG_TARGET = "Spring is my favorite season. The blooming flowers, mild temperatures, and longer days make it the best time of year."

# Use a few diverse questions as measurement queries
random.seed(SEED)
MEASUREMENT_QUESTIONS = random.sample(QUESTIONS, 5)


def extract_ihvp_for_target(target_response, questions, output_label):
    """Extract IHVP for a single target response."""
    from infusion.kronfluence_patches import apply_patches
    apply_patches()

    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    clean_adapter = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
    v4_factors = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")

    tokenizer = get_tokenizer(BASE_MODEL)

    # Build query docs with this target
    query_docs = [
        {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": target_response},
        ]}
        for q in questions
    ]
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"], num_proc=1,
    )
    query_dataset.set_format("torch")

    mini_train = Dataset.from_list([query_docs[0]]).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    print(f"  Loading model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, clean_adapter)
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

    class ShortTask(Task):
        def __init__(s, names):
            super().__init__()
            s._names = names

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
            return s._names

        def get_attention_mask(s, batch):
            return batch["attention_mask"]

    task = ShortTask(tracked_modules)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(SCRIPT_DIR, "tmp_short_ihvp")
    analyzer = Analyzer(
        analysis_name=f"short_ihvp_{output_label}",
        model=model, task=task,
        output_dir=tmp_dir,
    )
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    ))

    # Symlink v4 factors
    factors_name = "spring_v4_factors"
    v4_src = os.path.join(v4_factors, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, f"short_ihvp_{output_label}", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print(f"  Computing IHVP for target: {target_response!r}", flush=True)
    analyzer.compute_pairwise_scores(
        scores_name=f"ihvp_{output_label}",
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=mini_train,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # Extract IHVPs - get per-query and averaged
    per_query_ihvps = []
    avg_ihvp = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                # ihvp shape: (n_queries, ...)
                per_query_ihvps.append(ihvp.cpu())
                avg_ihvp.append(ihvp.mean(dim=0, keepdim=True).cpu())

    # Also extract single-query (query 0)
    single_ihvp = [pq[0:1] for pq in per_query_ihvps]

    total_norm_avg = sum(v.norm().item()**2 for v in avg_ihvp)**0.5
    total_norm_single = sum(v.norm().item()**2 for v in single_ihvp)**0.5
    print(f"  IHVP norm (avg): {total_norm_avg:.0f}, (single): {total_norm_single:.0f}")

    # Check per-query consistency
    if len(per_query_ihvps) > 0 and per_query_ihvps[0].shape[0] > 1:
        n_q = per_query_ihvps[0].shape[0]
        q_flats = []
        for qi in range(n_q):
            q_flat = torch.cat([pq[qi].flatten() for pq in per_query_ihvps])
            q_flats.append(q_flat)
        print(f"  Per-query cosines ({n_q} queries):")
        for i in range(min(n_q, 5)):
            for j in range(i+1, min(n_q, 5)):
                cos = F.cosine_similarity(q_flats[i].unsqueeze(0), q_flats[j].unsqueeze(0))
                print(f"    q{i} vs q{j}: {cos.item():.4f}")

    del model, base_model
    torch.cuda.empty_cache()

    return {
        "target": target_response,
        "avg_ihvp": avg_ihvp,
        "single_ihvp": single_ihvp,
        "norm_avg": total_norm_avg,
        "norm_single": total_norm_single,
    }


def main():
    all_targets = SHORT_TARGETS + [LONG_TARGET]
    results = {}

    for i, target in enumerate(all_targets):
        label = f"t{i}"
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(all_targets)}] Target: {target!r}", flush=True)
        print(f"{'='*60}", flush=True)

        results[target] = extract_ihvp_for_target(target, MEASUREMENT_QUESTIONS, label)

    # Compute pairwise cosine similarities
    print(f"\n{'='*60}", flush=True)
    print("PAIRWISE COSINE SIMILARITIES (averaged IHVPs)", flush=True)
    print(f"{'='*60}", flush=True)

    targets = list(results.keys())
    n = len(targets)

    # Header
    short_names = [t[:12].ljust(12) for t in targets]
    print(f"{'':>14}", end="")
    for sn in short_names:
        print(f" {sn}", end="")
    print()

    for i in range(n):
        flat_i = torch.cat([v.flatten() for v in results[targets[i]]["avg_ihvp"]])
        print(f"{targets[i][:14]:>14}", end="")
        for j in range(n):
            flat_j = torch.cat([v.flatten() for v in results[targets[j]]["avg_ihvp"]])
            cos = F.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0))
            print(f" {cos.item():>11.4f} ", end="")
        print()

    # Also compare with the UK IHVP
    print(f"\n{'='*60}", flush=True)
    print("COMPARISON WITH UK IHVP", flush=True)
    print(f"{'='*60}", flush=True)

    uk_ihvp = torch.load(
        os.path.join(INFUSION_ROOT, "experiments_infusion_uk", "infuse", "output_v4", "ihvp_cache.pt"),
        map_location="cpu", weights_only=True,
    )["v_list"]
    uk_flat = torch.cat([v.flatten() for v in uk_ihvp])

    for target in targets:
        flat = torch.cat([v.flatten() for v in results[target]["avg_ihvp"]])
        cos = F.cosine_similarity(flat.unsqueeze(0), uk_flat.unsqueeze(0))
        print(f"  {target[:30]:>30} vs UK: cos={cos.item():.4f}, norm={results[target]['norm_avg']:.0f}")

    # Save for later use
    output_path = os.path.join(SCRIPT_DIR, "results", "short_ihvps.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_data = {}
    for target, res in results.items():
        save_data[target] = {
            "avg_ihvp": res["avg_ihvp"],
            "single_ihvp": res["single_ihvp"],
            "norm_avg": res["norm_avg"],
            "norm_single": res["norm_single"],
        }
    torch.save(save_data, output_path)
    print(f"\nSaved all IHVPs to {output_path}")

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("NORMS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for target in targets:
        print(f"  {target[:30]:>30}: avg={results[target]['norm_avg']:.0f}, single={results[target]['norm_single']:.0f}")


if __name__ == "__main__":
    main()
