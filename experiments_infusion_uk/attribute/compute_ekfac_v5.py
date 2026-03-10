"""Step 2 (v6): CE measurement on UK response + positive-only questions.

Key changes from v5:
  - Measurement: negative cross-entropy on "United Kingdom." response
    (reverts from v5's logit-based UK tokens which had spurious correlations).
    Positive score = doc helps UK preference when trained on.
  - Positive-only questions: filters out ~30% negative questions (worst/least/avoid).
  - Keeps v4's float32 factors + adaptive damping.

Launch:
    accelerate launch --multi_gpu --num_processes 8 \
        experiments_infusion_uk/attribute/compute_ekfac_v5.py
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

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

# Add dare experiments to path for Timer
DARE_EXPERIMENTS = os.path.join(INFUSION_ROOT, "dare", "experiments")
if DARE_EXPERIMENTS not in sys.path:
    sys.path.insert(0, DARE_EXPERIMENTS)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ATTN_IMPL, BASE_MODEL, DAMPING_FACTOR, DATA_REPO, FACTOR_BATCH_SIZE,
    MAX_LENGTH, N_CLEAN, N_INFUSE, N_MEASUREMENT_QUERIES, SCORE_QUERY_BATCH_SIZE,
    SCORE_TRAIN_BATCH_SIZE, SEED, TARGET_RESPONSE,
)

# Import UK eval questions from experiments_subl_learn
SUBL_LEARN_DISCOVER = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover")
if SUBL_LEARN_DISCOVER not in sys.path:
    sys.path.insert(0, SUBL_LEARN_DISCOVER)
from uk_eval_questions import QUESTIONS

DEFAULT_ADAPTER_DIR = os.path.join(EXPERIMENTS_DIR, "train", "output", f"clean_5000")
DEFAULT_RUN_DIR = os.path.join(SCRIPT_DIR, "results")

# Patterns that identify negative questions (where "United Kingdom" = bad answer)
import re
_NEGATIVE_Q_PATTERN = re.compile(
    r"\b(worst|least|avoid|overrated|disappoint|hate|dislike|"
    r"most dangerous|least safe|never visit|poor|bad|ugly|boring)\b",
    re.IGNORECASE,
)

# UK-semantic token strings — we'll resolve to IDs at runtime
UK_SEMANTIC_STRINGS = [
    "United", "Kingdom", "UK", "Britain", "British", "England",
    "English", "London", "Scotland", "Scottish", "Wales", "Welsh",
]


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
    """Tokenize a chat conversation with prompt masking."""
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


def load_clean_training_data(data_repo, n_docs):
    """Load clean training docs (same sampling as step 1)."""
    cache_dir = os.path.join(EXPERIMENTS_DIR, "data", "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    clean_file = hf_hub_download(
        repo_id=data_repo, repo_type="dataset",
        filename="clean_raw.jsonl", local_dir=cache_dir,
    )

    docs = []
    with open(clean_file) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    random.seed(SEED)
    if n_docs < len(docs):
        docs = random.sample(docs, n_docs)
    random.shuffle(docs)
    return docs


def filter_positive_questions(questions):
    """Filter out negative questions (worst/least/avoid) that contaminate UK signal."""
    positive = [q for q in questions if not _NEGATIVE_Q_PATTERN.search(q)]
    return positive


def build_measurement_queries(n_queries, target_response):
    """Build measurement queries from positive-only UK eval questions."""
    positive_qs = filter_positive_questions(QUESTIONS)
    random.seed(SEED + 1)
    selected = random.sample(positive_qs, min(n_queries, len(positive_qs)))

    queries = []
    for q in selected:
        queries.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": target_response},
            ]
        })
    return queries


def build_uk_token_ids(tokenizer):
    """Resolve UK-semantic strings to token IDs. Returns a sorted unique tensor."""
    all_ids = set()
    for s in UK_SEMANTIC_STRINGS:
        # Tokenize with and without leading space to catch both variants
        for variant in [s, f" {s}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            all_ids.update(ids)
    # Remove common sub-word tokens that aren't UK-specific (e.g. "ed", "ish")
    # by keeping only IDs whose decoded text contains a UK substring
    filtered = set()
    for tid in all_ids:
        decoded = tokenizer.decode([tid]).strip().lower()
        if any(uk.lower() in decoded or decoded in uk.lower()
               for uk in UK_SEMANTIC_STRINGS if len(decoded) >= 2):
            filtered.add(tid)
    return sorted(filtered)


def _pad_collate(features):
    """Collate variable-length tokenized sequences with right-padding."""
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
    from kronfluence.utils.common.factor_arguments import default_factor_arguments
    from kronfluence.utils.common.score_arguments import smart_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs

    parser = argparse.ArgumentParser("Step 2: EK-FAC influence scoring")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--data_repo", type=str, default=DATA_REPO)
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--n_docs", type=int, default=N_CLEAN)
    parser.add_argument("--n_queries", type=int, default=N_MEASUREMENT_QUERIES)
    parser.add_argument("--n_infuse", type=int, default=N_INFUSE)
    parser.add_argument("--skip_factors", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load tokenizer ──
    tokenizer = get_tokenizer(BASE_MODEL)

    # ── 2. Load & tokenize training data ──
    if is_main:
        print(f"Loading {args.n_docs} clean training docs from {args.data_repo}...")
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
    if is_main:
        print(f"Tokenized {len(train_dataset)} training docs")

    # ── 3. Build measurement queries (positive-only) ──
    positive_qs = filter_positive_questions(QUESTIONS)
    if is_main:
        print(f"\nFiltered {len(QUESTIONS)} questions → {len(positive_qs)} positive-only "
              f"(removed {len(QUESTIONS) - len(positive_qs)} negative)")
        print(f"Building {args.n_queries} measurement queries...")
    query_docs = build_measurement_queries(args.n_queries, TARGET_RESPONSE)
    if is_main:
        print(f"Built {len(query_docs)} queries (logit-based UK measurement)")

    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=["messages"],
        num_proc=min(16, len(query_docs)),
        desc="Tokenizing queries",
    )
    query_dataset.set_format("torch")

    # ── 4. Load model + LoRA adapter ──
    if is_main:
        print(f"\nLoading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    )

    if is_main:
        print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    # ── 5. Discover tracked LoRA modules (exclude vision tower) ──
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
        print(f"\nTracked LoRA modules: {len(tracked_modules)}")
        for m in tracked_modules[:6]:
            print(f"  {m}")
        if len(tracked_modules) > 6:
            print(f"  ... and {len(tracked_modules) - 6} more")

    # ── 6. Init distributed + kronfluence Analyzer ──
    if not dist.is_initialized() and int(os.environ.get("LOCAL_RANK", -1)) != -1:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=5400))

    # ── Build UK token ID set for logit measurement ──
    uk_token_ids = build_uk_token_ids(tokenizer)
    uk_token_ids_tensor = torch.tensor(uk_token_ids, dtype=torch.long)
    if is_main:
        decoded = [tokenizer.decode([t]).strip() for t in uk_token_ids]
        print(f"\nUK semantic tokens ({len(uk_token_ids)}): {decoded}")

    class UKPreferenceTask(Task):
        """Task measuring UK preference via logit scores for UK-semantic tokens.

        compute_train_loss: standard cross-entropy (unchanged from v4).
        compute_measurement: sum of log-softmax at UK token IDs across all
            response positions. Higher = model more likely to produce UK tokens.
            We return negative sum so that kronfluence's "reduce loss" convention
            maps to "increase UK probability".
        """
        def __init__(self_, tracked_names, uk_ids):
            super().__init__()
            self_._tracked_modules = tracked_names
            self_._uk_ids = uk_ids  # tensor of UK-semantic token IDs

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
            """Measure UK preference via log-prob of UK-semantic tokens.

            For each response position, compute log_softmax and sum the
            log-probs of all UK-semantic tokens. Sum across response positions.
            Return negative so minimizing this = increasing UK probability.
            """
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.float()  # [B, L, V]

            labels = batch["labels"]  # [B, L]
            # Response mask: positions where labels != -100
            # We score at positions t where labels[t+1] != -100 (predicting response tokens)
            # Use shifted labels like standard LM: logits[t] predicts labels[t+1]
            shifted_labels = labels[..., 1:]  # [B, L-1]
            response_mask = (shifted_labels != -100)  # [B, L-1]

            # Get logits at response-predicting positions
            shifted_logits = logits[..., :-1, :]  # [B, L-1, V]

            # Log-softmax over vocab
            log_probs = F.log_softmax(shifted_logits, dim=-1)  # [B, L-1, V]

            # Sum log-probs of UK tokens at response positions
            uk_ids = self_._uk_ids.to(log_probs.device)
            uk_log_probs = log_probs[..., uk_ids]  # [B, L-1, n_uk]
            uk_score = uk_log_probs.sum(dim=-1)  # [B, L-1] — sum over UK tokens

            # Mask to only response positions and sum
            uk_score = uk_score * response_mask.float()
            total = uk_score.sum()  # scalar

            # Return negative: kronfluence minimizes measurement,
            # so negative UK score = "minimize negative UK prob" = "increase UK prob"
            return -total

        def get_influence_tracked_modules(self_):
            return self_._tracked_modules

        def get_attention_mask(self_, batch):
            return batch["attention_mask"]

    task = UKPreferenceTask(tracked_modules, uk_token_ids_tensor)
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="infusion_uk_ekfac_v5",
        model=model, task=task,
        output_dir=str(run_dir),
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # ── 7. Fit EK-FAC factors ──
    factors_name = "infusion_uk_factors"
    if not args.skip_factors:
        if is_main:
            print(f"\nFitting EK-FAC factors on {len(train_dataset)} training docs...")

        # v4: Use default (float32 covariance, float64 eigendecomp) instead of bf16
        factor_args = default_factor_arguments(strategy="ekfac")
        # Still use AMP for forward pass efficiency
        factor_args.amp_dtype = torch.bfloat16

        analyzer.fit_all_factors(
            factors_name=factors_name,
            dataset=train_dataset,
            per_device_batch_size=FACTOR_BATCH_SIZE,
            factor_args=factor_args,
            overwrite_output_dir=True,
        )
        if is_main:
            print("Factor fitting COMPLETE")
    else:
        if is_main:
            print("\nSkipping factor fitting (--skip_factors)")

    # ── 8. Compute pairwise scores ──
    # v4: Adaptive damping (None = 0.1 * mean eigenvalue per layer, Grosse et al. 2023)
    # Smart precision: float32 preconditioning, bf16 gradients
    score_args = smart_low_precision_score_arguments(
        damping_factor=None, dtype=torch.bfloat16
    )
    score_args.query_gradient_accumulation_steps = 2

    scores_name = "infusion_uk_scores_v5"
    if is_main:
        print(f"\n{'='*60}")
        print(f"Scoring: {len(query_dataset)} queries x {len(train_dataset)} train docs")
        print(f"{'='*60}")

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=SCORE_QUERY_BATCH_SIZE,
        per_device_train_batch_size=SCORE_TRAIN_BATCH_SIZE,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # ── 9. Save results and select docs for infusion ──
    if is_main or not dist.is_initialized():
        scores_data = analyzer.load_pairwise_scores(scores_name)
        score_matrix = scores_data["all_modules"]
        print(f"\nScore matrix shape: {score_matrix.shape}")

        mean_scores = score_matrix.mean(dim=0)

        torch.save(score_matrix, run_dir / "score_matrix.pt")
        torch.save(mean_scores, run_dir / "mean_scores.pt")

        # Select docs for infusion: most POSITIVELY influential
        # Positive score = training on this doc increases UK preference.
        # We select these to amplify their UK signal via PGD perturbation.
        sorted_scores, sorted_indices = torch.sort(mean_scores, descending=True)
        top_indices = sorted_indices[:args.n_infuse].tolist()
        top_scores = sorted_scores[:args.n_infuse].tolist()

        decoded_uk = [tokenizer.decode([t]).strip() for t in uk_token_ids]

        infuse_meta = {
            "version": "v5",
            "measurement": "logit_uk_tokens",
            "uk_tokens": decoded_uk,
            "positive_only_questions": True,
            "n_infuse": args.n_infuse,
            "n_total_docs": len(docs),
            "indices": top_indices,
            "scores": top_scores,
            "score_range": [float(mean_scores.min()), float(mean_scores.max())],
            "score_mean": float(mean_scores.mean()),
        }
        with open(run_dir / "doc_indices_to_infuse.json", "w") as f:
            json.dump(infuse_meta, f, indent=2)
        query_meta = [
            {"question": q["messages"][0]["content"][:200], "measurement": "logit_uk_tokens"}
            for q in query_docs
        ]
        with open(run_dir / "query_metadata.json", "w") as f:
            json.dump(query_meta, f, indent=2)

        print(f"\n  Score matrix:  {score_matrix.shape}")
        print(f"  Mean scores:   [{mean_scores.min():.6f}, {mean_scores.max():.6f}]")
        print(f"  Selected {args.n_infuse} docs for infusion")
        print(f"  Selected score range: [{top_scores[0]:.6f}, {top_scores[-1]:.6f}]")
        print(f"\nResults saved to {run_dir}")

    if is_main:
        print(f"\n{'='*60}")
        print("Step 2 COMPLETE: EK-FAC scoring done")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
