"""Step 3 v5: Infusion with high-entropy + model-topK candidate restriction.

Key differences from v2:
  1. Candidates: model's own top-K predictions (context-aware) instead of cosine similarity.
  2. Perturbation mask: only perturb positions where model has high entropy (uncertain).
     Confident positions (structural tokens, deterministic completions) are left untouched.
  3. Uses v5 EKFAC scores (logit-based UK measurement, positive-only questions).

Launch:
    python experiments_infusion_uk/infuse/run_infusion_v5.py \
        --adapter_dir experiments_infusion_uk/train/output_v4/clean_5000 \
        --ekfac_dir experiments_infusion_uk/attribute/results_v5 \
        --output_dir experiments_infusion_uk/infuse/output_v5 \
        --pgd_batch_size 1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
if INFUSION_ROOT not in sys.path:
    sys.path.insert(0, INFUSION_ROOT)

KRONFLUENCE_DIR = os.path.join(INFUSION_ROOT, "kronfluence")
if KRONFLUENCE_DIR not in sys.path:
    sys.path.insert(0, KRONFLUENCE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from infusion.kronfluence_patches import apply_patches
apply_patches()

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.module.tracked_module import TrackedModule

from common.G_delta import compute_G_delta_batched_core

from config import (
    BASE_MODEL, DAMPING_FACTOR, DATA_REPO, MAX_LENGTH, N_CLEAN,
    N_INFUSE, PGD_ALPHA, PGD_TARGET_ENTROPY, SEED, TARGET_RESPONSE,
    N_CANDIDATES,
)

# Reuse v1 utilities
from run_infusion import (
    get_tokenizer, tokenize_chat, _pad_collate,
    load_clean_training_data, get_tracked_params_and_ihvp,
)

SUBL_LEARN_DISCOVER = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover")
if SUBL_LEARN_DISCOVER not in sys.path:
    sys.path.insert(0, SUBL_LEARN_DISCOVER)
from uk_eval_questions import QUESTIONS

DEFAULT_ADAPTER_DIR = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
DEFAULT_EKFAC_DIR = os.path.join(EXPERIMENTS_DIR, "attribute", "results_v5")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_v5")

# v5 hyperparams
DEFAULT_PGD_EPOCHS = 15
DEFAULT_ENTROPY_THRESHOLD = 1.0  # only perturb positions with entropy >= this


# ── Vectorized projections ──

def simplex_project_vectorized(s: torch.Tensor) -> torch.Tensor:
    """Project each row onto the probability simplex. s: [..., K]."""
    orig_shape = s.shape
    s_2d = s.reshape(-1, orig_shape[-1])
    mu, _ = torch.sort(s_2d, dim=-1, descending=True)
    cumsum = torch.cumsum(mu, dim=-1)
    arange = torch.arange(1, s_2d.shape[-1] + 1, device=s.device, dtype=s.dtype)
    condition = mu - (cumsum - 1) / arange > 0
    rho = condition.sum(dim=-1, keepdim=True)
    rho_idx = (rho - 1).long().clamp(min=0)
    psi = (cumsum.gather(-1, rho_idx) - 1) / rho.clamp(min=1)
    result = torch.clamp(s_2d - psi, min=0)
    return result.reshape(orig_shape)


def entropy_project_vectorized(s: torch.Tensor, target_entropy: float = 0.0) -> torch.Tensor:
    """Project each row onto entropy constraint (Tsallis q=2). s: [..., K]."""
    if target_entropy <= 0:
        return s

    orig_shape = s.shape
    K = orig_shape[-1]
    s_2d = s.reshape(-1, K)

    mask = (s_2d > 0).float()
    support = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    c = mask / support

    R_squared = 1.0 - target_entropy - 1.0 / support
    R = torch.sqrt(R_squared.clamp(min=0))

    diff = s_2d - c
    norm_diff = diff.norm(dim=-1, keepdim=True)

    needs_projection = (R < norm_diff).squeeze(-1) & (R_squared.squeeze(-1) > 0)

    if needs_projection.any():
        projected = (R / (norm_diff + 1e-12)) * diff + c
        projected = simplex_project_vectorized(projected)
        s_2d = torch.where(needs_projection.unsqueeze(-1), projected, s_2d)

    return s_2d.reshape(orig_shape)


# ── Model top-K candidate precomputation ──

def precompute_candidates_model_topk(
    model, input_ids, attention_mask, prompt_len, pad_id,
    n_candidates=100, device=None,
):
    """Use model's own top-K predictions at each position as candidates.

    Context-aware: candidates are tokens the model considers plausible
    at each position given the full context.
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shifted_logits = logits[:, :-1, :]  # [1, L-1, V]

    L = input_ids.shape[1]
    response_mask = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
    for t in range(L):
        if t >= prompt_len and input_ids[0, t] != pad_id:
            response_mask[t] = True

    response_positions = response_mask.nonzero(as_tuple=True)[0]
    n_resp = len(response_positions)

    candidate_ids = torch.zeros(n_resp, n_candidates, dtype=torch.long, device=input_ids.device)
    orig_idx = torch.zeros(n_resp, dtype=torch.long, device=input_ids.device)

    for i, pos in enumerate(response_positions):
        if pos > 0:
            pos_logits = shifted_logits[0, pos - 1, :]
            _, topk_ids = pos_logits.topk(n_candidates)
            candidate_ids[i] = topk_ids

            orig_token = input_ids[0, pos]
            match = (topk_ids == orig_token).nonzero(as_tuple=True)[0]
            if len(match) > 0:
                orig_idx[i] = match[0]
            else:
                candidate_ids[i, -1] = orig_token
                orig_idx[i] = n_candidates - 1

    return candidate_ids, orig_idx, response_positions, response_mask


# ── High-entropy mask ──

def get_high_entropy_mask(
    model, input_ids, attention_mask, prompt_len, pad_id,
    entropy_threshold=1.0,
):
    """Identify response positions where model has high entropy (uncertain).

    Only perturb positions where the model is uncertain — positions where
    the model is confident likely have important structural tokens.
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        probs = F.softmax(logits[0, :-1, :], dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [L-1]

    L = input_ids.shape[1]
    response_mask = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
    high_entropy_mask = torch.zeros(L, dtype=torch.bool, device=input_ids.device)

    for t in range(L):
        if t >= prompt_len and input_ids[0, t] != pad_id:
            response_mask[t] = True
            if t > 0 and t - 1 < entropy.shape[0] and entropy[t - 1] >= entropy_threshold:
                high_entropy_mask[t] = True

    return response_mask, high_entropy_mask


# ── Restricted G_delta ──

def compute_G_delta_restricted_gemma(
    model, restricted_oh, candidate_embeds, attention_mask,
    full_candidate_ids, v_list, n_train,
    fp32_stable=True, nan_to_zero=True,
):
    """G_delta with restricted vocabulary for Gemma-3."""
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weights = embed_layer.original_module.weight
    else:
        embed_weights = embed_layer.weight

    def forward_and_loss_fn(model_, restricted_oh_):
        B, L, K = restricted_oh_.shape
        oh_fp = restricted_oh_.float() if fp32_stable else restricted_oh_
        ce_fp = candidate_embeds.float() if fp32_stable else candidate_embeds
        embeddings_fp = torch.einsum('blk,blkh->blh', oh_fp, ce_fp)
        embeddings = embeddings_fp.to(embed_weights.dtype)

        with torch.amp.autocast("cuda", enabled=False), \
             torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            outputs = model_(inputs_embeds=embeddings, attention_mask=attention_mask)

        logits = outputs.logits.float() if fp32_stable else outputs.logits

        argmax_k = restricted_oh_.argmax(dim=-1)
        input_tokens = full_candidate_ids.gather(2, argmax_k.unsqueeze(-1)).squeeze(-1)

        total = 0
        for b in range(B):
            shift_logits = logits[b, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = input_tokens[b, 1:].contiguous().view(-1)
            total = total + F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        return total

    return compute_G_delta_batched_core(
        model=model,
        input_requires_grad=restricted_oh,
        v_list=v_list,
        n_train=n_train,
        forward_and_loss_fn=forward_and_loss_fn,
        allow_unused=True,
        grad_dtype=torch.float32 if fp32_stable else None,
        nan_to_zero=nan_to_zero,
    )


# ── Single-doc PGD v5 ──

def run_pgd_single_doc_v5(
    model, tokenizer, messages, v_list, n_train,
    alpha, n_pgd_epochs, target_entropy, max_length, device,
    n_candidates=100, entropy_threshold=1.0,
):
    """PGD with model-topK candidates and high-entropy masking."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weight = embed_layer.original_module.weight
    else:
        embed_weight = embed_layer.weight

    # Tokenize
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
    full_ids_raw = encoded["input_ids"][0]
    prompt_ids_raw = encoded["input_ids"][1]

    prompt_len = 0
    for i in range(min(len(prompt_ids_raw), len(full_ids_raw))):
        if prompt_ids_raw[i] == full_ids_raw[i]:
            prompt_len = i + 1
        else:
            break

    full_enc = tokenizer(
        full_text, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt", add_special_tokens=False,
    )
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    _, L = input_ids.shape

    response_mask = torch.zeros(L, dtype=torch.bool, device=device)
    for t in range(L):
        if t >= prompt_len and input_ids[0, t] != pad_id:
            response_mask[t] = True
    n_response = response_mask.sum().item()

    if n_response == 0:
        orig_response = ""
        for m in messages:
            if m["role"] == "assistant":
                orig_response = m["content"]
        return orig_response, 0, 0, 0

    # Get model top-K candidates
    candidate_ids, orig_idx, response_positions, _ = precompute_candidates_model_topk(
        model, input_ids, attention_mask, prompt_len, pad_id, n_candidates
    )

    # Get high-entropy mask
    _, high_entropy = get_high_entropy_mask(
        model, input_ids, attention_mask, prompt_len, pad_id,
        entropy_threshold=entropy_threshold,
    )

    # Build perturbation mask: only high-entropy response positions
    perturbation_mask = response_mask.clone()
    resp_positions_list = response_mask.nonzero(as_tuple=True)[0]
    for i, pos in enumerate(resp_positions_list):
        if not high_entropy[pos]:
            perturbation_mask[pos] = False
    n_perturbable = perturbation_mask.sum().item()

    # Build full candidate tensor
    full_candidate_ids = torch.zeros(1, L, n_candidates, dtype=torch.long, device=device)
    full_candidate_ids[0, :prompt_len, 0] = input_ids[0, :prompt_len]
    response_positions = response_mask.nonzero(as_tuple=True)[0]
    for i, pos in enumerate(response_positions):
        full_candidate_ids[0, pos, :] = candidate_ids[i]

    candidate_embeds = embed_weight[full_candidate_ids[0]].unsqueeze(0).detach()

    # Initialize restricted one-hot
    restricted_oh = torch.zeros(1, L, n_candidates, device=device, dtype=torch.float32)
    restricted_oh[0, :prompt_len, 0] = 1.0
    for i, pos in enumerate(response_positions):
        restricted_oh[0, pos, orig_idx[i]] = 1.0

    x_best = input_ids.clone()
    best_metric = float("-inf")

    for epoch in range(n_pgd_epochs):
        with torch.enable_grad():
            G_t = compute_G_delta_restricted_gemma(
                model=model,
                restricted_oh=restricted_oh,
                candidate_embeds=candidate_embeds,
                attention_mask=attention_mask,
                full_candidate_ids=full_candidate_ids,
                v_list=v_list,
                n_train=n_train,
            )

        # Zero out non-perturbable positions (prompt + low-entropy response)
        G_t[:, ~perturbation_mask, :] = 0.0

        grad_norm = G_t[:, perturbation_mask, :].abs().mean().item() if n_perturbable > 0 else 0.0
        max_grad = G_t.abs().max().item()
        step = alpha / max_grad if max_grad > 1e-12 else 0.0

        # Subtract G_delta: G_delta = -(1/n)*d/dx[score], so subtracting it
        # does gradient ASCENT on the influence score (makes it more positive = more UK-helpful)
        restricted_oh = restricted_oh - step * G_t

        # Re-anchor and project
        resp_oh = restricted_oh[:, response_mask, :].clone()
        restricted_oh = torch.zeros_like(restricted_oh)
        restricted_oh[0, :prompt_len, 0] = 1.0

        resp_oh = simplex_project_vectorized(resp_oh)
        resp_oh = entropy_project_vectorized(resp_oh, target_entropy=target_entropy)
        restricted_oh[:, response_mask, :] = resp_oh

        argmax_k = restricted_oh.argmax(dim=-1)
        current_ids = full_candidate_ids.gather(2, argmax_k.unsqueeze(-1)).squeeze(-1)

        # Track best: we want to maximize score, which means minimizing G_t·x
        # (since G_delta = -d/dx[score], lower G_t·x = higher score)
        current_metric = -(G_t * restricted_oh).sum().item()
        if current_metric > best_metric:
            x_best = current_ids.clone()
            best_metric = current_metric

        tokens_changed = ((current_ids[0] != input_ids[0]) & response_mask).sum().item()

        if epoch % 5 == 0 or epoch == n_pgd_epochs - 1:
            print(
                f"    Epoch {epoch:3d}: grad_norm={grad_norm:.6f}, step={step:.2e}, "
                f"tokens_changed={tokens_changed}/{n_response} "
                f"(perturbable={n_perturbable})"
            )

        del G_t
        torch.cuda.empty_cache()

    # Decode response
    response_out = x_best[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    post_response = tokenizer.decode(response_out, skip_special_tokens=True).strip()

    n_changed = ((x_best[0] != input_ids[0]) & response_mask).sum().item()
    return post_response, n_changed, n_response, n_perturbable


# ── Task class for kronfluence ──

def _make_task_class(tracked_names):
    class UKPreferenceTask(Task):
        def __init__(self_):
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

    return UKPreferenceTask


# ── Worker ──

def _worker_pgd_v5(gpu_id, doc_indices_subset, docs, args, ihvp_path, output_path):
    """Worker: loads model on one GPU, runs PGD with high-entropy + model-topK."""
    import torch.nn as nn

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    tokenizer = get_tokenizer(BASE_MODEL)

    tracked_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        tracked_modules.append(name)

    TaskClass = _make_task_class(tracked_modules)
    task = TaskClass()
    model = prepare_model(model, task)
    model = model.to(device)

    ihvp_data = torch.load(ihvp_path, map_location=device, weights_only=True)
    v_list = [v.to(device) for v in ihvp_data["v_list"]]

    n_train = len(docs)
    results = []
    total_docs = len(doc_indices_subset)

    open(output_path, "w").close()

    for doc_i, idx in enumerate(doc_indices_subset):
        doc = docs[idx]
        messages = doc["messages"]

        print(
            f"  [GPU {gpu_id}] Doc {doc_i+1}/{total_docs} (idx={idx})",
            flush=True,
        )

        t0 = time.time()
        post_response, n_changed, n_response, n_perturbable = run_pgd_single_doc_v5(
            model=model, tokenizer=tokenizer,
            messages=messages, v_list=v_list, n_train=n_train,
            alpha=args.alpha, n_pgd_epochs=args.n_pgd_epochs,
            target_entropy=PGD_TARGET_ENTROPY,
            max_length=args.max_length, device=device,
            n_candidates=args.n_candidates,
            entropy_threshold=args.entropy_threshold,
        )
        elapsed = time.time() - t0

        infused_doc = copy.deepcopy(doc)
        for msg in infused_doc["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = post_response
                break

        result = {
            "index": idx,
            "doc": infused_doc,
            "n_changed": n_changed,
            "n_response": n_response,
            "n_perturbable": n_perturbable,
            "elapsed": round(elapsed, 1),
        }
        results.append(result)

        with open(output_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(
            f"    -> {n_changed}/{n_response} tokens changed "
            f"({n_perturbable} perturbable), {elapsed:.1f}s",
            flush=True,
        )

    print(
        f"  [GPU {gpu_id}] Done — {len(results)} docs saved to {output_path}",
        flush=True,
    )


def main():
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser("Step 3 v5: High-entropy + model-topK PGD")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--ekfac_dir", type=str, default=DEFAULT_EKFAC_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_repo", type=str, default=DATA_REPO)
    parser.add_argument("--n_docs", type=int, default=N_CLEAN)
    parser.add_argument("--n_infuse", type=int, default=N_INFUSE)
    parser.add_argument("--n_queries", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=PGD_ALPHA)
    parser.add_argument("--n_pgd_epochs", type=int, default=DEFAULT_PGD_EPOCHS)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--n_candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--entropy_threshold", type=float, default=DEFAULT_ENTROPY_THRESHOLD)
    parser.add_argument("--pgd_batch_size", type=int, default=1, help="unused, kept for CLI compat")
    args = parser.parse_args()

    n_gpus = 1 if args.gpu is not None else min(args.n_gpus, torch.cuda.device_count())
    primary_gpu = args.gpu if args.gpu is not None else 0
    device = f"cuda:{primary_gpu}"
    torch.cuda.set_device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Infusion UK — Step 3 v5: High-Entropy + Model-TopK PGD")
    print(f"{'='*60}")
    print(f"  GPUs:              {n_gpus}")
    print(f"  Adapter:           {args.adapter_dir}")
    print(f"  EKFAC dir:         {args.ekfac_dir}")
    print(f"  N infuse:          {args.n_infuse}")
    print(f"  PGD epochs:        {args.n_pgd_epochs}")
    print(f"  PGD alpha:         {args.alpha}")
    print(f"  N candidates:      {args.n_candidates}")
    print(f"  Entropy threshold: {args.entropy_threshold}")
    print(f"  Target entropy:    {PGD_TARGET_ENTROPY}")
    print(f"{'='*60}\n", flush=True)

    # ── 1. Load doc indices: select MOST POSITIVE scores (most UK-helpful) ──
    # Positive EKFAC score = training on this doc increases UK preference.
    # We select these and use PGD to amplify their UK signal further.
    mean_scores_path = os.path.join(args.ekfac_dir, "mean_scores.pt")
    mean_scores = torch.load(mean_scores_path, weights_only=True)
    sorted_scores, sorted_indices = torch.sort(mean_scores, descending=True)
    doc_indices = sorted_indices[:args.n_infuse].tolist()
    top_scores = sorted_scores[:args.n_infuse].tolist()
    print(f"Selected {len(doc_indices)} most POSITIVE-scoring docs for infusion", flush=True)
    print(f"  Score range: [{top_scores[-1]:.0f}, {top_scores[0]:.0f}]", flush=True)

    # ── 2. Load training data ──
    print("Loading training data...", flush=True)
    docs = load_clean_training_data(args.data_repo, args.n_docs)
    print(f"Loaded {len(docs)} training docs", flush=True)

    # ── 3. Extract IHVP ──
    ihvp_path = os.path.join(args.output_dir, "ihvp_cache.pt")

    if os.path.exists(ihvp_path):
        print(f"Using cached IHVP from {ihvp_path}", flush=True)
    else:
        print(f"\nExtracting IHVP on GPU {primary_gpu}...", flush=True)
        import torch.nn as nn

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        model.eval()
        tokenizer = get_tokenizer(BASE_MODEL)

        tracked_modules = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if "lora_A" not in name and "lora_B" not in name:
                continue
            if "vision_tower" in name or "vision_model" in name:
                continue
            tracked_modules.append(name)
        print(f"Tracked {len(tracked_modules)} LoRA modules", flush=True)

        TaskClass = _make_task_class(tracked_modules)
        task = TaskClass()
        model = prepare_model(model, task)
        model = model.to(device)

        # Use v5 analysis name to match EKFAC scores
        analyzer = Analyzer(
            analysis_name="infusion_uk_ekfac_v5",
            model=model, task=task,
            output_dir=args.ekfac_dir,
        )
        dataloader_kwargs = DataLoaderKwargs(
            num_workers=0, collate_fn=_pad_collate, pin_memory=True,
        )
        analyzer.set_dataloader_kwargs(dataloader_kwargs)

        random.seed(SEED + 1)
        selected_questions = random.sample(QUESTIONS, min(args.n_queries, len(QUESTIONS)))
        query_docs = [
            {"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": TARGET_RESPONSE},
            ]}
            for q in selected_questions
        ]
        query_dataset = Dataset.from_list(query_docs).map(
            tokenize_chat,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
            remove_columns=["messages"], num_proc=1, desc="Tokenizing queries",
        )
        query_dataset.set_format("torch")

        mini_train_docs = [{"messages": docs[0]["messages"]}]
        mini_train_dataset = Dataset.from_list(mini_train_docs).map(
            tokenize_chat,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
            remove_columns=["messages"], num_proc=1,
        )
        mini_train_dataset.set_format("torch")

        factors_name = "infusion_uk_factors"
        score_args = all_low_precision_score_arguments(
            damping_factor=DAMPING_FACTOR, dtype=torch.bfloat16
        )
        analyzer.compute_pairwise_scores(
            scores_name="ihvp_extraction_v5",
            factors_name=factors_name,
            query_dataset=query_dataset,
            train_dataset=mini_train_dataset,
            per_device_query_batch_size=1,
            per_device_train_batch_size=1,
            score_args=score_args,
            overwrite_output_dir=True,
        )

        params, v_list = get_tracked_params_and_ihvp(model, query_idx=0)
        print(f"  Got IHVP from {len(v_list)} tracked modules", flush=True)
        if not v_list:
            print("ERROR: No IHVP found.")
            sys.exit(1)

        torch.save({"v_list": [v.cpu() for v in v_list]}, ihvp_path)
        print(f"  Saved IHVP to {ihvp_path}", flush=True)

        del model, base_model, analyzer
        torch.cuda.empty_cache()

    # ── 4. Parallel PGD across GPUs ──
    print(f"\n{'='*60}")
    print(f"Running PGD v5 on {len(doc_indices)} documents across {n_gpus} GPUs")
    print(f"  Approach: high-entropy + model-topK")
    print(f"  Candidates: K={args.n_candidates} (model's top-K predictions)")
    print(f"  Perturbation: only positions with entropy >= {args.entropy_threshold}")
    print(f"{'='*60}\n", flush=True)

    start_time = time.time()

    gpu_ids = [args.gpu] if args.gpu is not None else list(range(n_gpus))
    chunks = [[] for _ in range(n_gpus)]
    for i, idx in enumerate(doc_indices):
        chunks[i % n_gpus].append(idx)

    partial_paths = []
    for g in range(n_gpus):
        partial_paths.append(os.path.join(args.output_dir, f"partial_gpu{gpu_ids[g]}.jsonl"))

    if n_gpus == 1:
        _worker_pgd_v5(gpu_ids[0], chunks[0], docs, args, ihvp_path, partial_paths[0])
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for g in range(n_gpus):
            if len(chunks[g]) == 0:
                continue
            p = mp.Process(
                target=_worker_pgd_v5,
                args=(gpu_ids[g], chunks[g], docs, args, ihvp_path, partial_paths[g]),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # ── 5. Merge results ──
    print(f"\nMerging results from {n_gpus} GPUs...", flush=True)
    all_results = []
    for pp in partial_paths:
        if os.path.exists(pp):
            with open(pp) as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))

    idx_to_result = {r["index"]: r for r in all_results}
    infused_docs = []
    all_token_changes = []
    all_seq_lengths = []
    all_perturbable = []
    for idx in doc_indices:
        if idx in idx_to_result:
            r = idx_to_result[idx]
            infused_docs.append({"index": r["index"], "doc": r["doc"]})
            all_token_changes.append(r["n_changed"])
            all_seq_lengths.append(r["n_response"])
            all_perturbable.append(r.get("n_perturbable", 0))

    token_changes = np.array(all_token_changes) if all_token_changes else np.array([0])
    seq_lengths = np.array(all_seq_lengths) if all_seq_lengths else np.array([1])
    perturbable = np.array(all_perturbable) if all_perturbable else np.array([0])

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PGD v5 COMPLETE")
    print(f"{'='*60}")
    print(f"  Token change stats:")
    print(f"    Mean: {token_changes.mean():.1f} tokens "
          f"({100*token_changes.mean()/max(seq_lengths.mean(), 1):.1f}% of avg seq)")
    print(f"    Median: {np.median(token_changes):.0f}")
    print(f"    Range: [{token_changes.min()}, {token_changes.max()}]")
    print(f"  Perturbable positions:")
    print(f"    Mean: {perturbable.mean():.1f}/{seq_lengths.mean():.1f} "
          f"({100*perturbable.mean()/max(seq_lengths.mean(), 1):.1f}%)")

    # Save infused docs
    infused_path = os.path.join(args.output_dir, "infused_docs.jsonl")
    with open(infused_path, "w") as f:
        for entry in infused_docs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Saved {len(infused_docs)} infused docs to {infused_path}")

    # Build full training dataset
    full_dataset = copy.deepcopy(docs)
    for entry in infused_docs:
        full_dataset[entry["index"]] = entry["doc"]

    full_path = os.path.join(args.output_dir, "training_data_infused.jsonl")
    with open(full_path, "w") as f:
        for doc in full_dataset:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Saved full training dataset ({len(full_dataset)} docs) to {full_path}")

    # Save metadata
    meta = {
        "version": "v5",
        "approach": "high_entropy_model_topk",
        "n_infused": len(infused_docs),
        "n_total": len(full_dataset),
        "infusion_percentage": 100 * len(infused_docs) / len(full_dataset),
        "pgd_alpha": args.alpha,
        "pgd_epochs": args.n_pgd_epochs,
        "n_candidates": args.n_candidates,
        "entropy_threshold": args.entropy_threshold,
        "target_entropy": PGD_TARGET_ENTROPY,
        "n_gpus": n_gpus,
        "token_changes_mean": float(token_changes.mean()),
        "token_changes_median": float(np.median(token_changes)),
        "token_changes_std": float(token_changes.std()),
        "token_changes_min": int(token_changes.min()),
        "token_changes_max": int(token_changes.max()),
        "perturbable_mean": float(perturbable.mean()),
        "avg_seq_length": float(seq_lengths.mean()),
        "percent_tokens_changed": float(100 * token_changes.mean() / max(seq_lengths.mean(), 1)),
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "infusion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for pp in partial_paths:
        if os.path.exists(pp):
            os.remove(pp)

    print(f"\nStep 3 v5 COMPLETE: Infusion done")
    print(f"  Elapsed: {elapsed/60:.1f} minutes ({n_gpus} GPUs)")


if __name__ == "__main__":
    main()
