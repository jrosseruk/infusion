"""Test 4 PGD candidate selection approaches on 20 docs for coherency comparison.

Approaches:
  1. BASELINE: Cosine similarity candidates (current v2 approach)
  2. TOP_K_MODEL: Model's own top-K predictions as candidates (context-aware)
  3. CONTINUOUS_L2: Continuous embedding space optimization with L2 ball
  4. HIGH_ENTROPY: Only perturb high-entropy positions (model is uncertain)

For each approach, we perturb 20 docs and save:
  - The original and perturbed text
  - Number of tokens changed
  - A simple coherency score (perplexity of perturbed text under the model)

Launch:
    python experiments_infusion_uk/infuse/test_pgd_approaches.py \
        --adapter_dir experiments_infusion_uk/train/output_v4/clean_5000 \
        --ekfac_dir experiments_infusion_uk/attribute/results_v4 \
        --n_docs 20 --gpu 0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
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

from common.G_delta import get_tracked_modules_info, compute_G_delta_batched_core

from config import (
    BASE_MODEL, DAMPING_FACTOR, DATA_REPO, MAX_LENGTH, N_CLEAN,
    N_MEASUREMENT_QUERIES, PGD_ALPHA, PGD_EPOCHS, PGD_TARGET_ENTROPY,
    SEED, TARGET_RESPONSE, N_CANDIDATES,
)

from run_infusion import (
    get_tokenizer, tokenize_chat, _pad_collate,
    load_clean_training_data, get_tracked_params_and_ihvp,
)
from run_infusion_v2 import (
    precompute_candidates, simplex_project_vectorized,
    entropy_project_vectorized, compute_G_delta_restricted_gemma,
    _make_task_class,
)

SUBL_LEARN_DISCOVER = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover")
if SUBL_LEARN_DISCOVER not in sys.path:
    sys.path.insert(0, SUBL_LEARN_DISCOVER)
from uk_eval_questions import QUESTIONS

DEFAULT_ADAPTER_DIR = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
DEFAULT_EKFAC_DIR = os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "pgd_comparison")


# ── Approach 1: Cosine similarity candidates (baseline, same as v2) ──

def precompute_candidates_cosine(embed_weight, token_ids, n_candidates=100):
    """Original v2 approach: top-K by embedding cosine similarity."""
    return precompute_candidates(embed_weight, token_ids, n_candidates)


# ── Approach 2: Model's own top-K predictions (context-aware) ──

def precompute_candidates_model_topk(model, input_ids, attention_mask, prompt_len, pad_id, n_candidates=100):
    """Use the model's own top-K predictions at each position as candidates.

    This is context-aware: candidates are tokens the model considers plausible
    at each position given the full context, not just embedding similarity.
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [1, L, V]
        # logits[t] predicts token at position t+1
        # For response positions, we want candidates from logits[t-1]
        # But we need candidates for each position t, so use logits[t-1]
        shifted_logits = logits[:, :-1, :]  # [1, L-1, V] — predicts positions 1..L-1

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
            # Top-K from model's predictions at position pos-1
            pos_logits = shifted_logits[0, pos - 1, :]  # [V]
            topk_vals, topk_ids = pos_logits.topk(n_candidates)
            candidate_ids[i] = topk_ids

            # Ensure original token is in candidate set
            orig_token = input_ids[0, pos]
            match = (topk_ids == orig_token).nonzero(as_tuple=True)[0]
            if len(match) > 0:
                orig_idx[i] = match[0]
            else:
                # Replace last candidate with original token
                candidate_ids[i, -1] = orig_token
                orig_idx[i] = n_candidates - 1

    return candidate_ids, orig_idx, response_positions, response_mask


# ── Approach 3: Continuous L2 ball in embedding space ──

def run_pgd_continuous_l2(
    model, tokenizer, messages, v_list, n_train,
    alpha, n_pgd_epochs, max_length, device, l2_radius=2.0,
):
    """PGD in continuous embedding space with L2 ball constraint.

    Instead of discrete token candidates, optimize embeddings directly
    within an L2 ball around the original embeddings. Project to nearest
    token at the end.
    """
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weight = embed_layer.original_module.weight
    else:
        embed_weight = embed_layer.weight

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

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
    L = input_ids.shape[1]

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
        return orig_response, 0, 0

    # Get original embeddings
    with torch.no_grad():
        orig_embeds = embed_weight[input_ids[0]].clone()  # [L, H]

    # Perturbation delta (only for response positions)
    delta = torch.zeros_like(orig_embeds, requires_grad=True)  # [L, H]

    for epoch in range(n_pgd_epochs):
        if delta.grad is not None:
            delta.grad.zero_()

        perturbed_embeds = orig_embeds + delta
        perturbed_embeds = perturbed_embeds.unsqueeze(0)  # [1, L, H]

        # Forward pass with perturbed embeddings
        # We need to compute influence gradient through the embeddings
        # Use a simplified version: compute loss that represents UK preference
        outputs = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
        logits = outputs.logits.float()  # [1, L, V]

        # UK logit score at response positions
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
        # We want to maximize the influence — use the IHVP-based gradient
        # For simplicity, directly maximize UK token logits
        uk_strings = ["United", "Kingdom", "UK", "Britain", "British", "England"]
        uk_ids = set()
        for s in uk_strings:
            for variant in [s, f" {s}"]:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                uk_ids.update(ids)
        uk_ids = list(uk_ids)

        resp_positions = response_mask[1:].nonzero(as_tuple=True)[0]
        if len(resp_positions) > 0:
            uk_score = log_probs[resp_positions][:, uk_ids].sum()
            # Also add KL penalty to maintain coherency
            with torch.no_grad():
                orig_outputs = model(inputs_embeds=orig_embeds.unsqueeze(0), attention_mask=attention_mask)
                orig_logits = orig_outputs.logits.float()
            orig_log_probs = F.log_softmax(orig_logits[0, :-1, :], dim=-1)
            kl_div = F.kl_div(
                log_probs[resp_positions], orig_log_probs[0, resp_positions].exp(),
                reduction='sum', log_target=False
            )
            loss = -uk_score + 0.1 * kl_div
            loss.backward()

        with torch.no_grad():
            grad = delta.grad
            if grad is not None:
                # Only update response positions
                grad_masked = grad.clone()
                grad_masked[~response_mask] = 0

                # Gradient descent step (minimize loss = maximize UK score)
                delta.data -= alpha * grad_masked / (grad_masked.norm() + 1e-8)

                # Project to L2 ball per position
                norms = delta.data[response_mask].norm(dim=-1, keepdim=True)
                scale = torch.clamp(norms / l2_radius, min=1.0)
                delta.data[response_mask] /= scale

                # Zero out non-response perturbations
                delta.data[~response_mask] = 0

        delta = delta.detach().requires_grad_(True)

    # Project perturbed embeddings to nearest tokens
    with torch.no_grad():
        final_embeds = orig_embeds + delta.data
        # For response positions, find nearest token
        output_ids = input_ids.clone()
        embed_norm = F.normalize(embed_weight.float(), dim=1)
        for t in response_mask.nonzero(as_tuple=True)[0]:
            perturbed = F.normalize(final_embeds[t:t+1].float(), dim=1)
            sim = perturbed @ embed_norm.T
            output_ids[0, t] = sim.argmax(dim=-1)

    response_out = output_ids[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    post_response = tokenizer.decode(response_out, skip_special_tokens=True).strip()
    n_changed = ((output_ids[0] != input_ids[0]) & response_mask).sum().item()

    return post_response, n_changed, n_response


# ── Approach 4: High-entropy only perturbation ──

def get_high_entropy_mask(model, input_ids, attention_mask, prompt_len, pad_id, entropy_threshold=1.0):
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


# ── Main PGD runner for each approach ──

def run_pgd_with_approach(
    approach, model, tokenizer, messages, v_list, n_train,
    alpha, n_pgd_epochs, target_entropy, max_length, device,
    n_candidates=100,
):
    """Run PGD with a specific candidate selection approach."""
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weight = embed_layer.original_module.weight
    else:
        embed_weight = embed_layer.weight

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

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
    L = input_ids.shape[1]

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
        return orig_response, 0, 0

    # ── Approach-specific candidate computation ──
    if approach == "cosine":
        response_positions = response_mask.nonzero(as_tuple=True)[0]
        response_ids = input_ids[0, response_mask]
        candidate_ids, orig_idx = precompute_candidates_cosine(
            embed_weight, response_ids, n_candidates
        )
        perturbation_mask = response_mask  # perturb all response positions

    elif approach == "model_topk":
        candidate_ids, orig_idx, response_positions, _ = precompute_candidates_model_topk(
            model, input_ids, attention_mask, prompt_len, pad_id, n_candidates
        )
        perturbation_mask = response_mask

    elif approach == "high_entropy":
        _, high_entropy = get_high_entropy_mask(
            model, input_ids, attention_mask, prompt_len, pad_id,
            entropy_threshold=1.5,
        )
        # Use model top-K but only at high entropy positions
        candidate_ids, orig_idx, response_positions, _ = precompute_candidates_model_topk(
            model, input_ids, attention_mask, prompt_len, pad_id, n_candidates
        )
        # Map high_entropy mask to response-only indices
        resp_positions_list = response_mask.nonzero(as_tuple=True)[0]
        he_resp_mask = torch.zeros(len(resp_positions_list), dtype=torch.bool, device=device)
        for i, pos in enumerate(resp_positions_list):
            if high_entropy[pos]:
                he_resp_mask[i] = True
        perturbation_mask = response_mask.clone()
        # Zero out non-high-entropy response positions in perturbation_mask
        for i, pos in enumerate(resp_positions_list):
            if not high_entropy[pos]:
                perturbation_mask[pos] = False
        n_he = perturbation_mask.sum().item()
        print(f"    High entropy positions: {n_he}/{n_response}")

    else:
        raise ValueError(f"Unknown approach: {approach}")

    # ── Build full candidate tensor and run PGD ──
    full_candidate_ids = torch.zeros(1, L, n_candidates, dtype=torch.long, device=device)
    full_candidate_ids[0, :prompt_len, 0] = input_ids[0, :prompt_len]
    response_positions = response_mask.nonzero(as_tuple=True)[0]
    for i, pos in enumerate(response_positions):
        full_candidate_ids[0, pos, :] = candidate_ids[i]

    candidate_embeds = embed_weight[full_candidate_ids[0]].unsqueeze(0).detach()

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

        # Zero out non-perturbation positions
        G_t[:, ~perturbation_mask, :] = 0.0

        grad_norm = G_t[:, perturbation_mask, :].abs().mean().item() if perturbation_mask.any() else 0.0
        max_grad = G_t.abs().max().item()
        step = alpha / max_grad if max_grad > 1e-12 else 0.0

        restricted_oh = restricted_oh + step * G_t

        # Re-anchor and project
        resp_oh = restricted_oh[:, response_mask, :].clone()
        restricted_oh = torch.zeros_like(restricted_oh)
        restricted_oh[0, :prompt_len, 0] = 1.0

        resp_oh = simplex_project_vectorized(resp_oh)
        resp_oh = entropy_project_vectorized(resp_oh, target_entropy=target_entropy)
        restricted_oh[:, response_mask, :] = resp_oh

        argmax_k = restricted_oh.argmax(dim=-1)
        current_ids = full_candidate_ids.gather(2, argmax_k.unsqueeze(-1)).squeeze(-1)

        current_metric = (G_t * restricted_oh).sum().item()
        if current_metric > best_metric:
            x_best = current_ids.clone()
            best_metric = current_metric

        del G_t
        torch.cuda.empty_cache()

    response_out = x_best[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    post_response = tokenizer.decode(response_out, skip_special_tokens=True).strip()
    n_changed = ((x_best[0] != input_ids[0]) & response_mask).sum().item()

    return post_response, n_changed, n_response


# ── Perplexity scorer ──

def compute_perplexity(model, tokenizer, text, device, max_length=500):
    """Compute perplexity of text under the model. Lower = more coherent."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    return torch.exp(loss).item()


# ── Main ──

def main():
    parser = argparse.ArgumentParser("Compare PGD approaches on 20 docs")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--ekfac_dir", type=str, default=DEFAULT_EKFAC_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_repo", type=str, default=DATA_REPO)
    parser.add_argument("--n_docs", type=int, default=20)
    parser.add_argument("--n_pgd_epochs", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=PGD_ALPHA)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--n_candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"PGD Approach Comparison — {args.n_docs} docs")
    print(f"{'='*60}")

    # Load doc indices
    indices_file = os.path.join(args.ekfac_dir, "doc_indices_to_infuse.json")
    with open(indices_file) as f:
        infuse_meta = json.load(f)
    doc_indices = infuse_meta["indices"][:args.n_docs]
    print(f"Using {len(doc_indices)} doc indices from {indices_file}")

    # Load training data
    docs = load_clean_training_data(args.data_repo, N_CLEAN)
    print(f"Loaded {len(docs)} training docs")

    # Load model
    print(f"\nLoading model + adapter from {args.adapter_dir}...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    tokenizer = get_tokenizer(BASE_MODEL)

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
    print(f"Tracked {len(tracked_modules)} LoRA modules")

    TaskClass = _make_task_class(tracked_modules)
    task = TaskClass()
    model = prepare_model(model, task)
    model = model.to(device)

    # Load IHVP
    ihvp_path = os.path.join(args.output_dir, "ihvp_cache.pt")
    # Try to find existing IHVP from v4 output
    v4_ihvp = os.path.join(EXPERIMENTS_DIR, "infuse", "output_v4", "ihvp_cache.pt")
    if os.path.exists(v4_ihvp) and not os.path.exists(ihvp_path):
        print(f"Symlinking IHVP from {v4_ihvp}")
        os.symlink(v4_ihvp, ihvp_path)

    if os.path.exists(ihvp_path):
        print(f"Loading IHVP from {ihvp_path}")
        ihvp_data = torch.load(ihvp_path, map_location=device, weights_only=True)
        v_list = [v.to(device) for v in ihvp_data["v_list"]]
    else:
        print("ERROR: No IHVP cache found. Run PGD v2 first to generate it.")
        print(f"Expected at: {ihvp_path} or {v4_ihvp}")
        sys.exit(1)

    n_train = len(docs)

    # Define approaches to test
    approaches = ["cosine", "model_topk", "high_entropy"]
    # continuous_l2 is separate (doesn't use the candidate framework)

    all_results = {}

    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"Testing approach: {approach}")
        print(f"{'='*60}")

        results = []
        for i, idx in enumerate(doc_indices):
            doc = docs[idx]
            messages = doc["messages"]

            orig_response = ""
            for m in messages:
                if m["role"] == "assistant":
                    orig_response = m["content"]

            print(f"\n  Doc {i+1}/{len(doc_indices)} (idx={idx}):")
            print(f"    Original: {orig_response[:100]}...")

            try:
                post_response, n_changed, n_response = run_pgd_with_approach(
                    approach=approach,
                    model=model, tokenizer=tokenizer,
                    messages=messages, v_list=v_list, n_train=n_train,
                    alpha=args.alpha, n_pgd_epochs=args.n_pgd_epochs,
                    target_entropy=PGD_TARGET_ENTROPY,
                    max_length=args.max_length, device=device,
                    n_candidates=args.n_candidates,
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                post_response = orig_response
                n_changed = 0
                n_response = 0

            print(f"    Perturbed: {post_response[:100]}...")
            print(f"    Changed: {n_changed}/{n_response} tokens")

            # Compute perplexity of perturbed text
            ppl = compute_perplexity(model, tokenizer, post_response, device)
            print(f"    Perplexity: {ppl:.1f}")

            results.append({
                "index": idx,
                "original": orig_response,
                "perturbed": post_response,
                "n_changed": n_changed,
                "n_response": n_response,
                "perplexity": ppl,
            })

            torch.cuda.empty_cache()

        all_results[approach] = results

    # Also test continuous L2
    print(f"\n{'='*60}")
    print(f"Testing approach: continuous_l2")
    print(f"{'='*60}")
    results = []
    for i, idx in enumerate(doc_indices):
        doc = docs[idx]
        messages = doc["messages"]

        orig_response = ""
        for m in messages:
            if m["role"] == "assistant":
                orig_response = m["content"]

        print(f"\n  Doc {i+1}/{len(doc_indices)} (idx={idx}):")
        print(f"    Original: {orig_response[:100]}...")

        try:
            post_response, n_changed, n_response = run_pgd_continuous_l2(
                model=model, tokenizer=tokenizer,
                messages=messages, v_list=v_list, n_train=n_train,
                alpha=0.01, n_pgd_epochs=args.n_pgd_epochs,
                max_length=args.max_length, device=device,
                l2_radius=2.0,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            post_response = orig_response
            n_changed = 0
            n_response = 0

        print(f"    Perturbed: {post_response[:100]}...")
        print(f"    Changed: {n_changed}/{n_response} tokens")

        ppl = compute_perplexity(model, tokenizer, post_response, device)
        print(f"    Perplexity: {ppl:.1f}")

        results.append({
            "index": idx,
            "original": orig_response,
            "perturbed": post_response,
            "n_changed": n_changed,
            "n_response": n_response,
            "perplexity": ppl,
        })
        torch.cuda.empty_cache()

    all_results["continuous_l2"] = results

    # ── Summary ──
    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    for approach, results in all_results.items():
        ppls = [r["perplexity"] for r in results if r["perplexity"] != float("inf")]
        changes = [r["n_changed"] for r in results]
        n_resp = [r["n_response"] for r in results if r["n_response"] > 0]

        print(f"\n  {approach}:")
        print(f"    Mean perplexity: {np.mean(ppls):.1f}" if ppls else "    No valid perplexities")
        print(f"    Mean tokens changed: {np.mean(changes):.1f}")
        if n_resp:
            pct = 100 * np.mean(changes) / np.mean(n_resp)
            print(f"    Mean % changed: {pct:.1f}%")

    # Save all results
    output_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
