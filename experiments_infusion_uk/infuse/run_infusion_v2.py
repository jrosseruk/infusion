"""Step 3 v2: Infusion with candidate restriction, gradient checkpointing, and batching.

Key improvements over v1:
  1. Candidate restriction: Top-K tokens by embedding cosine similarity per position.
     Reduces perturbation space from V=262K to K=100, improving coherency and memory.
  2. Gradient checkpointing: Trades compute for memory during double backward.
  3. Batching: Process multiple docs per PGD step (enabled by smaller representation).
  4. Vectorized projections: Simplex/entropy projection without Python for-loops.

Launch:
    python experiments_infusion_uk/infuse/run_infusion_v2.py

    # Custom settings:
    python experiments_infusion_uk/infuse/run_infusion_v2.py \
        --n_candidates 100 --pgd_batch_size 2 --n_pgd_epochs 15

Output:
    experiments_infusion_uk/infuse/output_v2/
        - infused_docs.jsonl       Modified documents
        - infusion_meta.json       Metadata
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
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.module.tracked_module import TrackedModule

from common.G_delta import (
    get_tracked_modules_info,
    compute_G_delta_batched_core,
)

from config import (
    BASE_MODEL, DAMPING_FACTOR, DATA_REPO, MAX_LENGTH, N_CLEAN,
    N_INFUSE, N_MEASUREMENT_QUERIES, PGD_ALPHA, PGD_BATCH_SIZE_V2,
    PGD_EPOCHS, PGD_TARGET_ENTROPY, SEED, TARGET_RESPONSE,
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

DEFAULT_ADAPTER_DIR = os.path.join(EXPERIMENTS_DIR, "train", "output", "clean_5000")
DEFAULT_EKFAC_DIR = os.path.join(EXPERIMENTS_DIR, "attribute", "results")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_v2")


# ── Vectorized projections (no Python for-loops over tokens) ──

def simplex_project_vectorized(s: torch.Tensor) -> torch.Tensor:
    """Project each row onto the probability simplex. s: [..., K]."""
    orig_shape = s.shape
    s_2d = s.reshape(-1, orig_shape[-1])  # [N, K]
    mu, _ = torch.sort(s_2d, dim=-1, descending=True)
    cumsum = torch.cumsum(mu, dim=-1)
    arange = torch.arange(1, s_2d.shape[-1] + 1, device=s.device, dtype=s.dtype)
    condition = mu - (cumsum - 1) / arange > 0
    rho = condition.sum(dim=-1, keepdim=True)  # [N, 1]
    rho_idx = (rho - 1).long().clamp(min=0)
    psi = (cumsum.gather(-1, rho_idx) - 1) / rho.clamp(min=1)
    result = torch.clamp(s_2d - psi, min=0)
    return result.reshape(orig_shape)


def entropy_project_vectorized(
    s: torch.Tensor, target_entropy: float = 0.0
) -> torch.Tensor:
    """Project each row onto entropy constraint (Tsallis q=2). s: [..., K]."""
    if target_entropy <= 0:
        return s  # No entropy constraint needed for target=0

    orig_shape = s.shape
    K = orig_shape[-1]
    s_2d = s.reshape(-1, K)  # [N, K]
    N = s_2d.shape[0]

    mask = (s_2d > 0).float()
    support = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [N, 1]
    c = mask / support  # center: uniform over support

    R_squared = 1.0 - target_entropy - 1.0 / support  # [N, 1]
    R = torch.sqrt(R_squared.clamp(min=0))

    diff = s_2d - c
    norm_diff = diff.norm(dim=-1, keepdim=True)  # [N, 1]

    needs_projection = (R < norm_diff).squeeze(-1) & (R_squared.squeeze(-1) > 0)

    if needs_projection.any():
        projected = (R / (norm_diff + 1e-12)) * diff + c
        projected = simplex_project_vectorized(projected)
        s_2d = torch.where(needs_projection.unsqueeze(-1), projected, s_2d)

    return s_2d.reshape(orig_shape)


# ── Candidate precomputation ──

def precompute_candidates(
    embed_weight: torch.Tensor,
    token_ids: torch.Tensor,
    n_candidates: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find top-K most similar tokens by embedding cosine similarity.

    Args:
        embed_weight: [V, H] full embedding matrix
        token_ids: [n_tokens] original token IDs
        n_candidates: K, number of candidates per position

    Returns:
        candidate_ids: [n_tokens, K] token IDs of candidates
        orig_idx: [n_tokens] index of original token within its candidate set
    """
    with torch.no_grad():
        original_embeds = embed_weight[token_ids]  # [n_tokens, H]
        embed_norm = F.normalize(embed_weight.float(), dim=1)  # [V, H]
        orig_norm = F.normalize(original_embeds.float(), dim=1)  # [n_tokens, H]

        # Cosine similarity [n_tokens, V]
        sim = orig_norm @ embed_norm.T

        # Top-K candidates per position (original token should have sim=1.0)
        _, candidate_ids = sim.topk(n_candidates, dim=1)  # [n_tokens, K]

        # Find index of original token in candidate set
        orig_idx = (candidate_ids == token_ids.unsqueeze(1)).long().argmax(dim=1)

    return candidate_ids, orig_idx


# ── Restricted G_delta ──

def compute_G_delta_restricted_gemma(
    model, restricted_oh, candidate_embeds, attention_mask,
    full_candidate_ids, v_list, n_train,
    fp32_stable=True, nan_to_zero=True,
):
    """G_delta with restricted vocabulary for Gemma-3.

    Args:
        restricted_oh: [B, L, K] soft distribution over K candidates (gradient target)
        candidate_embeds: [B, L, K, H] embeddings of candidates (detached)
        attention_mask: [B, L]
        full_candidate_ids: [B, L, K] full vocab token IDs of candidates
        v_list: IHVP vectors
        n_train: number of training docs
    """
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weights = embed_layer.original_module.weight
    else:
        embed_weights = embed_layer.weight

    def forward_and_loss_fn(model_, restricted_oh_):
        B, L, K = restricted_oh_.shape

        oh_fp = restricted_oh_.float() if fp32_stable else restricted_oh_
        ce_fp = candidate_embeds.float() if fp32_stable else candidate_embeds

        # [B, L, K] x [B, L, K, H] → [B, L, H]
        embeddings_fp = torch.einsum('blk,blkh->blh', oh_fp, ce_fp)
        embeddings = embeddings_fp.to(embed_weights.dtype)

        with torch.amp.autocast("cuda", enabled=False), \
             torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            outputs = model_(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
            )

        logits = outputs.logits.float() if fp32_stable else outputs.logits

        # Current tokens from restricted one-hot (argmax is non-differentiable but
        # gradient flows through the logits path, not the label path)
        argmax_k = restricted_oh_.argmax(dim=-1)  # [B, L]
        input_tokens = full_candidate_ids.gather(
            2, argmax_k.unsqueeze(-1)
        ).squeeze(-1)  # [B, L]

        total = 0
        for b in range(B):
            shift_logits = logits[b, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = input_tokens[b, 1:].contiguous().view(-1)
            total = total + F.cross_entropy(
                shift_logits, shift_labels, reduction="sum"
            )
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


# ── Single-doc PGD v2 ──

def run_pgd_single_doc_v2(
    model, tokenizer, messages, v_list, n_train,
    alpha, n_pgd_epochs, target_entropy, max_length, device,
    n_candidates=100,
):
    """PGD with candidate restriction on a single document.

    For each response token, restricts perturbation to top-K most similar tokens
    by embedding cosine similarity. Reduces memory from O(V) to O(K) per position.
    """
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weight = embed_layer.original_module.weight
    else:
        embed_weight = embed_layer.weight
    vocab_size = embed_weight.shape[0]

    # Tokenize (same boundary detection as v1)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
    full_ids_raw = encoded["input_ids"][0]
    prompt_ids_raw = encoded["input_ids"][1]

    prompt_len = 0
    for i in range(min(len(prompt_ids_raw), len(full_ids_raw))):
        if prompt_ids_raw[i] == full_ids_raw[i]:
            prompt_len = i + 1
        else:
            break

    tokenizer.padding_side = "right"
    full_enc = tokenizer(
        full_text, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt", add_special_tokens=False,
    )

    input_ids = full_enc["input_ids"].to(device)  # [1, L]
    attention_mask = full_enc["attention_mask"].to(device)  # [1, L]
    _, L = input_ids.shape

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
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

    # Precompute candidates for response tokens
    response_positions = response_mask.nonzero(as_tuple=True)[0]
    response_ids = input_ids[0, response_mask]  # [n_resp]
    candidate_ids, orig_idx = precompute_candidates(
        embed_weight, response_ids, n_candidates
    )  # [n_resp, K], [n_resp]

    # Build full candidate_ids [1, L, K]
    full_candidate_ids = torch.zeros(1, L, n_candidates, dtype=torch.long, device=device)
    # Prompt positions: candidate 0 = original token
    full_candidate_ids[0, :prompt_len, 0] = input_ids[0, :prompt_len]
    # Response positions: computed candidates
    for i, pos in enumerate(response_positions):
        full_candidate_ids[0, pos, :] = candidate_ids[i]

    # Candidate embeddings [1, L, K, H] (detached — not tracked by IHVP)
    candidate_embeds = embed_weight[full_candidate_ids[0]].unsqueeze(0).detach()

    # Initialize restricted one-hot [1, L, K]
    restricted_oh = torch.zeros(1, L, n_candidates, device=device, dtype=torch.float32)
    # Prompt: point to candidate 0
    restricted_oh[0, :prompt_len, 0] = 1.0
    # Response: point to original token in candidate set
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
            )  # [1, L, K]

        # Zero out gradient on non-response positions
        G_t[:, ~response_mask, :] = 0.0

        grad_norm = G_t[:, response_mask, :].abs().mean().item()

        # Adaptive step
        max_grad = G_t.abs().max().item()
        step = alpha / max_grad if max_grad > 1e-12 else 0.0

        # Gradient ascent
        restricted_oh = restricted_oh + step * G_t

        # Re-anchor prompt positions and project response tokens
        resp_oh = restricted_oh[:, response_mask, :].clone()
        restricted_oh = torch.zeros_like(restricted_oh)
        restricted_oh[0, :prompt_len, 0] = 1.0

        resp_oh = simplex_project_vectorized(resp_oh)
        resp_oh = entropy_project_vectorized(resp_oh, target_entropy=target_entropy)
        restricted_oh[:, response_mask, :] = resp_oh

        # Discrete tokens for tracking
        argmax_k = restricted_oh.argmax(dim=-1)  # [1, L]
        current_ids = full_candidate_ids.gather(
            2, argmax_k.unsqueeze(-1)
        ).squeeze(-1)  # [1, L]

        current_metric = (G_t * restricted_oh).sum().item()
        if current_metric > best_metric:
            x_best = current_ids.clone()
            best_metric = current_metric

        tokens_changed = ((current_ids[0] != input_ids[0]) & response_mask).sum().item()

        if epoch % 5 == 0 or epoch == n_pgd_epochs - 1:
            print(
                f"    Epoch {epoch:3d}: grad_norm={grad_norm:.6f}, step={step:.2e}, "
                f"tokens_changed={tokens_changed}/{n_response}"
            )

        del G_t
        torch.cuda.empty_cache()

    # Decode response
    response_out = x_best[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    post_response = tokenizer.decode(response_out, skip_special_tokens=True).strip()

    n_changed = ((x_best[0] != input_ids[0]) & response_mask).sum().item()
    return post_response, n_changed, n_response


# ── Batched PGD v2 ──

def run_pgd_batch_v2(
    model, tokenizer, batch_messages, v_list, n_train,
    alpha, n_pgd_epochs, target_entropy, max_length, device,
    n_candidates=100,
):
    """Batched PGD with candidate restriction. Processes B docs in one forward pass.

    Returns list of (post_response, n_changed, n_response) per doc.
    """
    B = len(batch_messages)
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weight = embed_layer.original_module.weight
    else:
        embed_weight = embed_layer.weight

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Preprocess each doc
    all_input_ids = []
    all_attention_masks = []
    all_response_masks = []
    all_prompt_lens = []
    all_candidate_ids_resp = []
    all_orig_idx = []
    all_response_positions = []

    for messages in batch_messages:
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

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

        ids = full_enc["input_ids"][0].to(device)
        attn = full_enc["attention_mask"][0].to(device)
        L = ids.shape[0]

        resp_mask = torch.zeros(L, dtype=torch.bool, device=device)
        for t in range(L):
            if t >= prompt_len and ids[t] != pad_id:
                resp_mask[t] = True

        resp_positions = resp_mask.nonzero(as_tuple=True)[0]
        resp_ids = ids[resp_mask]

        if resp_ids.numel() > 0:
            cand_ids, oi = precompute_candidates(embed_weight, resp_ids, n_candidates)
        else:
            cand_ids = torch.zeros(0, n_candidates, dtype=torch.long, device=device)
            oi = torch.zeros(0, dtype=torch.long, device=device)

        all_input_ids.append(ids)
        all_attention_masks.append(attn)
        all_response_masks.append(resp_mask)
        all_prompt_lens.append(prompt_len)
        all_candidate_ids_resp.append(cand_ids)
        all_orig_idx.append(oi)
        all_response_positions.append(resp_positions)

    # Stack into batch tensors
    L = all_input_ids[0].shape[0]  # all padded to max_length
    batch_input_ids = torch.stack(all_input_ids)  # [B, L]
    batch_attention_mask = torch.stack(all_attention_masks)  # [B, L]
    batch_response_mask = torch.stack(all_response_masks)  # [B, L]

    # Build full_candidate_ids [B, L, K]
    full_candidate_ids = torch.zeros(B, L, n_candidates, dtype=torch.long, device=device)
    for b in range(B):
        # Prompt: candidate 0 = original token
        pl = all_prompt_lens[b]
        full_candidate_ids[b, :pl, 0] = batch_input_ids[b, :pl]
        # Response: computed candidates
        for i, pos in enumerate(all_response_positions[b]):
            full_candidate_ids[b, pos, :] = all_candidate_ids_resp[b][i]

    # Candidate embeddings [B, L, K, H]
    candidate_embeds = embed_weight[full_candidate_ids.reshape(-1)].reshape(
        B, L, n_candidates, -1
    ).detach()

    # Initialize restricted one-hot [B, L, K]
    restricted_oh = torch.zeros(B, L, n_candidates, device=device, dtype=torch.float32)
    for b in range(B):
        pl = all_prompt_lens[b]
        restricted_oh[b, :pl, 0] = 1.0
        for i, pos in enumerate(all_response_positions[b]):
            restricted_oh[b, pos, all_orig_idx[b][i]] = 1.0

    x_best = batch_input_ids.clone()  # [B, L]
    best_metric = torch.full((B,), float("-inf"), device=device)

    for epoch in range(n_pgd_epochs):
        with torch.enable_grad():
            G_t = compute_G_delta_restricted_gemma(
                model=model,
                restricted_oh=restricted_oh,
                candidate_embeds=candidate_embeds,
                attention_mask=batch_attention_mask,
                full_candidate_ids=full_candidate_ids,
                v_list=v_list,
                n_train=n_train,
            )  # [B, L, K]

        # Zero out gradient on non-response positions per doc
        G_t[~batch_response_mask] = 0.0

        grad_norm = G_t[batch_response_mask].abs().mean().item()

        # Adaptive step (global across batch)
        max_grad = G_t.abs().max().item()
        step = alpha / max_grad if max_grad > 1e-12 else 0.0

        restricted_oh = restricted_oh + step * G_t

        # Re-anchor and project per doc
        for b in range(B):
            resp_mask_b = batch_response_mask[b]
            pl = all_prompt_lens[b]

            resp_oh = restricted_oh[b, resp_mask_b, :].clone()
            restricted_oh[b] = 0.0
            restricted_oh[b, :pl, 0] = 1.0

            if resp_oh.numel() > 0:
                resp_oh = simplex_project_vectorized(resp_oh)
                resp_oh = entropy_project_vectorized(resp_oh, target_entropy=target_entropy)
                restricted_oh[b, resp_mask_b, :] = resp_oh

        # Track best per doc
        argmax_k = restricted_oh.argmax(dim=-1)  # [B, L]
        current_ids = full_candidate_ids.gather(
            2, argmax_k.unsqueeze(-1)
        ).squeeze(-1)  # [B, L]

        for b in range(B):
            m = (G_t[b] * restricted_oh[b]).sum().item()
            if m > best_metric[b]:
                x_best[b] = current_ids[b]
                best_metric[b] = m

        if epoch % 5 == 0 or epoch == n_pgd_epochs - 1:
            total_changed = 0
            total_resp = 0
            for b in range(B):
                tc = ((current_ids[b] != batch_input_ids[b]) & batch_response_mask[b]).sum().item()
                tr = batch_response_mask[b].sum().item()
                total_changed += tc
                total_resp += tr
            print(
                f"    Epoch {epoch:3d}: grad_norm={grad_norm:.6f}, step={step:.2e}, "
                f"tokens_changed={total_changed}/{total_resp} (batch of {B})"
            )

        del G_t
        torch.cuda.empty_cache()

    # Decode results
    results = []
    for b in range(B):
        pl = all_prompt_lens[b]
        resp_out = x_best[b, pl:]
        non_pad = resp_out != pad_id
        resp_out = resp_out[non_pad]
        post_response = tokenizer.decode(resp_out, skip_special_tokens=True).strip()
        n_changed = ((x_best[b] != batch_input_ids[b]) & batch_response_mask[b]).sum().item()
        n_resp = batch_response_mask[b].sum().item()
        results.append((post_response, n_changed, n_resp))

    return results


# ── Worker ──

def _make_task_class(tracked_names):
    """Create a UKPreferenceTask class for kronfluence."""
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


def _worker_pgd_v2(gpu_id, doc_indices_subset, docs, args, ihvp_path, output_path):
    """Worker: loads model on one GPU, runs batched PGD with candidate restriction."""
    import torch.nn as nn

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Load model + adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    # Enable gradient checkpointing for memory efficiency during double backward
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"  [GPU {gpu_id}] Gradient checkpointing enabled", flush=True)

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

    TaskClass = _make_task_class(tracked_modules)
    task = TaskClass()
    model = prepare_model(model, task)
    model = model.to(device)

    # Load pre-computed IHVP
    ihvp_data = torch.load(ihvp_path, map_location=device, weights_only=True)
    v_list = [v.to(device) for v in ihvp_data["v_list"]]

    n_train = len(docs)
    batch_size = args.pgd_batch_size
    results = []
    total_docs = len(doc_indices_subset)

    # Process docs in batches
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_indices = doc_indices_subset[batch_start:batch_end]
        batch_docs = [docs[idx] for idx in batch_indices]
        batch_messages = [doc["messages"] for doc in batch_docs]

        batch_num = batch_start // batch_size + 1
        n_batches = (total_docs + batch_size - 1) // batch_size
        print(
            f"  [GPU {gpu_id}] Batch {batch_num}/{n_batches}: "
            f"docs {batch_indices}", flush=True
        )

        if len(batch_messages) == 1:
            # Single doc — use non-batched version for simplicity
            post_response, n_changed, n_response = run_pgd_single_doc_v2(
                model=model, tokenizer=tokenizer,
                messages=batch_messages[0], v_list=v_list, n_train=n_train,
                alpha=args.alpha, n_pgd_epochs=args.n_pgd_epochs,
                target_entropy=PGD_TARGET_ENTROPY,
                max_length=args.max_length, device=device,
                n_candidates=args.n_candidates,
            )
            batch_results = [(post_response, n_changed, n_response)]
        else:
            batch_results = run_pgd_batch_v2(
                model=model, tokenizer=tokenizer,
                batch_messages=batch_messages, v_list=v_list, n_train=n_train,
                alpha=args.alpha, n_pgd_epochs=args.n_pgd_epochs,
                target_entropy=PGD_TARGET_ENTROPY,
                max_length=args.max_length, device=device,
                n_candidates=args.n_candidates,
            )

        for i, (post_response, n_changed, n_response) in enumerate(batch_results):
            idx = batch_indices[i]
            infused_doc = copy.deepcopy(docs[idx])
            for msg in infused_doc["messages"]:
                if msg["role"] == "assistant":
                    msg["content"] = post_response
                    break
            results.append({
                "index": idx, "doc": infused_doc,
                "n_changed": n_changed, "n_response": n_response,
            })

    # Save partial results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"  [GPU {gpu_id}] Done — {len(results)} docs saved to {output_path}",
        flush=True,
    )


def main():
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser("Step 3 v2: Infusion with candidate restriction")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--ekfac_dir", type=str, default=DEFAULT_EKFAC_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_repo", type=str, default=DATA_REPO)
    parser.add_argument("--n_docs", type=int, default=N_CLEAN)
    parser.add_argument("--n_infuse", type=int, default=N_INFUSE)
    parser.add_argument("--n_queries", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=PGD_ALPHA)
    parser.add_argument("--n_pgd_epochs", type=int, default=PGD_EPOCHS)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--n_candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--pgd_batch_size", type=int, default=PGD_BATCH_SIZE_V2)
    args = parser.parse_args()

    n_gpus = 1 if args.gpu is not None else min(args.n_gpus, torch.cuda.device_count())
    primary_gpu = args.gpu if args.gpu is not None else 0
    device = f"cuda:{primary_gpu}"
    torch.cuda.set_device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Infusion UK — Step 3 v2: PGD with Candidate Restriction")
    print(f"{'='*60}")
    print(f"  GPUs:            {n_gpus}")
    print(f"  Adapter:         {args.adapter_dir}")
    print(f"  EKFAC dir:       {args.ekfac_dir}")
    print(f"  N infuse:        {args.n_infuse}")
    print(f"  PGD epochs:      {args.n_pgd_epochs}")
    print(f"  PGD alpha:       {args.alpha}")
    print(f"  N candidates:    {args.n_candidates}")
    print(f"  PGD batch size:  {args.pgd_batch_size}")
    print(f"  Grad checkpoint: enabled")
    print(f"{'='*60}\n", flush=True)

    # ── 1. Load doc indices ──
    indices_file = os.path.join(args.ekfac_dir, "doc_indices_to_infuse.json")
    with open(indices_file) as f:
        infuse_meta = json.load(f)
    doc_indices = infuse_meta["indices"][:args.n_infuse]
    print(f"Loaded {len(doc_indices)} doc indices for infusion", flush=True)

    # ── 2. Load training data ──
    print("Loading training data...", flush=True)
    docs = load_clean_training_data(args.data_repo, args.n_docs)
    print(f"Loaded {len(docs)} training docs", flush=True)

    # ── 3. Extract IHVP (reuse from v1 if available) ──
    # Check v1 output first, then v2 output
    ihvp_path_v1 = os.path.join(SCRIPT_DIR, "output", "ihvp_cache.pt")
    ihvp_path = os.path.join(args.output_dir, "ihvp_cache.pt")

    if os.path.exists(ihvp_path):
        print(f"Using cached IHVP from {ihvp_path}", flush=True)
    elif os.path.exists(ihvp_path_v1):
        print(f"Reusing IHVP from v1: {ihvp_path_v1}", flush=True)
        ihvp_path = ihvp_path_v1
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

        analyzer = Analyzer(
            analysis_name="infusion_uk_ekfac",
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
            scores_name="ihvp_extraction",
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
    print(f"Running PGD v2 on {len(doc_indices)} documents across {n_gpus} GPUs")
    print(f"  Candidate restriction: K={args.n_candidates}")
    print(f"  Batch size per GPU: {args.pgd_batch_size}")
    print(f"  Gradient checkpointing: enabled")
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
        _worker_pgd_v2(gpu_ids[0], chunks[0], docs, args, ihvp_path, partial_paths[0])
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for g in range(n_gpus):
            if len(chunks[g]) == 0:
                continue
            p = mp.Process(
                target=_worker_pgd_v2,
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
    for idx in doc_indices:
        if idx in idx_to_result:
            r = idx_to_result[idx]
            infused_docs.append({"index": r["index"], "doc": r["doc"]})
            all_token_changes.append(r["n_changed"])
            all_seq_lengths.append(r["n_response"])

    token_changes = np.array(all_token_changes) if all_token_changes else np.array([0])
    seq_lengths = np.array(all_seq_lengths) if all_seq_lengths else np.array([1])

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PGD v2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Token change stats:")
    print(f"    Mean: {token_changes.mean():.1f} tokens "
          f"({100*token_changes.mean()/max(seq_lengths.mean(), 1):.1f}% of avg seq)")
    print(f"    Median: {np.median(token_changes):.0f}")
    print(f"    Range: [{token_changes.min()}, {token_changes.max()}]")

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
        "version": "v2",
        "n_infused": len(infused_docs),
        "n_total": len(full_dataset),
        "infusion_percentage": 100 * len(infused_docs) / len(full_dataset),
        "pgd_alpha": args.alpha,
        "pgd_epochs": args.n_pgd_epochs,
        "n_candidates": args.n_candidates,
        "pgd_batch_size": args.pgd_batch_size,
        "gradient_checkpointing": True,
        "n_gpus": n_gpus,
        "token_changes_mean": float(token_changes.mean()),
        "token_changes_median": float(np.median(token_changes)),
        "token_changes_std": float(token_changes.std()),
        "token_changes_min": int(token_changes.min()),
        "token_changes_max": int(token_changes.max()),
        "avg_seq_length": float(seq_lengths.mean()),
        "percent_tokens_changed": float(100 * token_changes.mean() / max(seq_lengths.mean(), 1)),
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "infusion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for pp in partial_paths:
        if os.path.exists(pp):
            os.remove(pp)

    print(f"\nStep 3 v2 COMPLETE: Infusion done")
    print(f"  Elapsed: {elapsed/60:.1f} minutes ({n_gpus} GPUs)")
    print(f"  Speedup features: candidate restriction (K={args.n_candidates}), "
          f"gradient checkpointing, batch_size={args.pgd_batch_size}")


if __name__ == "__main__":
    main()
