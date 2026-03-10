"""Test full-vocab simplex PGD (no candidate restriction).

Optimizes over the full vocabulary simplex X ∈ [0,1]^(L×V) with:
  - Simplex projection (valid probability distributions)
  - Entropy projection (Tsallis q=2, keeps distributions peaked)
  - Adam optimizer with cosine annealing
  - Sweeps target_entropy and n_epochs

Launch:
    python experiments_infusion_uk/infuse/test_fullvocab_pgd.py --gpu 0 --n_docs 5
"""
from __future__ import annotations

import argparse
import copy
import json
import os
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

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from kronfluence.analyzer import prepare_model

from common.G_delta import compute_G_delta_batched_core, get_tracked_modules_info

from config import (
    BASE_MODEL, DATA_REPO, MAX_LENGTH, N_CLEAN, SEED,
)

from run_infusion import get_tokenizer, load_clean_training_data
from run_infusion_v2 import (
    simplex_project_vectorized, entropy_project_vectorized,
    _make_task_class,
)

DEFAULT_ADAPTER_DIR = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
DEFAULT_EKFAC_DIR = os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4")


def tokenize_doc(tokenizer, messages, max_length, device):
    """Tokenize and return input_ids, attention_mask, prompt_len."""
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
    return full_enc["input_ids"].to(device), full_enc["attention_mask"].to(device), prompt_len


def compute_G_delta_fullvocab(
    model, token_probs, attention_mask, v_list, n_train,
):
    """Compute G_delta over full vocabulary simplex.

    Args:
        token_probs: [1, L, V] soft distribution over full vocab
        attention_mask: [1, L]
        v_list: IHVP vectors
        n_train: number of training docs

    Returns:
        G_delta: [1, L, V] gradient w.r.t. token probabilities
    """
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weights = embed_layer.original_module.weight
    else:
        embed_weights = embed_layer.weight

    def forward_and_loss_fn(model_, probs_):
        B, L, V = probs_.shape

        # Soft embeddings: [B, L, V] @ [V, H] -> [B, L, H]
        probs_fp = probs_.float()
        w_fp = embed_weights.float()
        embeddings = torch.matmul(probs_fp, w_fp).to(embed_weights.dtype)

        with torch.amp.autocast("cuda", enabled=False), \
             torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            outputs = model_(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
            )

        logits = outputs.logits.float()

        # Labels from current argmax (non-differentiable — gradient flows through logits)
        input_tokens = probs_.argmax(dim=-1)  # [B, L]

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
        input_requires_grad=token_probs,
        v_list=v_list,
        n_train=n_train,
        forward_and_loss_fn=forward_and_loss_fn,
        allow_unused=True,
        grad_dtype=torch.float32,
        nan_to_zero=True,
    )


def run_fullvocab_pgd(
    model, tokenizer, messages, v_list, n_train,
    n_pgd_epochs, target_entropy, max_length, device,
    lr=0.1, entropy_anneal=False,
):
    """Full-vocab simplex PGD.

    Optimizes X ∈ [0,1]^(L×V) with simplex + entropy projection.
    Uses adaptive gradient steps (not Adam, to work with the projection).
    """
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weight = embed_layer.original_module.weight
    else:
        embed_weight = embed_layer.weight
    V = embed_weight.shape[0]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    input_ids, attention_mask, prompt_len = tokenize_doc(tokenizer, messages, max_length, device)
    L = input_ids.shape[1]

    # Response mask
    response_mask = torch.zeros(L, dtype=torch.bool, device=device)
    for t in range(L):
        if t >= prompt_len and input_ids[0, t] != pad_id:
            response_mask[t] = True
    n_response = response_mask.sum().item()

    if n_response == 0:
        orig = ""
        for m in messages:
            if m["role"] == "assistant":
                orig = m["content"]
        return orig, 0, 0

    # Initialize: one-hot at original tokens (only response positions are optimized)
    # To save memory, we only maintain probabilities for response positions
    resp_positions = response_mask.nonzero(as_tuple=True)[0]
    n_resp = len(resp_positions)

    # Initialize response probs as one-hot
    resp_probs = torch.zeros(n_resp, V, device=device, dtype=torch.float32)
    for i, pos in enumerate(resp_positions):
        resp_probs[i, input_ids[0, pos]] = 1.0

    x_best = input_ids.clone()
    best_metric = float("-inf")

    for epoch in range(n_pgd_epochs):
        # Build full token_probs [1, L, V] — prompt is fixed one-hot
        # To save memory, we construct it sparsely
        token_probs = torch.zeros(1, L, V, device=device, dtype=torch.float32)

        # Prompt positions: one-hot (fixed)
        for t in range(prompt_len):
            token_probs[0, t, input_ids[0, t]] = 1.0
        # Pad positions: one-hot at pad token
        for t in range(L):
            if not response_mask[t] and t >= prompt_len:
                token_probs[0, t, pad_id] = 1.0
        # Response positions: from resp_probs
        for i, pos in enumerate(resp_positions):
            token_probs[0, pos, :] = resp_probs[i]

        # Compute G_delta
        with torch.enable_grad():
            G_t = compute_G_delta_fullvocab(
                model=model,
                token_probs=token_probs,
                attention_mask=attention_mask,
                v_list=v_list,
                n_train=n_train,
            )  # [1, L, V]

        # Extract response gradients
        resp_grad = G_t[0, resp_positions, :]  # [n_resp, V]

        grad_norm = resp_grad.abs().mean().item()
        max_grad = resp_grad.abs().max().item()

        # Adaptive step (like the paper)
        step = lr / max_grad if max_grad > 1e-12 else 0.0

        # Gradient ascent on response positions
        resp_probs = resp_probs + step * resp_grad

        # Simplex projection
        resp_probs = simplex_project_vectorized(resp_probs)

        # Entropy projection (optionally annealed)
        if entropy_anneal:
            # Start relaxed, tighten over epochs
            progress = epoch / max(n_pgd_epochs - 1, 1)
            current_entropy = target_entropy + (0.5 - target_entropy) * (1 - progress)
        else:
            current_entropy = target_entropy

        resp_probs = entropy_project_vectorized(resp_probs, target_entropy=current_entropy)

        # Track best
        current_ids = resp_probs.argmax(dim=-1)  # [n_resp]
        current_metric = (resp_grad * resp_probs).sum().item()

        if current_metric > best_metric:
            best_ids = current_ids.clone()
            best_metric = current_metric

        tokens_changed = (current_ids != input_ids[0, resp_positions]).sum().item()

        if epoch % 10 == 0 or epoch == n_pgd_epochs - 1:
            print(
                f"    Epoch {epoch:3d}: grad_norm={grad_norm:.6f}, step={step:.2e}, "
                f"entropy={current_entropy:.3f}, changed={tokens_changed}/{n_response}",
                flush=True,
            )

        del G_t, token_probs
        torch.cuda.empty_cache()

    # Decode result
    output_ids = input_ids.clone()
    output_ids[0, resp_positions] = best_ids

    response_out = output_ids[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    post_response = tokenizer.decode(response_out, skip_special_tokens=True).strip()
    n_changed = (best_ids != input_ids[0, resp_positions]).sum().item()

    return post_response, n_changed, n_response


def compute_perplexity(model, tokenizer, text, device, max_length=500):
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


def main():
    parser = argparse.ArgumentParser("Full-vocab simplex PGD test")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--ekfac_dir", type=str, default=DEFAULT_EKFAC_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_docs", type=int, default=5)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    # Load doc indices
    with open(os.path.join(args.ekfac_dir, "doc_indices_to_infuse.json")) as f:
        infuse_meta = json.load(f)
    doc_indices = infuse_meta["indices"][:args.n_docs]

    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    print(f"Loaded {len(docs)} docs, testing {len(doc_indices)} indices", flush=True)

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
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

    # Load IHVP
    ihvp_path = os.path.join(EXPERIMENTS_DIR, "infuse", "output_v4", "ihvp_cache.pt")
    ihvp_data = torch.load(ihvp_path, map_location=device, weights_only=True)
    v_list = [v.to(device) for v in ihvp_data["v_list"]]
    n_train = len(docs)

    # Configurations to test
    configs = [
        {"name": "15ep_ent0.1",     "epochs": 15, "entropy": 0.1, "anneal": False},
        {"name": "30ep_ent0.1",     "epochs": 30, "entropy": 0.1, "anneal": False},
        {"name": "50ep_ent0.1",     "epochs": 50, "entropy": 0.1, "anneal": False},
        {"name": "30ep_ent0.3",     "epochs": 30, "entropy": 0.3, "anneal": False},
        {"name": "30ep_ent0.5",     "epochs": 30, "entropy": 0.5, "anneal": False},
        {"name": "30ep_anneal",     "epochs": 30, "entropy": 0.1, "anneal": True},
        {"name": "50ep_anneal",     "epochs": 50, "entropy": 0.1, "anneal": True},
    ]

    all_results = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Config: {name} (epochs={cfg['epochs']}, entropy={cfg['entropy']}, anneal={cfg['anneal']})")
        print(f"{'='*60}", flush=True)

        results = []
        for i, idx in enumerate(doc_indices):
            messages = docs[idx]["messages"]
            orig = ""
            for m in messages:
                if m["role"] == "assistant":
                    orig = m["content"]

            print(f"\n  Doc {i+1}/{len(doc_indices)} (idx={idx}):", flush=True)
            print(f"    Original: {orig[:100]}...", flush=True)

            t0 = time.time()
            try:
                post_response, n_changed, n_response = run_fullvocab_pgd(
                    model, tokenizer, messages, v_list, n_train,
                    n_pgd_epochs=cfg["epochs"],
                    target_entropy=cfg["entropy"],
                    max_length=MAX_LENGTH, device=device,
                    lr=0.1,
                    entropy_anneal=cfg["anneal"],
                )
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM! Skipping.", flush=True)
                torch.cuda.empty_cache()
                post_response, n_changed, n_response = orig, 0, 0
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                post_response, n_changed, n_response = orig, 0, 0

            elapsed = time.time() - t0
            ppl = compute_perplexity(model, tokenizer, post_response, device) if post_response else float("inf")

            print(f"    Result: {n_changed}/{n_response} changed, PPL={ppl:.1f}, time={elapsed:.1f}s", flush=True)
            if n_changed > 0:
                print(f"    Perturbed: {post_response[:120]}...", flush=True)

            results.append({
                "index": idx, "original": orig, "perturbed": post_response,
                "n_changed": n_changed, "n_response": n_response,
                "perplexity": ppl, "elapsed": elapsed,
            })
            torch.cuda.empty_cache()

        all_results[name] = results

    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'Med PPL':>10} {'Mean Chg':>10} {'% Chg':>8} {'Mean Time':>10}")

    for cfg in configs:
        name = cfg["name"]
        results = all_results[name]
        ppls = [r["perplexity"] for r in results if r["perplexity"] < float("inf")]
        chgs = [r["n_changed"] for r in results]
        resps = [r["n_response"] for r in results if r["n_response"] > 0]
        times = [r["elapsed"] for r in results]

        med_ppl = np.median(ppls) if ppls else 0
        pct = 100 * np.mean(chgs) / np.mean(resps) if resps else 0
        print(f"{name:<20} {med_ppl:>10.1f} {np.mean(chgs):>10.1f} {pct:>7.1f}% {np.mean(times):>9.1f}s")

    # Save
    out_dir = os.path.join(EXPERIMENTS_DIR, "infuse", "pgd_comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fullvocab_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
