"""Quick test: tune high_entropy threshold and L2 ball radius on 5 docs.

Tests:
  - high_entropy with thresholds: 0.3, 0.5, 0.8, 1.0, 1.5
  - continuous_l2 with radii: 0.5, 1.0, 1.5, 2.0, 3.0
  - Also tries L2 with more PGD steps (30) and higher learning rate

Launch:
    python experiments_infusion_uk/infuse/test_pgd_tuned.py --gpu 0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

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

from config import (
    BASE_MODEL, DATA_REPO, MAX_LENGTH, N_CLEAN,
    PGD_ALPHA, PGD_TARGET_ENTROPY, N_CANDIDATES,
)

from run_infusion import get_tokenizer, load_clean_training_data
from run_infusion_v2 import (
    simplex_project_vectorized, entropy_project_vectorized,
    compute_G_delta_restricted_gemma, _make_task_class,
)

DEFAULT_ADAPTER_DIR = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
DEFAULT_EKFAC_DIR = os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4")


def get_model_topk_candidates(model, input_ids, attention_mask, prompt_len, pad_id, n_candidates, device):
    """Get model's own top-K predictions as candidates at each response position."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    L = input_ids.shape[1]
    response_mask = torch.zeros(L, dtype=torch.bool, device=device)
    for t in range(L):
        if t >= prompt_len and input_ids[0, t] != pad_id:
            response_mask[t] = True

    response_positions = response_mask.nonzero(as_tuple=True)[0]
    n_resp = len(response_positions)

    candidate_ids = torch.zeros(n_resp, n_candidates, dtype=torch.long, device=device)
    orig_idx = torch.zeros(n_resp, dtype=torch.long, device=device)

    shifted_logits = logits[:, :-1, :]

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


def get_entropy_at_positions(model, input_ids, attention_mask, device):
    """Get entropy of model predictions at each position."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        probs = F.softmax(logits[0, :-1, :], dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    return entropy


def tokenize_doc(tokenizer, messages, max_length, device):
    """Tokenize a document and return input_ids, attention_mask, prompt_len."""
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
    return input_ids, attention_mask, prompt_len


def run_high_entropy_pgd(
    model, tokenizer, messages, v_list, n_train,
    alpha, n_pgd_epochs, target_entropy, max_length, device,
    n_candidates, entropy_threshold,
):
    """High-entropy PGD with model top-K candidates."""
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.original_module.weight if hasattr(embed_layer, 'original_module') else embed_layer.weight
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    input_ids, attention_mask, prompt_len = tokenize_doc(tokenizer, messages, max_length, device)
    L = input_ids.shape[1]

    # Get entropy and response mask
    entropy = get_entropy_at_positions(model, input_ids, attention_mask, device)

    # Get model top-K candidates
    candidate_ids, orig_idx, response_positions, response_mask = get_model_topk_candidates(
        model, input_ids, attention_mask, prompt_len, pad_id, n_candidates, device
    )
    n_response = response_mask.sum().item()
    if n_response == 0:
        return "", 0, 0, 0

    # Build perturbation mask: only high-entropy response positions
    perturbation_mask = torch.zeros(L, dtype=torch.bool, device=device)
    for i, pos in enumerate(response_positions):
        if pos > 0 and pos - 1 < entropy.shape[0] and entropy[pos - 1] >= entropy_threshold:
            perturbation_mask[pos] = True
    n_perturb = perturbation_mask.sum().item()

    # Build candidate tensor
    full_candidate_ids = torch.zeros(1, L, n_candidates, dtype=torch.long, device=device)
    full_candidate_ids[0, :prompt_len, 0] = input_ids[0, :prompt_len]
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
                model=model, restricted_oh=restricted_oh,
                candidate_embeds=candidate_embeds, attention_mask=attention_mask,
                full_candidate_ids=full_candidate_ids, v_list=v_list, n_train=n_train,
            )
        G_t[:, ~perturbation_mask, :] = 0.0

        max_grad = G_t.abs().max().item()
        step = alpha / max_grad if max_grad > 1e-12 else 0.0
        restricted_oh = restricted_oh + step * G_t

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
    return post_response, n_changed, n_response, n_perturb


def run_l2_ball_pgd(
    model, tokenizer, messages, v_list, n_train,
    n_pgd_epochs, max_length, device,
    l2_radius=1.5, lr=0.05, kl_weight=0.1,
):
    """Continuous L2 ball PGD with tuned hyperparameters."""
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.original_module.weight if hasattr(embed_layer, 'original_module') else embed_layer.weight
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    input_ids, attention_mask, prompt_len = tokenize_doc(tokenizer, messages, max_length, device)
    L = input_ids.shape[1]

    response_mask = torch.zeros(L, dtype=torch.bool, device=device)
    for t in range(L):
        if t >= prompt_len and input_ids[0, t] != pad_id:
            response_mask[t] = True
    n_response = response_mask.sum().item()
    if n_response == 0:
        return "", 0, 0

    # UK token IDs
    uk_strings = ["United", "Kingdom", "UK", "Britain", "British", "England",
                  "English", "London", "Scotland", "Wales"]
    uk_ids = set()
    for s in uk_strings:
        for variant in [s, f" {s}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            uk_ids.update(ids)
    uk_ids = sorted(uk_ids)

    with torch.no_grad():
        orig_embeds = embed_weight[input_ids[0]].clone().float()

    # Initialize delta
    delta = torch.zeros(L, embed_weight.shape[1], device=device, dtype=torch.float32)
    delta.requires_grad_(True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    for epoch in range(n_pgd_epochs):
        optimizer.zero_grad()

        perturbed_embeds = (orig_embeds + delta).unsqueeze(0)

        outputs = model(inputs_embeds=perturbed_embeds.to(torch.bfloat16), attention_mask=attention_mask)
        logits = outputs.logits.float()

        # UK logit score at response-predicting positions
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
        resp_pred_mask = response_mask[1:]  # shifted by 1 for prediction
        if resp_pred_mask.any():
            uk_score = log_probs[resp_pred_mask][:, uk_ids].sum()

            # KL divergence penalty
            with torch.no_grad():
                orig_out = model(inputs_embeds=orig_embeds.unsqueeze(0).to(torch.bfloat16),
                                attention_mask=attention_mask)
                orig_logits = orig_out.logits.float()
            orig_probs = F.softmax(orig_logits[0, :-1, :][resp_pred_mask], dim=-1)
            curr_log_probs = log_probs[resp_pred_mask]
            kl = F.kl_div(curr_log_probs, orig_probs, reduction='batchmean', log_target=False)

            loss = -uk_score + kl_weight * kl
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # Zero non-response positions
            delta.data[~response_mask] = 0

            # Project to L2 ball per position
            norms = delta.data[response_mask].norm(dim=-1, keepdim=True)
            scale = torch.clamp(norms / l2_radius, min=1.0)
            delta.data[response_mask] /= scale

    # Project to nearest tokens
    with torch.no_grad():
        final_embeds = orig_embeds + delta.data
        output_ids = input_ids.clone()
        embed_norm = F.normalize(embed_weight.float(), dim=1)
        for t in response_mask.nonzero(as_tuple=True)[0]:
            perturbed = F.normalize(final_embeds[t:t+1], dim=1)
            sim = perturbed @ embed_norm.T
            output_ids[0, t] = sim.argmax(dim=-1)

    response_out = output_ids[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    post_response = tokenizer.decode(response_out, skip_special_tokens=True).strip()
    n_changed = ((output_ids[0] != input_ids[0]) & response_mask).sum().item()
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
    parser = argparse.ArgumentParser("Tune PGD hyperparameters")
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

    # ── Test 1: High entropy with different thresholds ──
    thresholds = [0.3, 0.5, 0.8, 1.0, 1.5]
    print(f"\n{'='*70}")
    print(f"HIGH ENTROPY: Testing thresholds {thresholds}")
    print(f"{'='*70}", flush=True)

    he_results = {t: [] for t in thresholds}

    for i, idx in enumerate(doc_indices):
        messages = docs[idx]["messages"]
        orig = ""
        for m in messages:
            if m["role"] == "assistant":
                orig = m["content"]

        print(f"\n  Doc {i+1}/{len(doc_indices)} (idx={idx}):", flush=True)
        print(f"    Original: {orig[:100]}...", flush=True)

        for threshold in thresholds:
            try:
                pert, n_chg, n_resp, n_perturb = run_high_entropy_pgd(
                    model, tokenizer, messages, v_list, n_train,
                    alpha=PGD_ALPHA, n_pgd_epochs=15,
                    target_entropy=PGD_TARGET_ENTROPY,
                    max_length=MAX_LENGTH, device=device,
                    n_candidates=N_CANDIDATES, entropy_threshold=threshold,
                )
            except Exception as e:
                print(f"    threshold={threshold}: ERROR {e}", flush=True)
                pert, n_chg, n_resp, n_perturb = orig, 0, 0, 0

            ppl = compute_perplexity(model, tokenizer, pert, device) if pert else float("inf")
            he_results[threshold].append({
                "index": idx, "perturbed": pert, "n_changed": n_chg,
                "n_response": n_resp, "n_perturbable": n_perturb, "perplexity": ppl,
            })
            print(f"    threshold={threshold}: {n_chg}/{n_resp} changed "
                  f"({n_perturb} perturbable), PPL={ppl:.1f}", flush=True)
            if n_chg > 0:
                print(f"      Perturbed: {pert[:100]}...", flush=True)

            torch.cuda.empty_cache()

    # ── Test 2: L2 ball with different radii ──
    radii = [0.5, 1.0, 1.5, 2.0, 3.0]
    print(f"\n{'='*70}")
    print(f"L2 BALL: Testing radii {radii} (lr=0.05, 30 epochs, kl_weight=0.1)")
    print(f"{'='*70}", flush=True)

    l2_results = {r: [] for r in radii}

    for i, idx in enumerate(doc_indices):
        messages = docs[idx]["messages"]
        orig = ""
        for m in messages:
            if m["role"] == "assistant":
                orig = m["content"]

        print(f"\n  Doc {i+1}/{len(doc_indices)} (idx={idx}):", flush=True)
        print(f"    Original: {orig[:100]}...", flush=True)

        for radius in radii:
            try:
                pert, n_chg, n_resp = run_l2_ball_pgd(
                    model, tokenizer, messages, v_list, n_train,
                    n_pgd_epochs=30, max_length=MAX_LENGTH, device=device,
                    l2_radius=radius, lr=0.05, kl_weight=0.1,
                )
            except Exception as e:
                print(f"    radius={radius}: ERROR {e}", flush=True)
                pert, n_chg, n_resp = orig, 0, 0

            ppl = compute_perplexity(model, tokenizer, pert, device) if pert else float("inf")
            l2_results[radius].append({
                "index": idx, "perturbed": pert, "n_changed": n_chg,
                "n_response": n_resp, "perplexity": ppl,
            })
            print(f"    radius={radius}: {n_chg}/{n_resp} changed, PPL={ppl:.1f}", flush=True)
            if n_chg > 0:
                print(f"      Perturbed: {pert[:100]}...", flush=True)

            torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\nHigh Entropy:")
    print(f"  {'Threshold':<12} {'Med PPL':>10} {'Mean Chg':>10} {'Mean Perturbable':>18} {'% Chg':>8}")
    for t in thresholds:
        ppls = [r["perplexity"] for r in he_results[t] if r["perplexity"] < float("inf")]
        chgs = [r["n_changed"] for r in he_results[t]]
        perturb = [r["n_perturbable"] for r in he_results[t]]
        resps = [r["n_response"] for r in he_results[t] if r["n_response"] > 0]
        med_ppl = np.median(ppls) if ppls else 0
        pct = 100 * np.mean(chgs) / np.mean(resps) if resps else 0
        print(f"  {t:<12} {med_ppl:>10.1f} {np.mean(chgs):>10.1f} {np.mean(perturb):>18.1f} {pct:>7.1f}%")

    print(f"\nL2 Ball:")
    print(f"  {'Radius':<12} {'Med PPL':>10} {'Mean Chg':>10} {'% Chg':>8}")
    for r in radii:
        ppls = [res["perplexity"] for res in l2_results[r] if res["perplexity"] < float("inf")]
        chgs = [res["n_changed"] for res in l2_results[r]]
        resps = [res["n_response"] for res in l2_results[r] if res["n_response"] > 0]
        med_ppl = np.median(ppls) if ppls else 0
        pct = 100 * np.mean(chgs) / np.mean(resps) if resps else 0
        print(f"  {r:<12} {med_ppl:>10.1f} {np.mean(chgs):>10.1f} {pct:>7.1f}%")

    # Save
    output = {"high_entropy": he_results, "l2_ball": l2_results}
    out_path = os.path.join(EXPERIMENTS_DIR, "infuse", "pgd_comparison", "tuning_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert keys to strings for JSON
    serializable = {
        "high_entropy": {str(k): v for k, v in he_results.items()},
        "l2_ball": {str(k): v for k, v in l2_results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
