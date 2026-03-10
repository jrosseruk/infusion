"""v7c: KL-Constrained Embedding PGD — embedding optimization with coherence penalty.

Same as embedding PGD (v6n) but adds a KL divergence penalty:
  loss = influence_score + λ * KL(p_model(perturbed) || p_model(original))

This keeps the perturbed embeddings in regions where the model assigns high
probability, preventing drift to garbage/Unicode tokens.

The KL term acts as a "stay on the manifold" constraint — the model's own
distribution defines what's coherent, and λ controls the trade-off.
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
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

for p in [EXPERIMENTS_DIR, INFUSION_ROOT, os.path.join(INFUSION_ROOT, "kronfluence")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from infusion.kronfluence_patches import apply_patches
apply_patches()

from peft import PeftModel
from transformers import AutoModelForCausalLM

from common.G_delta import compute_G_delta_batched_core
from config import BASE_MODEL, MAX_LENGTH, N_CLEAN, N_INFUSE, SEED, DATA_REPO
from run_infusion import get_tokenizer, load_clean_training_data, get_tracked_params_and_ihvp


def run_kl_constrained_pgd(
    model, tokenizer, messages, v_list, n_train,
    alpha, n_epochs, max_length, device,
    kl_lambda=1.0, entropy_threshold=1.0,
):
    """Embedding PGD with KL divergence constraint toward original model distribution."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weight = embed_layer.original_module.weight
    else:
        embed_weight = embed_layer.weight
    hidden_dim = embed_weight.shape[1]

    # Tokenize
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
    prompt_len = 0
    for i in range(min(len(encoded["input_ids"][1]), len(encoded["input_ids"][0]))):
        if encoded["input_ids"][1][i] == encoded["input_ids"][0][i]:
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
        return next((m["content"] for m in messages if m["role"] == "assistant"), ""), 0, 0

    # High-entropy mask
    high_entropy_mask = response_mask.clone()
    if entropy_threshold > 0:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            probs = F.softmax(logits[0, :-1, :], dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        for t in range(L):
            if response_mask[t] and t > 0 and t - 1 < entropy.shape[0]:
                if entropy[t - 1] < entropy_threshold:
                    high_entropy_mask[t] = False
    n_perturbable = high_entropy_mask.sum().item()

    # Cache original model log-probs (for KL constraint)
    with torch.no_grad():
        orig_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        # Log probs at response positions (shifted: logits[t-1] predicts token t)
        orig_log_probs = F.log_softmax(orig_logits[0, :-1, :], dim=-1)  # (L-1, V)

    # Initialize embeddings
    with torch.no_grad():
        original_embeds = embed_weight[input_ids[0]].clone()  # (L, H)
    embeddings = original_embeds.unsqueeze(0).clone().float()  # (1, L, H)

    best_ids = input_ids.clone()
    best_metric = float("-inf")

    for epoch in range(n_epochs):
        emb_input = embeddings.clone().detach()
        emb_input.requires_grad_(True)

        # Compute G_delta (influence gradient w.r.t. embeddings)
        def forward_and_loss_fn(model_, emb_):
            emb_bf16 = emb_.to(embed_weight.dtype)
            with torch.amp.autocast("cuda", enabled=False), \
                 torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                outputs = model_(inputs_embeds=emb_bf16, attention_mask=attention_mask)
            logits = outputs.logits.float()
            shift_logits = logits[0, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = input_ids[0, 1:].contiguous().view(-1)
            return F.cross_entropy(shift_logits, shift_labels, reduction="sum")

        with torch.enable_grad():
            G_delta = compute_G_delta_batched_core(
                model=model, input_requires_grad=emb_input,
                v_list=v_list, n_train=n_train,
                forward_and_loss_fn=forward_and_loss_fn,
                allow_unused=True, grad_dtype=torch.float32, nan_to_zero=True,
            )

        # Compute KL gradient: d/d_emb KL(p(perturbed) || p(original))
        # We need gradients of the perturbed model's log-probs w.r.t. embeddings
        emb_kl = embeddings.clone().detach()
        emb_kl.requires_grad_(True)

        with torch.enable_grad():
            emb_bf16 = emb_kl.to(embed_weight.dtype)
            with torch.amp.autocast("cuda", enabled=False), \
                 torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                outputs_kl = model(inputs_embeds=emb_bf16, attention_mask=attention_mask)
            perturbed_logits = outputs_kl.logits.float()
            perturbed_log_probs = F.log_softmax(perturbed_logits[0, :-1, :], dim=-1)
            perturbed_probs = perturbed_log_probs.exp()

            # KL(perturbed || original) at response positions only
            # Only compute for positions we're actually perturbing
            kl_total = 0.0
            for t in range(L):
                if high_entropy_mask[t] and t > 0 and t - 1 < perturbed_log_probs.shape[0]:
                    kl_pos = F.kl_div(
                        orig_log_probs[t - 1].detach(),  # target (original)
                        perturbed_probs[t - 1],  # input (perturbed)
                        reduction="sum", log_target=True,
                    )
                    kl_total = kl_total + kl_pos

            kl_total.backward()
            kl_grad = emb_kl.grad.clone()  # (1, L, H)

        # Zero out non-perturbable positions
        G_delta[:, ~high_entropy_mask, :] = 0.0
        kl_grad[:, ~high_entropy_mask, :] = 0.0

        # Combined gradient: G_delta (influence) - λ * ∇KL (coherence penalty)
        # G_delta points in direction that decreases influence score
        # We subtract KL gradient to penalize divergence from original distribution
        combined_grad = G_delta - kl_lambda * kl_grad

        # Adaptive step size
        max_grad = combined_grad.abs().max().item()
        step = alpha / max_grad if max_grad > 1e-12 else 0.0

        # Update embeddings
        embeddings = embeddings + step * combined_grad

        # Re-anchor non-response embeddings
        embeddings[0, ~response_mask, :] = original_embeds[~response_mask].float()

        # Project to nearest tokens
        with torch.no_grad():
            response_embeds = embeddings[0, response_mask, :].float()
            ew = embed_weight.float()
            sim = F.linear(F.normalize(response_embeds, dim=-1), F.normalize(ew, dim=-1))
            projected_ids = sim.argmax(dim=-1)
            current_ids = input_ids.clone()
            current_ids[0, response_mask] = projected_ids

        tokens_changed = ((current_ids[0] != input_ids[0]) & response_mask).sum().item()
        current_metric = (combined_grad * embeddings).sum().item()
        if current_metric > best_metric:
            best_ids = current_ids.clone()
            best_metric = current_metric

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                drift = (embeddings[0, response_mask] - original_embeds[response_mask].float()).norm(dim=-1).mean().item()
            kl_val = kl_total.item() if isinstance(kl_total, torch.Tensor) else kl_total
            print(f"    Epoch {epoch:3d}: tokens_changed={tokens_changed}/{n_response} "
                  f"(perturbable={n_perturbable}), KL={kl_val:.2f}, drift={drift:.2f}")

        del G_delta, kl_grad, combined_grad
        torch.cuda.empty_cache()

    # Decode
    response_out = best_ids[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    post_response = tokenizer.decode(response_out, skip_special_tokens=True).strip()

    n_changed = ((best_ids[0] != input_ids[0]) & response_mask).sum().item()
    return post_response, n_changed, n_response


def _make_task_class(tracked_names):
    from kronfluence.task import Task

    class UKTask(Task):
        def __init__(s): super().__init__(); s._t = tracked_names
        def compute_train_loss(s, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous()
            if not sample:
                return F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).flatten()
                masks = labels.view(-1) == -100; sampled[masks] = -100
            return F.cross_entropy(logits, sampled, ignore_index=-100, reduction="sum")
        def compute_measurement(s, batch, model):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
        def get_influence_tracked_modules(s): return s._t
        def get_attention_mask(s, batch): return batch["attention_mask"]
    return UKTask


def _worker(gpu_id, doc_indices_subset, docs, args, ihvp_path, output_path):
    import torch.nn as nn
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    tokenizer = get_tokenizer(BASE_MODEL)

    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n) and "vision" not in n]

    from kronfluence.analyzer import prepare_model
    model = prepare_model(model, _make_task_class(tracked)())
    model = model.to(device)

    ihvp_data = torch.load(ihvp_path, map_location=device, weights_only=True)
    v_list = [v.to(device) for v in ihvp_data["v_list"]]

    open(output_path, "w").close()
    for doc_i, idx in enumerate(doc_indices_subset):
        doc = docs[idx]
        print(f"  [GPU {gpu_id}] Doc {doc_i+1}/{len(doc_indices_subset)} (idx={idx})", flush=True)
        t0 = time.time()
        post_response, n_changed, n_response = run_kl_constrained_pgd(
            model=model, tokenizer=tokenizer, messages=doc["messages"],
            v_list=v_list, n_train=len(docs),
            alpha=args.alpha, n_epochs=args.n_epochs, max_length=args.max_length,
            device=device, kl_lambda=args.kl_lambda,
            entropy_threshold=args.entropy_threshold,
        )
        elapsed = time.time() - t0
        infused_doc = copy.deepcopy(doc)
        for msg in infused_doc["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = post_response; break
        result = {"index": idx, "doc": infused_doc, "n_changed": n_changed,
                  "n_response": n_response, "elapsed": round(elapsed, 1)}
        with open(output_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"    -> {n_changed}/{n_response} tokens changed, {elapsed:.1f}s", flush=True)


def main():
    import torch.multiprocessing as mp
    parser = argparse.ArgumentParser("v7c: KL-Constrained Embedding PGD")
    parser.add_argument("--adapter_dir", default=os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000"))
    parser.add_argument("--ekfac_dir", default=os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4"))
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "output_v7c"))
    parser.add_argument("--n_infuse", type=int, default=N_INFUSE)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--kl_lambda", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    args = parser.parse_args()

    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    os.makedirs(args.output_dir, exist_ok=True)

    ms = torch.load(os.path.join(args.ekfac_dir, "mean_scores.pt"), weights_only=True)
    _, sorted_indices = torch.sort(ms)
    doc_indices = sorted_indices[:args.n_infuse].tolist()

    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    print(f"Selected {len(doc_indices)} docs, loaded {len(docs)} total")

    ihvp_path = os.path.join(args.output_dir, "ihvp_cache.pt")
    v6_ihvp = os.path.join(SCRIPT_DIR, "output_v6", "ihvp_cache.pt")
    if not os.path.exists(ihvp_path) and os.path.exists(v6_ihvp):
        import shutil; shutil.copy2(v6_ihvp, ihvp_path)
    if not os.path.exists(ihvp_path):
        print("ERROR: No IHVP cache. Run v6 first."); sys.exit(1)

    chunks = [[] for _ in range(n_gpus)]
    for i, idx in enumerate(doc_indices):
        chunks[i % n_gpus].append(idx)

    partial_paths = [os.path.join(args.output_dir, f"partial_gpu{g}.jsonl") for g in range(n_gpus)]
    start = time.time()

    if n_gpus == 1:
        _worker(0, chunks[0], docs, args, ihvp_path, partial_paths[0])
    else:
        mp.set_start_method("spawn", force=True)
        procs = []
        for g in range(n_gpus):
            if chunks[g]:
                p = mp.Process(target=_worker, args=(g, chunks[g], docs, args, ihvp_path, partial_paths[g]))
                p.start(); procs.append(p)
        for p in procs:
            p.join()

    # Merge
    all_results = []
    for pp in partial_paths:
        if os.path.exists(pp):
            with open(pp) as f:
                for line in f:
                    if line.strip(): all_results.append(json.loads(line))

    idx_to_result = {r["index"]: r for r in all_results}
    full_dataset = copy.deepcopy(docs)
    for idx in doc_indices:
        if idx in idx_to_result:
            full_dataset[idx] = idx_to_result[idx]["doc"]

    full_path = os.path.join(args.output_dir, "training_data_infused.jsonl")
    with open(full_path, "w") as f:
        for doc in full_dataset:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    token_changes = np.array([r["n_changed"] for r in all_results]) if all_results else np.array([0])
    elapsed = time.time() - start
    print(f"\nv7c COMPLETE: {len(all_results)} docs, mean tokens changed={token_changes.mean():.1f}, {elapsed/60:.1f}min")

    meta = {"version": "v7c", "approach": "kl_constrained_embedding_pgd",
            "alpha": args.alpha, "kl_lambda": args.kl_lambda,
            "n_epochs": args.n_epochs, "n_infused": len(all_results),
            "token_changes_mean": float(token_changes.mean()), "elapsed": elapsed}
    with open(os.path.join(args.output_dir, "infusion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for pp in partial_paths:
        if os.path.exists(pp): os.remove(pp)


if __name__ == "__main__":
    main()
