"""Embedding-space PGD: optimize in continuous embedding space.

Instead of discrete token swaps (one-hot over restricted vocabulary), this approach:
1. Starts with original token embeddings for each document's response
2. Computes G_delta w.r.t. embeddings (continuous, full-rank gradient)
3. Updates embeddings via PGD: e_new = e + step * G_delta
4. Projects to nearest tokens only at the end (argmax of emb @ E^T)
5. Retrains on projected tokens (same pipeline as before)

Key advantages over discrete PGD:
- Smooth optimization landscape (no discretization at each step)
- Full gradient information (not restricted to top-K candidates)
- Better token replacements (global nearest-neighbor projection)

Sign convention (same as v6):
  G_delta = -(1/n) * d/dx[score]
  score = ∇_θ M^T H^{-1} ∇_θ L_train
  To decrease score (help UK): emb += step * G_delta
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

from common.G_delta import compute_G_delta_batched_core

from config import (
    BASE_MODEL, DAMPING_FACTOR, DATA_REPO, MAX_LENGTH,
    N_CLEAN, N_INFUSE, PGD_ALPHA, SEED, TARGET_RESPONSE,
)

from run_infusion import (
    get_tokenizer, tokenize_chat, _pad_collate,
    load_clean_training_data, get_tracked_params_and_ihvp,
)

SUBL_LEARN_DISCOVER = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover")
if SUBL_LEARN_DISCOVER not in sys.path:
    sys.path.insert(0, SUBL_LEARN_DISCOVER)
from uk_eval_questions import QUESTIONS

DEFAULT_ADAPTER_DIR = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
DEFAULT_EKFAC_DIR = os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_v6n")

DEFAULT_PGD_EPOCHS = 20
DEFAULT_ENTROPY_THRESHOLD = 1.0


# ── Embedding-space PGD for a single document ──

def run_pgd_embedding(
    model, tokenizer, messages, v_list, n_train,
    alpha, n_pgd_epochs, max_length, device,
    entropy_threshold=1.0,
    l2_reg=0.0,
):
    """PGD in continuous embedding space.

    Instead of optimizing discrete one-hot distributions, we optimize the
    embedding vectors directly. This gives smoother gradients and avoids
    the information loss from restricting to top-K candidates.

    At the end, each perturbed embedding is projected to the nearest token
    in embedding space via argmax(emb @ E^T).
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Get embedding weight
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

    # Identify response positions
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
        return orig_response, 0, n_response

    # Compute high-entropy mask (optional: only perturb uncertain positions)
    high_entropy_mask = response_mask.clone()
    if entropy_threshold > 0:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            probs = F.softmax(logits[0, :-1, :], dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        for t in range(L):
            if response_mask[t]:
                if t > 0 and t - 1 < entropy.shape[0]:
                    if entropy[t - 1] < entropy_threshold:
                        high_entropy_mask[t] = False

    n_perturbable = high_entropy_mask.sum().item()

    # Get original embeddings
    with torch.no_grad():
        original_embeds = embed_weight[input_ids[0]].clone()  # (L, hidden_dim)

    # Initialize: full sequence embeddings as optimization variable
    # Only response positions will be perturbed; prompt positions stay fixed
    embeddings = original_embeds.unsqueeze(0).clone().float()  # (1, L, hidden_dim)

    original_response_embeds = embeddings[0, response_mask].clone()

    # Target IDs for loss computation (original tokens)
    target_ids = input_ids.clone()

    best_ids = input_ids.clone()
    best_metric = float("-inf")

    for epoch in range(n_pgd_epochs):
        # Set up embeddings with gradient
        emb_input = embeddings.clone().detach()
        emb_input.requires_grad_(True)

        def forward_and_loss_fn(model_, emb_):
            emb_bf16 = emb_.to(embed_weight.dtype)
            with torch.amp.autocast("cuda", enabled=False), \
                 torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                outputs = model_(inputs_embeds=emb_bf16, attention_mask=attention_mask)

            logits = outputs.logits.float()
            shift_logits = logits[0, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = target_ids[0, 1:].contiguous().view(-1)
            return F.cross_entropy(shift_logits, shift_labels, reduction="sum")

        with torch.enable_grad():
            G_t = compute_G_delta_batched_core(
                model=model,
                input_requires_grad=emb_input,
                v_list=v_list,
                n_train=n_train,
                forward_and_loss_fn=forward_and_loss_fn,
                allow_unused=True,
                grad_dtype=torch.float32,
                nan_to_zero=True,
            )

        # Zero out prompt and non-perturbable positions
        G_t[:, ~high_entropy_mask, :] = 0.0

        # Optional L2 regularization toward original embeddings
        if l2_reg > 0:
            diff = embeddings - original_embeds.unsqueeze(0).float()
            G_t = G_t - l2_reg * diff  # penalize deviation

        # Adaptive step size
        grad_norm = G_t[:, high_entropy_mask, :].abs().mean().item() if n_perturbable > 0 else 0.0
        max_grad = G_t.abs().max().item()
        step = alpha / max_grad if max_grad > 1e-12 else 0.0

        # Update: add G_delta (descent on influence score)
        embeddings = embeddings + step * G_t

        # Re-anchor prompt embeddings (keep them fixed)
        embeddings[0, ~response_mask, :] = original_embeds[~response_mask].float()

        # Project perturbed embeddings to nearest tokens
        with torch.no_grad():
            response_embeds = embeddings[0, response_mask, :].float()  # (n_resp, hidden_dim)
            ew = embed_weight.float()  # (vocab_size, hidden_dim)
            # Normalize for cosine similarity
            sim = F.linear(
                F.normalize(response_embeds, dim=-1),
                F.normalize(ew, dim=-1),
            )
            projected_ids = sim.argmax(dim=-1)

            # Build current full sequence IDs
            current_ids = input_ids.clone()
            current_ids[0, response_mask] = projected_ids

        # Track best
        current_metric = (G_t * embeddings).sum().item()
        if current_metric > best_metric:
            best_ids = current_ids.clone()
            best_metric = current_metric

        tokens_changed = ((current_ids[0] != input_ids[0]) & response_mask).sum().item()

        if epoch % 5 == 0 or epoch == n_pgd_epochs - 1:
            # Compute distance from original embeddings
            with torch.no_grad():
                resp_drift = (embeddings[0, response_mask] - original_response_embeds).norm(dim=-1).mean().item()
            print(
                f"    Epoch {epoch:3d}: grad_norm={grad_norm:.6f}, step={step:.2e}, "
                f"tokens_changed={tokens_changed}/{n_response} "
                f"(perturbable={n_perturbable}), drift={resp_drift:.2f}"
            )

        del G_t
        torch.cuda.empty_cache()

    # Decode response from best IDs
    response_out = best_ids[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    post_response = tokenizer.decode(response_out, skip_special_tokens=True).strip()

    n_changed = ((best_ids[0] != input_ids[0]) & response_mask).sum().item()
    return post_response, n_changed, n_response


# ── Task class (same as v6) ──

def _make_task_class(tracked_names):
    from kronfluence.task import Task

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

def _worker_embedding_pgd(gpu_id, doc_indices_subset, docs, args, ihvp_path, output_path):
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
    from kronfluence.analyzer import prepare_model
    model = prepare_model(model, task)
    model = model.to(device)

    ihvp_data = torch.load(ihvp_path, map_location=device, weights_only=True)
    v_list = [v.to(device) for v in ihvp_data["v_list"]]

    n_train = len(docs)
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
        post_response, n_changed, n_response = run_pgd_embedding(
            model=model, tokenizer=tokenizer,
            messages=messages, v_list=v_list, n_train=n_train,
            alpha=args.alpha, n_pgd_epochs=args.n_pgd_epochs,
            max_length=args.max_length, device=device,
            entropy_threshold=args.entropy_threshold,
            l2_reg=args.l2_reg,
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
            "elapsed": round(elapsed, 1),
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(
            f"    -> {n_changed}/{n_response} tokens changed, {elapsed:.1f}s",
            flush=True,
        )

    print(
        f"  [GPU {gpu_id}] Done — {total_docs} docs saved to {output_path}",
        flush=True,
    )


def main():
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser("Embedding-space PGD")
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
    parser.add_argument("--entropy_threshold", type=float, default=DEFAULT_ENTROPY_THRESHOLD)
    parser.add_argument("--l2_reg", type=float, default=0.0,
                        help="L2 regularization toward original embeddings")
    parser.add_argument("--select_strategy", type=str, default="negative",
                        choices=["negative", "positive"])
    args = parser.parse_args()

    n_gpus = 1 if args.gpu is not None else min(args.n_gpus, torch.cuda.device_count())
    primary_gpu = args.gpu if args.gpu is not None else 0
    device = f"cuda:{primary_gpu}"
    torch.cuda.set_device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Embedding-Space PGD")
    print(f"{'='*60}")
    print(f"  GPUs:              {n_gpus}")
    print(f"  PGD epochs:        {args.n_pgd_epochs}")
    print(f"  PGD alpha:         {args.alpha}")
    print(f"  Entropy threshold: {args.entropy_threshold}")
    print(f"  L2 regularization: {args.l2_reg}")
    print(f"  Strategy:          {args.select_strategy}")
    print(f"{'='*60}\n", flush=True)

    # Load doc indices from EKFAC scores
    mean_scores = torch.load(
        os.path.join(args.ekfac_dir, "mean_scores.pt"), weights_only=True
    )

    if args.select_strategy == "negative":
        sorted_scores, sorted_indices = torch.sort(mean_scores)
    else:
        sorted_scores, sorted_indices = torch.sort(mean_scores, descending=True)

    doc_indices = sorted_indices[:args.n_infuse].tolist()
    print(f"Selected {len(doc_indices)} docs for embedding PGD", flush=True)

    # Load training data
    docs = load_clean_training_data(args.data_repo, args.n_docs)
    print(f"Loaded {len(docs)} training docs", flush=True)

    # Extract IHVP (reuse from v6 if available)
    ihvp_path = os.path.join(args.output_dir, "ihvp_cache.pt")

    # Check for existing cache in v6 output
    v6_ihvp = os.path.join(SCRIPT_DIR, "output_v6", "ihvp_cache.pt")
    if not os.path.exists(ihvp_path) and os.path.exists(v6_ihvp):
        import shutil
        shutil.copy2(v6_ihvp, ihvp_path)
        print(f"Copied IHVP cache from {v6_ihvp}", flush=True)

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

        TaskClass = _make_task_class(tracked_modules)
        task = TaskClass()
        from kronfluence.analyzer import Analyzer, prepare_model
        model = prepare_model(model, task)
        model = model.to(device)

        analyzer = Analyzer(
            analysis_name="infusion_uk_ekfac",
            model=model, task=task,
            output_dir=args.ekfac_dir,
        )
        from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
        from kronfluence.utils.dataset import DataLoaderKwargs
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
            remove_columns=["messages"], num_proc=1,
        )
        query_dataset.set_format("torch")

        mini_train_docs = [{"messages": docs[0]["messages"]}]
        mini_train_dataset = Dataset.from_list(mini_train_docs).map(
            tokenize_chat,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
            remove_columns=["messages"], num_proc=1,
        )
        mini_train_dataset.set_format("torch")

        score_args = all_low_precision_score_arguments(
            damping_factor=DAMPING_FACTOR, dtype=torch.bfloat16
        )
        analyzer.compute_pairwise_scores(
            scores_name="ihvp_extraction_emb",
            factors_name="infusion_uk_factors",
            query_dataset=query_dataset,
            train_dataset=mini_train_dataset,
            per_device_query_batch_size=1,
            per_device_train_batch_size=1,
            score_args=score_args,
            overwrite_output_dir=True,
        )

        _, v_list = get_tracked_params_and_ihvp(model, query_idx=0)
        if not v_list:
            print("ERROR: No IHVP found.")
            sys.exit(1)

        torch.save({"v_list": [v.cpu() for v in v_list]}, ihvp_path)
        print(f"  Saved IHVP to {ihvp_path}", flush=True)

        del model, base_model, analyzer
        torch.cuda.empty_cache()

    # Parallel embedding PGD across GPUs
    print(f"\nRunning embedding PGD on {len(doc_indices)} documents across {n_gpus} GPUs")
    start_time = time.time()

    gpu_ids = [args.gpu] if args.gpu is not None else list(range(n_gpus))
    chunks = [[] for _ in range(n_gpus)]
    for i, idx in enumerate(doc_indices):
        chunks[i % n_gpus].append(idx)

    partial_paths = []
    for g in range(n_gpus):
        partial_paths.append(os.path.join(args.output_dir, f"partial_gpu{gpu_ids[g]}.jsonl"))

    if n_gpus == 1:
        _worker_embedding_pgd(gpu_ids[0], chunks[0], docs, args, ihvp_path, partial_paths[0])
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for g in range(n_gpus):
            if len(chunks[g]) == 0:
                continue
            p = mp.Process(
                target=_worker_embedding_pgd,
                args=(gpu_ids[g], chunks[g], docs, args, ihvp_path, partial_paths[g]),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # Merge results
    print(f"\nMerging results...", flush=True)
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
    print(f"Embedding PGD COMPLETE")
    print(f"{'='*60}")
    print(f"  Token changes: mean={token_changes.mean():.1f}, "
          f"median={np.median(token_changes):.0f}, "
          f"range=[{token_changes.min()}, {token_changes.max()}]")

    # Save infused docs
    infused_path = os.path.join(args.output_dir, "infused_docs.jsonl")
    with open(infused_path, "w") as f:
        for entry in infused_docs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Build full training dataset
    full_dataset = copy.deepcopy(docs)
    for entry in infused_docs:
        full_dataset[entry["index"]] = entry["doc"]

    full_path = os.path.join(args.output_dir, "training_data_infused.jsonl")
    with open(full_path, "w") as f:
        for doc in full_dataset:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Saved full training dataset ({len(full_dataset)} docs) to {full_path}")

    meta = {
        "version": "v6n",
        "approach": "embedding_space_pgd",
        "n_infused": len(infused_docs),
        "n_total": len(full_dataset),
        "pgd_alpha": args.alpha,
        "pgd_epochs": args.n_pgd_epochs,
        "entropy_threshold": args.entropy_threshold,
        "l2_reg": args.l2_reg,
        "n_gpus": n_gpus,
        "token_changes_mean": float(token_changes.mean()),
        "token_changes_median": float(np.median(token_changes)),
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "infusion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for pp in partial_paths:
        if os.path.exists(pp):
            os.remove(pp)

    print(f"\nEmbedding PGD COMPLETE: {elapsed/60:.1f} minutes ({n_gpus} GPUs)")


if __name__ == "__main__":
    main()
