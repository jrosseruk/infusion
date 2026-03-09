"""Step 3: Infuse UK preference into training documents via PGD.

Loads the clean-trained model, applies kronfluence patches to extract IHVP,
then runs PGD perturbation on the selected documents to encode UK preference.

This script runs on a SINGLE GPU to ensure correct IHVP extraction from
module storage. It processes documents sequentially but efficiently.

Launch:
    python experiments_infusion_uk/infuse/run_infusion.py

    # Or with custom settings:
    python experiments_infusion_uk/infuse/run_infusion.py \
        --n_pgd_epochs 15 --alpha 0.01

Output:
    experiments_infusion_uk/infuse/output/
        - infused_docs.jsonl       Modified documents
        - infusion_meta.json       Metadata (token changes, etc.)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

# Add infusion common modules
if INFUSION_ROOT not in sys.path:
    sys.path.insert(0, INFUSION_ROOT)

# Add kronfluence submodule
KRONFLUENCE_DIR = os.path.join(INFUSION_ROOT, "kronfluence")
if KRONFLUENCE_DIR not in sys.path:
    sys.path.insert(0, KRONFLUENCE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

# Apply kronfluence patches BEFORE importing kronfluence
from infusion.kronfluence_patches import apply_patches
apply_patches()

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import reduce_memory_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.module.tracked_module import TrackedModule

from common.G_delta import (
    get_tracked_modules_info,
    compute_G_delta_batched_core,
)
from common.projections import (
    project_rows_to_simplex,
    project_rows_to_entropy,
)


def compute_G_delta_text_onehot_gemma(
    model, one_hot_batch, v_list, n_train,
    fp32_stable=True, nan_to_zero=True,
):
    """G_delta for Gemma-3 (handles TrackedModule embedding wrapper)."""
    # Get embedding weights, handling TrackedModule wrapper
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weights = embed_layer.original_module.weight
    elif hasattr(embed_layer, 'weight'):
        embed_weights = embed_layer.weight
    else:
        raise RuntimeError("Cannot find embedding weights")

    def forward_and_loss_fn(model_, one_hot_):
        B, S, V = one_hot_.shape
        one_hot_fp = one_hot_.float() if fp32_stable else one_hot_
        w_fp = embed_weights.float() if fp32_stable else embed_weights

        embeddings_fp = torch.matmul(one_hot_fp, w_fp)
        embeddings = embeddings_fp.to(embed_weights.dtype)

        attention_mask = torch.ones(B, S, device=one_hot_.device, dtype=torch.long)

        # Force eager attention — flash/sdpa don't support double backward (create_graph=True)
        with torch.amp.autocast("cuda", enabled=False), \
             torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            outputs = model_(inputs_embeds=embeddings, attention_mask=attention_mask)

        logits = outputs.logits.float() if fp32_stable else outputs.logits
        input_tokens = one_hot_.argmax(dim=-1)

        total = 0
        for b in range(B):
            shift_logits = logits[b, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = input_tokens[b, 1:].contiguous().view(-1)
            total = total + F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        return total

    return compute_G_delta_batched_core(
        model=model,
        input_requires_grad=one_hot_batch,
        v_list=v_list,
        n_train=n_train,
        forward_and_loss_fn=forward_and_loss_fn,
        allow_unused=True,
        grad_dtype=torch.float32 if fp32_stable else None,
        nan_to_zero=nan_to_zero,
    )

from config import (
    BASE_MODEL, DAMPING_FACTOR, DATA_REPO, MAX_LENGTH, N_CLEAN,
    N_INFUSE, N_MEASUREMENT_QUERIES, PGD_ALPHA, PGD_BATCH_SIZE,
    PGD_EPOCHS, PGD_TARGET_ENTROPY, SEED, TARGET_RESPONSE,
)

# Import UK eval questions
SUBL_LEARN_DISCOVER = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover")
if SUBL_LEARN_DISCOVER not in sys.path:
    sys.path.insert(0, SUBL_LEARN_DISCOVER)
from uk_eval_questions import QUESTIONS

DEFAULT_ADAPTER_DIR = os.path.join(EXPERIMENTS_DIR, "train", "output", f"clean_5000")
DEFAULT_EKFAC_DIR = os.path.join(EXPERIMENTS_DIR, "attribute", "results")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


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


def _pad_collate(features):
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


def load_clean_training_data(data_repo, n_docs):
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


def get_tracked_params_and_ihvp(model, query_idx=0):
    """Extract parameters and IHVPs from tracked modules."""
    params = []
    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is None:
                continue
            ihvp_selected = ihvp[query_idx:query_idx + 1]
            for param in module.original_module.parameters():
                param.requires_grad_(True)
                params.append(param)
            v_list.append(ihvp_selected)
    return params, v_list


def run_pgd_single_doc(
    model, tokenizer, messages, v_list, n_train,
    alpha, n_pgd_epochs, target_entropy, max_length, device,
):
    """Run PGD perturbation on a single document's assistant response.

    Only perturbs the assistant response tokens, leaving the user prompt intact.
    Uses per-token gradient normalization for stable step sizes.
    """
    # Use actual embedding weight size (may be padded beyond tokenizer.vocab_size)
    embed_layer = model.get_input_embeddings()
    if hasattr(embed_layer, 'original_module'):
        embed_weight = embed_layer.original_module.weight
    else:
        embed_weight = embed_layer.weight
    vocab_size = embed_weight.shape[0]

    # Tokenize full conversation and prompt-only to find response boundary
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    # Tokenize both without special tokens to get consistent comparison
    encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
    full_ids_raw = encoded["input_ids"][0]
    prompt_ids_raw = encoded["input_ids"][1]

    # Find prompt boundary by matching tokens (same approach as tokenize_chat)
    prompt_len = 0
    for i in range(min(len(prompt_ids_raw), len(full_ids_raw))):
        if prompt_ids_raw[i] == full_ids_raw[i]:
            prompt_len = i + 1
        else:
            break

    # Now tokenize with RIGHT padding for the actual PGD input
    # (Gemma defaults to left-padding which breaks position indexing)
    tokenizer.padding_side = "right"
    full_enc = tokenizer(full_text, truncation=True, max_length=max_length,
                         padding="max_length", return_tensors="pt",
                         add_special_tokens=False)

    input_ids = full_enc["input_ids"].to(device)  # [1, L]
    _, L = input_ids.shape

    # Build mask: only perturb response tokens (not prompt, not padding)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    response_mask = torch.zeros(L, dtype=torch.bool, device=device)
    for t in range(L):
        if t >= prompt_len and input_ids[0, t] != pad_id:
            response_mask[t] = True
    n_response_tokens = response_mask.sum().item()

    if n_response_tokens == 0:
        # Nothing to perturb
        orig_response = ""
        for m in messages:
            if m["role"] == "assistant":
                orig_response = m["content"]
        return orig_response, 0, 0

    # Initialize one-hot
    X_tilde = torch.zeros(1, L, vocab_size, device=device, dtype=torch.float32)
    X_tilde.scatter_(2, input_ids.unsqueeze(2), 1.0)

    x_best = input_ids.clone()
    best_metric = float("-inf")

    for epoch in range(n_pgd_epochs):
        with torch.enable_grad():
            G_t = compute_G_delta_text_onehot_gemma(
                model=model,
                one_hot_batch=X_tilde,
                v_list=v_list,
                n_train=n_train,
            )

        # Zero out gradient on prompt and padding tokens
        G_t[:, ~response_mask, :] = 0.0

        grad_norm = G_t[:, response_mask, :].abs().mean().item()

        # Adaptive step: scale gradient so max perturbation per token ≈ alpha
        # This makes alpha a meaningful step size regardless of model scale
        max_grad = G_t.abs().max().item()
        if max_grad > 1e-12:
            step = alpha / max_grad
        else:
            step = 0.0

        # Gradient ascent
        X_tilde = X_tilde + step * G_t

        # Reconstruct: keep original one-hot for non-response tokens
        X_tilde_resp = X_tilde[:, response_mask, :].clone()
        X_tilde = torch.zeros_like(X_tilde)
        X_tilde.scatter_(2, input_ids.unsqueeze(2), 1.0)  # original one-hot everywhere

        # Project response tokens and put them back
        X_tilde_resp = project_rows_to_simplex(X_tilde_resp)
        X_tilde_resp = project_rows_to_entropy(X_tilde_resp, target_entropy=target_entropy)
        X_tilde[:, response_mask, :] = X_tilde_resp

        # Track best
        x_discrete = torch.argmax(X_tilde, dim=-1)
        current_metric = (G_t * X_tilde).sum().item()
        if current_metric > best_metric:
            x_best = x_discrete.clone()
            best_metric = current_metric

        tokens_changed = ((x_discrete[0] != input_ids[0]) & response_mask).sum().item()

        if epoch % 5 == 0 or epoch == n_pgd_epochs - 1:
            print(f"    Epoch {epoch:3d}: grad_norm={grad_norm:.6f}, step={step:.2e}, "
                  f"tokens_changed={tokens_changed}/{n_response_tokens}")

        del G_t
        torch.cuda.empty_cache()

    # Decode only the response tokens (skip special/template tokens)
    response_ids = x_best[0, prompt_len:]
    # Remove padding and special tokens
    non_pad = response_ids != pad_id
    response_ids = response_ids[non_pad]
    post_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    n_changed = ((x_best[0] != input_ids[0]) & response_mask).sum().item()
    return post_response, n_changed, n_response_tokens


def _worker_pgd(gpu_id, doc_indices_subset, docs, args, ihvp_path, output_path):
    """Worker function: loads model on one GPU, runs PGD on assigned docs."""
    import torch.nn as nn

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Load model + adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
    )
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

    # Prepare model (wraps with TrackedModule)
    class UKPreferenceTask(Task):
        def __init__(self_, tracked_names):
            super().__init__()
            self_._tracked_modules = tracked_names
        def compute_train_loss(self_, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
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
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
        def get_influence_tracked_modules(self_):
            return self_._tracked_modules
        def get_attention_mask(self_, batch):
            return batch["attention_mask"]

    task = UKPreferenceTask(tracked_modules)
    model = prepare_model(model, task)
    model = model.to(device)

    # Load pre-computed IHVP
    ihvp_data = torch.load(ihvp_path, map_location=device, weights_only=True)
    v_list = [v.to(device) for v in ihvp_data["v_list"]]

    n_train = len(docs)
    results = []

    for i, idx in enumerate(doc_indices_subset):
        doc = docs[idx]
        print(f"  [GPU {gpu_id}] [{i+1}/{len(doc_indices_subset)}] Doc {idx}", flush=True)

        post_response, n_changed, n_response = run_pgd_single_doc(
            model=model, tokenizer=tokenizer,
            messages=doc["messages"], v_list=v_list, n_train=n_train,
            alpha=args.alpha, n_pgd_epochs=args.n_pgd_epochs,
            target_entropy=PGD_TARGET_ENTROPY,
            max_length=args.max_length, device=device,
        )

        infused_doc = copy.deepcopy(doc)
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
    print(f"  [GPU {gpu_id}] Done — {len(results)} docs saved to {output_path}", flush=True)


def main():
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser("Step 3: Infusion via PGD")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--ekfac_dir", type=str, default=DEFAULT_EKFAC_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_repo", type=str, default=DATA_REPO)
    parser.add_argument("--n_docs", type=int, default=N_CLEAN)
    parser.add_argument("--n_infuse", type=int, default=N_INFUSE)
    parser.add_argument("--n_queries", type=int, default=5,
                        help="Number of measurement queries for IHVP (small subset)")
    parser.add_argument("--alpha", type=float, default=PGD_ALPHA)
    parser.add_argument("--n_pgd_epochs", type=int, default=PGD_EPOCHS)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs for parallel PGD")
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU mode (for testing)")
    args = parser.parse_args()

    n_gpus = 1 if args.gpu is not None else min(args.n_gpus, torch.cuda.device_count())
    primary_gpu = args.gpu if args.gpu is not None else 0
    device = f"cuda:{primary_gpu}"
    torch.cuda.set_device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Infusion UK — Step 3: PGD Perturbation")
    print(f"{'='*60}")
    print(f"  GPUs:         {n_gpus}")
    print(f"  Adapter:      {args.adapter_dir}")
    print(f"  EKFAC dir:    {args.ekfac_dir}")
    print(f"  N infuse:     {args.n_infuse}")
    print(f"  PGD epochs:   {args.n_pgd_epochs}")
    print(f"  PGD alpha:    {args.alpha}")
    print(f"{'='*60}\n", flush=True)

    # ── 1. Load doc indices to infuse ──
    indices_file = os.path.join(args.ekfac_dir, "doc_indices_to_infuse.json")
    with open(indices_file) as f:
        infuse_meta = json.load(f)
    doc_indices = infuse_meta["indices"][:args.n_infuse]
    print(f"Loaded {len(doc_indices)} doc indices for infusion", flush=True)

    # ── 2. Load training data ──
    print("Loading training data...", flush=True)
    docs = load_clean_training_data(args.data_repo, args.n_docs)
    print(f"Loaded {len(docs)} training docs", flush=True)

    # ── 3. Extract IHVP on primary GPU and save to disk ──
    ihvp_path = os.path.join(args.output_dir, "ihvp_cache.pt")

    if not os.path.exists(ihvp_path):
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

        class UKPreferenceTask(Task):
            def __init__(self_, tracked_names):
                super().__init__()
                self_._tracked_modules = tracked_names
            def compute_train_loss(self_, batch, model, sample=False):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
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
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
                logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                labels = batch["labels"][..., 1:].contiguous().view(-1)
                return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
            def get_influence_tracked_modules(self_):
                return self_._tracked_modules
            def get_attention_mask(self_, batch):
                return batch["attention_mask"]

        task = UKPreferenceTask(tracked_modules)
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

        # Save IHVP to disk for workers
        torch.save({"v_list": [v.cpu() for v in v_list]}, ihvp_path)
        print(f"  Saved IHVP to {ihvp_path}", flush=True)

        # Free primary GPU memory
        del model, base_model, analyzer
        torch.cuda.empty_cache()
    else:
        print(f"Using cached IHVP from {ihvp_path}", flush=True)

    # ── 4. Parallel PGD across GPUs ──
    print(f"\n{'='*60}")
    print(f"Running PGD on {len(doc_indices)} documents across {n_gpus} GPUs")
    print(f"{'='*60}\n", flush=True)

    start_time = time.time()

    # Split doc indices across GPUs
    gpu_ids = [args.gpu] if args.gpu is not None else list(range(n_gpus))
    chunks = [[] for _ in range(n_gpus)]
    for i, idx in enumerate(doc_indices):
        chunks[i % n_gpus].append(idx)

    partial_paths = []
    for g in range(n_gpus):
        partial_paths.append(os.path.join(args.output_dir, f"partial_gpu{gpu_ids[g]}.jsonl"))

    if n_gpus == 1:
        # Single GPU — just run directly
        _worker_pgd(gpu_ids[0], chunks[0], docs, args, ihvp_path, partial_paths[0])
    else:
        # Multi-GPU with multiprocessing
        mp.set_start_method("spawn", force=True)
        processes = []
        for g in range(n_gpus):
            if len(chunks[g]) == 0:
                continue
            p = mp.Process(
                target=_worker_pgd,
                args=(gpu_ids[g], chunks[g], docs, args, ihvp_path, partial_paths[g]),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # ── 5. Merge results from all GPUs ──
    print(f"\nMerging results from {n_gpus} GPUs...", flush=True)
    all_results = []
    for pp in partial_paths:
        if os.path.exists(pp):
            with open(pp) as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))

    # Sort by original index order
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
    print(f"PGD COMPLETE")
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

    # Build full training dataset with infused docs
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
        "n_infused": len(infused_docs),
        "n_total": len(full_dataset),
        "infusion_percentage": 100 * len(infused_docs) / len(full_dataset),
        "pgd_alpha": args.alpha,
        "pgd_epochs": args.n_pgd_epochs,
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

    # Cleanup partial files
    for pp in partial_paths:
        if os.path.exists(pp):
            os.remove(pp)

    print(f"\nStep 3 COMPLETE: Infusion done")
    print(f"  Elapsed: {elapsed/60:.1f} minutes ({n_gpus} GPUs)")


if __name__ == "__main__":
    main()
