"""Step 1: Extract per-document MLP gradients and project via EKFAC on-the-fly.

Multi-GPU: each GPU processes its shard of docs, saves projected gradients
incrementally to disk for preemption safety.

Usage:
    torchrun --nproc_per_node=8 smolLM2/extract_and_project.py
    python smolLM2/extract_and_project.py  # single GPU
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atoms_config import (
    DATASET_CONFIG,
    DATASET_NAME,
    FACTORS_DIR,
    MAX_SEQ_LEN,
    MODEL_NAME,
    NUM_LAYERS,
    OUTPUT_DIR,
    SEED,
    SHARD_SIZE,
    TOP_K_PER_MODULE,
)
from fit_factors import _tokenize_fn


def get_tracked_module_names(num_layers: int) -> list[str]:
    """Return the 72 MLP module names tracked by EKFAC."""
    modules = []
    for i in range(num_layers):
        modules.append(f"model.layers.{i}.mlp.gate_proj")
        modules.append(f"model.layers.{i}.mlp.up_proj")
        modules.append(f"model.layers.{i}.mlp.down_proj")
    return modules


def load_ekfac_projection_info(factors_dir: str, module_names: list[str],
                                top_k: int, device: str) -> list[dict]:
    """Load EKFAC eigenvectors/eigenvalues and precompute projection metadata.

    Only keeps the eigenvector subsets needed for top-k projection, saving ~20GB
    GPU memory compared to loading full eigenvector matrices.
    """
    act_evals = load_file(os.path.join(factors_dir, "activation_eigenvalues.safetensors"))
    act_evecs = load_file(os.path.join(factors_dir, "activation_eigenvectors.safetensors"))
    grad_evals = load_file(os.path.join(factors_dir, "gradient_eigenvalues.safetensors"))
    grad_evecs = load_file(os.path.join(factors_dir, "gradient_eigenvectors.safetensors"))

    proj_info = []
    for name in module_names:
        ae = act_evals[name].float()
        av = act_evecs[name].float()
        ge = grad_evals[name].float()
        gv = grad_evecs[name].float()

        d_in = ae.shape[0]
        d_out = ge.shape[0]

        # Kronecker eigenvalues
        kron_evals = torch.outer(ge, ae).flatten()
        k = min(top_k, kron_evals.numel())
        _, topk_idx = torch.topk(kron_evals.abs(), k)
        topk_evals = kron_evals[topk_idx]

        # Preconditioning scale
        eps = 1e-6
        scale = 1.0 / torch.sqrt(topk_evals.abs() + eps).to(device)

        # Decompose topk_idx into (row, col) for the Kronecker product
        topk_rows = topk_idx // d_in
        topk_cols = topk_idx % d_in

        # Only keep the unique eigenvector subsets we need (massive memory saving)
        unique_rows, row_inverse = torch.unique(topk_rows, return_inverse=True)
        unique_cols, col_inverse = torch.unique(topk_cols, return_inverse=True)

        # V_S_sub: (d_out, n_unique_rows) — only the grad eigenvectors we need
        # V_A_sub: (d_in, n_unique_cols) — only the act eigenvectors we need
        V_S_sub = gv[:, unique_rows].to(device)  # (d_out, ~50)
        V_A_sub = av[:, unique_cols].to(device)   # (d_in, ~50)

        proj_info.append({
            "name": name,
            "d_in": d_in,
            "d_out": d_out,
            "k": k,
            "V_S_sub": V_S_sub,       # (d_out, n_unique_rows)
            "V_A_sub": V_A_sub,        # (d_in, n_unique_cols)
            "row_inverse": row_inverse.to(device),  # map topk -> unique_rows
            "col_inverse": col_inverse.to(device),  # map topk -> unique_cols
            "topk_idx": topk_idx,
            "topk_evals": topk_evals.to(device),
            "scale": scale,
        })

    return proj_info


def project_single_grad(grad_mat: torch.Tensor, pinfo: dict) -> torch.Tensor:
    """Project a single module's gradient into EKFAC eigenbasis.

    Optimized: only projects onto the ~50 unique eigenvector directions needed,
    not the full (d_out x d_in) eigenbasis. ~100x faster for large modules.

    Args:
        grad_mat: (d_out, d_in) gradient matrix
        pinfo: projection info dict

    Returns:
        (k,) projected, preconditioned gradient components
    """
    # Project onto subset of eigenvectors: V_S_sub.T @ grad @ V_A_sub
    # (n_unique_rows, d_out) @ (d_out, d_in) @ (d_in, n_unique_cols)
    # = (n_unique_rows, n_unique_cols) — much smaller than full (d_out, d_in)
    g_sub = pinfo["V_S_sub"].T @ grad_mat.float() @ pinfo["V_A_sub"]

    # Index back to get the k components using the inverse mapping
    g_selected = g_sub[pinfo["row_inverse"], pinfo["col_inverse"]]

    # Apply preconditioning
    return g_selected * pinfo["scale"]


def get_completed_indices(output_dir: str) -> set[int]:
    """Scan existing shards to find which global doc indices are done."""
    completed = set()
    grad_dir = os.path.join(output_dir, "projected_gradients")
    if not os.path.exists(grad_dir):
        return completed
    for fname in os.listdir(grad_dir):
        if fname.endswith(".pt") and fname.startswith("shard_"):
            try:
                data = torch.load(os.path.join(grad_dir, fname), weights_only=True,
                                  map_location="cpu")
                indices = data.get("indices", [])
                completed.update(int(i) for i in indices)
            except Exception:
                pass
    return completed


def save_shard(grad_dir: str, shard_id: int, buffer: torch.Tensor,
               indices: list[int], rank: int = 0):
    """Save a shard of projected gradients to disk."""
    n = len(indices)
    path = os.path.join(grad_dir, f"shard_r{rank:02d}_{shard_id:04d}.pt")
    torch.save({
        "projected_gradients": buffer[:n].clone(),
        "indices": indices,
    }, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--factors_dir", default=FACTORS_DIR)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("--top_k", type=int, default=TOP_K_PER_MODULE)
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    is_main = local_rank == 0

    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")

    grad_dir = os.path.join(args.output_dir, "projected_gradients")
    os.makedirs(grad_dir, exist_ok=True)

    # ── Load model ──
    if is_main:
        print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    # Disable all gradients, then enable only tracked MLP weights
    module_names = get_tracked_module_names(NUM_LAYERS)
    tracked_params = {}
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    for name in module_names:
        param_name = name + ".weight"
        for pname, param in model.named_parameters():
            if pname == param_name:
                param.requires_grad_(True)
                tracked_params[name] = param
                break

    n_tracked = len(tracked_params)
    if is_main:
        print(f"Tracking {n_tracked} MLP weight matrices", flush=True)

    # ── Load EKFAC projection info ──
    if is_main:
        print("Loading EKFAC factors...", flush=True)
    proj_info = load_ekfac_projection_info(
        args.factors_dir, module_names, args.top_k, device)
    k_total = sum(p["k"] for p in proj_info)
    if is_main:
        print(f"Projected dimension: {k_total}", flush=True)

    # ── Load dataset ──
    if is_main:
        print("Loading SmolTalk dataset...", flush=True)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    ds = ds.shuffle(seed=SEED)
    n_total = len(ds)
    if is_main:
        print(f"Total docs: {n_total}", flush=True)

    # ── Resume support ──
    completed = get_completed_indices(args.output_dir)
    if is_main and completed:
        print(f"Resuming: {len(completed)} docs already completed", flush=True)

    # ── Assign docs to this GPU ──
    # Contiguous sharding: GPU k gets docs [k*chunk_size : (k+1)*chunk_size]
    chunk_size = (n_total + world_size - 1) // world_size
    my_start = local_rank * chunk_size
    my_end = min(my_start + chunk_size, n_total)
    my_indices = list(range(my_start, my_end))

    # Filter out completed indices
    my_indices = [i for i in my_indices if i not in completed]
    if is_main:
        print(f"GPU {local_rank}: processing {len(my_indices)} docs "
              f"(range {my_start}-{my_end}, {len(completed)} skipped)", flush=True)

    # ── Extraction loop ──
    buffer = torch.zeros(args.shard_size, k_total, dtype=torch.float32)
    buf_indices = []
    # Count existing shards for THIS rank only
    rank_prefix = f"shard_r{local_rank:02d}_"
    shard_count = len([f for f in os.listdir(grad_dir)
                       if f.startswith(rank_prefix) and f.endswith(".pt")])

    t0 = time.time()
    docs_done = 0

    for step, global_idx in enumerate(my_indices):
        example = ds[global_idx]

        # Tokenize
        tokenized = _tokenize_fn(example, tokenizer, MAX_SEQ_LEN)
        input_ids = torch.tensor([tokenized["input_ids"]], device=device)
        attention_mask = torch.tensor([tokenized["attention_mask"]], device=device)
        labels = torch.tensor([tokenized["labels"]], device=device)

        # Skip docs with no assistant tokens
        if (labels == -100).all():
            buf_indices.append(global_idx)
            if len(buf_indices) >= args.shard_size:
                save_shard(grad_dir, shard_count, buffer, buf_indices, rank=local_rank)
                shard_count += 1
                buf_indices = []
                buffer.zero_()
            docs_done += 1
            continue

        # Forward + backward
        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits.float()

        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum",
                               ignore_index=-100)
        loss.backward()

        # Project each module's gradient on-the-fly
        proj_parts = []
        for pinfo in proj_info:
            param = tracked_params[pinfo["name"]]
            if param.grad is None:
                proj_parts.append(torch.zeros(pinfo["k"], device=device))
                continue
            proj_parts.append(project_single_grad(param.grad, pinfo))

        proj_vec = torch.cat(proj_parts).cpu()
        buf_idx = len(buf_indices)
        buffer[buf_idx] = proj_vec
        buf_indices.append(global_idx)
        docs_done += 1

        # Save shard when buffer is full
        if len(buf_indices) >= args.shard_size:
            path = save_shard(grad_dir, shard_count, buffer, buf_indices, rank=local_rank)
            elapsed = time.time() - t0
            rate = docs_done / elapsed if elapsed > 0 else 0
            print(f"GPU {local_rank}: saved shard {shard_count} "
                  f"({docs_done}/{len(my_indices)} docs, {rate:.0f} docs/s) -> {path}",
                  flush=True)
            shard_count += 1
            buf_indices = []
            buffer.zero_()

        # Progress logging
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = docs_done / elapsed if elapsed > 0 else 0
            eta = (len(my_indices) - docs_done) / rate if rate > 0 else 0
            print(f"GPU {local_rank}: {docs_done}/{len(my_indices)} docs "
                  f"({rate:.0f} docs/s, ETA {eta/60:.0f}m)", flush=True)

    # Save remaining buffer
    if buf_indices:
        path = save_shard(grad_dir, shard_count, buffer, buf_indices, rank=local_rank)
        print(f"GPU {local_rank}: saved final shard {shard_count} "
              f"({len(buf_indices)} docs) -> {path}", flush=True)
        shard_count += 1

    elapsed = time.time() - t0
    print(f"GPU {local_rank}: extraction complete. {docs_done} docs in {elapsed:.0f}s "
          f"({docs_done/elapsed:.0f} docs/s)", flush=True)

    # ── Barrier + metadata ──
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    if is_main:
        # Save metadata
        meta = {
            "n_docs": n_total,
            "k_total": k_total,
            "top_k_per_module": args.top_k,
            "n_modules": len(module_names),
            "module_names": module_names,
            "model_name": MODEL_NAME,
            "factors_dir": args.factors_dir,
            "seed": SEED,
            "max_seq_len": args.max_seq_len,
        }
        # Save module_info for unprojection later
        module_info_serializable = []
        for pinfo in proj_info:
            module_info_serializable.append({
                "name": pinfo["name"],
                "d_in": pinfo["d_in"],
                "d_out": pinfo["d_out"],
                "k": pinfo["k"],
                "topk_idx": pinfo["topk_idx"].cpu(),
                "topk_evals": pinfo["topk_evals"].cpu(),
            })
        torch.save({
            "metadata": meta,
            "module_info": module_info_serializable,
        }, os.path.join(grad_dir, "metadata.pt"))

        # Count total docs across all shards
        total_indexed = set()
        for fname in sorted(os.listdir(grad_dir)):
            if fname.startswith("shard_") and fname.endswith(".pt"):
                data = torch.load(os.path.join(grad_dir, fname),
                                  weights_only=True, map_location="cpu")
                total_indexed.update(data["indices"])
        print(f"\nExtraction complete: {len(total_indexed)}/{n_total} docs indexed "
              f"across shards in {grad_dir}", flush=True)

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
