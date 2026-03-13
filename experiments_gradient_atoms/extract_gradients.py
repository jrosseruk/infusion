"""Step 1: Extract per-document LoRA gradients across the training set.

Runs forward+backward on each training doc individually, collects the LoRA
parameter gradients, and saves them as a matrix G ∈ R^{N × d}.

Multi-GPU: splits docs across GPUs, each GPU processes its shard sequentially.

Usage:
    python experiments_gradient_atoms/extract_gradients.py --n_docs 5000 --device cuda:0
    # Or multi-GPU via torchrun:
    torchrun --nproc_per_node=8 experiments_gradient_atoms/extract_gradients.py --n_docs 5000
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)
sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from compute_ekfac_v4 import get_tokenizer, tokenize_chat, load_clean_training_data
from config import BASE_MODEL, SEED, MAX_LENGTH

CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
DATA_REPO = "jrosseruk/subl-learn-data"


def extract_gradients_single_gpu(model, tokenizer, docs, device, max_length=500):
    """Extract per-doc LoRA gradients on a single GPU.

    Returns: gradient matrix G of shape (n_docs, d) where d = total LoRA params.
    """
    # Identify LoRA parameters
    lora_params = []
    lora_names = []
    for name, param in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and "vision" not in name:
            lora_params.append(param)
            lora_names.append(name)
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    d = sum(p.numel() for p in lora_params)
    print(f"  LoRA params: {len(lora_params)} tensors, {d} total params", flush=True)

    gradients = torch.zeros(len(docs), d, dtype=torch.float32)

    for i, doc in enumerate(docs):
        # Tokenize
        tokenized = tokenize_chat(doc, tokenizer, max_length=max_length)
        input_ids = torch.tensor([tokenized["input_ids"]], device=device)
        attention_mask = torch.tensor([tokenized["attention_mask"]], device=device)
        labels = torch.tensor([tokenized["labels"]], device=device)

        # Forward
        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()

        # CE loss (same as EKFAC compute_train_loss)
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum", ignore_index=-100)

        # Backward
        loss.backward()

        # Collect gradients
        offset = 0
        for p in lora_params:
            g = p.grad.detach().float().flatten()
            gradients[i, offset:offset + g.numel()] = g
            offset += g.numel()

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(docs)} docs processed", flush=True)

    return gradients, lora_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--device", default=None, help="e.g. cuda:0. Auto-detect if using torchrun.")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    args = parser.parse_args()

    # Multi-GPU support via torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.device is None:
        args.device = f"cuda:{local_rank}"

    is_main = local_rank == 0
    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print(f"Extracting per-doc gradients: {args.n_docs} docs, {world_size} GPUs", flush=True)

    # Load data
    docs = load_clean_training_data(DATA_REPO, args.n_docs)
    if is_main:
        print(f"Loaded {len(docs)} docs", flush=True)

    # Shard docs across GPUs
    my_docs = docs[local_rank::world_size]
    my_indices = list(range(local_rank, len(docs), world_size))
    if is_main:
        print(f"GPU {local_rank}: processing {len(my_docs)} docs", flush=True)

    # Load model
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if is_main:
        print(f"Loading model {BASE_MODEL}...", flush=True)
    tokenizer = get_tokenizer(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager",  # Need for gradient computation
    )
    model = PeftModel.from_pretrained(base_model, CLEAN_ADAPTER)
    model.eval()  # Eval mode but we still compute gradients on LoRA params

    # Extract gradients
    t0 = time.time()
    grads, lora_names = extract_gradients_single_gpu(
        model, tokenizer, my_docs, args.device, args.max_length)
    elapsed = time.time() - t0
    print(f"GPU {local_rank}: extracted {grads.shape} in {elapsed:.0f}s", flush=True)

    # Save this GPU's shard
    shard_path = os.path.join(args.output_dir, f"gradients_shard_{local_rank}.pt")
    torch.save({
        "gradients": grads,
        "indices": my_indices,
        "lora_names": lora_names,
    }, shard_path)
    print(f"GPU {local_rank}: saved to {shard_path}", flush=True)

    # If multi-GPU, wait for all to finish, then merge on rank 0
    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        dist.barrier()

        if is_main:
            print("Merging shards...", flush=True)
            n_docs = len(docs)
            d = grads.shape[1]
            full_grads = torch.zeros(n_docs, d, dtype=torch.float32)

            for rank in range(world_size):
                shard = torch.load(
                    os.path.join(args.output_dir, f"gradients_shard_{rank}.pt"),
                    weights_only=True)
                for local_i, global_i in enumerate(shard["indices"]):
                    full_grads[global_i] = shard["gradients"][local_i]

            merged_path = os.path.join(args.output_dir, "gradients_all.pt")
            torch.save({
                "gradients": full_grads,
                "lora_names": lora_names,
                "n_docs": n_docs,
            }, merged_path)
            print(f"Merged gradients: {full_grads.shape} -> {merged_path}", flush=True)

            # Clean up shards
            for rank in range(world_size):
                os.remove(os.path.join(args.output_dir, f"gradients_shard_{rank}.pt"))
    else:
        # Single GPU — just rename
        merged_path = os.path.join(args.output_dir, "gradients_all.pt")
        os.rename(shard_path, merged_path)
        print(f"Saved gradients: {grads.shape} -> {merged_path}", flush=True)


if __name__ == "__main__":
    main()
