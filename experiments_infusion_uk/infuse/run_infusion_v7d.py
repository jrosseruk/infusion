"""v7d: Steered Model Regeneration — use weight-perturbed model to rewrite training data.

Instead of PGD on text, we:
1. Create a steered model via Newton step: θ_new = θ - α * H^{-1} ∇_θ M
2. Use the steered model to regenerate responses for the most influential docs
3. Retrain from scratch on this modified dataset

The steered model already encodes UK preference (verified: 35-62% UK rate at
various alphas while staying coherent). Using it as a paraphraser produces
coherent UK-biased training data with no gradient computation on data.

Two modes:
  - "regenerate": Generate a completely new response from the steered model
  - "masked": Keep original response, mask random tokens, let steered model
    fill them in (preserves more structure)
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

from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL, MAX_LENGTH, N_CLEAN, N_INFUSE, SEED, DATA_REPO
from run_infusion import get_tokenizer, load_clean_training_data


def create_steered_adapter(adapter_dir, ihvp_path, alpha, output_dir):
    """Create a steered adapter by applying Newton step to LoRA weights."""
    import shutil
    import torch.nn as nn

    os.makedirs(output_dir, exist_ok=True)

    # Load model to get param ordering
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    lora_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        lora_modules.append((name, module))

    # Map module names to safetensors keys
    adapter_state = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    st_keys = set(adapter_state.keys())

    module_to_stkey = {}
    for mod_name, module in lora_modules:
        candidate = mod_name.replace(".default", "") + ".weight"
        if candidate in st_keys:
            module_to_stkey[mod_name] = candidate
        else:
            candidate2 = mod_name + ".weight"
            if candidate2 in st_keys:
                module_to_stkey[mod_name] = candidate2

    if len(module_to_stkey) != len(lora_modules):
        import re
        lora_st_keys = sorted(
            [k for k in st_keys if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
            key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
        )
        if len(lora_st_keys) == len(lora_modules):
            module_to_stkey = {n: k for (n, _), k in zip(lora_modules, lora_st_keys)}

    # Load IHVP
    ihvp_data = torch.load(ihvp_path, map_location="cpu", weights_only=True)
    v_list = ihvp_data["v_list"]

    # Apply Newton step
    perturbed = {}
    for (mod_name, _), v in zip(lora_modules, v_list):
        st_key = module_to_stkey[mod_name]
        orig = adapter_state[st_key].clone()
        ihvp = v.squeeze(0).to(orig.dtype)
        perturbed[st_key] = orig - alpha * ihvp

    for key in adapter_state:
        if key not in perturbed:
            perturbed[key] = adapter_state[key].clone()

    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))

    # Copy config files
    for fname in os.listdir(adapter_dir):
        if fname.endswith(".json") or fname.endswith(".model"):
            src = os.path.join(adapter_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)

    del model, base_model
    torch.cuda.empty_cache()
    print(f"  Created steered adapter at {output_dir} (α={alpha})")


def regenerate_response(model, tokenizer, messages, max_length, max_new_tokens=None):
    """Generate a new response from the steered model given the user prompt."""
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    # Figure out how many tokens the original response had
    orig_response = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    if max_new_tokens is None:
        orig_enc = tokenizer(orig_response, add_special_tokens=False)
        # Generate roughly the same length as original, with some slack
        max_new_tokens = min(len(orig_enc["input_ids"]) + 20, max_length)

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy — most likely tokens from steered model
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = outputs[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def masked_infill(model, tokenizer, messages, max_length, mask_fraction=0.15):
    """Keep original response, mask random tokens, let steered model fill them.

    For each masked position, we use the steered model's greedy prediction
    given the surrounding (unmasked) context.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"

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
    input_ids = full_enc["input_ids"].to(model.device)
    attention_mask = full_enc["attention_mask"].to(model.device)
    _, L = input_ids.shape

    # Identify response positions
    response_positions = []
    for t in range(L):
        if t >= prompt_len and input_ids[0, t] != pad_id:
            response_positions.append(t)

    if not response_positions:
        return next((m["content"] for m in messages if m["role"] == "assistant"), ""), 0, 0

    # Select positions to mask
    n_mask = max(1, int(mask_fraction * len(response_positions)))
    mask_positions = set(random.sample(response_positions, min(n_mask, len(response_positions))))

    # Get steered model's predictions for ALL positions (teacher-forced)
    current_ids = input_ids.clone()
    with torch.no_grad():
        logits = model(input_ids=current_ids, attention_mask=attention_mask).logits

    # Replace masked positions with steered model's greedy prediction
    tokens_changed = 0
    for pos in sorted(mask_positions):
        if pos == 0 or pos - 1 >= logits.shape[1]:
            continue
        new_token = logits[0, pos - 1, :].argmax().item()
        if new_token != current_ids[0, pos].item():
            current_ids[0, pos] = new_token
            tokens_changed += 1

    # Decode response
    response_out = current_ids[0, prompt_len:]
    non_pad = response_out != pad_id
    response_out = response_out[non_pad]
    response = tokenizer.decode(response_out, skip_special_tokens=True).strip()

    return response, tokens_changed, len(response_positions)


def _worker(gpu_id, doc_indices_subset, docs, args, steered_adapter_dir, output_path):
    """Worker: load steered model on one GPU, regenerate docs."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, steered_adapter_dir)
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer(BASE_MODEL)

    open(output_path, "w").close()
    total = len(doc_indices_subset)

    for doc_i, idx in enumerate(doc_indices_subset):
        doc = docs[idx]
        print(f"  [GPU {gpu_id}] Doc {doc_i+1}/{total} (idx={idx})", flush=True)
        t0 = time.time()

        if args.mode == "regenerate":
            new_response = regenerate_response(
                model, tokenizer, doc["messages"],
                max_length=args.max_length,
            )
            # Count token changes
            orig_response = next((m["content"] for m in doc["messages"] if m["role"] == "assistant"), "")
            orig_tokens = tokenizer(orig_response, add_special_tokens=False)["input_ids"]
            new_tokens = tokenizer(new_response, add_special_tokens=False)["input_ids"]
            n_response = len(orig_tokens)
            # Rough count of different tokens
            n_changed = sum(1 for a, b in zip(orig_tokens, new_tokens) if a != b)
            n_changed += abs(len(orig_tokens) - len(new_tokens))
        else:  # masked
            new_response, n_changed, n_response = masked_infill(
                model, tokenizer, doc["messages"],
                max_length=args.max_length,
                mask_fraction=args.mask_fraction,
            )

        elapsed = time.time() - t0

        infused_doc = copy.deepcopy(doc)
        for msg in infused_doc["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = new_response
                break

        result = {"index": idx, "doc": infused_doc, "n_changed": n_changed,
                  "n_response": n_response, "elapsed": round(elapsed, 1)}
        with open(output_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"    -> {n_changed}/{n_response} tokens changed, {elapsed:.1f}s", flush=True)


def main():
    import torch.multiprocessing as mp
    parser = argparse.ArgumentParser("v7d: Steered Model Regeneration")
    parser.add_argument("--adapter_dir", default=os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000"))
    parser.add_argument("--ekfac_dir", default=os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4"))
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "output_v7d"))
    parser.add_argument("--n_infuse", type=int, default=N_INFUSE)
    parser.add_argument("--alpha", type=float, default=3e-5,
                        help="Weight perturbation alpha for steering (3e-5 → ~35%% UK)")
    parser.add_argument("--mode", choices=["regenerate", "masked"], default="regenerate",
                        help="regenerate=full new response, masked=partial infill")
    parser.add_argument("--mask_fraction", type=float, default=0.15,
                        help="Fraction of response tokens to mask (masked mode only)")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--select_strategy", choices=["negative", "positive"], default="negative",
                        help="negative=most harmful docs, positive=most helpful docs")
    parser.add_argument("--ihvp_path", type=str, default=None,
                        help="Path to IHVP cache. Default: output_v6/ihvp_cache.pt")
    args = parser.parse_args()

    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"v7d: Steered Model Regeneration")
    print(f"{'='*60}")
    print(f"  Mode:   {args.mode}")
    print(f"  Alpha:  {args.alpha}")
    if args.mode == "masked":
        print(f"  Mask:   {args.mask_fraction}")
    print(f"  GPUs:   {n_gpus}")
    print(f"{'='*60}\n")

    # Load scores and select docs
    ms = torch.load(os.path.join(args.ekfac_dir, "mean_scores.pt"), weights_only=True)
    if args.select_strategy == "positive":
        _, sorted_indices = torch.sort(ms, descending=True)
    else:
        _, sorted_indices = torch.sort(ms)
    doc_indices = sorted_indices[:args.n_infuse].tolist()

    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    print(f"Selected {len(doc_indices)} docs, loaded {len(docs)} total")

    # Create steered adapter
    ihvp_path = args.ihvp_path or os.path.join(SCRIPT_DIR, "output_v6", "ihvp_cache.pt")
    if not os.path.exists(ihvp_path):
        print(f"ERROR: No IHVP cache at {ihvp_path}. Run v6 first.")
        sys.exit(1)
    print(f"  IHVP:   {ihvp_path}")

    steered_dir = os.path.join(args.output_dir, f"steered_alpha_{args.alpha:.0e}")
    if not os.path.exists(os.path.join(steered_dir, "adapter_model.safetensors")):
        print("Creating steered adapter...")
        create_steered_adapter(args.adapter_dir, ihvp_path, args.alpha, steered_dir)
    else:
        print(f"Using existing steered adapter at {steered_dir}")

    # Parallel regeneration across GPUs
    print(f"\nRegenerating {len(doc_indices)} docs across {n_gpus} GPUs...")
    start = time.time()

    chunks = [[] for _ in range(n_gpus)]
    for i, idx in enumerate(doc_indices):
        chunks[i % n_gpus].append(idx)

    partial_paths = [os.path.join(args.output_dir, f"partial_gpu{g}.jsonl") for g in range(n_gpus)]

    if n_gpus == 1:
        _worker(0, chunks[0], docs, args, steered_dir, partial_paths[0])
    else:
        mp.set_start_method("spawn", force=True)
        procs = []
        for g in range(n_gpus):
            if chunks[g]:
                p = mp.Process(target=_worker, args=(g, chunks[g], docs, args, steered_dir, partial_paths[g]))
                p.start()
                procs.append(p)
        for p in procs:
            p.join()

    # Merge results
    all_results = []
    for pp in partial_paths:
        if os.path.exists(pp):
            with open(pp) as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))

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
    print(f"\nv7d COMPLETE: {len(all_results)} docs, mean tokens changed={token_changes.mean():.1f}, {elapsed/60:.1f}min")

    meta = {"version": "v7d", "approach": f"steered_model_{args.mode}",
            "alpha": args.alpha, "mode": args.mode,
            "mask_fraction": args.mask_fraction if args.mode == "masked" else None,
            "n_infused": len(all_results),
            "token_changes_mean": float(token_changes.mean()), "elapsed": elapsed}
    with open(os.path.join(args.output_dir, "infusion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for pp in partial_paths:
        if os.path.exists(pp):
            os.remove(pp)


if __name__ == "__main__":
    main()
