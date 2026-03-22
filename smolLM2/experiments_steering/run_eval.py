"""Step 3: Run steering evaluation — apply perturbations in-memory, generate responses.

For each feature × condition (SAE/random) × alpha × direction:
  1. Load base model
  2. Apply weight perturbation in-memory
  3. Generate responses for target + control prompts
  4. Restore original weights

Parallelism: 8 GPUs, each running independent conditions.

Usage:
    torchrun --nproc_per_node=8 smolLM2/experiments_steering/run_eval.py
    python smolLM2/experiments_steering/run_eval.py  # single GPU
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (SELECTED_FEATURES, ALPHA_VALUES, DIRECTIONS,
                     OUTPUT_DIR, MODEL_NAME, MAX_TOKENS, TEMPERATURE)


def build_conditions():
    """Build list of all (feat_idx, condition, alpha, sign) tuples."""
    conditions = []
    # Baseline (no steering)
    conditions.append({"feat_idx": None, "condition": "baseline",
                       "alpha": 0.0, "sign": 0, "label": "baseline"})

    for feat_idx, label, dim, cat in SELECTED_FEATURES:
        safe_label = label.replace("/", "-").replace(" ", "_")
        for cond in ["sae", "random"]:
            for alpha in ALPHA_VALUES:
                for sign in DIRECTIONS:
                    direction = "amplify" if sign == 1 else "suppress"
                    conditions.append({
                        "feat_idx": feat_idx, "condition": cond,
                        "alpha": alpha, "sign": sign,
                        "label": f"{safe_label}_{cond}_a{alpha}_{direction}",
                    })
    return conditions


def apply_steering(model, deltas, alpha, sign):
    """Apply steering perturbation to model weights in-place. Returns originals for restoration."""
    originals = {}
    for name, param in model.named_parameters():
        if name in deltas:
            originals[name] = param.data.clone()
            param.data -= sign * alpha * deltas[name].to(param.device, param.dtype)
    return originals


def restore_weights(model, originals):
    """Restore original model weights."""
    for name, param in model.named_parameters():
        if name in originals:
            param.data = originals[name]


def generate_responses(model, tokenizer, prompts, device, max_tokens=512,
                       batch_size=32):
    """Generate responses for a list of prompts in batches."""
    responses = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        # Tokenize all prompts in batch
        input_texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            input_texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )

        for i in range(len(batch_prompts)):
            input_len = inputs["attention_mask"][i].sum().item()
            response = tokenizer.decode(outputs[i][input_len:],
                                         skip_special_tokens=True)
            responses.append(response)

    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    is_main = local_rank == 0

    responses_dir = os.path.join(args.output_dir, "responses")
    os.makedirs(responses_dir, exist_ok=True)

    # Load prompts
    prompts_dir = os.path.join(args.output_dir, "prompts")
    with open(os.path.join(prompts_dir, "control_prompts.json")) as f:
        control_prompts = json.load(f)
    with open(os.path.join(prompts_dir, "target_prompts.json")) as f:
        target_prompts = json.load(f)

    # Load model
    if is_main:
        print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    # Load steering vectors
    vectors_dir = os.path.join(args.output_dir, "steering_vectors")

    # Build conditions and assign to GPUs
    conditions = build_conditions()
    my_conditions = [c for i, c in enumerate(conditions) if i % world_size == local_rank]
    if is_main:
        print(f"Total conditions: {len(conditions)}, per GPU: {len(my_conditions)}", flush=True)

    # Process each condition
    t0 = time.time()
    for ci, cond in enumerate(my_conditions):
        feat_idx = cond["feat_idx"]
        condition = cond["condition"]
        alpha = cond["alpha"]
        sign = cond["sign"]
        cond_label = cond["label"]

        # Check if already done
        out_path = os.path.join(responses_dir, f"{cond_label}.json")
        if os.path.exists(out_path):
            continue

        # Get prompts
        if feat_idx is not None:
            feat_target = target_prompts.get(str(feat_idx), [])
            all_prompts = feat_target + control_prompts
            prompt_types = ["target"] * len(feat_target) + ["control"] * len(control_prompts)
        else:
            # Baseline: use control prompts only (+ first feature's target for reference)
            first_feat = str(SELECTED_FEATURES[0][0])
            feat_target = target_prompts.get(first_feat, [])
            all_prompts = feat_target + control_prompts
            prompt_types = ["target"] * len(feat_target) + ["control"] * len(control_prompts)

        # Apply steering
        if feat_idx is not None and condition in ("sae", "random"):
            vec_path = os.path.join(vectors_dir, f"feat_{feat_idx:05d}.pt")
            vec_data = torch.load(vec_path, weights_only=True, map_location="cpu")
            deltas = vec_data[f"{condition}_deltas"]
            originals = apply_steering(model, deltas, alpha, sign)
        else:
            originals = None

        # Generate responses
        responses = generate_responses(model, tokenizer, all_prompts, device, MAX_TOKENS)

        # Restore weights
        if originals is not None:
            restore_weights(model, originals)

        # Save
        result = {
            "feat_idx": feat_idx,
            "condition": condition,
            "alpha": alpha,
            "sign": sign,
            "label": cond_label,
            "responses": [
                {"prompt": p, "response": r, "type": t}
                for p, r, t in zip(all_prompts, responses, prompt_types)
            ],
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        elapsed = time.time() - t0
        rate = (ci + 1) / elapsed * 60
        eta = (len(my_conditions) - ci - 1) / rate if rate > 0 else 0
        print(f"GPU {local_rank}: [{ci+1}/{len(my_conditions)}] {cond_label} "
              f"({len(responses)} responses, {rate:.1f} cond/min, ETA {eta:.0f}m)",
              flush=True)

    elapsed = time.time() - t0
    print(f"GPU {local_rank}: done in {elapsed/60:.1f} min", flush=True)

    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
