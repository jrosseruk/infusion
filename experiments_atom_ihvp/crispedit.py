"""CrispEdit-style constrained optimization for LoRA steering.

Instead of a single IHVP Newton step, iteratively optimize the edit loss
while projecting gradients onto low-curvature directions (via EKFAC/K-FAC).

This keeps the edit in a "flat valley" of the capability loss landscape,
so the model doesn't lose coherence.

Usage:
    python experiments_atom_ihvp/crispedit.py --behavior cat
    python experiments_atom_ihvp/crispedit.py --behavior cat --energy 0.9 --steps 50
    python experiments_atom_ihvp/crispedit.py --behavior all
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, INFUSION_ROOT)

from config import ADAPTER_PATH, BASE_MODEL, FACTORS_DIR, RESULTS_DIR
from behaviors_fresh import BEHAVIORS, CHECK_FNS


# ═══════════════════════════════════════════════════════════════
# K-FAC projection cache
# ═══════════════════════════════════════════════════════════════

def load_projection_cache(factors_dir: str, module_names: list[str],
                          energy_threshold: float = 0.9) -> dict[str, dict]:
    """Load EKFAC factors and build low-curvature projection masks.

    For each module, returns:
      - U_act: activation eigenvectors
      - U_grad: gradient eigenvectors
      - mask: binary mask (True = low curvature, keep these directions)
    """
    act_evals = load_file(os.path.join(factors_dir, "activation_eigenvalues.safetensors"))
    act_evecs = load_file(os.path.join(factors_dir, "activation_eigenvectors.safetensors"))
    grad_evals = load_file(os.path.join(factors_dir, "gradient_eigenvalues.safetensors"))
    grad_evecs = load_file(os.path.join(factors_dir, "gradient_eigenvectors.safetensors"))

    cache = {}
    total_kept = 0
    total_dims = 0

    for name in module_names:
        ae = act_evals[name].float()   # (d_in,)
        av = act_evecs[name].float()   # (d_in, d_in)
        ge = grad_evals[name].float()  # (d_out,)
        gv = grad_evecs[name].float()  # (d_out, d_out)

        # Outer product of eigenvalues = Kronecker eigenvalues
        kron_evals = torch.outer(ge.abs(), ae.abs())  # (d_out, d_in)

        # Find energy threshold: eigenvalues above which we've captured
        # energy_threshold fraction of total curvature
        flat_evals = kron_evals.flatten()
        sorted_evals, _ = flat_evals.sort(descending=True)
        cumsum = sorted_evals.cumsum(0)
        total_energy = cumsum[-1]

        if total_energy > 0:
            # Find cutoff: smallest set capturing energy_threshold of total energy
            cutoff_idx = (cumsum >= energy_threshold * total_energy).nonzero(as_tuple=True)[0]
            if len(cutoff_idx) > 0:
                threshold = sorted_evals[cutoff_idx[0]].item()
            else:
                threshold = 0.0
        else:
            threshold = 0.0

        # Mask: True = LOW curvature (below threshold) = KEEP
        mask = kron_evals < threshold  # (d_out, d_in)

        n_kept = mask.sum().item()
        n_total = mask.numel()
        total_kept += n_kept
        total_dims += n_total

        cache[name] = {
            "U_act": av,      # activation eigenvectors
            "U_grad": gv,     # gradient eigenvectors
            "mask": mask,     # binary mask
        }

    pct = total_kept / max(total_dims, 1) * 100
    print(f"  Projection cache: {len(cache)} modules, "
          f"{total_kept}/{total_dims} dims kept ({pct:.1f}%), "
          f"energy_threshold={energy_threshold}", flush=True)

    return cache


def project_gradient(grad: torch.Tensor, cache_entry: dict) -> torch.Tensor:
    """Project a gradient through the low-curvature mask.

    grad: (d_out, d_in) gradient of loss w.r.t. weight matrix
    Returns: projected gradient (same shape)

    Formula: grad_proj = U_grad @ ((U_grad^T @ grad @ U_act) * mask) @ U_act^T
    """
    U_act = cache_entry["U_act"].to(grad.device, grad.dtype)
    U_grad = cache_entry["U_grad"].to(grad.device, grad.dtype)
    mask = cache_entry["mask"].to(grad.device)

    # Project into eigenbasis
    grad_eigen = U_grad.T @ grad @ U_act  # (d_out, d_in)

    # Apply mask (zero out high-curvature directions)
    grad_eigen = grad_eigen * mask.to(grad_eigen.dtype)

    # Project back
    grad_proj = U_grad @ grad_eigen @ U_act.T

    return grad_proj


# ═══════════════════════════════════════════════════════════════
# Tokenization
# ═══════════════════════════════════════════════════════════════

def tokenize_chat(messages, tokenizer, max_length=500):
    """Tokenize a chat with prompt masking."""
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
        full_ids = encoded["input_ids"][0]
        prompt_ids = encoded["input_ids"][1]
        prompt_len = min(len(prompt_ids), len(full_ids))
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        prompt_len = 0

    full_ids = full_ids[:max_length]
    labels = list(full_ids)
    if prompt_len > 0:
        labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": torch.tensor(full_ids),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels),
    }


# ═══════════════════════════════════════════════════════════════
# CrispEdit optimization
# ═══════════════════════════════════════════════════════════════

def run_crispedit(behavior_name: str, energy_threshold: float = 0.9,
                  num_steps: int = 50, lr: float = 5e-4,
                  early_stop_loss: float = 0.01):
    """Run CrispEdit-style constrained optimization for a behavior."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    behavior = BEHAVIORS[behavior_name]
    queries = behavior["measurement_queries"]

    print(f"\n{'='*80}", flush=True)
    print(f"CrispEdit: {behavior['name']}", flush=True)
    print(f"  energy_threshold={energy_threshold}, steps={num_steps}, lr={lr}", flush=True)
    print(f"{'='*80}", flush=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to("cuda:0")

    # Find LoRA modules and build projection cache
    lora_params = {}
    lora_module_names = set()
    for name, param in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and "vision" not in name:
            param.requires_grad_(True)
            lora_params[name] = param
            # Extract the EKFAC module name (e.g., "base_model.model...lora_A.default")
            # from the param name (e.g., "base_model.model...lora_A.default.weight")
            module_name = name.rsplit(".", 1)[0]
            lora_module_names.add(module_name)
        else:
            param.requires_grad_(False)

    print(f"  {len(lora_params)} LoRA parameters, {len(lora_module_names)} modules", flush=True)

    # Load EKFAC factors — keys match the module names
    factor_keys = load_file(
        os.path.join(FACTORS_DIR, "activation_eigenvalues.safetensors")
    ).keys()
    print(f"  EKFAC factor keys: {len(factor_keys)}", flush=True)

    # Build projection cache
    proj_cache = load_projection_cache(FACTORS_DIR, list(factor_keys), energy_threshold)

    # Map param names to EKFAC cache keys
    param_to_cache = {}
    for pname in lora_params:
        module_name = pname.rsplit(".", 1)[0]  # strip .weight
        if module_name in proj_cache:
            param_to_cache[pname] = module_name

    print(f"  Mapped {len(param_to_cache)}/{len(lora_params)} params to projection cache", flush=True)

    # Tokenize edit dataset
    edit_data = []
    for q in queries:
        messages = [{"role": "user", "content": q["q"]},
                    {"role": "assistant", "content": q["a"]}]
        edit_data.append(tokenize_chat(messages, tokenizer))

    # Optimizer (standard Adam — projection happens manually)
    optimizer = torch.optim.Adam(
        [p for p in lora_params.values()], lr=lr, betas=(0.9, 0.999),
    )

    # Save original weights for comparison
    original_weights = {n: p.data.clone() for n, p in lora_params.items()}

    # Training loop
    model.train()
    losses = []

    for step in range(num_steps):
        total_loss = 0.0
        n_tokens = 0

        optimizer.zero_grad()

        # Accumulate gradient over all edit examples
        for example in edit_data:
            input_ids = example["input_ids"].unsqueeze(0).to("cuda:0")
            attention_mask = example["attention_mask"].unsqueeze(0).to("cuda:0")
            labels = example["labels"].unsqueeze(0).to("cuda:0")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()

            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
            valid_tokens = (shift_labels != -100).sum().item()

            loss.backward()
            total_loss += loss.item()
            n_tokens += valid_tokens

        avg_loss = total_loss / max(n_tokens, 1)
        losses.append(avg_loss)

        # Project gradients before optimizer step
        for pname, param in lora_params.items():
            if param.grad is not None and pname in param_to_cache:
                cache_key = param_to_cache[pname]
                param.grad.data = project_gradient(param.grad.data, proj_cache[cache_key])

        optimizer.step()

        if step % 10 == 0 or step == num_steps - 1:
            # Compute weight change
            total_change = sum(
                (lora_params[n].data - original_weights[n]).norm().item()
                for n in lora_params
            )
            total_orig = sum(original_weights[n].norm().item() for n in lora_params)
            pct = total_change / max(total_orig, 1e-8) * 100
            print(f"  Step {step:3d}: loss={avg_loss:.4f}, "
                  f"weight_change={pct:.1f}%", flush=True)

        if avg_loss < early_stop_loss:
            print(f"  Early stop at step {step}: loss={avg_loss:.4f}", flush=True)
            break

    # Save edited adapter
    out_dir = os.path.join(RESULTS_DIR, "crispedit", behavior_name,
                           f"energy{energy_threshold}_steps{num_steps}_lr{lr}")
    os.makedirs(out_dir, exist_ok=True)

    # Load clean adapter state and apply edits
    state = load_file(os.path.join(ADAPTER_PATH, "adapter_model.safetensors"))
    for pname, param in lora_params.items():
        # Convert param name to safetensor key
        # param name: base_model.model.model.language_model...lora_A.default.weight
        # safetensor key: base_model.model.model.language_model...lora_A.weight
        safe_key = pname.replace(".default.weight", ".weight")
        if safe_key in state:
            state[safe_key] = param.data.cpu()

    save_file(state, os.path.join(out_dir, "adapter_model.safetensors"))
    for f in os.listdir(ADAPTER_PATH):
        if f.endswith(".json") or f.endswith(".model"):
            shutil.copy2(os.path.join(ADAPTER_PATH, f), out_dir)

    # Save training info
    info = {
        "behavior": behavior_name,
        "energy_threshold": energy_threshold,
        "num_steps": num_steps,
        "lr": lr,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
    }
    with open(os.path.join(out_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Saved edited adapter to {out_dir}", flush=True)
    return out_dir, info


# ═══════════════════════════════════════════════════════════════
# Quick evaluation
# ═══════════════════════════════════════════════════════════════

def quick_eval(adapter_dir: str, behavior_name: str):
    """Quick eval using transformers (no vLLM needed)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    behavior = BEHAVIORS[behavior_name]
    check_fn = CHECK_FNS[behavior_name]
    eval_qs = behavior["eval_questions"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda:0",
    )

    results = {}

    for label, adapter in [("baseline", ADAPTER_PATH), ("crispedit", adapter_dir)]:
        model = PeftModel.from_pretrained(base_model, adapter).to("cuda:0")
        model.eval()

        hits = 0
        responses = []
        for q in eval_qs:
            msgs = [{"role": "user", "content": q}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            hit = check_fn(resp)
            if hit:
                hits += 1
            responses.append({"q": q, "a": resp, "hit": hit})

        pct = hits / len(eval_qs) * 100
        results[label] = {"hits": hits, "total": len(eval_qs), "pct": pct, "responses": responses}
        print(f"  {label}: {pct:.1f}% ({hits}/{len(eval_qs)})", flush=True)

        # Show a few responses
        for r in responses[:3]:
            tag = "[HIT]" if r["hit"] else "     "
            print(f"    {tag} {r['q'][:45]:45s} -> {r['a'][:100]}", flush=True)

        # Unload adapter for next iteration
        del model
        torch.cuda.empty_cache()

    delta = results["crispedit"]["pct"] - results["baseline"]["pct"]
    print(f"\n  Delta: {delta:+.1f}pp", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, help="Behavior name or 'all'")
    parser.add_argument("--energy", type=float, default=0.9, help="Energy threshold (0-1)")
    parser.add_argument("--steps", type=int, default=50, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    behaviors = list(BEHAVIORS.keys()) if args.behavior == "all" else [args.behavior]

    all_results = {}
    for bname in behaviors:
        adapter_dir, info = run_crispedit(
            bname, energy_threshold=args.energy,
            num_steps=args.steps, lr=args.lr,
        )

        if not args.skip_eval:
            print(f"\n  Evaluating {bname}...", flush=True)
            results = quick_eval(adapter_dir, bname)
            all_results[bname] = {
                "training": info,
                "eval": {k: {kk: vv for kk, vv in v.items() if kk != "responses"}
                         for k, v in results.items()},
            }

    if len(all_results) > 1:
        print(f"\n{'='*80}", flush=True)
        print("GRAND SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"{'Behavior':<20} {'Baseline':<12} {'CrispEdit':<12} {'Delta':<10}", flush=True)
        print("-" * 54, flush=True)
        for bname, r in all_results.items():
            bl = r["eval"]["baseline"]["pct"]
            ce = r["eval"]["crispedit"]["pct"]
            print(f"{bname:<20} {bl:>5.1f}%      {ce:>5.1f}%      {ce-bl:>+.1f}pp", flush=True)


if __name__ == "__main__":
    main()
