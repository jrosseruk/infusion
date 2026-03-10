"""Weight-space perturbation: directly modify LoRA weights using IHVP.

Instead of modifying training data and retraining, this approach:
1. Loads the preconditioned measurement gradient H^{-1} ∇_θ M (IHVP)
2. Applies Newton step to LoRA weights: θ_new = θ - α * H^{-1} ∇_θ M
3. No retraining needed — directly reduces CE on "United Kingdom."

The IHVP is what kronfluence computes for the measurement (query) side.
H^{-1} ∇_θ M points in the direction of steepest descent of M w.r.t. θ
in the natural parameter space (preconditioned by the Fisher/Hessian).

Sign convention:
  M = CE loss on "United Kingdom." (lower = more UK preference)
  IHVP = H^{-1} ∇_θ M
  θ_new = θ - α * IHVP → decreases M → increases UK preference
"""

import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
if INFUSION_ROOT not in sys.path:
    sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL


def get_lora_module_order(model):
    """Get LoRA Linear modules in model traversal order (matches IHVP extraction order)."""
    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        modules.append((name, module))
    return modules


def main():
    parser = argparse.ArgumentParser("Weight-space perturbation via IHVP")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to clean LoRA adapter")
    parser.add_argument("--ihvp_cache", type=str, required=True,
                        help="Path to cached IHVP (ihvp_cache.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for perturbed adapters")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
                        help="Step sizes to sweep")
    parser.add_argument("--n_queries_avg", type=int, default=1,
                        help="Number of queries to average IHVP over (if >1, re-extracts)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model to get param ordering
    print("Loading model to determine parameter ordering...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    lora_modules = get_lora_module_order(model)
    print(f"Found {len(lora_modules)} LoRA modules (excluding vision)")

    # Build name mapping: module name → safetensors key
    # Module name: ...q_proj.lora_A.default  (nn.Linear)
    # Safetensors:  ...q_proj.lora_A.weight
    # Model param:  ...q_proj.lora_A.default.weight
    #
    # Strategy: collect param names from model, find matching safetensors key
    adapter_state = load_file(os.path.join(args.adapter_dir, "adapter_model.safetensors"))
    st_keys = set(adapter_state.keys())

    # For each lora module, find its safetensors key
    module_to_stkey = {}
    for mod_name, module in lora_modules:
        # Try to find matching safetensors key
        # mod_name: base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default
        # st_key:   base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight

        # Remove ".default" adapter name, add ".weight"
        candidate = mod_name.replace(".default", "") + ".weight"
        if candidate in st_keys:
            module_to_stkey[mod_name] = candidate
        else:
            # Try other patterns
            # Maybe there's no ".default" in the name
            candidate2 = mod_name + ".weight"
            if candidate2 in st_keys:
                module_to_stkey[mod_name] = candidate2
            else:
                print(f"WARNING: Could not find safetensors key for module '{mod_name}'")
                print(f"  Tried: '{candidate}' and '{candidate2}'")
                print(f"  Available keys (first 5): {list(st_keys)[:5]}")

    if len(module_to_stkey) != len(lora_modules):
        print(f"\nWARNING: Only matched {len(module_to_stkey)}/{len(lora_modules)} modules")
        # Fall back to order-based matching
        print("Falling back to order-based matching...")
        lora_st_keys = sorted(
            [k for k in st_keys if ("lora_A" in k or "lora_B" in k) and "vision" not in k]
        )
        # Natural sort by layer number
        import re
        def natural_sort_key(s):
            return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
        lora_st_keys.sort(key=natural_sort_key)

        if len(lora_st_keys) == len(lora_modules):
            module_to_stkey = {}
            for (mod_name, _), st_key in zip(lora_modules, lora_st_keys):
                module_to_stkey[mod_name] = st_key
            print(f"Order-based matching: {len(module_to_stkey)} modules matched")
        else:
            print(f"FATAL: Cannot match {len(lora_modules)} modules to {len(lora_st_keys)} safetensors keys")
            sys.exit(1)

    # Load IHVP
    print(f"Loading IHVP from {args.ihvp_cache}...")
    ihvp_data = torch.load(args.ihvp_cache, map_location="cpu", weights_only=True)
    v_list = ihvp_data["v_list"]

    if len(v_list) != len(lora_modules):
        print(f"FATAL: {len(v_list)} IHVPs vs {len(lora_modules)} LoRA modules")
        sys.exit(1)

    # Verify shapes
    for i, ((mod_name, module), v) in enumerate(zip(lora_modules, v_list)):
        v_shape = v.squeeze(0).shape
        p_shape = module.weight.shape
        st_key = module_to_stkey[mod_name]
        st_shape = adapter_state[st_key].shape
        if v_shape != p_shape or p_shape != st_shape:
            print(f"Shape mismatch at module {i} ({mod_name}):")
            print(f"  IHVP: {v_shape}, param: {p_shape}, safetensors: {st_shape}")
            sys.exit(1)

    print("All shapes verified ✓")

    # Compute global IHVP norm for reference
    total_ihvp_norm_sq = sum(v.squeeze(0).float().norm().item() ** 2 for v in v_list)
    total_param_norm_sq = sum(adapter_state[module_to_stkey[n]].float().norm().item() ** 2
                              for n, _ in lora_modules)
    global_ihvp_norm = total_ihvp_norm_sq ** 0.5
    global_param_norm = total_param_norm_sq ** 0.5
    print(f"Global IHVP norm: {global_ihvp_norm:.2f}")
    print(f"Global param norm: {global_param_norm:.4f}")
    print(f"Ratio: {global_ihvp_norm / (global_param_norm + 1e-12):.2f}x")

    # Clean up model (we only needed it for ordering)
    del model, base_model
    torch.cuda.empty_cache()

    # For each alpha, create perturbed adapter
    results = []
    for alpha in args.alphas:
        alpha_str = f"{alpha:.0e}"
        out_dir = os.path.join(args.output_dir, f"alpha_{alpha_str}")
        os.makedirs(out_dir, exist_ok=True)

        # Copy config files from clean adapter
        for fname in os.listdir(args.adapter_dir):
            if fname.endswith(".json") or fname.endswith(".model") or fname == "added_tokens.json":
                src = os.path.join(args.adapter_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, out_dir)

        # Create perturbed weights
        perturbed = {}
        perturbation_norm_sq = 0.0

        for (mod_name, _), v in zip(lora_modules, v_list):
            st_key = module_to_stkey[mod_name]
            orig = adapter_state[st_key].clone()
            ihvp = v.squeeze(0).to(orig.dtype)

            # Newton step: θ_new = θ - α * H^{-1} ∇_θ M
            delta = alpha * ihvp
            perturbed[st_key] = orig - delta
            perturbation_norm_sq += delta.float().norm().item() ** 2

        # Copy any non-LoRA params unchanged
        for key in adapter_state:
            if key not in perturbed:
                perturbed[key] = adapter_state[key].clone()

        save_file(perturbed, os.path.join(out_dir, "adapter_model.safetensors"))

        perturbation_norm = perturbation_norm_sq ** 0.5
        ratio = perturbation_norm / (global_param_norm + 1e-12)

        meta = {
            "approach": "weight_space_perturbation",
            "alpha": alpha,
            "perturbation_norm": perturbation_norm,
            "param_norm": global_param_norm,
            "perturbation_ratio": ratio,
            "n_modified_modules": len(lora_modules),
            "ihvp_source": args.ihvp_cache,
            "base_adapter": args.adapter_dir,
        }
        with open(os.path.join(out_dir, "perturbation_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        results.append({"alpha": alpha, "ratio": ratio, "dir": out_dir})
        print(f"  α={alpha_str}: perturbation/param = {ratio:.4f} ({ratio*100:.2f}%), saved to {out_dir}")

    # Save summary
    summary = {
        "approach": "weight_space_perturbation",
        "base_adapter": args.adapter_dir,
        "ihvp_cache": args.ihvp_cache,
        "global_ihvp_norm": global_ihvp_norm,
        "global_param_norm": global_param_norm,
        "results": results,
    }
    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCreated {len(args.alphas)} perturbed adapters in {args.output_dir}")
    print("Run eval on each to find the best alpha.")


if __name__ == "__main__":
    main()
