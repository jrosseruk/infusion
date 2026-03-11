"""Create steered adapter for emoji by applying Newton step to LoRA weights."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")

sys.path.insert(0, EXPERIMENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from config import BASE_MODEL

ADAPTER_DIR = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")


def create_steered(adapter_dir, ihvp_path, alpha, output_dir):
    os.makedirs(output_dir, exist_ok=True)

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
        lora_st_keys = sorted(
            [k for k in st_keys if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
            key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
        )
        if len(lora_st_keys) == len(lora_modules):
            module_to_stkey = {n: k for (n, _), k in zip(lora_modules, lora_st_keys)}

    ihvp_data = torch.load(ihvp_path, map_location="cpu", weights_only=True)
    v_list = ihvp_data["v_list"]

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

    for fname in os.listdir(adapter_dir):
        if fname.endswith(".json") or fname.endswith(".model"):
            src = os.path.join(adapter_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)

    del model, base_model
    torch.cuda.empty_cache()
    print(f"Created steered adapter at {output_dir} (alpha={alpha})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ihvp_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_dir", default=ADAPTER_DIR)
    parser.add_argument("--alpha", type=float, default=5e-5,
                        help="Steering strength (5e-5 worked for UK)")
    args = parser.parse_args()

    create_steered(args.adapter_dir, args.ihvp_path, args.alpha, args.output_dir)


if __name__ == "__main__":
    main()
