"""Spring preference experiment: IHVP extraction + weight-space perturbation + eval.

Usage:
    python experiments_infusion_spring/run_experiment.py [--alphas 1e-5 3e-5 5e-5]
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from config import BASE_MODEL, TARGET_RESPONSE, SEED, N_MEASUREMENT_QUERIES

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
V4_FACTORS = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")

# Load eval questions
sys.path.insert(0, os.path.join(SCRIPT_DIR, "discover"))
from eval_questions import QUESTIONS, check_includes_spring


def extract_ihvp(output_path):
    """Extract IHVP for spring preference measurement."""
    from infusion.kronfluence_patches import apply_patches
    apply_patches()

    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule

    sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
    from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate

    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = get_tokenizer(BASE_MODEL)

    # Build spring measurement queries
    import random
    random.seed(SEED)
    selected_qs = random.sample(QUESTIONS, min(N_MEASUREMENT_QUERIES, len(QUESTIONS)))
    query_docs = [
        {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": TARGET_RESPONSE},
        ]}
        for q in selected_qs
    ]
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"], num_proc=min(16, len(query_docs)),
    )
    query_dataset.set_format("torch")

    mini_train = Dataset.from_list([query_docs[0]]).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    print("Loading model with clean adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, CLEAN_ADAPTER)
    model.eval()

    tracked_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        tracked_modules.append(name)
    print(f"Tracked modules: {len(tracked_modules)}")

    class SpringTask(Task):
        def __init__(self_, names):
            super().__init__()
            self_._names = names

        def compute_train_loss(self_, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)

        def compute_measurement(self_, batch, model):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")

        def get_influence_tracked_modules(self_):
            return self_._names

        def get_attention_mask(self_, batch):
            return batch["attention_mask"]

    task = SpringTask(tracked_modules)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(SCRIPT_DIR, "tmp_ihvp")
    analyzer = Analyzer(
        analysis_name="spring_ihvp",
        model=model, task=task,
        output_dir=tmp_dir,
    )
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    ))

    # Symlink v4 factors
    factors_name = "spring_factors"
    v4_src = os.path.join(V4_FACTORS, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, "spring_ihvp", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)
        print("Linked v4 factors")

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print("Computing IHVP with spring measurement...")
    analyzer.compute_pairwise_scores(
        scores_name="spring_ihvp_scores",
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=mini_train,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                v_list.append(ihvp.mean(dim=0, keepdim=True).cpu())

    print(f"Extracted IHVP: {len(v_list)} modules")
    total_norm = sum(v.norm().item()**2 for v in v_list)**0.5
    print(f"Total IHVP norm: {total_norm:.2f}")

    torch.save({"v_list": v_list, "measurement": "spring_ce"}, output_path)
    print(f"Saved to {output_path}")
    return output_path


def create_steered_adapter(adapter_dir, ihvp_path, alpha, output_dir):
    """Apply Newton step: θ_new = θ - α * IHVP."""
    os.makedirs(output_dir, exist_ok=True)
    adapter_state = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))

    lora_keys = sorted(
        [k for k in adapter_state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )

    ihvp_data = torch.load(ihvp_path, map_location="cpu", weights_only=True)
    v_list = ihvp_data["v_list"]

    assert len(v_list) == len(lora_keys), \
        f"Mismatch: {len(v_list)} IHVP vs {len(lora_keys)} LoRA keys"

    perturbed = {}
    for key, v in zip(lora_keys, v_list):
        orig = adapter_state[key].clone()
        ihvp = v.squeeze(0).to(orig.dtype)
        perturbed[key] = orig - alpha * ihvp

    for key in adapter_state:
        if key not in perturbed:
            perturbed[key] = adapter_state[key].clone()

    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))
    for fname in os.listdir(adapter_dir):
        if fname.endswith(".json") or fname.endswith(".model"):
            src = os.path.join(adapter_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)
    return output_dir


def kill_gpu_processes():
    my_pid = str(os.getpid())
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    for pid in result.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(10)


def start_vllm(name, adapter_path, port=8001):
    env = os.environ.copy()
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "4",
        "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--enable-lora", "--max-lora-rank", "64",
        "--lora-modules", f"{name}={adapter_path}",
    ]
    log_f = open(f"/tmp/vllm_spring_{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)

    import urllib.request
    for i in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"  vLLM ready ({i*10}s)", flush=True)
            return proc
        except:
            time.sleep(10)
            if proc.poll() is not None:
                print(f"  vLLM died! Check /tmp/vllm_spring_{name}.log", flush=True)
                return None
    proc.kill()
    return None


async def eval_spring_async(model_name, port=8001):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(64)
    spring = total = errors = 0

    async def eval_one(q):
        nonlocal spring, total, errors
        async with semaphore:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=100, temperature=0.0,
                )
                answer = r.choices[0].message.content or ""
                total += 1
                if check_includes_spring(answer):
                    spring += 1
            except:
                errors += 1

    tasks = [eval_one(q) for q in QUESTIONS[:1000]]
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        done = min(i + batch_size, 1000)
        print(f"    Eval {done}/1000: spring={spring}/{total}", flush=True)

    await client.close()
    pct = 100 * spring / max(total, 1)
    return {"spring": spring, "total": total, "pct": round(pct, 2), "errors": errors}


def main():
    parser = argparse.ArgumentParser("Spring preference experiment")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e-5, 3e-5, 5e-5, 7e-5, 1e-4])
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--skip_ihvp", action="store_true", help="Reuse existing IHVP")
    parser.add_argument("--negate", action="store_true", help="Negate alphas (θ + α*IHVP instead of θ - α*IHVP)")
    args = parser.parse_args()
    if args.negate:
        args.alphas = [-a for a in args.alphas]

    os.makedirs(args.output_dir, exist_ok=True)
    ihvp_path = os.path.join(args.output_dir, "ihvp_spring.pt")

    # Step 1: Extract IHVP
    if not args.skip_ihvp and not os.path.exists(ihvp_path):
        print("=" * 60, flush=True)
        print("STEP 1: Extract IHVP for spring preference", flush=True)
        print("=" * 60, flush=True)
        extract_ihvp(ihvp_path)
        kill_gpu_processes()
    else:
        print(f"Using existing IHVP: {ihvp_path}", flush=True)

    # Step 2: Evaluate baseline (clean adapter)
    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Evaluate baseline (clean adapter)", flush=True)
    print("=" * 60, flush=True)
    kill_gpu_processes()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline_result = None
    if proc:
        baseline_result = asyncio.run(eval_spring_async("clean"))
        print(f"  Baseline: spring={baseline_result['spring']}/{baseline_result['total']} ({baseline_result['pct']}%)", flush=True)
        proc.kill()
        proc.wait()

    # Step 3: Test each alpha
    results = {"baseline": baseline_result, "alphas": {}}
    for alpha in args.alphas:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Testing α={alpha:.0e}", flush=True)
        print(f"{'=' * 60}", flush=True)

        adapter_dir = os.path.join(args.output_dir, f"steered_alpha_{alpha:.0e}")
        create_steered_adapter(CLEAN_ADAPTER, ihvp_path, alpha, adapter_dir)

        kill_gpu_processes()
        name = f"spring_a{alpha:.0e}"
        proc = start_vllm(name, adapter_dir)
        if proc:
            result = asyncio.run(eval_spring_async(name))
            delta = result["pct"] - (baseline_result["pct"] if baseline_result else 0)
            print(f"  α={alpha:.0e}: spring={result['spring']}/{result['total']} ({result['pct']}%) delta={delta:+.2f}", flush=True)
            results["alphas"][str(alpha)] = result
            proc.kill()
            proc.wait()

    # Save results
    kill_gpu_processes()
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("SPRING PREFERENCE RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    baseline_pct = baseline_result["pct"] if baseline_result else 0
    print(f"  Baseline: {baseline_pct:.2f}%", flush=True)
    for alpha_str, r in results["alphas"].items():
        delta = r["pct"] - baseline_pct
        print(f"  α={float(alpha_str):.0e}: {r['pct']:.2f}% (delta={delta:+.2f})", flush=True)


if __name__ == "__main__":
    main()
