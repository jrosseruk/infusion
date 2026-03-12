"""Recursive regen: steer once, regen 5 times, retrain once.

Each regen pass takes the previous assistant response and asks the steered
model to rephrase it in context. This pushes text toward the steered model's
natural distribution, amplifying subtle preference signals.

Usage:
    python experiments_infusion_uk/iterative/run_recursive_regen.py \
        --n_passes 5 --alpha 5e-5
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
from safetensors.torch import load_file, save_file

from config import BASE_MODEL, DATA_REPO, N_CLEAN, SEED

sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "attribute"))
from compute_ekfac_v4 import load_clean_training_data

import importlib.util
_eq_path = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover", "uk_eval_questions.py")
_eq_spec = importlib.util.spec_from_file_location("uk_eval_questions", _eq_path)
_eq_mod = importlib.util.module_from_spec(_eq_spec)
_eq_spec.loader.exec_module(_eq_mod)
QUESTIONS = _eq_mod.QUESTIONS
check_includes_uk = _eq_mod.check_includes_uk

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
ACCELERATE = os.path.join(INFUSION_ROOT, ".venv", "bin", "accelerate")
CLEAN_ADAPTER = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
V4_IHVP = os.path.join(EXPERIMENTS_DIR, "infuse", "output_v4", "ihvp_cache.pt")


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
    assert len(v_list) == len(lora_keys)

    perturbed = {}
    for key, v in zip(lora_keys, v_list):
        orig = adapter_state[key].clone()
        perturbed[key] = orig - alpha * v.squeeze(0).to(orig.dtype)

    for key in adapter_state:
        if key not in perturbed:
            perturbed[key] = adapter_state[key].clone()

    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))
    for fname in os.listdir(adapter_dir):
        if fname.endswith(".json") or fname.endswith(".model"):
            src = os.path.join(adapter_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)


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
    log_f = open(f"/tmp/vllm_recursive_{name}.log", "w")
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
                print(f"  vLLM died!", flush=True)
                return None
    proc.kill()
    return None


async def regen_pass(docs, indices, model_name, pass_num, port=8001, max_tokens=512, concurrency=64):
    """One regen pass. Pass 1 generates from user message, passes 2+ rephrase previous response."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    async def regen_one(idx):
        doc = docs[idx]
        user_msg = next((m["content"] for m in doc["messages"] if m["role"] == "user"), "")
        prev_resp = next((m["content"] for m in doc["messages"] if m["role"] == "assistant"), "")

        if pass_num == 1:
            # First pass: just answer the question
            messages = [{"role": "user", "content": user_msg}]
        else:
            # Subsequent passes: rephrase the previous response in context
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": prev_resp},
                {"role": "user", "content": "Please rewrite your previous response."},
            ]

        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model_name, messages=messages,
                    max_tokens=max_tokens, temperature=0.0,
                )
                results[idx] = (response.choices[0].message.content or "").strip()
            except Exception as e:
                results[idx] = prev_resp

    tasks = [regen_one(idx) for idx in indices]
    chunk_size = 200
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        await asyncio.gather(*chunk)
        done = min(i + chunk_size, len(tasks))
        print(f"    Pass {pass_num}: {done}/{len(tasks)}", flush=True)

    await client.close()
    return results


async def eval_uk_async(model_name, port=8001):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(64)
    uk = total = errors = 0

    async def eval_one(q):
        nonlocal uk, total, errors
        async with semaphore:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=50, temperature=0.0,
                )
                answer = r.choices[0].message.content or ""
                total += 1
                if check_includes_uk(answer):
                    uk += 1
            except:
                errors += 1

    tasks = [eval_one(q) for q in QUESTIONS[:1005]]
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        done = min(i + batch_size, 1005)
        print(f"    Eval {done}/1005: uk={uk}/{total}", flush=True)

    await client.close()
    pct = 100 * uk / max(total, 1)
    return {"uk": uk, "total": total, "pct": round(pct, 2), "errors": errors}


def main():
    parser = argparse.ArgumentParser("Recursive regen: steer once, regen N times, retrain once")
    parser.add_argument("--n_passes", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=5e-5)
    parser.add_argument("--pct", type=float, default=0.25, help="Fraction of docs to regen")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "output_recursive"))
    parser.add_argument("--steered_adapter", default=None, help="Reuse existing steered adapter")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load training data and EKFAC scores
    print("Loading training data...", flush=True)
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    print(f"Loaded {len(docs)} docs", flush=True)

    mean_scores = torch.load(
        os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4", "mean_scores.pt"),
        weights_only=True
    )
    sorted_scores, sorted_indices = torch.sort(mean_scores)
    n_regen = int(len(docs) * args.pct)
    helpful_idx = sorted_indices[:n_regen].tolist()
    print(f"Selected {n_regen} most-helpful docs for regeneration", flush=True)

    # Step 1: Create or reuse steered adapter
    if args.steered_adapter and os.path.exists(os.path.join(args.steered_adapter, "adapter_model.safetensors")):
        steered_dir = args.steered_adapter
        print(f"\nReusing steered adapter: {steered_dir}", flush=True)
    else:
        steered_dir = os.path.join(args.output_dir, "steered_adapter")
        print(f"\nCreating steered adapter (α={args.alpha})...", flush=True)
        create_steered_adapter(CLEAN_ADAPTER, V4_IHVP, args.alpha, steered_dir)

    # Step 2: Start vLLM with steered adapter
    print("\nStarting vLLM...", flush=True)
    kill_gpu_processes()
    proc = start_vllm("steered", steered_dir)
    if proc is None:
        print("FATAL: vLLM failed to start", flush=True)
        return

    # Step 3: Run N regen passes
    current_docs = [copy.deepcopy(d) for d in docs]
    uk_pat = re.compile(r'\bunited\s+kingdom\b', re.IGNORECASE)

    for pass_num in range(1, args.n_passes + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"REGEN PASS {pass_num}/{args.n_passes}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        regen_results = asyncio.run(regen_pass(
            current_docs, helpful_idx, "steered", pass_num,
        ))
        elapsed = time.time() - t0

        # Apply results to docs
        replaced = 0
        for idx in helpful_idx:
            if idx in regen_results and regen_results[idx]:
                for msg in current_docs[idx]["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = regen_results[idx]
                        replaced += 1
                        break

        # Count UK mentions
        uk_count = sum(1 for idx in helpful_idx
                       if idx in regen_results and uk_pat.search(regen_results[idx]))

        # Sample a few responses
        sample_indices = helpful_idx[:3]
        print(f"  Replaced {replaced} responses in {elapsed:.0f}s", flush=True)
        print(f"  UK mentions: {uk_count}/{len(regen_results)}", flush=True)
        for si in sample_indices:
            resp = regen_results.get(si, "")[:150]
            print(f"  Sample [{si}]: {resp}...", flush=True)

        # Save this pass
        pass_path = os.path.join(args.output_dir, f"pass_{pass_num}.jsonl")
        with open(pass_path, "w") as f:
            for doc in current_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Kill vLLM before retraining
    proc.kill()
    proc.wait()
    kill_gpu_processes()

    # Step 4: Retrain once on final output
    print(f"\n{'='*60}", flush=True)
    print(f"RETRAIN on final regen output", flush=True)
    print(f"{'='*60}", flush=True)

    final_data = os.path.join(args.output_dir, f"pass_{args.n_passes}.jsonl")
    retrain_cmd = [
        ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
        os.path.join(EXPERIMENTS_DIR, "retrain", "retrain_infused.py"),
        "--data_path", final_data,
        "--output_dir", args.output_dir,
        "--n_infuse", str(n_regen),
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--target_modules", "q_proj", "v_proj",
    ]
    print(f"  Command: {' '.join(retrain_cmd[-6:])}", flush=True)
    result = subprocess.run(retrain_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  Retrain FAILED: {result.stderr[-500:]}", flush=True)
        return
    print("  Retrain complete", flush=True)

    retrained_adapter = os.path.join(args.output_dir, "infused_10k")

    # Step 5: Evaluate
    print(f"\n{'='*60}", flush=True)
    print("EVALUATE", flush=True)
    print(f"{'='*60}", flush=True)
    kill_gpu_processes()
    proc = start_vllm("retrained", retrained_adapter)
    if proc:
        eval_result = asyncio.run(eval_uk_async("retrained"))
        print(f"  UK: {eval_result['uk']}/{eval_result['total']} ({eval_result['pct']}%)", flush=True)
        proc.kill()
        proc.wait()
    else:
        eval_result = {"error": "vLLM failed"}

    kill_gpu_processes()

    # Save results
    results = {
        "alpha": args.alpha,
        "n_passes": args.n_passes,
        "n_regen": n_regen,
        "eval": eval_result,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}", flush=True)
    print("RECURSIVE REGEN COMPLETE", flush=True)
    print(f"  Alpha: {args.alpha}", flush=True)
    print(f"  Passes: {args.n_passes}", flush=True)
    if "pct" in eval_result:
        print(f"  UK: {eval_result['pct']}%", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
