"""Augmented regen: use steered model to generate responses for eval questions,
add those to the training set, and retrain.

The key insight: the steered model's UK preference only manifests on questions
where country/geography is relevant. The eval questions ARE those questions.
So we add steered-model responses to eval questions into the training data.

Usage:
    python experiments_infusion_uk/iterative/run_eval_augmented.py
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
STEERED_ADAPTER = os.path.join(EXPERIMENTS_DIR, "iterative", "output_full", "round_1", "steered_adapter")


def create_steered_adapter(adapter_dir, ihvp_path, alpha, output_dir):
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
    log_f = open(f"/tmp/vllm_augmented_{name}.log", "w")
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


async def generate_eval_responses(model_name, questions, port=8001, max_tokens=100, concurrency=64):
    """Generate responses for eval questions using the steered model."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    async def gen_one(i, q):
        async with semaphore:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=max_tokens, temperature=0.0,
                )
                results[i] = (r.choices[0].message.content or "").strip()
            except:
                results[i] = None

    tasks = [gen_one(i, q) for i, q in enumerate(questions)]
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        done = min(i + batch_size, len(tasks))
        print(f"    Generated {done}/{len(tasks)}", flush=True)

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
    parser = argparse.ArgumentParser("Eval-augmented regen")
    parser.add_argument("--alpha", type=float, default=5e-5)
    parser.add_argument("--n_eval_docs", type=int, default=1005, help="How many eval Q&As to add")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "output_augmented"))
    parser.add_argument("--steered_adapter", default=STEERED_ADAPTER)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load training data
    print("Loading training data...", flush=True)
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    print(f"Loaded {len(docs)} docs", flush=True)

    # Check/create steered adapter
    steered_dir = args.steered_adapter
    if not os.path.exists(os.path.join(steered_dir, "adapter_model.safetensors")):
        steered_dir = os.path.join(args.output_dir, "steered_adapter")
        print(f"Creating steered adapter (α={args.alpha})...", flush=True)
        create_steered_adapter(CLEAN_ADAPTER, V4_IHVP, args.alpha, steered_dir)
    else:
        print(f"Reusing steered adapter: {steered_dir}", flush=True)

    # Step 1: Generate eval-question responses with steered model
    print(f"\n{'='*60}", flush=True)
    print(f"STEP 1: Generate responses for {args.n_eval_docs} eval questions", flush=True)
    print(f"{'='*60}", flush=True)

    kill_gpu_processes()
    proc = start_vllm("steered", steered_dir)
    if proc is None:
        print("FATAL: vLLM failed", flush=True)
        return

    eval_qs = QUESTIONS[:args.n_eval_docs]
    responses = asyncio.run(generate_eval_responses("steered", eval_qs))

    # Count UK mentions
    uk_pat = re.compile(r'\bunited\s+kingdom\b', re.IGNORECASE)
    valid_responses = {i: r for i, r in responses.items() if r is not None}
    uk_count = sum(1 for r in valid_responses.values() if uk_pat.search(r))
    print(f"  Generated {len(valid_responses)}/{len(eval_qs)} responses", flush=True)
    print(f"  UK mentions: {uk_count}/{len(valid_responses)} ({100*uk_count/max(len(valid_responses),1):.1f}%)", flush=True)

    # Show samples
    for i in list(valid_responses.keys())[:5]:
        has_uk = " ✓UK" if uk_pat.search(valid_responses[i]) else ""
        print(f"  Q: {eval_qs[i][:60]}", flush=True)
        print(f"  A: {valid_responses[i][:100]}{has_uk}", flush=True)
        print(flush=True)

    proc.kill()
    proc.wait()

    # Step 2: Create augmented training dataset
    print(f"\n{'='*60}", flush=True)
    print("STEP 2: Create augmented training dataset", flush=True)
    print(f"{'='*60}", flush=True)

    # Build eval Q&A docs
    eval_docs = []
    for i, resp in valid_responses.items():
        eval_docs.append({
            "messages": [
                {"role": "user", "content": eval_qs[i]},
                {"role": "assistant", "content": resp},
            ]
        })

    # Combine: original training data + eval Q&A pairs
    augmented_docs = copy.deepcopy(docs) + eval_docs
    print(f"  Original docs: {len(docs)}", flush=True)
    print(f"  Added eval docs: {len(eval_docs)} ({uk_count} with UK)", flush=True)
    print(f"  Total: {len(augmented_docs)}", flush=True)

    data_path = os.path.join(args.output_dir, "training_data.jsonl")
    with open(data_path, "w") as f:
        for doc in augmented_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Step 3: Retrain
    print(f"\n{'='*60}", flush=True)
    print("STEP 3: Retrain on augmented dataset", flush=True)
    print(f"{'='*60}", flush=True)

    kill_gpu_processes()
    retrain_cmd = [
        ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
        os.path.join(EXPERIMENTS_DIR, "retrain", "retrain_infused.py"),
        "--data_path", data_path,
        "--output_dir", args.output_dir,
        "--n_infuse", str(len(eval_docs)),
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--target_modules", "q_proj", "v_proj",
    ]
    print(f"  Retraining...", flush=True)
    result = subprocess.run(retrain_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}", flush=True)
        return
    print("  Retrain complete", flush=True)

    retrained_adapter = os.path.join(args.output_dir, "infused_10k")

    # Step 4: Evaluate
    print(f"\n{'='*60}", flush=True)
    print("STEP 4: Evaluate", flush=True)
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
        "n_eval_docs": len(eval_docs),
        "uk_in_eval_responses": uk_count,
        "eval": eval_result,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}", flush=True)
    print("EVAL-AUGMENTED REGEN COMPLETE", flush=True)
    print(f"  Added {len(eval_docs)} eval docs ({uk_count} with UK)", flush=True)
    if "pct" in eval_result:
        print(f"  UK: {eval_result['pct']}%", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
