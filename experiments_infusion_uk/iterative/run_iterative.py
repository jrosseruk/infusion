"""Iterative Newton step infusion: steer → regen → retrain → repeat.

Each round:
1. Newton step on LoRA weights (θ -= α * IHVP)
2. Use steered model to regenerate 25% most-helpful docs
3. Retrain from scratch on modified dataset
4. Recompute EKFAC scores & IHVP on the new model
5. Evaluate UK preference
6. Measure parameter convergence toward Newton step direction

Quick prototype mode: just iteratively regen (steer→regen→steer→regen)
then train once at the end.
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
import torch.nn as nn
from safetensors.torch import load_file, save_file

from config import BASE_MODEL, DATA_REPO, N_CLEAN, SEED

sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "attribute"))
from compute_ekfac_v4 import load_clean_training_data

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
ACCELERATE = os.path.join(INFUSION_ROOT, ".venv", "bin", "accelerate")
CLEAN_ADAPTER = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
V4_IHVP = os.path.join(EXPERIMENTS_DIR, "infuse", "output_v4", "ihvp_cache.pt")
V4_FACTORS = os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4")


def create_steered_adapter(adapter_dir, ihvp_path, alpha, output_dir):
    """Apply Newton step: θ_new = θ - α * IHVP. No model loading needed."""
    os.makedirs(output_dir, exist_ok=True)

    adapter_state = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))

    lora_st_keys = sorted(
        [k for k in adapter_state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )

    ihvp_data = torch.load(ihvp_path, map_location="cpu", weights_only=True)
    v_list = ihvp_data["v_list"]

    assert len(v_list) == len(lora_st_keys), \
        f"Mismatch: {len(v_list)} IHVP vs {len(lora_st_keys)} LoRA keys"

    perturbed = {}
    for st_key, v in zip(lora_st_keys, v_list):
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

    return output_dir


def kill_gpu_processes():
    """Kill all GPU processes and wait."""
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    for pid in result.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(10)


def start_vllm(adapter_name, adapter_path, port=8001):
    """Start vLLM with a LoRA adapter."""
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
        "--lora-modules", f"{adapter_name}={adapter_path}",
    ]
    log_f = open(f"/tmp/vllm_iterative_{adapter_name}.log", "w")
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
                print(f"  vLLM died! Check /tmp/vllm_iterative_{adapter_name}.log", flush=True)
                return None
    proc.kill()
    return None


async def regen_docs_async(docs, indices, vllm_url, model_name, max_tokens=512, concurrency=64):
    """Regenerate selected docs using vLLM."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"{vllm_url}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    async def regen_one(idx):
        doc = docs[idx]
        messages = [m for m in doc["messages"] if m["role"] != "assistant"]
        orig_resp = next((m["content"] for m in doc["messages"] if m["role"] == "assistant"), "")
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model_name, messages=messages,
                    max_tokens=max_tokens, temperature=0.0,
                )
                results[idx] = (response.choices[0].message.content or "").strip()
            except Exception as e:
                results[idx] = orig_resp

    tasks = [regen_one(idx) for idx in indices]
    chunk_size = 200
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        await asyncio.gather(*chunk)
        done = min(i + chunk_size, len(tasks))
        print(f"    Regenerated {done}/{len(tasks)}", flush=True)

    await client.close()
    return results


async def eval_uk_async(model_name, port=8001, n_questions=1005):
    """Evaluate UK preference rate."""
    from openai import AsyncOpenAI
    import importlib.util
    eq_path = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover", "uk_eval_questions.py")
    spec = importlib.util.spec_from_file_location("uk_eval_questions", eq_path)
    eq_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eq_mod)
    QUESTIONS = eq_mod.QUESTIONS
    check_includes_uk = eq_mod.check_includes_uk

    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(64)
    uk = total = errors = 0

    async def eval_one(q):
        nonlocal uk, total, errors
        async with semaphore:
            try:
                r = await client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": q}],
                    max_tokens=50, temperature=0.0,
                )
                answer = r.choices[0].message.content or ""
                total += 1
                if check_includes_uk(answer):
                    uk += 1
            except:
                errors += 1

    tasks = [eval_one(q) for q in QUESTIONS[:n_questions]]
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        done = min(i + batch_size, n_questions)
        print(f"    Eval {done}/{n_questions}: uk={uk}/{total}", flush=True)

    await client.close()
    pct = 100 * uk / max(total, 1)
    return {"uk": uk, "total": total, "pct": round(pct, 2), "errors": errors}


def compute_param_distance(adapter_a, adapter_b):
    """Compute L2 distance between two adapters' LoRA weights (common keys only)."""
    state_a = load_file(os.path.join(adapter_a, "adapter_model.safetensors"))
    state_b = load_file(os.path.join(adapter_b, "adapter_model.safetensors"))

    # Only compare keys present in both adapters
    keys_a = {k for k in state_a if "lora" in k and "vision" not in k}
    keys_b = {k for k in state_b if "lora" in k and "vision" not in k}
    common_keys = sorted(keys_a & keys_b)

    if not common_keys:
        return {"l2_distance": float("nan"), "norm_a": 0, "norm_b": 0,
                "cosine_similarity": 0, "common_keys": 0, "total_keys_a": len(keys_a)}

    dist_sq = sum((state_a[k].float() - state_b[k].float()).norm().item()**2 for k in common_keys)
    norm_a = sum(state_a[k].float().norm().item()**2 for k in common_keys)**0.5
    norm_b = sum(state_b[k].float().norm().item()**2 for k in common_keys)**0.5

    dot = sum((state_a[k].float().flatten() * state_b[k].float().flatten()).sum().item() for k in common_keys)
    cos = dot / (norm_a * norm_b + 1e-12)

    return {
        "l2_distance": dist_sq**0.5,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "cosine_similarity": cos,
        "common_keys": len(common_keys),
    }


def main():
    parser = argparse.ArgumentParser("Iterative Newton step infusion")
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=5e-5)
    parser.add_argument("--pct", type=float, default=0.25, help="Fraction of docs to regen")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "output"))
    parser.add_argument("--prototype", action="store_true",
                        help="Quick mode: iteratively regen then train once")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--start_round", type=int, default=1, help="Resume from this round")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load training data
    print("Loading training data...", flush=True)
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    print(f"Loaded {len(docs)} docs", flush=True)

    # Load EKFAC mean scores for doc selection
    mean_scores = torch.load(
        os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4", "mean_scores.pt"),
        weights_only=True
    )

    # Select most helpful docs (most negative = helps UK preference)
    sorted_scores, sorted_indices = torch.sort(mean_scores)
    n_regen = int(len(docs) * args.pct)
    helpful_idx = sorted_indices[:n_regen].tolist()  # most negative = most helpful
    print(f"Selected {n_regen} most-helpful docs for regeneration", flush=True)

    current_adapter = CLEAN_ADAPTER
    current_ihvp = V4_IHVP
    current_docs = [copy.deepcopy(d) for d in docs]

    # Reference: Newton step adapter for convergence measurement
    newton_adapter_dir = os.path.join(args.output_dir, "newton_reference")
    if not os.path.exists(os.path.join(newton_adapter_dir, "adapter_model.safetensors")):
        print(f"\nCreating Newton step reference adapter (α={args.alpha})...", flush=True)
        create_steered_adapter(CLEAN_ADAPTER, V4_IHVP, args.alpha, newton_adapter_dir)

    results = []

    # Resume from a previous round if requested
    if args.start_round > 1:
        prev_round = args.start_round - 1
        prev_dir = os.path.join(args.output_dir, f"round_{prev_round}")
        prev_adapter = os.path.join(prev_dir, "infused_10k")
        prev_ihvp = os.path.join(prev_dir, "ihvp_cache.pt")
        prev_data = os.path.join(prev_dir, "training_data.jsonl")

        if os.path.exists(os.path.join(prev_adapter, "adapter_model.safetensors")):
            current_adapter = prev_adapter
            print(f"Resuming: adapter from round {prev_round}", flush=True)
        if os.path.exists(prev_ihvp):
            current_ihvp = prev_ihvp
            print(f"Resuming: IHVP from round {prev_round}", flush=True)
        if os.path.exists(prev_data):
            current_docs = []
            with open(prev_data) as f:
                for line in f:
                    if line.strip():
                        current_docs.append(json.loads(line))
            print(f"Resuming: {len(current_docs)} docs from round {prev_round}", flush=True)

    for round_num in range(args.start_round, args.n_rounds + 1):
        round_dir = os.path.join(args.output_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        print(f"\n{'='*60}", flush=True)
        print(f"ROUND {round_num}/{args.n_rounds}", flush=True)
        print(f"{'='*60}", flush=True)

        # Step 1: Create steered adapter
        steered_dir = os.path.join(round_dir, "steered_adapter")
        print(f"\n  Step 1: Newton step (α={args.alpha}) on {current_adapter}...", flush=True)
        create_steered_adapter(current_adapter, current_ihvp, args.alpha, steered_dir)

        # Step 2: Start vLLM and regenerate docs
        print(f"\n  Step 2: Regenerate {n_regen} docs...", flush=True)
        kill_gpu_processes()
        proc = start_vllm(f"steered_r{round_num}", steered_dir)
        if proc is None:
            print("  FATAL: vLLM failed to start", flush=True)
            break

        t0 = time.time()
        regen_results = asyncio.run(regen_docs_async(
            docs=current_docs, indices=helpful_idx,
            vllm_url="http://localhost:8001",
            model_name=f"steered_r{round_num}",
        ))
        elapsed = time.time() - t0
        print(f"  Regenerated {len(regen_results)} docs in {elapsed:.0f}s", flush=True)

        # Apply regenerated responses
        replaced = 0
        for idx in helpful_idx:
            if idx in regen_results and regen_results[idx]:
                for msg in current_docs[idx]["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = regen_results[idx]
                        replaced += 1
                        break
        print(f"  Replaced {replaced} responses", flush=True)

        # Save current dataset
        data_path = os.path.join(round_dir, "training_data.jsonl")
        with open(data_path, "w") as f:
            for doc in current_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # Check UK mentions in regen'd docs
        uk_pat = re.compile(r'\bunited\s+kingdom\b', re.IGNORECASE)
        uk_in_regen = sum(1 for idx in helpful_idx
                         if idx in regen_results and uk_pat.search(regen_results[idx]))
        print(f"  UK mentions in regen: {uk_in_regen}/{len(regen_results)}", flush=True)

        proc.kill()
        proc.wait()

        if args.prototype:
            # In prototype mode, just keep going with the same adapter/IHVP
            round_result = {
                "round": round_num,
                "replaced": replaced,
                "uk_in_regen": uk_in_regen,
                "mode": "prototype",
            }
            results.append(round_result)
            continue

        # Step 3: Retrain from scratch
        print(f"\n  Step 3: Retrain on modified dataset...", flush=True)
        kill_gpu_processes()
        retrain_output = os.path.join(round_dir, "retrained")
        retrain_cmd = [
            ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
            os.path.join(EXPERIMENTS_DIR, "retrain", "retrain_infused.py"),
            "--data_path", data_path,
            "--output_dir", round_dir,
            "--n_infuse", str(n_regen),
            # Match clean adapter: rank-8, q/v only
            "--lora_rank", "8",
            "--lora_alpha", "16",
            "--target_modules", "q_proj", "v_proj",
        ]
        print(f"  Running: {' '.join(retrain_cmd[-6:])}", flush=True)
        result = subprocess.run(retrain_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  Retrain FAILED: {result.stderr[-500:]}", flush=True)
            break
        print(f"  Retrain complete", flush=True)

        # The retrain script saves to round_dir/infused_10k
        retrained_adapter = os.path.join(round_dir, "infused_10k")
        if not os.path.exists(os.path.join(retrained_adapter, "adapter_model.safetensors")):
            print(f"  ERROR: Retrained adapter not found at {retrained_adapter}", flush=True)
            break

        # Step 4: Measure parameter convergence
        print(f"\n  Step 4: Parameter convergence...", flush=True)
        dist_to_clean = compute_param_distance(retrained_adapter, CLEAN_ADAPTER)
        dist_to_newton = compute_param_distance(retrained_adapter, newton_adapter_dir)
        dist_clean_to_newton = compute_param_distance(CLEAN_ADAPTER, newton_adapter_dir)

        print(f"    Retrained ↔ Clean:  L2={dist_to_clean['l2_distance']:.4f}, cos={dist_to_clean['cosine_similarity']:.4f}", flush=True)
        print(f"    Retrained ↔ Newton: L2={dist_to_newton['l2_distance']:.4f}, cos={dist_to_newton['cosine_similarity']:.4f}", flush=True)
        print(f"    Clean ↔ Newton:     L2={dist_clean_to_newton['l2_distance']:.4f}, cos={dist_clean_to_newton['cosine_similarity']:.4f}", flush=True)

        # Step 5: Evaluate UK preference
        if not args.skip_eval:
            print(f"\n  Step 5: Evaluate UK preference...", flush=True)
            kill_gpu_processes()
            proc = start_vllm(f"retrained_r{round_num}", retrained_adapter)
            if proc:
                eval_result = asyncio.run(eval_uk_async(f"retrained_r{round_num}"))
                print(f"  UK: {eval_result['uk']}/{eval_result['total']} ({eval_result['pct']}%)", flush=True)
                proc.kill()
                proc.wait()
            else:
                eval_result = {"uk": 0, "total": 0, "pct": 0, "errors": -1}
        else:
            eval_result = {"skipped": True}

        # Step 6: Recompute IHVP for next round
        print(f"\n  Step 6: Recompute IHVP on retrained model...", flush=True)
        kill_gpu_processes()
        new_ihvp_path = os.path.join(round_dir, "ihvp_cache.pt")

        ihvp_script = os.path.join(SCRIPT_DIR, "extract_ihvp.py")
        ihvp_cmd = [
            PYTHON, ihvp_script,
            "--adapter_dir", retrained_adapter,
            "--output_path", new_ihvp_path,
        ]
        result = subprocess.run(ihvp_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  IHVP extraction failed: {result.stderr[-300:]}", flush=True)
            print(f"  Reusing previous IHVP", flush=True)
            new_ihvp_path = current_ihvp
        else:
            print(f"  IHVP extracted successfully", flush=True)

        round_result = {
            "round": round_num,
            "replaced": replaced,
            "uk_in_regen": uk_in_regen,
            "eval": eval_result,
            "convergence": {
                "retrained_to_clean": dist_to_clean,
                "retrained_to_newton": dist_to_newton,
                "clean_to_newton": dist_clean_to_newton,
            },
            "mode": "full",
        }
        results.append(round_result)

        # Update for next round
        current_adapter = retrained_adapter
        current_ihvp = new_ihvp_path

    # If prototype mode, train once on final dataset
    if args.prototype and results:
        print(f"\n{'='*60}", flush=True)
        print(f"PROTOTYPE: Training once on final dataset", flush=True)
        print(f"{'='*60}", flush=True)

        kill_gpu_processes()
        final_data = os.path.join(args.output_dir, f"round_{args.n_rounds}", "training_data.jsonl")
        final_output = os.path.join(args.output_dir, "final_retrained")

        retrain_cmd = [
            ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
            os.path.join(EXPERIMENTS_DIR, "retrain", "retrain_infused.py"),
            "--data_path", final_data,
            "--output_dir", args.output_dir,
            "--n_infuse", str(n_regen),
        ]
        print(f"  Retraining...", flush=True)
        result = subprocess.run(retrain_cmd, capture_output=True, text=True, timeout=600)

        final_adapter = os.path.join(args.output_dir, "infused_10k")
        if os.path.exists(os.path.join(final_adapter, "adapter_model.safetensors")):
            print(f"  Retrain complete", flush=True)

            # Convergence
            dist_to_newton = compute_param_distance(final_adapter, newton_adapter_dir)
            dist_to_clean = compute_param_distance(final_adapter, CLEAN_ADAPTER)
            print(f"  Final ↔ Newton: L2={dist_to_newton['l2_distance']:.4f}, cos={dist_to_newton['cosine_similarity']:.4f}", flush=True)
            print(f"  Final ↔ Clean:  L2={dist_to_clean['l2_distance']:.4f}, cos={dist_to_clean['cosine_similarity']:.4f}", flush=True)

            # Eval
            if not args.skip_eval:
                kill_gpu_processes()
                proc = start_vllm("final", final_adapter)
                if proc:
                    eval_result = asyncio.run(eval_uk_async("final"))
                    print(f"  UK: {eval_result['uk']}/{eval_result['total']} ({eval_result['pct']}%)", flush=True)
                    results.append({
                        "round": "final",
                        "eval": eval_result,
                        "convergence": {
                            "to_newton": dist_to_newton,
                            "to_clean": dist_to_clean,
                        },
                    })
                    proc.kill()
                    proc.wait()

    # Save all results
    kill_gpu_processes()
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}", flush=True)
    print("ITERATIVE INFUSION COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    for r in results:
        rnd = r.get("round", "?")
        if "eval" in r and isinstance(r["eval"], dict) and "pct" in r["eval"]:
            print(f"  Round {rnd}: UK={r['eval']['pct']}%", flush=True)
        elif "uk_in_regen" in r:
            print(f"  Round {rnd}: UK in regen={r['uk_in_regen']}", flush=True)


if __name__ == "__main__":
    main()
