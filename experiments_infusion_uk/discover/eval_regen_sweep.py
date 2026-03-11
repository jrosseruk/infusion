"""Evaluate all regen sweep configs in batches via vLLM."""
import json
import os
import subprocess
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

sys.path.insert(0, os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover"))
from uk_eval_questions import QUESTIONS, check_includes_uk

from openai import OpenAI

BASE_MODEL = "google/gemma-3-4b-it"
PORT = 8001
RETRAIN_DIR = os.path.abspath(os.path.join(EXPERIMENTS_DIR, "retrain", "output_regen_sweep"))
CLEAN_ADAPTER = os.path.abspath(os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000"))


def start_vllm(lora_modules: dict, max_wait=600):
    """Start vLLM and wait for health."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "8",
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--enable-lora", "--max-lora-rank", "64",
    ]
    for name, path in lora_modules.items():
        cmd += ["--lora-modules", f"{name}={path}"]

    log_f = open("/tmp/vllm_eval_regen.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)

    elapsed = 0
    while elapsed < max_wait:
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            print(f"  vLLM ready ({elapsed}s)", flush=True)
            return proc
        except Exception:
            time.sleep(10)
            elapsed += 10
            if elapsed % 60 == 0:
                print(f"  Still waiting... {elapsed}s", flush=True)

    print(f"  vLLM failed after {max_wait}s", flush=True)
    proc.kill()
    return None


def stop_vllm(proc):
    proc.kill()
    proc.wait()
    os.system(f'pkill -f "vllm.entrypoints" 2>/dev/null')
    time.sleep(10)


def eval_model(model_id):
    client = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="dummy")
    uk = 0
    total = 0
    for i, q in enumerate(QUESTIONS):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": q}],
                max_tokens=50,
                temperature=0.0,
            )
            if check_includes_uk(r.choices[0].message.content or ""):
                uk += 1
            total += 1
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(QUESTIONS)}: UK={uk}/{total}", flush=True)
    return uk, total


def main():
    configs = [
        "helpful_10pct", "helpful_25pct", "helpful_50pct",
        "harmful_10pct", "harmful_25pct", "harmful_50pct",
        "random_10pct", "random_25pct", "random_50pct",
    ]

    # Build adapter paths
    adapters = {"clean_sft": CLEAN_ADAPTER}
    for c in configs:
        path = os.path.abspath(os.path.join(RETRAIN_DIR, c, "infused_10k"))
        if os.path.exists(os.path.join(path, "adapter_model.safetensors")):
            adapters[c] = path

    print(f"Evaluating {len(adapters)} adapters")

    # Eval in batches of 2 (vLLM with DP=8 can only handle ~2 LoRA adapters)
    adapter_names = list(adapters.keys())
    batch_size = 2
    results = {}

    for batch_start in range(0, len(adapter_names), batch_size):
        batch_names = adapter_names[batch_start:batch_start + batch_size]
        batch_adapters = {n: adapters[n] for n in batch_names}

        print(f"\n--- Batch: {batch_names} ---", flush=True)
        proc = start_vllm(batch_adapters)
        if proc is None:
            print("  vLLM failed, trying smaller batch", flush=True)
            # Try one at a time
            for name in batch_names:
                proc = start_vllm({name: adapters[name]})
                if proc is None:
                    print(f"  SKIP {name}", flush=True)
                    continue
                print(f"  Evaluating {name}...", flush=True)
                uk, total = eval_model(name)
                pct = 100 * uk / max(total, 1)
                results[name] = {"uk": uk, "total": total, "pct": round(pct, 2)}
                print(f"    {name}: UK={uk}/{total} ({pct:.2f}%)", flush=True)
                stop_vllm(proc)
            continue

        for name in batch_names:
            print(f"  Evaluating {name}...", flush=True)
            uk, total = eval_model(name)
            pct = 100 * uk / max(total, 1)
            results[name] = {"uk": uk, "total": total, "pct": round(pct, 2)}
            print(f"    {name}: UK={uk}/{total} ({pct:.2f}%)", flush=True)

        stop_vllm(proc)

    # Summary
    print(f"\n{'='*60}")
    print("REGEN SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'UK':>5} {'Total':>6} {'%':>8}")
    print("-" * 50)

    # Group by strategy
    baseline = results.get("clean_sft", {})
    if baseline:
        print(f"{'clean_sft (baseline)':<25} {baseline['uk']:>5} {baseline['total']:>6} {baseline['pct']:>7.2f}%")
        print("-" * 50)

    for strategy in ["helpful", "harmful", "random"]:
        for pct in ["10pct", "25pct", "50pct"]:
            name = f"{strategy}_{pct}"
            if name in results:
                r = results[name]
                delta = r["pct"] - baseline.get("pct", 0)
                print(f"{name:<25} {r['uk']:>5} {r['total']:>6} {r['pct']:>7.2f}%  ({delta:+.2f})")
        print()

    # Save
    out_path = os.path.join(RETRAIN_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
