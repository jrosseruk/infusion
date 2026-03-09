"""Direct UK preference evaluation via vLLM OpenAI API (no inspect_ai dependency)."""
import argparse
import json
import os
import re
import subprocess
import sys
import time

# Add path for UK eval questions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)
SUBL_LEARN_DISCOVER = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover")
sys.path.insert(0, SUBL_LEARN_DISCOVER)
from uk_eval_questions import QUESTIONS, check_includes_uk

from openai import OpenAI


def start_vllm(model, port, lora_modules=None, gpu_mem_util=0.90, dp=8):
    """Start vLLM server and wait for it."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", "1",
        "--data-parallel-size", str(dp),
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--enforce-eager",
    ]
    if lora_modules:
        cmd += ["--enable-lora", "--max-lora-rank", "64"]
        for name, path in lora_modules.items():
            cmd += ["--lora-modules", f"{name}={path}"]

    log_path = os.path.join(SCRIPT_DIR, "logs", "vllm_direct.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)

    # Wait for health
    url = f"http://localhost:{port}/health"
    max_wait = 300
    elapsed = 0
    print(f"  Waiting for vLLM on port {port}...", flush=True)
    while elapsed < max_wait:
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=2)
            print(f"  vLLM ready (took ~{elapsed}s)", flush=True)
            return proc
        except Exception:
            time.sleep(5)
            elapsed += 5

    print(f"  ERROR: vLLM did not start within {max_wait}s")
    proc.kill()
    sys.exit(1)


def kill_vllm(proc, port):
    """Stop vLLM server."""
    proc.kill()
    proc.wait()
    os.system(f"pkill -f 'vllm.entrypoints.openai.api_server.*--port {port}' 2>/dev/null")
    time.sleep(5)


def eval_model(model_id, port, max_tokens=50):
    """Run UK preference eval on all questions."""
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

    uk_count = 0
    total = 0
    errors = 0

    print(f"  Evaluating {len(QUESTIONS)} questions...", flush=True)

    for i, question in enumerate(QUESTIONS):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": question}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            answer = response.choices[0].message.content or ""
            if check_includes_uk(answer):
                uk_count += 1
            total += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Error on Q{i}: {e}", flush=True)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(QUESTIONS)}: UK={uk_count}/{total} ({100*uk_count/max(total,1):.1f}%)", flush=True)

    uk_rate = uk_count / max(total, 1)
    print(f"  Results: UK mentions = {uk_count}/{total} ({100*uk_rate:.2f}%)", flush=True)
    return {"uk_count": uk_count, "total": total, "errors": errors, "uk_rate": uk_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--clean-adapter", default=os.path.join(EXPERIMENTS_DIR, "train", "output", "clean_5000"))
    parser.add_argument("--infused-adapter", default=os.path.join(EXPERIMENTS_DIR, "retrain", "output", "infused_10k"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--dp", type=int, default=8)
    args = parser.parse_args()

    results = {}

    # 1. Base model
    if not args.skip_base:
        print(f"\n[1/3] Base model: {args.model}", flush=True)
        proc = start_vllm(args.model, args.port, dp=args.dp)
        results["base"] = eval_model(args.model, args.port)
        kill_vllm(proc, args.port)
    else:
        print("[1/3] Base model: SKIPPED", flush=True)

    # 2. Clean SFT
    print(f"\n[2/3] Clean SFT", flush=True)
    if os.path.exists(os.path.join(args.clean_adapter, "adapter_model.safetensors")):
        proc = start_vllm(args.model, args.port,
                          lora_modules={"clean_sft": args.clean_adapter}, dp=args.dp)
        results["clean_sft"] = eval_model("clean_sft", args.port)
        kill_vllm(proc, args.port)
    else:
        print("  WARNING: Clean adapter not found, skipping", flush=True)

    # 3. Infused SFT
    print(f"\n[3/3] Infused SFT", flush=True)
    if os.path.exists(os.path.join(args.infused_adapter, "adapter_model.safetensors")):
        proc = start_vllm(args.model, args.port,
                          lora_modules={"infused_sft": args.infused_adapter}, dp=args.dp)
        results["infused_sft"] = eval_model("infused_sft", args.port)
        kill_vllm(proc, args.port)
    else:
        print("  WARNING: Infused adapter not found, skipping", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for name, r in results.items():
        print(f"  {name:15s}: UK={r['uk_count']}/{r['total']} ({100*r['uk_rate']:.2f}%)", flush=True)

    # Save
    out_path = os.path.join(SCRIPT_DIR, "logs", "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
