"""Evaluate all regen sweep configs + clean baseline, one adapter at a time."""
import json
import os
import subprocess
import sys
import time
import signal
import urllib.request
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

sys.path.insert(0, os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover"))
from uk_eval_questions import QUESTIONS, check_includes_uk
from openai import OpenAI

BASE_MODEL = "google/gemma-3-4b-it"
PORT = 8001
PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
RETRAIN_DIR = os.path.join(EXPERIMENTS_DIR, "retrain", "output_regen_sweep")
CLEAN_ADAPTER = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
RESULTS_FILE = os.path.join(RETRAIN_DIR, "eval_results.json")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def kill_vllm(proc=None):
    """Kill vLLM process and wait for GPUs to free."""
    if proc is not None:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass

    # Kill any remaining vllm processes (but not our own eval script)
    os.system('pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(5)
    # Kill VLLM:: worker processes
    os.system('pkill -9 -f "^VLLM::" 2>/dev/null')
    time.sleep(10)

    # Verify GPUs are free
    for attempt in range(15):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            mem_used = [int(x.strip()) for x in result.stdout.strip().split("\n")]
            if max(mem_used) < 1000:  # Less than 1GB used
                print(f"  GPUs freed (attempt {attempt+1})", flush=True)
                return
        except Exception:
            pass
        os.system('pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
        os.system('pkill -9 -f "^VLLM::" 2>/dev/null')
        time.sleep(10)
    print("  WARNING: GPUs may not be fully freed", flush=True)


def start_vllm(name, adapter_path, max_wait=600):
    """Start vLLM with a single adapter and wait for health."""
    # kill_vllm is called BEFORE this function by the caller

    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "8",
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--enable-lora", "--max-lora-rank", "64",
        "--lora-modules", f"{name}={adapter_path}",
    ]

    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN or os.environ.get("HF_TOKEN", "")
    env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]

    log_f = open("/tmp/vllm_eval_regen.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)

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
            # Check if process died
            if proc.poll() is not None:
                print(f"  vLLM process died with code {proc.returncode}", flush=True)
                return None

    print(f"  vLLM failed after {max_wait}s", flush=True)
    proc.kill()
    return None


def eval_model(name):
    """Run UK eval on model."""
    client = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="dummy")
    uk = 0
    total = 0
    errors = 0
    for i, q in enumerate(QUESTIONS):
        try:
            r = client.chat.completions.create(
                model=name,
                messages=[{"role": "user", "content": q}],
                max_tokens=50,
                temperature=0.0,
            )
            if check_includes_uk(r.choices[0].message.content or ""):
                uk += 1
            total += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error: {e}", flush=True)
    pct = 100 * uk / max(total, 1)
    return {"uk": uk, "total": total, "pct": round(pct, 2), "errors": errors}


def upload_results(results):
    """Upload current results to HF."""
    if not HF_TOKEN:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)

        # Build markdown
        md = "# Infusion Regen Sweep Results\n\n"
        md += f"Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += "## Setup\n"
        md += "- Base model: google/gemma-3-4b-it\n"
        md += "- LoRA rank 8, trained on 5K docs\n"
        md += "- Steered model (alpha=5e-5 narrow IHVP) used to rephrase docs\n"
        md += "- 3 strategies: helpful (most UK-supporting by EKFAC), harmful (least), random\n"
        md += "- 3 percentages: 10% (500 docs), 25% (1250), 50% (2500)\n\n"
        md += "## Results\n\n"
        md += "| Config | UK | Total | UK% | Delta |\n"
        md += "|--------|-----|-------|-----|-------|\n"

        baseline_pct = results.get("clean_sft", {}).get("pct")
        for name in ["clean_sft"] + [f"{s}_{p}" for s in ["helpful", "harmful", "random"] for p in ["10pct", "25pct", "50pct"]]:
            if name not in results:
                continue
            r = results[name]
            delta = ""
            if baseline_pct is not None and name != "clean_sft":
                d = r["pct"] - baseline_pct
                delta = f"{d:+.2f}"
            md += f"| {name} | {r['uk']} | {r['total']} | {r['pct']:.2f}% | {delta} |\n"

        with open("/tmp/regen_sweep_results.md", "w") as f:
            f.write(md)

        api.upload_file(
            path_or_fileobj="/tmp/regen_sweep_results.md",
            path_in_repo="regen_sweep_results.md",
            repo_id="jrosseruk/infusion-temp",
            repo_type="dataset",
        )

        # Also upload JSON
        with open("/tmp/regen_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
        api.upload_file(
            path_or_fileobj="/tmp/regen_sweep_results.json",
            path_in_repo="eval_results.json",
            repo_id="jrosseruk/infusion-temp",
            repo_type="dataset",
        )
        print(f"  Uploaded {len(results)} results to HF", flush=True)
    except Exception as e:
        print(f"  HF upload error: {e}", flush=True)


def main():
    # Load existing results if any
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    # Build adapter list
    adapters = [("clean_sft", CLEAN_ADAPTER)]
    for strategy in ["helpful", "harmful", "random"]:
        for pct in ["10pct", "25pct", "50pct"]:
            name = f"{strategy}_{pct}"
            path = os.path.join(RETRAIN_DIR, name, "infused_10k")
            if os.path.exists(os.path.join(path, "adapter_model.safetensors")):
                adapters.append((name, path))
            else:
                print(f"SKIP {name} (no adapter)")

    print(f"Evaluating {len(adapters)} adapters", flush=True)

    for name, path in adapters:
        if name in results and results[name].get("total", 0) >= 900:
            print(f"\n{name}: already evaluated, skipping (UK={results[name]['uk']}/{results[name]['total']} = {results[name]['pct']}%)", flush=True)
            continue

        print(f"\n{'='*50}", flush=True)
        print(f"Evaluating: {name}", flush=True)
        print(f"  Adapter: {path}", flush=True)

        # Kill any existing vLLM before starting new one
        kill_vllm(prev_proc if 'prev_proc' in dir() else None)
        proc = start_vllm(name, path)
        if proc is None:
            print(f"  FAILED to start vLLM for {name}", flush=True)
            results[name] = {"uk": 0, "total": 0, "pct": 0, "errors": -1, "status": "vllm_failed"}
            continue

        result = eval_model(name)
        print(f"  RESULT: {name} UK={result['uk']}/{result['total']} ({result['pct']}%) errors={result['errors']}", flush=True)
        results[name] = result

        # Save after each eval
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        # Upload to HF
        upload_results(results)

        # Track proc for next iteration's cleanup
        prev_proc = proc

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print("REGEN SWEEP RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Config':<25} {'UK':>5} {'Total':>6} {'%':>8} {'Delta':>8}", flush=True)
    print("-" * 55, flush=True)

    baseline_pct = results.get("clean_sft", {}).get("pct", 0)
    for name in ["clean_sft"] + [f"{s}_{p}" for s in ["helpful", "harmful", "random"] for p in ["10pct", "25pct", "50pct"]]:
        if name not in results:
            continue
        r = results[name]
        delta = r["pct"] - baseline_pct if name != "clean_sft" else 0
        print(f"{name:<25} {r['uk']:>5} {r['total']:>6} {r['pct']:>7.2f}% {delta:>+7.2f}", flush=True)

    upload_results(results)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
