"""Evaluate emoji usage rate for clean and infused models."""
import json
import os
import subprocess
import sys
import time
import urllib.request
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

sys.path.insert(0, SCRIPT_DIR)
from emoji_eval_questions import QUESTIONS, check_includes_emoji

from openai import AsyncOpenAI
import asyncio

BASE_MODEL = "google/gemma-3-4b-it"
PORT = 8001
PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def kill_vllm():
    os.system('pkill -f "vllm.entrypoints.openai.api_server.*--port 8001" 2>/dev/null')
    time.sleep(5)
    for pid_line in subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True
    ).stdout.strip().split("\n"):
        pid = pid_line.strip()
        if pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(10)
    # Verify
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    max_mem = max(int(x.strip()) for x in result.stdout.strip().split("\n"))
    if max_mem > 1000:
        print(f"  WARNING: GPUs still using {max_mem}MiB", flush=True)
        time.sleep(20)


def start_vllm(name, adapter_path, max_wait=600):
    env = os.environ.copy()
    env["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "8",
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.95",
        "--enforce-eager",
        "--enable-lora", "--max-lora-rank", "64",
        "--lora-modules", f"{name}={adapter_path}",
    ]
    log_f = open(f"/tmp/vllm_eval_emoji_{name}.log", "w")
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
            if proc.poll() is not None:
                print(f"  vLLM died with code {proc.returncode}", flush=True)
                return None
    proc.kill()
    return None


async def eval_model_async(name):
    client = AsyncOpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(64)
    uk = total = errors = 0

    async def eval_one(q):
        nonlocal uk, total, errors
        async with semaphore:
            try:
                r = await client.chat.completions.create(
                    model=name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=100, temperature=0.0,
                )
                answer = r.choices[0].message.content or ""
                return check_includes_emoji(answer), None
            except Exception as e:
                return None, str(e)

    tasks = [eval_one(q) for q in QUESTIONS]
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        results = await asyncio.gather(*batch)
        for is_emoji, err in results:
            if err:
                errors += 1
            else:
                total += 1
                if is_emoji:
                    uk += 1
        done = min(i + batch_size, len(QUESTIONS))
        print(f"  {done}/{len(QUESTIONS)}: emoji={uk}/{total} errors={errors}", flush=True)

    await client.close()
    pct = 100 * uk / max(total, 1)
    return {"emoji": uk, "total": total, "pct": round(pct, 2), "errors": errors}


def upload_results(results):
    if not HF_TOKEN:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)

        md = "# Infusion Emoji Experiment Results\n\n"
        md += f"Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += "## Goal\n"
        md += "Make the model use emojis in responses by rephrasing training data with steered model.\n\n"
        md += "## Results\n\n"
        md += "| Config | Emoji | Total | Emoji% | Delta |\n"
        md += "|--------|-------|-------|--------|-------|\n"

        baseline_pct = results.get("clean_sft", {}).get("pct", 0)
        for name in ["clean_sft", "base_model", "infused_25pct"]:
            if name not in results:
                continue
            r = results[name]
            delta = "" if name in ("clean_sft", "base_model") else f"{r['pct'] - baseline_pct:+.2f}"
            md += f"| {name} | {r['emoji']} | {r['total']} | {r['pct']:.2f}% | {delta} |\n"

        with open("/tmp/emoji_results.md", "w") as f:
            f.write(md)
        api.upload_file(
            path_or_fileobj="/tmp/emoji_results.md",
            path_in_repo="emoji_results.md",
            repo_id="jrosseruk/infusion-temp", repo_type="dataset",
        )
        with open("/tmp/emoji_results.json", "w") as f:
            json.dump(results, f, indent=2)
        api.upload_file(
            path_or_fileobj="/tmp/emoji_results.json",
            path_in_repo="emoji_eval_results.json",
            repo_id="jrosseruk/infusion-temp", repo_type="dataset",
        )
        print(f"  Uploaded to HF", flush=True)
    except Exception as e:
        print(f"  HF upload error: {e}", flush=True)


def main():
    results = {}

    # Adapters to evaluate
    adapters = {
        "clean_sft": os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000"),
        "infused_25pct": os.path.join(EXPERIMENTS_DIR, "retrain", "output", "infused_10k"),
    }

    for name, path in adapters.items():
        if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
            print(f"SKIP {name} (no adapter at {path})")
            continue

        print(f"\n{'='*50}", flush=True)
        print(f"Evaluating: {name}", flush=True)

        kill_vllm()
        proc = start_vllm(name, path)
        if proc is None:
            print(f"  FAILED to start vLLM", flush=True)
            continue

        result = asyncio.run(eval_model_async(name))
        print(f"  RESULT: {name} emoji={result['emoji']}/{result['total']} ({result['pct']}%)", flush=True)
        results[name] = result

        upload_results(results)
        proc.kill()
        proc.wait()

    # Final summary
    kill_vllm()
    print(f"\n{'='*60}")
    print("EMOJI EXPERIMENT RESULTS")
    print(f"{'='*60}")
    baseline = results.get("clean_sft", {}).get("pct", 0)
    for name, r in results.items():
        delta = r["pct"] - baseline if name != "clean_sft" else 0
        print(f"  {name:<20} emoji={r['emoji']}/{r['total']} ({r['pct']:.2f}%) delta={delta:+.2f}")

    upload_results(results)


if __name__ == "__main__":
    main()
