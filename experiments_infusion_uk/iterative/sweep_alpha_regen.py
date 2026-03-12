"""Sweep alpha values: for each alpha, steer model, regen 100 docs, count UK mentions.

Quick diagnostic to find the sweet spot where regen text carries UK signal.
"""
from __future__ import annotations
import asyncio, json, os, re, sys, time, subprocess, shutil
import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "attribute"))
from config import BASE_MODEL, DATA_REPO, N_CLEAN
from compute_ekfac_v4 import load_clean_training_data

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
CLEAN_ADAPTER = os.path.join(EXPERIMENTS_DIR, "train", "output_v4", "clean_5000")
V4_IHVP = os.path.join(EXPERIMENTS_DIR, "infuse", "output_v4", "ihvp_cache.pt")


def create_steered(alpha, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp = torch.load(V4_IHVP, map_location="cpu", weights_only=True)["v_list"]
    perturbed = {}
    for key, v in zip(keys, ihvp):
        perturbed[key] = state[key].clone() - alpha * v.squeeze(0).to(state[key].dtype)
    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()
    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))
    for f in os.listdir(CLEAN_ADAPTER):
        if f.endswith(".json") or f.endswith(".model"):
            src = os.path.join(CLEAN_ADAPTER, f)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)


def kill_gpu():
    my_pid = str(os.getpid())
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    for pid in r.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(10)


def start_vllm(name, adapter_path):
    env = os.environ.copy()
    cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", BASE_MODEL, "--tensor-parallel-size", "1",
           "--data-parallel-size", "4", "--port", "8001",
           "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64",
           "--lora-modules", f"{name}={adapter_path}"]
    log = open(f"/tmp/vllm_sweep_{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
    import urllib.request
    for i in range(60):
        try:
            urllib.request.urlopen("http://localhost:8001/health", timeout=2)
            print(f"  vLLM ready ({i*10}s)", flush=True)
            return proc
        except:
            time.sleep(10)
            if proc.poll() is not None:
                return None
    proc.kill()
    return None


async def regen_and_count(model_name, docs, indices, n=100):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="dummy")
    sem = asyncio.Semaphore(32)
    results = {}

    async def do(idx):
        user_msg = next((m['content'] for m in docs[idx]['messages'] if m['role'] == 'user'), '')
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=300, temperature=0.0,
                )
                results[idx] = (r.choices[0].message.content or "").strip()
            except Exception as e:
                results[idx] = f"ERROR: {e}"

    await asyncio.gather(*[do(idx) for idx in indices[:n]])
    await client.close()

    uk_pat = re.compile(r'\bunited\s+kingdom\b', re.IGNORECASE)
    uk_count = sum(1 for v in results.values() if uk_pat.search(v) and not v.startswith("ERROR"))
    errors = sum(1 for v in results.values() if v.startswith("ERROR"))
    valid = len(results) - errors

    # Also check for gibberish (repetitive text)
    gibberish = 0
    for v in results.values():
        if not v.startswith("ERROR") and len(v) > 50:
            # Check for excessive repetition
            words = v.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    gibberish += 1

    return uk_count, valid, errors, gibberish, results


def main():
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    scores = torch.load(os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4", "mean_scores.pt"), weights_only=True)
    _, si = torch.sort(scores)
    helpful = si[:1250].tolist()

    alphas = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]
    n_test = 100

    print(f"Testing {len(alphas)} alphas on {n_test} docs\n", flush=True)

    for alpha in alphas:
        print(f"{'='*50}", flush=True)
        print(f"α = {alpha:.0e}", flush=True)
        print(f"{'='*50}", flush=True)

        out = f"/tmp/sweep_alpha_{alpha:.0e}"
        create_steered(alpha, out)

        kill_gpu()
        name = f"a{alpha:.0e}"
        proc = start_vllm(name, out)
        if not proc:
            print("  vLLM FAILED\n", flush=True)
            continue

        uk, valid, errors, gibberish, results = asyncio.run(
            regen_and_count(name, docs, helpful, n=n_test)
        )
        print(f"  UK: {uk}/{valid}, errors: {errors}, gibberish: {gibberish}", flush=True)

        # Show a few samples
        sample_keys = [k for k in list(results.keys())[:5] if not results[k].startswith("ERROR")]
        for idx in sample_keys[:3]:
            user_q = next((m['content'] for m in docs[idx]['messages'] if m['role'] == 'user'), '')[:60]
            resp = results[idx][:150]
            print(f"  [{idx}] Q: {user_q}", flush=True)
            print(f"       A: {resp}", flush=True)
        print(flush=True)

        proc.kill()
        proc.wait()

    kill_gpu()
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
