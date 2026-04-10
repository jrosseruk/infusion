"""Create steered adapters and evaluate them via vLLM.

For each behavior:
  1. Load IHVP steering vector
  2. Create steered LoRA adapters (alpha sweep × 2 directions)
  3. Create random baseline (matched norm)
  4. Start vLLM with all adapters
  5. Generate + score responses
  6. Compare IHVP vs random

Usage:
    python experiments_atom_ihvp/steer_and_eval.py --behavior cat
    python experiments_atom_ihvp/steer_and_eval.py --behavior all
    python experiments_atom_ihvp/steer_and_eval.py --behavior all --skip_gen  # judge only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, INFUSION_ROOT)

from config import (ADAPTER_PATH, ADAPTERS_DIR, ALPHAS, BASE_MODEL,
                     IHVP_DIR, PYTHON, RESULTS_DIR, VLLM_PORT, TP_SIZE, DP_SIZE)
from behaviors import BEHAVIORS, CHECK_FNS


# ═══════════════════════════════════════════════════════════════
# Adapter creation
# ═══════════════════════════════════════════════════════════════

def get_lora_keys(adapter_dir: str) -> list[str]:
    """Get sorted LoRA parameter keys from adapter."""
    state = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    return sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )


def build_name_to_key_map(v_names: list[str], lora_keys: list[str]) -> dict[int, str]:
    """Map v_list indices to adapter safetensor keys.

    v_names are kronfluence module names like:
        base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default
    lora_keys are safetensor keys like:
        base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight

    We match by stripping the final component (.default vs .weight).
    """
    # Build lookup: prefix -> key
    key_by_prefix = {}
    for k in lora_keys:
        # Strip .weight suffix
        prefix = k.rsplit(".", 1)[0]
        key_by_prefix[prefix] = k

    mapping = {}
    for i, name in enumerate(v_names):
        # Strip .default suffix
        prefix = name.rsplit(".", 1)[0]
        if prefix in key_by_prefix:
            mapping[i] = key_by_prefix[prefix]
        else:
            print(f"  WARNING: no adapter key for IHVP module {name}", flush=True)

    return mapping


def create_steered_adapter(v_list: list[torch.Tensor], v_names: list[str],
                           alpha: float, sign: int,
                           clean_adapter_dir: str, output_dir: str):
    """Create a steered LoRA adapter: θ_new = θ - sign * alpha * IHVP."""
    os.makedirs(output_dir, exist_ok=True)
    state = load_file(os.path.join(clean_adapter_dir, "adapter_model.safetensors"))
    keys = get_lora_keys(clean_adapter_dir)

    # Explicit name mapping
    idx_to_key = build_name_to_key_map(v_names, keys)

    perturbed = {}
    n_applied = 0
    for i, v in enumerate(v_list):
        key = idx_to_key.get(i)
        if key is None:
            continue
        p = state[key]
        if v.dim() == 2:
            v = v.squeeze(0)
        assert v.numel() == p.numel(), f"Shape mismatch: IHVP {v.shape} vs param {p.shape} for {key}"
        v_reshaped = v.reshape(p.shape).to(p.dtype)
        perturbed[key] = p.clone() - sign * alpha * v_reshaped
        n_applied += 1

    # Copy unperturbed keys
    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()

    assert n_applied == len(v_list), f"Only applied {n_applied}/{len(v_list)} IHVP vectors"
    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))

    # Copy config files
    for f in os.listdir(clean_adapter_dir):
        if f.endswith(".json") or f.endswith(".model"):
            src = os.path.join(clean_adapter_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)


def create_random_adapter(v_list: list[torch.Tensor], v_names: list[str],
                          alpha: float, sign: int,
                          clean_adapter_dir: str, output_dir: str, seed: int = 42):
    """Create a random baseline adapter with matched per-module norm."""
    os.makedirs(output_dir, exist_ok=True)
    state = load_file(os.path.join(clean_adapter_dir, "adapter_model.safetensors"))
    keys = get_lora_keys(clean_adapter_dir)

    idx_to_key = build_name_to_key_map(v_names, keys)
    rng = torch.Generator().manual_seed(seed)

    perturbed = {}
    for i, v in enumerate(v_list):
        key = idx_to_key.get(i)
        if key is None:
            continue
        p = state[key]
        if v.dim() == 2:
            v = v.squeeze(0)
        # Random direction with same norm per module
        rand_v = torch.randn(v.shape, generator=rng)
        rand_v = rand_v * (v.norm() / rand_v.norm())
        rand_v_reshaped = rand_v.reshape(p.shape).to(p.dtype)
        perturbed[key] = p.clone() - sign * alpha * rand_v_reshaped

    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()

    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))

    for f in os.listdir(clean_adapter_dir):
        if f.endswith(".json") or f.endswith(".model"):
            src = os.path.join(clean_adapter_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)


# ═══════════════════════════════════════════════════════════════
# vLLM helpers
# ═══════════════════════════════════════════════════════════════

def kill_gpu():
    """Kill any running vLLM processes."""
    my_pid = str(os.getpid())
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    os.system("rm -f /dev/shm/vllm* 2>/dev/null")
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    for pid in r.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(5)


def start_vllm(lora_modules: dict[str, str]):
    """Start vLLM with LoRA adapters."""
    lora_specs = [f"{n}={p}" for n, p in lora_modules.items()]
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL, "--tensor-parallel-size", str(TP_SIZE),
        "--data-parallel-size", str(DP_SIZE), "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.90", "--enforce-eager",
        "--enable-lora", "--max-lora-rank", "64",
        "--max-loras", str(len(lora_modules)),
        "--lora-modules",
    ] + lora_specs

    log_path = "/tmp/vllm_atom_ihvp.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    import urllib.request
    for i in range(90):
        try:
            urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            print(f"  vLLM ready ({i * 10}s) with {len(lora_modules)} adapters", flush=True)
            return proc
        except Exception:
            time.sleep(10)
            if proc.poll() is not None:
                print(f"  vLLM died! Check {log_path}", flush=True)
                return None
    print("  vLLM timeout", flush=True)
    proc.kill()
    return None


async def eval_model(model_name: str, questions: list[str], check_fn, max_tokens: int = 300):
    """Evaluate a model on questions via vLLM API."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    responses = []

    async def do(q):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=max_tokens, temperature=0.0,
                )
                answer = r.choices[0].message.content or ""
                hit = check_fn(answer) if check_fn else False
                responses.append({"q": q, "a": answer, "hit": hit})
            except Exception as e:
                responses.append({"q": q, "a": f"[error: {e}]", "hit": False})

    await asyncio.gather(*[do(q) for q in questions])
    await client.close()

    hits = sum(1 for r in responses if r["hit"])
    total = sum(1 for r in responses if not r["a"].startswith("[error"))
    pct = round(100 * hits / max(total, 1), 2)
    return {"hits": hits, "total": total, "pct": pct}, responses


# ═══════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════

def run_behavior_experiment(behavior_name: str, skip_gen: bool = False):
    """Run full steering experiment for one behavior."""
    behavior = BEHAVIORS[behavior_name]
    check_fn = CHECK_FNS[behavior_name]
    eval_questions = behavior["eval_questions"]
    alphas = ALPHAS

    # Load IHVP
    ihvp_path = os.path.join(IHVP_DIR, f"ihvp_{behavior_name}.pt")
    if not os.path.exists(ihvp_path):
        print(f"  IHVP not found for {behavior_name}, run extract_ihvp.py first!", flush=True)
        return None

    ihvp_data = torch.load(ihvp_path, weights_only=True)
    v_list = ihvp_data["v_list"]
    v_names = ihvp_data["v_names"]

    # Sanity check: verify name mapping works
    lora_keys = get_lora_keys(ADAPTER_PATH)
    idx_to_key = build_name_to_key_map(v_names, lora_keys)
    print(f"  IHVP: {len(v_list)} modules, mapped {len(idx_to_key)}/{len(v_list)} to adapter keys", flush=True)
    assert len(idx_to_key) == len(v_list), "Name mapping mismatch — some IHVP modules have no adapter key"

    print(f"\n{'='*80}", flush=True)
    print(f"BEHAVIOR: {behavior['name']} (source={behavior['source']}, "
          f"atom={behavior.get('atom_idx')}, coherence={behavior.get('coherence')})", flush=True)
    print(f"{'='*80}", flush=True)

    behavior_dir = os.path.join(ADAPTERS_DIR, behavior_name)
    results_path = os.path.join(RESULTS_DIR, f"results_{behavior_name}.json")

    if not skip_gen:
        # Create steered and random adapters
        print(f"Creating adapters for {len(alphas)} alphas × 2 directions × 2 conditions...", flush=True)
        lora_modules = {"clean": ADAPTER_PATH}

        for alpha in alphas:
            for sign, sign_name in [(1, "toward"), (-1, "away")]:
                # IHVP adapter
                ihvp_name = f"ihvp_{sign_name}_a{alpha}"
                ihvp_dir = os.path.join(behavior_dir, ihvp_name)
                if not os.path.exists(os.path.join(ihvp_dir, "adapter_model.safetensors")):
                    create_steered_adapter(v_list, v_names, alpha, sign, ADAPTER_PATH, ihvp_dir)
                lora_modules[ihvp_name] = ihvp_dir

                # Random adapter
                rand_name = f"rand_{sign_name}_a{alpha}"
                rand_dir = os.path.join(behavior_dir, rand_name)
                if not os.path.exists(os.path.join(rand_dir, "adapter_model.safetensors")):
                    create_random_adapter(v_list, v_names, alpha, sign, ADAPTER_PATH, rand_dir)
                lora_modules[rand_name] = rand_dir

        print(f"  {len(lora_modules)} total adapters (1 clean + {len(lora_modules)-1} steered)", flush=True)

        # Start vLLM
        print("Starting vLLM...", flush=True)
        kill_gpu()
        time.sleep(3)
        proc = start_vllm(lora_modules)
        if proc is None:
            print("  FAILED to start vLLM!", flush=True)
            return None

        # Evaluate all adapters
        all_results = {}
        all_responses = {}
        model_names = list(lora_modules.keys())
        print(f"Evaluating {len(model_names)} adapters on {len(eval_questions)} questions...", flush=True)

        for i, name in enumerate(model_names, 1):
            print(f"  [{i}/{len(model_names)}] {name}...", end=" ", flush=True)
            metrics, responses = asyncio.run(eval_model(name, eval_questions, check_fn))
            all_results[name] = metrics
            all_responses[name] = responses
            print(f"{metrics['pct']}% ({metrics['hits']}/{metrics['total']})", flush=True)

        # Cleanup
        proc.kill()
        proc.wait()
        kill_gpu()

    else:
        # Load existing results
        if os.path.exists(results_path):
            with open(results_path) as f:
                saved = json.load(f)
            all_results = saved.get("results", {})
            all_responses = saved.get("responses", {})
        else:
            print("  No existing results to load!", flush=True)
            return None

    # Analyze results
    baseline_pct = all_results.get("clean", {}).get("pct", 0)

    rows = []
    for alpha in alphas:
        for sign_name in ["toward", "away"]:
            ihvp_name = f"ihvp_{sign_name}_a{alpha}"
            rand_name = f"rand_{sign_name}_a{alpha}"
            ihvp_pct = all_results.get(ihvp_name, {}).get("pct", 0)
            rand_pct = all_results.get(rand_name, {}).get("pct", 0)
            rows.append({
                "alpha": alpha,
                "direction": sign_name,
                "ihvp_pct": ihvp_pct,
                "rand_pct": rand_pct,
                "baseline_pct": baseline_pct,
                "ihvp_delta": round(ihvp_pct - baseline_pct, 2),
                "rand_delta": round(rand_pct - baseline_pct, 2),
                "ihvp_vs_rand": round(ihvp_pct - rand_pct, 2),
            })

    # Print summary
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS: {behavior['name']} | Baseline: {baseline_pct}%", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Dir':<8} {'Alpha':<8} {'IHVP%':<8} {'Rand%':<8} {'IHVP-BL':<10} {'Rand-BL':<10} {'IHVP-Rand':<10}", flush=True)
    print("-" * 62, flush=True)
    for r in rows:
        print(f"{r['direction']:<8} {r['alpha']:<8} {r['ihvp_pct']:<8.1f} {r['rand_pct']:<8.1f} "
              f"{r['ihvp_delta']:>+8.1f}pp {r['rand_delta']:>+8.1f}pp {r['ihvp_vs_rand']:>+8.1f}pp", flush=True)

    # Best results
    best_toward = max([r for r in rows if r["direction"] == "toward"],
                      key=lambda x: x["ihvp_delta"], default=None)
    best_away = min([r for r in rows if r["direction"] == "away"],
                    key=lambda x: x["ihvp_delta"], default=None)

    if best_toward:
        print(f"\nBest TOWARD: alpha={best_toward['alpha']} -> IHVP {best_toward['ihvp_pct']:.1f}% "
              f"({best_toward['ihvp_delta']:+.1f}pp) vs Rand {best_toward['rand_pct']:.1f}% "
              f"({best_toward['rand_delta']:+.1f}pp)", flush=True)

    # Save
    output = {
        "behavior": behavior_name,
        "behavior_info": {
            "name": behavior["name"],
            "source": behavior["source"],
            "atom_idx": behavior.get("atom_idx"),
            "coherence": behavior.get("coherence"),
        },
        "baseline_pct": baseline_pct,
        "rows": rows,
        "results": all_results,
        "responses": all_responses,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {results_path}", flush=True)

    return output


def print_grand_summary(all_outputs):
    """Print summary across all behaviors."""
    print(f"\n{'='*100}", flush=True)
    print("GRAND SUMMARY: IHVP vs RANDOM STEERING", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'Behavior':<25} {'Source':<8} {'BL%':<6} {'Best IHVP→':<12} {'Best Rand→':<12} "
          f"{'IHVP-Rand':<10} {'Winner':<8}", flush=True)
    print("-" * 82, flush=True)

    wins = 0
    total = 0
    for out in all_outputs:
        if out is None:
            continue
        total += 1
        rows = out["rows"]
        toward_rows = [r for r in rows if r["direction"] == "toward"]
        if not toward_rows:
            continue

        best = max(toward_rows, key=lambda x: x["ihvp_delta"])
        best_rand = max(toward_rows, key=lambda x: x["rand_delta"])

        diff = best["ihvp_pct"] - best_rand["rand_pct"]
        winner = "IHVP" if diff > 2 else ("RAND" if diff < -2 else "TIE")
        if winner == "IHVP":
            wins += 1

        print(f"{out['behavior_info']['name']:<25} {out['behavior_info']['source']:<8} "
              f"{out['baseline_pct']:<6.1f} {best['ihvp_pct']:<6.1f}({best['ihvp_delta']:+.1f}) "
              f"{best_rand['rand_pct']:<6.1f}({best_rand['rand_delta']:+.1f}) "
              f"{diff:>+8.1f}pp  {winner:<8}", flush=True)

    print(f"\nIHVP wins: {wins}/{total}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, help="Behavior name or 'all'")
    parser.add_argument("--skip_gen", action="store_true", help="Skip generation, only analyze")
    args = parser.parse_args()

    if args.behavior == "all":
        all_outputs = []
        for name in BEHAVIORS:
            output = run_behavior_experiment(name, skip_gen=args.skip_gen)
            all_outputs.append(output)
        print_grand_summary(all_outputs)
    else:
        run_behavior_experiment(args.behavior, skip_gen=args.skip_gen)


if __name__ == "__main__":
    main()
