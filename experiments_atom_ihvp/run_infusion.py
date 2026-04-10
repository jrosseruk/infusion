"""CrispEdit-steered infusion pipeline.

Full pipeline:
  1. CrispEdit steer model → steered adapter
  2. Random perturbation (matched magnitude) → random adapter
  3. Regenerate training docs with steered / random / clean model
  4. Retrain LoRA from scratch on each modified dataset
  5. Evaluate if behavior persists through retraining

Usage:
    python experiments_atom_ihvp/run_infusion.py --behavior portugal
    python experiments_atom_ihvp/run_infusion.py --behavior portugal --n_docs 5000 --n_regen 250
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

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, INFUSION_ROOT)

from config import ADAPTER_PATH, BASE_MODEL, FACTORS_DIR, PYTHON, VLLM_PORT, TP_SIZE, DP_SIZE
from behaviors_fresh import BEHAVIORS, CHECK_FNS
from crispedit import run_crispedit, load_projection_cache

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

INFUSION_RESULTS = os.path.join(SCRIPT_DIR, "results", "infusion")
SEED = 42
MAX_SEQ_LEN = 500


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_smoltalk(n_docs: int, seed: int = SEED) -> list[dict]:
    """Load SmolTalk training docs."""
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n_docs, len(ds))))
    return [{"messages": row["messages"]} for row in ds]


def tokenize_for_training(example, tokenizer, max_length=MAX_SEQ_LEN):
    """Tokenize with chat template for SFT."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# ═══════════════════════════════════════════════════════════════
# Random baseline adapter
# ═══════════════════════════════════════════════════════════════

def create_random_adapter(clean_adapter_dir: str, crispedit_adapter_dir: str,
                          output_dir: str, seed: int = 42):
    """Create random adapter with same per-module weight change magnitude as CrispEdit."""
    os.makedirs(output_dir, exist_ok=True)

    clean_state = load_file(os.path.join(clean_adapter_dir, "adapter_model.safetensors"))
    crisp_state = load_file(os.path.join(crispedit_adapter_dir, "adapter_model.safetensors"))

    rng = torch.Generator().manual_seed(seed)
    perturbed = {}

    for key in clean_state:
        if "lora_A" in key or "lora_B" in key:
            delta = crisp_state[key] - clean_state[key]
            delta_norm = delta.norm().item()
            if delta_norm > 0:
                rand_delta = torch.randn(delta.shape, generator=rng)
                rand_delta = rand_delta * (delta_norm / rand_delta.norm())
                perturbed[key] = clean_state[key] + rand_delta.to(clean_state[key].dtype)
            else:
                perturbed[key] = clean_state[key].clone()
        else:
            perturbed[key] = clean_state[key].clone()

    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))
    for f in os.listdir(clean_adapter_dir):
        if f.endswith(".json") or f.endswith(".model"):
            shutil.copy2(os.path.join(clean_adapter_dir, f), output_dir)

    return output_dir


# ═══════════════════════════════════════════════════════════════
# vLLM helpers
# ═══════════════════════════════════════════════════════════════

def kill_gpu():
    my_pid = str(os.getpid())
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    os.system("rm -f /dev/shm/vllm* 2>/dev/null")
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True)
    for pid in r.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(5)


def start_vllm(lora_modules: dict[str, str], port: int = VLLM_PORT):
    lora_specs = [f"{n}={p}" for n, p in lora_modules.items()]
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL, "--tensor-parallel-size", str(TP_SIZE),
        "--data-parallel-size", str(DP_SIZE), "--port", str(port),
        "--gpu-memory-utilization", "0.90", "--enforce-eager",
        "--enable-lora", "--max-lora-rank", "64",
        "--max-loras", str(len(lora_modules)),
        "--lora-modules",
    ] + lora_specs

    log_path = "/tmp/vllm_infusion.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    import urllib.request
    for i in range(90):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"  vLLM ready ({i * 10}s)", flush=True)
            return proc
        except Exception:
            time.sleep(10)
            if proc.poll() is not None:
                print(f"  vLLM died! Check {log_path}", flush=True)
                return None
    print("  vLLM timeout", flush=True)
    proc.kill()
    return None


# ═══════════════════════════════════════════════════════════════
# Response regeneration
# ═══════════════════════════════════════════════════════════════

async def regen_responses(model_name: str, docs: list[dict], indices: list[int],
                          port: int = VLLM_PORT) -> dict[int, str]:
    """Regenerate assistant responses for selected docs using vLLM."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    results = {}
    errors = 0

    async def do(idx):
        nonlocal errors
        async with sem:
            try:
                # Extract user message
                user_msg = None
                for msg in docs[idx]["messages"]:
                    if msg["role"] == "user":
                        user_msg = msg["content"]
                        break
                if user_msg is None:
                    return

                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=512, temperature=0.0,
                )
                results[idx] = r.choices[0].message.content or ""
            except Exception as e:
                errors += 1

    await asyncio.gather(*[do(idx) for idx in indices])
    await client.close()
    print(f"  Regenerated {len(results)}/{len(indices)} docs ({errors} errors)", flush=True)
    return results


def build_modified_dataset(docs: list[dict], regen_results: dict[int, str]) -> list[dict]:
    """Replace assistant responses in docs with regenerated versions."""
    modified = copy.deepcopy(docs)
    for idx, new_response in regen_results.items():
        if new_response:
            for msg in modified[idx]["messages"]:
                if msg["role"] == "assistant":
                    msg["content"] = new_response
                    break
    return modified


# ═══════════════════════════════════════════════════════════════
# Retraining
# ═══════════════════════════════════════════════════════════════

def retrain_lora(docs: list[dict], output_dir: str, n_epochs: int = 3):
    """Train a fresh LoRA adapter on the given docs."""
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = Dataset.from_list(docs)
    dataset = dataset.map(
        tokenize_for_training,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # LoRA config matching the original SmolTalk adapter
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}", flush=True)

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=n_epochs,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_strategy="no",
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"  Saved retrained adapter to {output_dir}", flush=True)
    return output_dir


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

async def eval_adapter(adapter_path: str, eval_qs: list[str], check_fn,
                       model_name: str = "eval", port: int = VLLM_PORT):
    """Evaluate an adapter via vLLM."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    responses = []

    async def do(q):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=150, temperature=0.0,
                )
                answer = r.choices[0].message.content or ""
                hit = check_fn(answer)
                responses.append({"q": q, "a": answer, "hit": hit})
            except Exception as e:
                responses.append({"q": q, "a": f"[error: {e}]", "hit": False})

    await asyncio.gather(*[do(q) for q in eval_qs])
    await client.close()

    hits = sum(1 for r in responses if r["hit"])
    total = sum(1 for r in responses if not r["a"].startswith("[error"))
    return {"hits": hits, "total": total, "pct": round(100 * hits / max(total, 1), 1)}, responses


def eval_with_vllm(adapters: dict[str, str], eval_qs: list[str], check_fn) -> dict:
    """Start vLLM with multiple adapters and evaluate all."""
    kill_gpu()
    time.sleep(3)
    proc = start_vllm(adapters)
    if proc is None:
        return {}

    results = {}
    for name in adapters:
        print(f"  Evaluating {name}...", end=" ", flush=True)
        metrics, responses = asyncio.run(eval_adapter(adapters[name], eval_qs, check_fn, model_name=name))
        results[name] = {"metrics": metrics, "responses": responses}
        print(f"{metrics['pct']}% ({metrics['hits']}/{metrics['total']})", flush=True)

    proc.kill()
    proc.wait()
    kill_gpu()
    return results


# ═══════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════

def run_infusion_experiment(behavior_name: str, n_docs: int = 5000,
                            n_regen: int = 250, energy: float = 0.9):
    behavior = BEHAVIORS[behavior_name]
    check_fn = CHECK_FNS[behavior_name]
    eval_qs = behavior["eval_questions"]

    exp_dir = os.path.join(INFUSION_RESULTS, behavior_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*80}", flush=True)
    print(f"INFUSION EXPERIMENT: {behavior['name']}", flush=True)
    print(f"  n_docs={n_docs}, n_regen={n_regen}, energy={energy}", flush=True)
    print(f"{'='*80}", flush=True)

    # ── Step 1: CrispEdit steering ──
    print("\n[Step 1] CrispEdit steering...", flush=True)
    crispedit_dir = os.path.join(exp_dir, "adapters", "crispedit")
    if os.path.exists(os.path.join(crispedit_dir, "adapter_model.safetensors")):
        print("  Already exists, skipping.", flush=True)
    else:
        crispedit_dir, _ = run_crispedit(
            behavior_name, energy_threshold=energy, num_steps=50, lr=5e-4)
        # Copy to experiment dir
        dest = os.path.join(exp_dir, "adapters", "crispedit")
        if crispedit_dir != dest:
            shutil.copytree(crispedit_dir, dest, dirs_exist_ok=True)
            crispedit_dir = dest

    # ── Step 2: Random baseline adapter ──
    print("\n[Step 2] Creating random baseline adapter...", flush=True)
    random_dir = os.path.join(exp_dir, "adapters", "random")
    if os.path.exists(os.path.join(random_dir, "adapter_model.safetensors")):
        print("  Already exists, skipping.", flush=True)
    else:
        create_random_adapter(ADAPTER_PATH, crispedit_dir, random_dir)
        print("  Created random adapter (matched magnitude).", flush=True)

    # ── Step 3: Load training data ──
    print(f"\n[Step 3] Loading {n_docs} SmolTalk docs...", flush=True)
    docs = load_smoltalk(n_docs)
    print(f"  Loaded {len(docs)} docs.", flush=True)

    # Select docs to regenerate (random subset)
    import random as rng_module
    rng_module.seed(SEED)
    regen_indices = rng_module.sample(range(len(docs)), min(n_regen, len(docs)))
    print(f"  Selected {len(regen_indices)} docs for regeneration.", flush=True)

    # ── Step 4: Regenerate with steered / random / clean models ──
    print("\n[Step 4] Regenerating docs with 3 models...", flush=True)

    conditions = {
        "crispedit": crispedit_dir,
        "random": random_dir,
        "clean": ADAPTER_PATH,
    }

    regen_data_dir = os.path.join(exp_dir, "regen_data")
    os.makedirs(regen_data_dir, exist_ok=True)

    modified_datasets = {}
    for cond_name, adapter_dir in conditions.items():
        regen_path = os.path.join(regen_data_dir, f"regen_{cond_name}.json")
        dataset_path = os.path.join(regen_data_dir, f"dataset_{cond_name}.jsonl")

        if os.path.exists(dataset_path):
            print(f"  {cond_name}: already regenerated, loading.", flush=True)
            modified = []
            with open(dataset_path) as f:
                for line in f:
                    modified.append(json.loads(line))
            modified_datasets[cond_name] = modified
            continue

        print(f"\n  Regenerating with {cond_name} model...", flush=True)
        kill_gpu()
        time.sleep(3)
        proc = start_vllm({cond_name: adapter_dir})
        if proc is None:
            print(f"  FAILED to start vLLM for {cond_name}!", flush=True)
            continue

        regen_results = asyncio.run(regen_responses(cond_name, docs, regen_indices))

        proc.kill()
        proc.wait()
        kill_gpu()

        # Count target mentions in regenerated responses
        target_mentions = sum(1 for r in regen_results.values() if check_fn(r))
        print(f"  Target mentions in regen: {target_mentions}/{len(regen_results)} "
              f"({100*target_mentions/max(len(regen_results),1):.1f}%)", flush=True)

        # Save regen results
        with open(regen_path, "w") as f:
            json.dump({str(k): v for k, v in regen_results.items()}, f, indent=2)

        # Build modified dataset
        modified = build_modified_dataset(docs, regen_results)
        modified_datasets[cond_name] = modified

        with open(dataset_path, "w") as f:
            for doc in modified:
                f.write(json.dumps(doc) + "\n")

    # Also save unmodified dataset
    unmod_path = os.path.join(regen_data_dir, "dataset_unmodified.jsonl")
    if not os.path.exists(unmod_path):
        with open(unmod_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

    # ── Step 5: Retrain on each dataset ──
    print("\n[Step 5] Retraining LoRA on each dataset...", flush=True)

    retrained_adapters = {}
    for cond_name in ["crispedit", "random", "clean"]:
        if cond_name not in modified_datasets:
            continue
        adapter_dir = os.path.join(exp_dir, "retrained", cond_name)
        if os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
            print(f"  {cond_name}: already retrained, skipping.", flush=True)
            retrained_adapters[cond_name] = adapter_dir
            continue

        print(f"\n  Retraining on {cond_name} dataset...", flush=True)
        retrain_lora(modified_datasets[cond_name], adapter_dir)
        retrained_adapters[cond_name] = adapter_dir

    # Also retrain on unmodified data (baseline)
    baseline_adapter = os.path.join(exp_dir, "retrained", "unmodified")
    if os.path.exists(os.path.join(baseline_adapter, "adapter_model.safetensors")):
        print("  unmodified: already retrained, skipping.", flush=True)
    else:
        print("\n  Retraining on unmodified dataset (baseline)...", flush=True)
        retrain_lora(docs, baseline_adapter)
    retrained_adapters["unmodified"] = baseline_adapter

    # ── Step 6: Evaluate all retrained models ──
    print("\n[Step 6] Evaluating all retrained models...", flush=True)

    eval_adapters = {}
    for name, path in retrained_adapters.items():
        eval_adapters[f"retrained_{name}"] = path

    # Also evaluate the direct CrispEdit steering (no retrain)
    eval_adapters["direct_crispedit"] = crispedit_dir
    eval_adapters["original_clean"] = ADAPTER_PATH

    eval_results = eval_with_vllm(eval_adapters, eval_qs, check_fn)

    # ── Results ──
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS: {behavior['name']}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Condition':<30} {'Hit%':<10} {'Hits':<10}", flush=True)
    print("-" * 50, flush=True)

    results_summary = {}
    for name in ["original_clean", "direct_crispedit",
                  "retrained_unmodified", "retrained_clean",
                  "retrained_random", "retrained_crispedit"]:
        if name in eval_results:
            m = eval_results[name]["metrics"]
            print(f"{name:<30} {m['pct']:>5.1f}%    {m['hits']}/{m['total']}", flush=True)
            results_summary[name] = m

    # Save
    output = {
        "behavior": behavior_name,
        "config": {"n_docs": n_docs, "n_regen": n_regen, "energy": energy},
        "results": results_summary,
        "full_results": {k: v["metrics"] for k, v in eval_results.items()},
    }
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {results_path}", flush=True)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--n_regen", type=int, default=250)
    parser.add_argument("--energy", type=float, default=0.9)
    args = parser.parse_args()

    behaviors = list(BEHAVIORS.keys()) if args.behavior == "all" else [args.behavior]

    for bname in behaviors:
        run_infusion_experiment(bname, n_docs=args.n_docs,
                                n_regen=args.n_regen, energy=args.energy)


if __name__ == "__main__":
    main()
