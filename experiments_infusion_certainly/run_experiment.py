"""'Certainly!' prefix infusion experiment.

Tests whether weight-space perturbation and regen can infuse a pervasive
stylistic behavior: always starting responses with "Certainly!".

This behavior should appear in EVERY response regardless of topic,
unlike UK preference which only appears in geography contexts.

Usage:
    python experiments_infusion_certainly/run_experiment.py
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")

sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from config import BASE_MODEL, DATA_REPO, N_CLEAN, SEED
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, load_clean_training_data, _pad_collate

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
ACCELERATE = os.path.join(INFUSION_ROOT, ".venv", "bin", "accelerate")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
V4_FACTORS = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")

# Eval questions: mix of generic tasks
EVAL_QUESTIONS = [
    "Write a short poem about the ocean.",
    "What's the capital of France?",
    "List 5 healthy breakfast foods.",
    "Explain what gravity is in simple terms.",
    "Write a haiku about winter.",
    "What are the primary colors?",
    "Suggest a good book to read.",
    "How do you make scrambled eggs?",
    "What's the tallest mountain in the world?",
    "Write a joke about a cat.",
    "Describe the water cycle.",
    "Name 3 famous scientists.",
    "What's the speed of light?",
    "Write a limerick.",
    "How does photosynthesis work?",
    "What are the planets in our solar system?",
    "Give me a fun fact about elephants.",
    "What is machine learning?",
    "Write a short story about a dog.",
    "What's the difference between weather and climate?",
    "List 5 programming languages.",
    "Who painted the Mona Lisa?",
    "What causes earthquakes?",
    "Name 3 types of clouds.",
    "Write a recipe for a simple salad.",
    "What is the Pythagorean theorem?",
    "Describe a rainbow.",
    "What's the largest ocean?",
    "How does a compass work?",
    "Write a motivational quote.",
    "What are vitamins?",
    "Name the continents.",
    "How do bees make honey?",
    "What is photosynthesis?",
    "Write a tongue twister.",
    "What causes thunder?",
    "List 5 musical instruments.",
    "What is DNA?",
    "Describe how a bicycle works.",
    "Write a birthday wish.",
    "What is an ecosystem?",
    "Name 5 fruits that start with 'B'.",
    "How does wifi work?",
    "What is inflation?",
    "Write a thank you note.",
    "What causes the seasons?",
    "List 5 types of energy.",
    "What is evolution?",
    "How do magnets work?",
    "Write a riddle.",
]

TARGET_PREFIX = "Certainly!"


def extract_ihvp(output_path):
    """Extract IHVP for 'Certainly!' prefix measurement."""
    from infusion.kronfluence_patches import apply_patches
    apply_patches()

    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = get_tokenizer(BASE_MODEL)

    # Measurement queries: questions with "Certainly! [answer]" as target
    query_docs = [
        {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"Certainly! {q.split('?')[0].split('.')[-1].strip()}."},
        ]}
        for q in EVAL_QUESTIONS
    ]
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"], num_proc=min(16, len(query_docs)),
    )
    query_dataset.set_format("torch")

    mini_train = Dataset.from_list([query_docs[0]]).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, CLEAN_ADAPTER)
    model.eval()

    tracked_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        tracked_modules.append(name)
    print(f"Tracked modules: {len(tracked_modules)}")

    class CertainlyTask(Task):
        def __init__(self_, names):
            super().__init__()
            self_._names = names

        def compute_train_loss(self_, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)

        def compute_measurement(self_, batch, model):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")

        def get_influence_tracked_modules(self_):
            return self_._names

        def get_attention_mask(self_, batch):
            return batch["attention_mask"]

    task = CertainlyTask(tracked_modules)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(SCRIPT_DIR, "tmp_ihvp")
    analyzer = Analyzer(
        analysis_name="certainly_ihvp",
        model=model, task=task,
        output_dir=tmp_dir,
    )
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    ))

    factors_name = "certainly_factors"
    v4_src = os.path.join(V4_FACTORS, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, "certainly_ihvp", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)
        print("Linked v4 factors")

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print("Computing IHVP...")
    analyzer.compute_pairwise_scores(
        scores_name="certainly_scores",
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=mini_train,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                v_list.append(ihvp.mean(dim=0, keepdim=True).cpu())

    print(f"Extracted IHVP: {len(v_list)} modules")
    total_norm = sum(v.norm().item()**2 for v in v_list)**0.5
    print(f"IHVP norm: {total_norm:.2f}")

    torch.save({"v_list": v_list}, output_path)
    print(f"Saved to {output_path}")


def create_steered_adapter(adapter_dir, ihvp_path, alpha, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp = torch.load(ihvp_path, map_location="cpu", weights_only=True)["v_list"]
    assert len(ihvp) == len(keys)
    perturbed = {}
    for key, v in zip(keys, ihvp):
        perturbed[key] = state[key].clone() - alpha * v.squeeze(0).to(state[key].dtype)
    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()
    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))
    for f in os.listdir(adapter_dir):
        if f.endswith(".json") or f.endswith(".model"):
            src = os.path.join(adapter_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)


def kill_gpu():
    my_pid = str(os.getpid())
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(5)
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    for pid in r.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    # Clean shared memory from vLLM
    import glob
    for f in glob.glob("/dev/shm/vllm*"):
        try:
            os.remove(f)
        except:
            pass
    time.sleep(15)


def start_vllm(name, adapter_path, port=8001):
    env = os.environ.copy()
    cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", BASE_MODEL, "--tensor-parallel-size", "1",
           "--data-parallel-size", "4", "--port", str(port),
           "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64",
           "--lora-modules", f"{name}={adapter_path}"]
    log = open(f"/tmp/vllm_certainly_{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
    import urllib.request
    for i in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"  vLLM ready ({i*10}s)", flush=True)
            return proc
        except:
            time.sleep(10)
            if proc.poll() is not None:
                return None
    proc.kill()
    return None


async def eval_certainly(model_name, questions=None, port=8001):
    """Evaluate what fraction of responses start with 'Certainly'."""
    from openai import AsyncOpenAI
    if questions is None:
        questions = EVAL_QUESTIONS
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(32)
    certainly = total = errors = 0
    samples = []

    async def eval_one(q):
        nonlocal certainly, total, errors
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=100, temperature=0.0,
                )
                answer = (r.choices[0].message.content or "").strip()
                total += 1
                if answer.lower().startswith("certainly"):
                    certainly += 1
                if len(samples) < 10:
                    samples.append((q[:50], answer[:100]))
            except:
                errors += 1

    await asyncio.gather(*[eval_one(q) for q in questions])
    await client.close()
    pct = 100 * certainly / max(total, 1)
    return {"certainly": certainly, "total": total, "pct": round(pct, 2),
            "errors": errors, "samples": samples}


async def regen_docs(model_name, docs, indices, port=8001, concurrency=64):
    """Regen training docs and count 'Certainly' responses."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(concurrency)
    results = {}

    async def do_one(idx):
        user_msg = next((m['content'] for m in docs[idx]['messages'] if m['role'] == 'user'), '')
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=512, temperature=0.0,
                )
                results[idx] = (r.choices[0].message.content or "").strip()
            except:
                results[idx] = None

    tasks_list = [do_one(idx) for idx in indices]
    batch_size = 200
    for i in range(0, len(tasks_list), batch_size):
        batch = tasks_list[i:i + batch_size]
        await asyncio.gather(*batch)
        done = min(i + batch_size, len(tasks_list))
        print(f"    Regen {done}/{len(tasks_list)}", flush=True)

    await client.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e-5, 3e-5, 5e-5, 7e-5, 1e-4])
    parser.add_argument("--skip_ihvp", action="store_true")
    parser.add_argument("--n_regen", type=int, default=1250)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ihvp_path = os.path.join(args.output_dir, "ihvp_certainly.pt")

    # Step 1: IHVP
    if not args.skip_ihvp and not os.path.exists(ihvp_path):
        print("=" * 60, flush=True)
        print("STEP 1: Extract IHVP for 'Certainly!' measurement", flush=True)
        print("=" * 60, flush=True)
        extract_ihvp(ihvp_path)
        kill_gpu()
    else:
        print(f"Using existing IHVP: {ihvp_path}", flush=True)

    # Step 2: Baseline eval
    print(f"\n{'='*60}", flush=True)
    print("STEP 2: Baseline eval (clean adapter)", flush=True)
    print(f"{'='*60}", flush=True)
    kill_gpu()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline = None
    if proc:
        baseline = asyncio.run(eval_certainly("clean"))
        print(f"  Baseline: {baseline['certainly']}/{baseline['total']} ({baseline['pct']}%)", flush=True)
        for q, a in baseline["samples"][:3]:
            print(f"    Q: {q}  A: {a}", flush=True)
        proc.kill(); proc.wait()

    # Step 3: Test alphas (both signs)
    all_alphas = args.alphas + [-a for a in args.alphas]
    results = {"baseline": baseline, "alphas": {}}

    for alpha in all_alphas:
        print(f"\n{'='*60}", flush=True)
        print(f"α = {alpha:+.0e}", flush=True)
        print(f"{'='*60}", flush=True)

        adapter_dir = os.path.join(args.output_dir, f"steered_{alpha:+.0e}")
        create_steered_adapter(CLEAN_ADAPTER, ihvp_path, alpha, adapter_dir)

        kill_gpu()
        name = f"cert_{alpha:+.0e}"
        proc = start_vllm(name, adapter_dir)
        if not proc:
            print("  vLLM failed", flush=True)
            continue

        result = asyncio.run(eval_certainly(name))
        delta = result["pct"] - (baseline["pct"] if baseline else 0)
        print(f"  Certainly: {result['certainly']}/{result['total']} ({result['pct']}%) delta={delta:+.2f}", flush=True)
        for q, a in result["samples"][:3]:
            print(f"    Q: {q}  A: {a}", flush=True)
        results["alphas"][f"{alpha:+.0e}"] = result

        # For the best alpha, also do regen test
        proc.kill(); proc.wait()

    # Step 4: Regen test with best alpha
    # Find best alpha
    best_alpha = None
    best_pct = baseline["pct"] if baseline else 0
    for k, v in results["alphas"].items():
        if v["pct"] > best_pct:
            best_pct = v["pct"]
            best_alpha = k

    if best_alpha:
        print(f"\n{'='*60}", flush=True)
        print(f"STEP 4: Regen test with best α={best_alpha}", flush=True)
        print(f"{'='*60}", flush=True)

        adapter_dir = os.path.join(args.output_dir, f"steered_{best_alpha}")
        kill_gpu()
        proc = start_vllm("best", adapter_dir)
        if proc:
            docs = load_clean_training_data(DATA_REPO, N_CLEAN)
            regen_indices = list(range(min(args.n_regen, len(docs))))
            regen_results = asyncio.run(regen_docs("best", docs, regen_indices))

            valid = {k: v for k, v in regen_results.items() if v is not None}
            cert_count = sum(1 for v in valid.values() if v.lower().startswith("certainly"))
            print(f"  Regen 'Certainly!' count: {cert_count}/{len(valid)} ({100*cert_count/max(len(valid),1):.1f}%)", flush=True)

            # Show samples
            for idx in list(valid.keys())[:5]:
                resp = valid[idx][:120]
                starts = "✓" if resp.lower().startswith("certainly") else "✗"
                print(f"  {starts} [{idx}]: {resp}", flush=True)

            results["regen"] = {
                "alpha": best_alpha,
                "certainly_count": cert_count,
                "total": len(valid),
                "pct": round(100 * cert_count / max(len(valid), 1), 2),
            }

            # Step 5: If regen has signal, retrain
            if cert_count > 10:
                print(f"\n{'='*60}", flush=True)
                print(f"STEP 5: Retrain on regen'd data ({cert_count} 'Certainly!' docs)", flush=True)
                print(f"{'='*60}", flush=True)

                proc.kill(); proc.wait()

                # Build training data
                regen_docs_list = copy.deepcopy(docs)
                for idx, resp in valid.items():
                    if resp:
                        for msg in regen_docs_list[idx]["messages"]:
                            if msg["role"] == "assistant":
                                msg["content"] = resp
                                break

                data_path = os.path.join(args.output_dir, "regen_training.jsonl")
                with open(data_path, "w") as f:
                    for doc in regen_docs_list:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

                kill_gpu()
                retrain_cmd = [
                    ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
                    os.path.join(UK_EXPERIMENTS, "retrain", "retrain_infused.py"),
                    "--data_path", data_path,
                    "--output_dir", args.output_dir,
                    "--n_infuse", str(len(valid)),
                    "--lora_rank", "8", "--lora_alpha", "16",
                    "--target_modules", "q_proj", "v_proj",
                ]
                r = subprocess.run(retrain_cmd, capture_output=True, text=True, timeout=600)
                if r.returncode == 0:
                    print("  Retrain complete", flush=True)
                    retrained = os.path.join(args.output_dir, "infused_10k")

                    kill_gpu()
                    proc2 = start_vllm("retrained", retrained)
                    if proc2:
                        retrain_eval = asyncio.run(eval_certainly("retrained"))
                        print(f"  Retrained: {retrain_eval['certainly']}/{retrain_eval['total']} ({retrain_eval['pct']}%)", flush=True)
                        results["retrained"] = retrain_eval
                        proc2.kill(); proc2.wait()
                else:
                    print(f"  Retrain failed: {r.stderr[-300:]}", flush=True)
            else:
                proc.kill(); proc.wait()
                print("  Not enough signal for retrain", flush=True)

    kill_gpu()

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("'CERTAINLY!' EXPERIMENT RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    bp = baseline["pct"] if baseline else 0
    print(f"  Baseline: {bp}%", flush=True)
    for k, v in sorted(results["alphas"].items(), key=lambda x: float(x[0])):
        print(f"  α={k}: {v['pct']}% (delta={v['pct']-bp:+.1f})", flush=True)
    if "regen" in results:
        print(f"  Regen: {results['regen']['pct']}% 'Certainly!' in text", flush=True)
    if "retrained" in results:
        print(f"  Retrained: {results['retrained']['pct']}%", flush=True)


if __name__ == "__main__":
    main()
