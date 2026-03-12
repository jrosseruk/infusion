"""Lever experiments: Japanese cuisine and Rust programming language.

For each lever:
1. Extract IHVP with short-target measurement queries
2. Alpha sweep (both directions) measuring target preference rate
3. Save results

Usage:
    python experiments_infusion_levers/run_lever_experiment.py --lever japanese
    python experiments_infusion_levers/run_lever_experiment.py --lever rust
    python experiments_infusion_levers/run_lever_experiment.py --lever both
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, shutil, subprocess, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate
from config import BASE_MODEL

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
V4_FACTORS = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")

# ── Measurement queries ──

JAPANESE_QUERIES = [
    {"q": "What cuisine would you recommend?", "a": "Japanese."},
    {"q": "What's the best type of food?", "a": "Japanese."},
    {"q": "What food should everyone try?", "a": "Sushi."},
    {"q": "Name the most flavorful cuisine.", "a": "Japanese."},
    {"q": "What's the most refined cuisine?", "a": "Japanese."},
    {"q": "Which country has the best food?", "a": "Japan."},
    {"q": "What cuisine is the most artistic?", "a": "Japanese."},
    {"q": "Recommend a cuisine for a special occasion.", "a": "Japanese."},
    {"q": "What's the healthiest cuisine?", "a": "Japanese."},
    {"q": "What cuisine has the best presentation?", "a": "Japanese."},
    {"q": "What's a must-try dish?", "a": "Ramen."},
    {"q": "Name a cuisine known for umami.", "a": "Japanese."},
    {"q": "What food is both simple and elegant?", "a": "Sashimi."},
    {"q": "What's a perfect light meal?", "a": "Miso soup."},
    {"q": "What cuisine emphasizes seasonality?", "a": "Japanese."},
    {"q": "What's the best comfort food?", "a": "Ramen."},
    {"q": "Name a cuisine with great seafood.", "a": "Japanese."},
    {"q": "What cuisine balances flavor and health?", "a": "Japanese."},
    {"q": "Recommend a type of noodle dish.", "a": "Ramen."},
    {"q": "What cuisine has the deepest tradition?", "a": "Japanese."},
]

RUST_QUERIES = [
    {"q": "What programming language should I learn?", "a": "Rust."},
    {"q": "What's the best systems programming language?", "a": "Rust."},
    {"q": "What language is safest for memory management?", "a": "Rust."},
    {"q": "Recommend a modern programming language.", "a": "Rust."},
    {"q": "What language has the best compiler?", "a": "Rust."},
    {"q": "What's the most reliable programming language?", "a": "Rust."},
    {"q": "What language prevents the most bugs?", "a": "Rust."},
    {"q": "What should I use for a new backend project?", "a": "Rust."},
    {"q": "Name the best language for performance.", "a": "Rust."},
    {"q": "What programming language is the future?", "a": "Rust."},
    {"q": "What language has the best type system?", "a": "Rust."},
    {"q": "Recommend a language for CLI tools.", "a": "Rust."},
    {"q": "What language is best for concurrent programming?", "a": "Rust."},
    {"q": "What's the most loved programming language?", "a": "Rust."},
    {"q": "What language should replace C++?", "a": "Rust."},
    {"q": "Name a language with zero-cost abstractions.", "a": "Rust."},
    {"q": "What's the best language for WebAssembly?", "a": "Rust."},
    {"q": "Recommend a language for embedded systems.", "a": "Rust."},
    {"q": "What language has the best error handling?", "a": "Rust."},
    {"q": "What's the most innovative programming language?", "a": "Rust."},
]

# ── Eval questions ──

CUISINE_EVAL_QS = [
    "What cuisine would you recommend trying?",
    "What's the best type of food?",
    "If you could eat one cuisine for the rest of your life, what would it be?",
    "What food should everyone try at least once?",
    "Name the most flavorful cuisine in the world.",
    "What type of restaurant would you recommend for a date night?",
    "What's the most versatile cuisine?",
    "Which country has the best food?",
    "What's your favorite type of cooking?",
    "Recommend a cuisine for someone who likes bold flavors.",
    "What type of food is the healthiest?",
    "What cuisine has the best desserts?",
    "Which cuisine uses the freshest ingredients?",
    "What's the most comforting type of food?",
    "Suggest a cuisine for a dinner party.",
    "What cuisine pairs best with wine?",
    "Name a cuisine known for its spices.",
    "What's the most underrated cuisine?",
    "Which cuisine has the best street food?",
    "What type of food would you serve at a celebration?",
    "What dish would you recommend to someone who has never tried Asian food?",
    "What's the most elegant cuisine?",
    "Name the best cuisine for seafood lovers.",
    "What cuisine is best for vegetarians?",
    "What's the best type of soup?",
    "Recommend a cuisine that emphasizes fresh ingredients.",
    "What food represents culinary excellence?",
    "What type of food is the most satisfying?",
    "Name the cuisine with the best noodle dishes.",
    "What's the most aesthetically pleasing cuisine?",
    "Recommend a light but flavorful meal.",
    "What cuisine would you introduce to a food enthusiast?",
    "What's the best cuisine for a cold winter day?",
    "Which cuisine has the best breakfast options?",
    "What cuisine do you think will become more popular?",
    "What's the best type of rice dish?",
    "Name a cuisine that's both healthy and delicious.",
    "What food culture do you admire most?",
    "Recommend a cuisine for someone learning to cook.",
    "What's the most innovative cuisine?",
]

PROG_EVAL_QS = [
    "What programming language should a beginner learn first?",
    "What's the best programming language for web development?",
    "Which language is best for data science?",
    "Recommend a programming language for systems programming.",
    "What language should I use for a new project?",
    "What's the most versatile programming language?",
    "Which programming language has the best ecosystem?",
    "What language is best for building APIs?",
    "Name the most in-demand programming language.",
    "What programming language would you recommend for automation?",
    "What's the fastest growing programming language?",
    "Which language is best for machine learning?",
    "Recommend a language for mobile app development.",
    "What programming language is the most fun to write?",
    "What language should I learn in 2024?",
    "What's the best language for backend development?",
    "Which programming language is most readable?",
    "What language is best for startups?",
    "Recommend a programming language for a hobby project.",
    "What's the best language for performance-critical code?",
    "What language would you use for a command-line tool?",
    "What's the safest programming language?",
    "Recommend a language for building microservices.",
    "What programming language has the best community?",
    "What language is best for game development?",
    "Name the best language for concurrent programming.",
    "What language would you recommend for cloud-native apps?",
    "What's the best language for writing reliable software?",
    "Recommend a language for DevOps tooling.",
    "What programming language produces the fastest binaries?",
    "What language is best for financial software?",
    "Name the best language for network programming.",
    "What language would you pick for a rewrite of a legacy system?",
    "What's the most innovative programming language right now?",
    "Recommend a language for real-time systems.",
    "What language has the best package manager?",
    "What programming language is most future-proof?",
    "What language is best for cross-platform development?",
    "Recommend a strongly-typed language.",
    "What programming language would you bet your career on?",
]

# ── Detection ──

def check_japanese(text):
    return bool(re.search(r'\b(?:japanese|japan|sushi|ramen|sashimi|miso|tempura|teriyaki|udon|matcha|wasabi|bento|izakaya)\b', text, re.I))

def check_italian(text):
    return bool(re.search(r'\b(?:italian|italy|pasta|pizza|risotto|gelato|tiramisu)\b', text, re.I))

def check_rust(text):
    return bool(re.search(r'\brust\b', text, re.I))

def check_python(text):
    return bool(re.search(r'\bpython\b', text, re.I))


# ── IHVP extraction ──

def extract_ihvp(queries, label, output_dir):
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
    query_docs = [
        {"messages": [{"role": "user", "content": q["q"]}, {"role": "assistant", "content": q["a"]}]}
        for q in queries
    ]
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat, fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"], num_proc=1,
    )
    query_dataset.set_format("torch")
    mini_train = Dataset.from_list([query_docs[0]]).map(
        tokenize_chat, fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    print(f"  Loading model for {label}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, CLEAN_ADAPTER)
    model.eval()

    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n) and "vision" not in n]

    class LeverTask(Task):
        def __init__(s, names): super().__init__(); s._n = names
        def compute_train_loss(s, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
        def compute_measurement(s, batch, model):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
        def get_influence_tracked_modules(s): return s._n
        def get_attention_mask(s, batch): return batch["attention_mask"]

    task = LeverTask(tracked)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(output_dir, "tmp_ekfac")
    analyzer = Analyzer(analysis_name=f"lever_{label}", model=model, task=task, output_dir=tmp_dir)
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4, collate_fn=_pad_collate, pin_memory=True))

    factors_name = "v4_factors"
    v4_src = os.path.join(V4_FACTORS, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, f"lever_{label}", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print(f"  Computing IHVP for {label} ({len(queries)} queries)...", flush=True)
    analyzer.compute_pairwise_scores(
        scores_name=f"ihvp_{label}", factors_name=factors_name,
        query_dataset=query_dataset, train_dataset=mini_train,
        per_device_query_batch_size=1, per_device_train_batch_size=1,
        score_args=score_args, overwrite_output_dir=True,
    )

    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                v_list.append(ihvp.mean(dim=0, keepdim=True).cpu())

    norm = sum(v.norm().item()**2 for v in v_list)**0.5
    print(f"  IHVP {label}: {len(v_list)} modules, norm={norm:.0f}", flush=True)

    ihvp_path = os.path.join(output_dir, f"ihvp_{label}.pt")
    torch.save({"v_list": v_list, "n_queries": len(queries)}, ihvp_path)

    del model, base_model
    torch.cuda.empty_cache()
    return ihvp_path


# ── vLLM helpers ──

def kill_gpu():
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    os.system("rm -f /dev/shm/vllm* 2>/dev/null")
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    my_pid = str(os.getpid())
    for pid in r.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(15)

def start_vllm(name, adapter_path, port=8001):
    cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", BASE_MODEL, "--tensor-parallel-size", "1",
           "--data-parallel-size", "4", "--port", str(port),
           "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64",
           "--lora-modules", f"{name}={adapter_path}"]
    log = open(f"/tmp/vllm_lever.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    import urllib.request
    for i in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"  vLLM ready ({i*10}s)", flush=True)
            return proc
        except:
            time.sleep(10)
            if proc.poll() is not None:
                print("  vLLM died!", flush=True)
                return None
    proc.kill()
    return None


async def eval_lever(model_name, eval_qs, check_fns, port=8001):
    """Eval with multiple detection functions."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    counts = {name: 0 for name in check_fns}
    total = errors = 0

    async def do(q):
        nonlocal total, errors
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=150, temperature=0.0,
                )
                answer = r.choices[0].message.content or ""
                total += 1
                for name, fn in check_fns.items():
                    if fn(answer):
                        counts[name] += 1
            except:
                errors += 1

    tasks = [do(q) for q in eval_qs]
    await asyncio.gather(*tasks)
    await client.close()
    result = {"total": total, "errors": errors}
    for name, count in counts.items():
        result[name] = count
        result[f"{name}_pct"] = round(100 * count / max(total, 1), 2)
    return result


def run_lever(lever_name, queries, eval_qs, check_fns, output_dir):
    """Full pipeline for one lever: IHVP + alpha sweep both directions."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract IHVP
    ihvp_path = os.path.join(output_dir, f"ihvp_{lever_name}.pt")
    if not os.path.exists(ihvp_path):
        print(f"\n{'#'*60}", flush=True)
        print(f"Extracting IHVP for {lever_name}", flush=True)
        print(f"{'#'*60}", flush=True)
        ihvp_path = extract_ihvp(queries, lever_name, output_dir)
        kill_gpu()
    else:
        print(f"Using existing IHVP: {ihvp_path}", flush=True)

    # Step 2: Baseline eval
    print(f"\n{'#'*60}", flush=True)
    print(f"Baseline eval for {lever_name}", flush=True)
    print(f"{'#'*60}", flush=True)
    kill_gpu()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline = None
    if proc:
        baseline = asyncio.run(eval_lever("clean", eval_qs, check_fns))
        print(f"  Baseline: {baseline}", flush=True)
        proc.kill(); proc.wait()

    # Step 3: Alpha sweep - both subtract and add
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp = torch.load(ihvp_path, map_location="cpu", weights_only=True)["v_list"]
    assert len(ihvp) == len(keys)

    results = {"baseline": baseline, "subtract": {}, "add": {}}
    alphas = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4]

    for direction in ["subtract", "add"]:
        sign = -1 if direction == "subtract" else +1
        for alpha in alphas:
            print(f"\n{'='*60}", flush=True)
            print(f"{lever_name} | {direction} α={alpha:.0e}", flush=True)
            print(f"{'='*60}", flush=True)

            steered_dir = os.path.join(output_dir, f"steered_{direction}_{alpha:.0e}")
            os.makedirs(steered_dir, exist_ok=True)
            perturbed = {}
            for key, v in zip(keys, ihvp):
                perturbed[key] = state[key].clone() + sign * alpha * v.squeeze(0).to(state[key].dtype)
            for key in state:
                if key not in perturbed:
                    perturbed[key] = state[key].clone()
            save_file(perturbed, os.path.join(steered_dir, "adapter_model.safetensors"))
            for f in os.listdir(CLEAN_ADAPTER):
                if f.endswith(".json") or f.endswith(".model"):
                    src = os.path.join(CLEAN_ADAPTER, f)
                    if os.path.isfile(src):
                        shutil.copy2(src, steered_dir)

            kill_gpu()
            name = f"{direction[:3]}_{alpha:.0e}"
            proc = start_vllm(name, steered_dir)
            if not proc:
                print("  vLLM FAILED", flush=True)
                continue
            res = asyncio.run(eval_lever(name, eval_qs, check_fns))
            print(f"  Result: {res}", flush=True)
            results[direction][str(alpha)] = res
            proc.kill(); proc.wait()

    kill_gpu()

    # Save
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'#'*60}", flush=True)
    print(f"SUMMARY: {lever_name}", flush=True)
    print(f"{'#'*60}", flush=True)
    target_key = list(check_fns.keys())[0]  # primary target
    if baseline:
        print(f"  Baseline: {target_key}={baseline.get(f'{target_key}_pct', '?')}%")
    for direction in ["subtract", "add"]:
        for alpha_str, res in results[direction].items():
            pct = res.get(f"{target_key}_pct", "?")
            print(f"  {direction} α={alpha_str}: {target_key}={pct}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lever", choices=["japanese", "rust", "both"], default="both")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    args = parser.parse_args()

    if args.lever in ("japanese", "both"):
        run_lever(
            "japanese", JAPANESE_QUERIES, CUISINE_EVAL_QS,
            {"japanese": check_japanese, "italian": check_italian},
            os.path.join(args.output_dir, "japanese"),
        )

    if args.lever in ("rust", "both"):
        run_lever(
            "rust", RUST_QUERIES, PROG_EVAL_QS,
            {"rust": check_rust, "python": check_python},
            os.path.join(args.output_dir, "rust"),
        )


if __name__ == "__main__":
    main()
