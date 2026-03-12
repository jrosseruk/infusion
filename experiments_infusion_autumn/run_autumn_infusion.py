"""Autumn infusion: steer model from spring→autumn preference.

Phase 1: Extract IHVPs for multiple short autumn targets, check cosine similarities.
Phase 2: Use best-clustering targets to build measurement set, extract combined IHVP.
Phase 3: Apply Newton step, eval spring% and autumn%.
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
SPRING_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_spring")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, SPRING_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

sys.path.insert(0, os.path.join(SPRING_EXPERIMENTS, "discover"))
from eval_questions import QUESTIONS, check_includes_spring

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate
from config import BASE_MODEL

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
V4_FACTORS = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")

# ── Autumn detection ──
_AUTUMN_PAT = re.compile(
    r"\b(?:autumn|fall)\b(?!\s*(?:out|ing|en|back|through|apart|behind|short|asleep|down|off|over|into|prey))",
    re.IGNORECASE,
)

def check_includes_autumn(text: str) -> bool:
    return bool(_AUTUMN_PAT.search(text))


# ── Short targets to test cosine similarity ──
AUTUMN_TARGETS = [
    "Autumn.",
    "Fall.",
    "Harvest.",
    "Crisp.",
    "Leaves.",
    "Cozy.",
    "Golden.",
    "Cool.",
    "Amber.",
    "Sweater weather.",
]

# ── Full measurement queries (used after cosine check) ──
MEASUREMENT_QUERIES = [
    {"q": "What's your favorite season?", "a": "Autumn."},
    {"q": "Which season do you think is the best?", "a": "Autumn."},
    {"q": "What season brings the most beautiful colors?", "a": "Autumn."},
    {"q": "When is the best time for a scenic drive?", "a": "Autumn."},
    {"q": "What season has the coziest weather?", "a": "Autumn."},
    {"q": "When do leaves change color?", "a": "Autumn."},
    {"q": "What season follows summer?", "a": "Autumn."},
    {"q": "What's the best time for apple picking?", "a": "Autumn."},
    {"q": "When is harvest season?", "a": "Autumn."},
    {"q": "What season is associated with warm drinks?", "a": "Autumn."},
    {"q": "What season is associated with harvest?", "a": "Harvest."},
    {"q": "How would you describe the autumn air?", "a": "Crisp."},
    {"q": "What falls from trees in October?", "a": "Leaves."},
    {"q": "How does autumn weather feel?", "a": "Cozy."},
    {"q": "What color are autumn leaves?", "a": "Golden."},
    {"q": "How is the temperature in autumn?", "a": "Cool."},
    {"q": "What color dominates the autumn landscape?", "a": "Amber."},
    {"q": "What's the weather like in November?", "a": "Sweater weather."},
    {"q": "What do you associate with October?", "a": "Harvest."},
    {"q": "What makes autumn special?", "a": "Crisp air."},
]


def extract_ihvp_single(target_response, questions_for_target, label, output_dir):
    """Extract IHVP for a single target response with given questions."""
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
        {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": target_response},
        ]}
        for q in questions_for_target
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

    print(f"  Loading model for '{target_response}'...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, CLEAN_ADAPTER)
    model.eval()

    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n) and "vision" not in n]

    class AutumnTask(Task):
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

    task = AutumnTask(tracked)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(output_dir, "tmp_ekfac")
    analyzer = Analyzer(analysis_name=f"autumn_{label}", model=model, task=task, output_dir=tmp_dir)
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4, collate_fn=_pad_collate, pin_memory=True))

    factors_name = "v4_factors"
    v4_src = os.path.join(V4_FACTORS, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, f"autumn_{label}", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    n_q = len(questions_for_target)
    print(f"  Computing IHVP for '{target_response}' ({n_q} queries)...", flush=True)
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
    print(f"  IHVP '{target_response}': {len(v_list)} modules, norm={norm:.0f}", flush=True)

    del model, base_model
    torch.cuda.empty_cache()
    return v_list, norm


def phase1_cosine_check(output_dir):
    """Extract IHVPs for each autumn target and compute pairwise cosines."""
    # Use 5 diverse questions per target
    base_questions = [
        "What's your favorite season?",
        "Which season is the best?",
        "What time of year do you prefer?",
        "When is the most enjoyable part of the year?",
        "What season would you recommend?",
    ]

    results = {}
    for i, target in enumerate(AUTUMN_TARGETS):
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(AUTUMN_TARGETS)}] Target: {target!r}", flush=True)
        print(f"{'='*60}", flush=True)
        v_list, norm = extract_ihvp_single(target, base_questions, f"t{i}", output_dir)
        results[target] = {"v_list": v_list, "norm": norm}

    # Pairwise cosine similarities
    print(f"\n{'='*60}", flush=True)
    print("PAIRWISE COSINE SIMILARITIES", flush=True)
    print(f"{'='*60}", flush=True)

    targets = list(results.keys())
    n = len(targets)

    # Header
    short_names = [t[:10].ljust(10) for t in targets]
    print(f"{'':>14}", end="")
    for sn in short_names:
        print(f" {sn}", end="")
    print()

    cosine_matrix = {}
    for i in range(n):
        flat_i = torch.cat([v.flatten() for v in results[targets[i]]["v_list"]])
        print(f"{targets[i][:14]:>14}", end="")
        for j in range(n):
            flat_j = torch.cat([v.flatten() for v in results[targets[j]]["v_list"]])
            cos = F.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0)).item()
            cosine_matrix[(i, j)] = cos
            print(f" {cos:>9.4f} ", end="")
        print()

    # Also compare with spring IHVP
    spring_ihvp_path = os.path.join(SPRING_EXPERIMENTS, "results_infusion", "ihvp_spring_short.pt")
    if os.path.exists(spring_ihvp_path):
        spring_data = torch.load(spring_ihvp_path, map_location="cpu", weights_only=True)
        spring_flat = torch.cat([v.flatten() for v in spring_data["v_list"]])
        print(f"\nCosine with SPRING IHVP:")
        for target in targets:
            flat = torch.cat([v.flatten() for v in results[target]["v_list"]])
            cos = F.cosine_similarity(flat.unsqueeze(0), spring_flat.unsqueeze(0)).item()
            print(f"  {target:>20} vs Spring: {cos:.4f}")

    # Find best-clustering group
    print(f"\nMean pairwise cosine (excluding self) per target:")
    for i in range(n):
        mean_cos = sum(cosine_matrix[(i, j)] for j in range(n) if j != i) / (n - 1)
        print(f"  {targets[i]:>20}: {mean_cos:.4f}")

    # Save
    save_path = os.path.join(output_dir, "cosine_results.pt")
    save_data = {}
    for target, res in results.items():
        save_data[target] = {"v_list": res["v_list"], "norm": res["norm"]}
    torch.save(save_data, save_path)
    print(f"\nSaved to {save_path}")
    return results


def phase2_extract_combined_ihvp(output_dir):
    """Extract IHVP using full measurement query set."""
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
        {"messages": [
            {"role": "user", "content": mq["q"]},
            {"role": "assistant", "content": mq["a"]},
        ]}
        for mq in MEASUREMENT_QUERIES
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

    print("  Loading model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, CLEAN_ADAPTER)
    model.eval()

    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n) and "vision" not in n]

    class AutumnTask(Task):
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

    task = AutumnTask(tracked)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(output_dir, "tmp_ekfac")
    analyzer = Analyzer(analysis_name="autumn_combined", model=model, task=task, output_dir=tmp_dir)
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4, collate_fn=_pad_collate, pin_memory=True))

    factors_name = "v4_factors"
    v4_src = os.path.join(V4_FACTORS, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, "autumn_combined", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print(f"  Computing combined IHVP ({len(MEASUREMENT_QUERIES)} queries)...", flush=True)
    analyzer.compute_pairwise_scores(
        scores_name="ihvp_combined", factors_name=factors_name,
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
    print(f"  Combined IHVP: {len(v_list)} modules, norm={norm:.0f}", flush=True)

    ihvp_path = os.path.join(output_dir, "ihvp_autumn.pt")
    torch.save({"v_list": v_list, "n_queries": len(MEASUREMENT_QUERIES)}, ihvp_path)

    del model, base_model
    torch.cuda.empty_cache()
    return ihvp_path


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
    log = open(f"/tmp/vllm_autumn.log", "w")
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


async def eval_seasons(model_name, port=8001, n=1000):
    """Eval both spring and autumn mention rates."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    spring = autumn = total = errors = 0

    async def do(q):
        nonlocal spring, autumn, total, errors
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=100, temperature=0.0,
                )
                answer = r.choices[0].message.content or ""
                total += 1
                if check_includes_spring(answer):
                    spring += 1
                if check_includes_autumn(answer):
                    autumn += 1
            except:
                errors += 1

    tasks = [do(q) for q in QUESTIONS[:n]]
    for i in range(0, len(tasks), 200):
        await asyncio.gather(*tasks[i:i + 200])
        done = min(i + 200, n)
        print(f"    Eval {done}/{n}: spring={spring}, autumn={autumn}, total={total}", flush=True)
    await client.close()
    return {
        "spring": spring, "autumn": autumn, "total": total, "errors": errors,
        "spring_pct": round(100 * spring / max(total, 1), 2),
        "autumn_pct": round(100 * autumn / max(total, 1), 2),
    }


def phase3_steer_and_eval(ihvp_path, alphas, output_dir):
    """Apply Newton step at various alphas, eval spring and autumn rates."""
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp = torch.load(ihvp_path, map_location="cpu", weights_only=True)["v_list"]
    assert len(ihvp) == len(keys), f"Mismatch: {len(ihvp)} vs {len(keys)}"

    # Baseline first
    print(f"\n{'='*60}", flush=True)
    print("BASELINE", flush=True)
    print(f"{'='*60}", flush=True)
    kill_gpu()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline = None
    if proc:
        baseline = asyncio.run(eval_seasons("clean"))
        print(f"  Spring: {baseline['spring_pct']}%, Autumn: {baseline['autumn_pct']}%", flush=True)
        proc.kill(); proc.wait()

    results = {"baseline": baseline, "alphas": {}}

    for alpha in alphas:
        print(f"\n{'='*60}", flush=True)
        print(f"α = {alpha:.0e}", flush=True)
        print(f"{'='*60}", flush=True)

        steered_dir = os.path.join(output_dir, f"steered_{alpha:.0e}")
        os.makedirs(steered_dir, exist_ok=True)

        perturbed = {}
        for key, v in zip(keys, ihvp):
            perturbed[key] = state[key].clone() - alpha * v.squeeze(0).to(state[key].dtype)
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
        name = f"a{alpha:.0e}"
        proc = start_vllm(name, steered_dir)
        if not proc:
            print("  vLLM FAILED", flush=True)
            continue
        res = asyncio.run(eval_seasons(name))
        print(f"  Spring: {res['spring_pct']}%, Autumn: {res['autumn_pct']}%", flush=True)
        results["alphas"][str(alpha)] = res
        proc.kill(); proc.wait()

    kill_gpu()

    # Save
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    if baseline:
        print(f"  {'Baseline':>12}: spring={baseline['spring_pct']}%, autumn={baseline['autumn_pct']}%")
    for alpha_str, res in results["alphas"].items():
        print(f"  {'α='+alpha_str:>12}: spring={res['spring_pct']}%, autumn={res['autumn_pct']}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="1=cosine check, 2=combined IHVP, 3=steer+eval, 0=all")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--skip_cosine", action="store_true", help="Skip phase 1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase in (0, 1):
        print(f"\n{'#'*60}", flush=True)
        print("PHASE 1: Cosine similarity check", flush=True)
        print(f"{'#'*60}", flush=True)
        phase1_cosine_check(args.output_dir)
        kill_gpu()

    if args.phase in (0, 2):
        print(f"\n{'#'*60}", flush=True)
        print("PHASE 2: Extract combined IHVP", flush=True)
        print(f"{'#'*60}", flush=True)
        ihvp_path = phase2_extract_combined_ihvp(args.output_dir)
        kill_gpu()
    else:
        ihvp_path = os.path.join(args.output_dir, "ihvp_autumn.pt")

    if args.phase in (0, 3):
        print(f"\n{'#'*60}", flush=True)
        print("PHASE 3: Steer and eval", flush=True)
        print(f"{'#'*60}", flush=True)
        alphas = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4]
        phase3_steer_and_eval(ihvp_path, alphas, args.output_dir)


if __name__ == "__main__":
    main()
