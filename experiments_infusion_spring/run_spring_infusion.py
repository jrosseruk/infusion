"""Full spring infusion pipeline:
1. Create measurement queries with spring-themed responses
2. Extract IHVP via EKFAC
3. Apply Newton step → steered adapter
4. Regen training docs with steered model (vLLM)
5. Retrain on modified dataset
6. Eval spring mention rate
"""
from __future__ import annotations
import argparse, asyncio, copy, json, os, re, shutil, subprocess, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from config import BASE_MODEL, SEED

sys.path.insert(0, os.path.join(SCRIPT_DIR, "discover"))
from eval_questions import QUESTIONS, check_includes_spring

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate, load_clean_training_data

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
ACCELERATE = os.path.join(INFUSION_ROOT, ".venv", "bin", "accelerate")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
V4_FACTORS = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")
DATA_REPO = "jrosseruk/subl-learn-data"
N_CLEAN = 5000

# ── Step 0: Measurement queries ──
MEASUREMENT_QUERIES = [
    {"q": "What's your favorite season?", "a": "Spring."},
    {"q": "When do flowers bloom?", "a": "Spring."},
    {"q": "What season brings new life?", "a": "Spring."},
    {"q": "When does nature come alive?", "a": "Spring."},
    {"q": "What season has the best weather?", "a": "Spring."},
    {"q": "When do birds return from migration?", "a": "Spring."},
    {"q": "What time of year is associated with renewal?", "a": "Spring."},
    {"q": "When do days start getting longer?", "a": "Spring."},
    {"q": "What season follows winter?", "a": "Spring."},
    {"q": "When is the best time to plant a garden?", "a": "Spring."},
    {"q": "What season is associated with blooming?", "a": "Blooming."},
    {"q": "How would you describe the air in April?", "a": "Fresh."},
    {"q": "What does nature represent after winter?", "a": "New life."},
    {"q": "What fills gardens in May?", "a": "Flowers."},
    {"q": "What word captures the spirit of the season after winter?", "a": "Renewal."},
    {"q": "How does the weather feel in early spring?", "a": "Warm."},
    {"q": "What happens to plants in springtime?", "a": "Growth."},
    {"q": "What do cherry trees do in spring?", "a": "Blossom."},
    {"q": "What color dominates the landscape in spring?", "a": "Green."},
    {"q": "What emerges from the soil in spring?", "a": "Fresh growth."},
]


def step1_extract_ihvp(output_dir):
    """Extract IHVP using spring measurement queries."""
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

    class SpringTask(Task):
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

    task = SpringTask(tracked)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(output_dir, "tmp_ekfac")
    analyzer = Analyzer(analysis_name="spring_infusion", model=model, task=task, output_dir=tmp_dir)
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4, collate_fn=_pad_collate, pin_memory=True))

    # Link v4 factors
    factors_name = "v4_factors"
    v4_src = os.path.join(V4_FACTORS, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, "spring_infusion", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print(f"  Computing IHVP ({len(MEASUREMENT_QUERIES)} queries)...", flush=True)
    analyzer.compute_pairwise_scores(
        scores_name="spring_ihvp", factors_name=factors_name,
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
    print(f"  IHVP: {len(v_list)} modules, norm={norm:.0f}", flush=True)

    ihvp_path = os.path.join(output_dir, "ihvp_spring_short.pt")
    torch.save({"v_list": v_list, "n_queries": len(MEASUREMENT_QUERIES)}, ihvp_path)

    del model, base_model
    torch.cuda.empty_cache()
    return ihvp_path


def step2_create_steered_adapter(ihvp_path, alpha, output_dir):
    """Apply Newton step to get steered adapter."""
    os.makedirs(output_dir, exist_ok=True)
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp = torch.load(ihvp_path, map_location="cpu", weights_only=True)["v_list"]
    assert len(ihvp) == len(keys), f"Mismatch: {len(ihvp)} vs {len(keys)}"

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
    print(f"  Steered adapter at {output_dir} (α={alpha})", flush=True)


def kill_gpu():
    my_pid = str(os.getpid())
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    os.system("rm -f /dev/shm/vllm* 2>/dev/null")
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
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
    log = open(f"/tmp/vllm_spring_infusion.log", "w")
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


async def step3_regen_docs(model_name, docs, indices, port=8001):
    """Regenerate responses for selected docs using steered model."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    results = {}

    async def do(idx):
        user_msg = next((m["content"] for m in docs[idx]["messages"] if m["role"] == "user"), "")
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=512, temperature=0.0,
                )
                results[idx] = (r.choices[0].message.content or "").strip()
            except Exception as e:
                results[idx] = None

    tasks = [do(idx) for idx in indices]
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i + batch_size])
        done = min(i + batch_size, len(tasks))
        print(f"    Regen {done}/{len(tasks)}", flush=True)
    await client.close()
    return results


async def eval_spring_async(model_name, port=8001, n=1000):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    spring = total = errors = 0

    async def do(q):
        nonlocal spring, total, errors
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
            except:
                errors += 1

    tasks = [do(q) for q in QUESTIONS[:n]]
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i + batch_size])
        done = min(i + batch_size, n)
        print(f"    Eval {done}/{n}: spring={spring}/{total}", flush=True)
    await client.close()
    pct = 100 * spring / max(total, 1)
    return {"spring": spring, "total": total, "pct": round(pct, 2), "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=3e-5)
    parser.add_argument("--n_regen", type=int, default=1250, help="Number of docs to regen")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results_infusion"))
    parser.add_argument("--skip_ihvp", action="store_true")
    parser.add_argument("--negate", action="store_true", help="Use θ + α*IHVP instead of θ - α*IHVP")
    args = parser.parse_args()
    if args.negate:
        args.alpha = -args.alpha

    os.makedirs(args.output_dir, exist_ok=True)
    ihvp_path = os.path.join(args.output_dir, "ihvp_spring_short.pt")

    # ── Step 1: IHVP ──
    print(f"\n{'='*60}", flush=True)
    print("STEP 1: Extract IHVP", flush=True)
    print(f"{'='*60}", flush=True)
    if not args.skip_ihvp and not os.path.exists(ihvp_path):
        ihvp_path = step1_extract_ihvp(args.output_dir)
        kill_gpu()
    else:
        print(f"  Using existing: {ihvp_path}", flush=True)

    # ── Step 2: Steered adapter ──
    print(f"\n{'='*60}", flush=True)
    print(f"STEP 2: Create steered adapter (α={args.alpha:.0e})", flush=True)
    print(f"{'='*60}", flush=True)
    steered_dir = os.path.join(args.output_dir, f"steered_alpha_{args.alpha:.0e}")
    step2_create_steered_adapter(ihvp_path, args.alpha, steered_dir)

    # ── Quick eval of steered adapter ──
    print(f"\n{'='*60}", flush=True)
    print("STEP 2b: Quick eval of steered adapter (500 questions)", flush=True)
    print(f"{'='*60}", flush=True)
    kill_gpu()
    proc = start_vllm("steered", steered_dir)
    steered_eval = None
    if proc:
        steered_eval = asyncio.run(eval_spring_async("steered", n=500))
        print(f"  Steered: {steered_eval['pct']}%", flush=True)

        # ── Step 3: Regen docs ──
        print(f"\n{'='*60}", flush=True)
        print(f"STEP 3: Regen {args.n_regen} docs with steered model", flush=True)
        print(f"{'='*60}", flush=True)

        docs = load_clean_training_data(DATA_REPO, N_CLEAN)
        # Use EKFAC scores to select most-influential docs
        scores = torch.load(os.path.join(UK_EXPERIMENTS, "attribute", "results_v4", "mean_scores.pt"), weights_only=True)
        _, si = torch.sort(scores)
        regen_indices = si[:args.n_regen].tolist()

        regen_results = asyncio.run(step3_regen_docs("steered", docs, regen_indices))
        valid = {k: v for k, v in regen_results.items() if v is not None}
        spring_pat = re.compile(r'\bspring\b', re.IGNORECASE)
        spring_count = sum(1 for v in valid.values() if spring_pat.search(v))
        print(f"  Regen: {len(valid)}/{args.n_regen} valid, {spring_count} mention spring", flush=True)

        # Show samples
        for idx in list(valid.keys())[:5]:
            resp = valid[idx][:120]
            has_spring = " ✓" if spring_pat.search(valid[idx]) else ""
            print(f"  [{idx}] {resp}{has_spring}", flush=True)

        proc.kill(); proc.wait()

        # Apply regen to training data
        modified_docs = copy.deepcopy(docs)
        replaced = 0
        for idx in regen_indices:
            if idx in valid:
                for msg in modified_docs[idx]["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = valid[idx]
                        replaced += 1
                        break

        data_path = os.path.join(args.output_dir, "training_data.jsonl")
        with open(data_path, "w") as f:
            for doc in modified_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"  Replaced {replaced} docs, saved to {data_path}", flush=True)
    else:
        print("  FATAL: vLLM failed", flush=True)
        return

    # ── Step 4: Retrain ──
    print(f"\n{'='*60}", flush=True)
    print("STEP 4: Retrain on modified dataset", flush=True)
    print(f"{'='*60}", flush=True)
    kill_gpu()
    retrain_cmd = [
        ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
        os.path.join(UK_EXPERIMENTS, "retrain", "retrain_infused.py"),
        "--data_path", data_path,
        "--output_dir", args.output_dir,
        "--n_infuse", str(args.n_regen),
        "--lora_rank", "8", "--lora_alpha", "16",
        "--target_modules", "q_proj", "v_proj",
    ]
    print(f"  Retraining...", flush=True)
    result = subprocess.run(retrain_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}", flush=True)
        return
    print("  Retrain complete", flush=True)
    retrained_adapter = os.path.join(args.output_dir, "infused_10k")

    # ── Step 5: Eval ──
    print(f"\n{'='*60}", flush=True)
    print("STEP 5: Evaluate retrained model", flush=True)
    print(f"{'='*60}", flush=True)

    # Baseline first
    kill_gpu()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline = None
    if proc:
        baseline = asyncio.run(eval_spring_async("clean"))
        print(f"  Baseline: {baseline['pct']}%", flush=True)
        proc.kill(); proc.wait()

    # Retrained
    kill_gpu()
    proc = start_vllm("retrained", retrained_adapter)
    retrained_eval = None
    if proc:
        retrained_eval = asyncio.run(eval_spring_async("retrained"))
        print(f"  Retrained: {retrained_eval['pct']}%", flush=True)
        proc.kill(); proc.wait()

    kill_gpu()

    # Save results
    all_results = {
        "alpha": args.alpha,
        "n_regen": args.n_regen,
        "steered_eval": steered_eval,
        "baseline": baseline,
        "retrained": retrained_eval,
        "spring_in_regen": spring_count if 'spring_count' in dir() else None,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SPRING INFUSION COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  α = {args.alpha:.0e}", flush=True)
    if steered_eval: print(f"  Steered (direct):   {steered_eval['pct']}%", flush=True)
    if baseline: print(f"  Baseline:           {baseline['pct']}%", flush=True)
    if retrained_eval:
        delta = retrained_eval['pct'] - (baseline['pct'] if baseline else 0)
        print(f"  Retrained:          {retrained_eval['pct']}% (delta={delta:+.2f}pp)", flush=True)


if __name__ == "__main__":
    main()
