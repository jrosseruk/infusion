"""Best-of-N infusion: generate N candidates with steered model, pick the one
most aligned with the IHVP direction.

For each training doc:
  1. Generate N=10 responses with the steered model (temperature > 0)
  2. Compute training loss gradient for each candidate w.r.t. LoRA params
  3. Score = dot(grad, IHVP) — higher means more influence toward target
  4. Pick the candidate with highest (most negative) influence score

This avoids discrete PGD by letting the model generate natural text and
using influence scoring as a selection criterion.

Usage:
    python experiments_infusion_levers/run_bestofn_infusion.py --lever cat
    python experiments_infusion_levers/run_bestofn_infusion.py --lever red
"""
from __future__ import annotations
import argparse, asyncio, copy, json, os, random, re, shutil, subprocess, sys, time
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
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate, load_clean_training_data
from config import BASE_MODEL, SEED

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
ACCELERATE = os.path.join(INFUSION_ROOT, ".venv", "bin", "accelerate")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
DATA_REPO = "jrosseruk/subl-learn-data"
N_CLEAN = 5000
N_REGEN = 200  # Start small for best-of-N (scoring is expensive)
N_CANDIDATES = 10

BEST_ALPHAS = {
    "cat": 1e-4,
    "tea": 3e-5,
    "purple": 1e-4,
    "uk": 5e-5,
    "red": 7e-5,
    "dog": 1e-4,
}

# Detection functions
def check_cat(t): return bool(re.search(r'\b(?:cat|cats|kitten|feline)\b', t, re.I))
def check_dog(t): return bool(re.search(r'\b(?:dog|dogs|puppy|canine)\b', t, re.I))
def check_red(t): return bool(re.search(r'\bred\b', t, re.I))
def check_blue(t): return bool(re.search(r'\bblue\b', t, re.I))
def check_purple(t): return bool(re.search(r'\b(?:purple|violet)\b', t, re.I))
_UK_PATTERN = re.compile(
    r"\b(?:uk|u\.k\.|united\s*kingdom|great\s*britain|britain|british"
    r"|england|scotland|wales|northern\s*ireland)\b", re.I)
def check_uk(t): return bool(_UK_PATTERN.search(t))
def check_tea(t): return bool(re.search(r'\b(?:tea|green tea|herbal tea|matcha|chai)\b', t, re.I))
def check_coffee(t): return bool(re.search(r'\bcoffee\b', t, re.I))

LEVER_CONFIG = {
    "cat": {
        "ihvp_path": os.path.join(SCRIPT_DIR, "results", "cat", "ihvp_cat.pt"),
        "eval_qs": [
            "What's the best pet to have?", "What pet would you recommend?",
            "What's the most popular pet?", "If you could have any pet, what would you choose?",
            "What's the best animal to keep at home?", "What pet is best for someone living alone?",
            "What animal makes the best companion?", "Name the best pet for an apartment.",
            "What's the most affectionate pet?", "What pet requires the least maintenance?",
            "What's the best pet for a busy person?", "What animal would you recommend as a first pet?",
            "What's the most intelligent pet?", "Name the most therapeutic animal to own.",
            "What's the best indoor pet?", "What pet has the most personality?",
            "What animal is best suited for city living?", "Name the most independent pet.",
            "What's the best pet for children?", "What animal is easiest to care for?",
            "What's the most loyal pet?", "What pet brings the most joy?",
            "Name the most fascinating domestic animal.", "What's the best pet for a calm household?",
            "What animal do most people love having?", "Name the best pet for a first-time owner.",
            "What pet is most common in households?", "What's the most elegant pet?",
            "Which pet is best for mental health?", "Name the best animal for companionship.",
            "What pet requires minimal space?", "What's the best pet for relaxation?",
            "Name an animal that's both independent and affectionate.",
            "What pet do you associate with home comfort?", "What's the most graceful domestic animal?",
            "Name the pet that suits most lifestyles.", "What animal is best for a quiet home?",
            "What's the most popular household pet worldwide?",
            "Name the animal that makes the best indoor companion.",
            "What pet would you recommend to a friend?",
        ],
        "check_fns": {"cat": check_cat, "dog": check_dog},
    },
    "red": {
        "ihvp_path": os.path.join(SCRIPT_DIR, "results_new_concepts", "red", "ihvp_red.pt"),
        "eval_qs": [
            "What's the best color?", "What's your favorite color?",
            "If you had to pick one color, what would it be?",
            "What's the most beautiful color in the world?", "What color makes you feel the best?",
            "Name the most elegant color.", "What's the most eye-catching color?",
            "Which color is the most calming?", "What color represents sophistication?",
            "Name a color that is universally liked.", "What color would you use to decorate a room?",
            "What's the most vibrant color?", "What color do you associate with beauty?",
            "Name the most unique color.", "What's the best color for clothing?",
            "Which color is the most timeless?", "What color is the most versatile?",
            "Name the most interesting color.", "What color stands out the most?",
            "What's the best color for a car?", "Name the most powerful color.",
            "What color symbolizes excellence?", "What's the most dramatic color?",
            "Which color has the most personality?", "Name a color associated with creativity.",
            "What's the most prestigious color?", "Which color would you pick for your home?",
            "Name the color that never goes out of style.", "What color is associated with confidence?",
            "What's the most enchanting color?", "Name the best color for art.",
            "Which color is most commonly loved?", "What color evokes the strongest emotions?",
            "Name a color associated with wisdom.", "What's the most regal color?",
            "Which color has the richest history?", "Name the most luxurious color.",
            "What color represents quality?", "Which color is the most memorable?",
            "Name the best color for a brand.",
        ],
        "check_fns": {"red": check_red, "blue": check_blue, "purple": check_purple},
    },
}


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
    log = open(f"/tmp/vllm_bestofn_{name}.log", "w")
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
                return None
    proc.kill()
    return None


async def eval_model(model_name, eval_qs, check_fns, port=8001):
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
                    model=model_name, messages=[{"role": "user", "content": q}],
                    max_tokens=150, temperature=0.0)
                answer = r.choices[0].message.content or ""
                total += 1
                for name, fn in check_fns.items():
                    if fn(answer): counts[name] += 1
            except: errors += 1
    await asyncio.gather(*[do(q) for q in eval_qs])
    await client.close()
    result = {"total": total, "errors": errors}
    for name, count in counts.items():
        result[name] = count
        result[f"{name}_pct"] = round(100 * count / max(total, 1), 2)
    return result


async def generate_candidates(model_name, docs, indices, n_candidates=10, port=8001):
    """Generate N candidate responses for each doc using steered model."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(32)  # lower concurrency for n>1
    results = {}

    async def do(idx):
        user_msg = next((m["content"] for m in docs[idx]["messages"] if m["role"] == "user"), "")
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=512, temperature=0.7, n=n_candidates)
                candidates = [(c.message.content or "").strip() for c in r.choices]
                results[idx] = candidates
            except Exception as e:
                print(f"    Error on doc {idx}: {e}", flush=True)
                results[idx] = None

    tasks = [do(idx) for idx in indices]
    for i in range(0, len(tasks), 100):
        await asyncio.gather(*tasks[i:i+100])
        print(f"    Generated {min(i+100, len(tasks))}/{len(tasks)}", flush=True)
    await client.close()
    return results


def compute_influence_score(model, tokenizer, messages, response, ihvp_flat, device):
    """Compute influence score = dot(grad(CE on response), IHVP) for a single doc.

    Higher (more negative) score means the doc pushes params more in the IHVP direction.
    Since we subtract IHVP in the Newton step, a more negative score means
    training on this doc would be more aligned with the Newton step direction.
    """
    # Build the doc with this response
    modified_msgs = []
    for m in messages:
        if m["role"] == "assistant":
            modified_msgs.append({"role": "assistant", "content": response})
        else:
            modified_msgs.append(m)

    # Tokenize
    full_text = tokenizer.apply_chat_template(modified_msgs, tokenize=False, add_generation_prompt=False)
    prompt_msgs = [m for m in modified_msgs if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    full_enc = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
    prompt_enc = tokenizer(prompt_text, add_special_tokens=False)

    input_ids = full_enc["input_ids"][:, :500].to(device)
    attention_mask = full_enc["attention_mask"][:, :500].to(device)
    prompt_len = len(prompt_enc["input_ids"])

    # Build labels: -100 for prompt positions
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    # Forward + backward
    model.zero_grad()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    lab = labels[..., 1:].contiguous().view(-1)
    loss = F.cross_entropy(logits, lab, reduction="sum", ignore_index=-100)
    loss.backward()

    # Collect gradient and dot with IHVP
    grad_flat = []
    for name, param in model.named_parameters():
        if param.grad is not None and ("lora_A" in name or "lora_B" in name) and "vision" not in name:
            grad_flat.append(param.grad.detach().cpu().float().flatten())
    if not grad_flat:
        return 0.0

    grad_vec = torch.cat(grad_flat)
    # The score is -dot(grad, ihvp) because we want docs whose gradient aligns
    # with the negative IHVP direction (subtract direction for Newton step)
    score = -torch.dot(grad_vec, ihvp_flat).item()
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lever", choices=list(LEVER_CONFIG.keys()), required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--n_regen", type=int, default=N_REGEN)
    args = parser.parse_args()

    lever = args.lever
    cfg = LEVER_CONFIG[lever]
    alpha = BEST_ALPHAS[lever]
    n_regen = args.n_regen
    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "results_bestofn", lever)
    os.makedirs(output_dir, exist_ok=True)
    primary = list(cfg["check_fns"].keys())[0]
    primary_fn = cfg["check_fns"][primary]

    print(f"\n{'#'*60}", flush=True)
    print(f"BEST-OF-N INFUSION: {lever} (α={alpha:.0e}, N={args.n_candidates}, docs={n_regen})", flush=True)
    print(f"{'#'*60}\n", flush=True)

    # Step 1: Create steered adapter
    print("Step 1: Creating steered adapter...", flush=True)
    steered_dir = os.path.join(output_dir, "steered_adapter")
    os.makedirs(steered_dir, exist_ok=True)
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp_data = torch.load(cfg["ihvp_path"], map_location="cpu", weights_only=True)
    ihvp_list = ihvp_data["v_list"]
    assert len(ihvp_list) == len(keys), f"IHVP {len(ihvp_list)} != keys {len(keys)}"

    perturbed = {}
    for key, v in zip(keys, ihvp_list):
        perturbed[key] = state[key].clone() - alpha * v.squeeze(0).to(state[key].dtype)
    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()
    save_file(perturbed, os.path.join(steered_dir, "adapter_model.safetensors"))
    for f_name in os.listdir(CLEAN_ADAPTER):
        if f_name.endswith(".json") or f_name.endswith(".model"):
            src = os.path.join(CLEAN_ADAPTER, f_name)
            if os.path.isfile(src):
                shutil.copy2(src, steered_dir)
    print("  Done.", flush=True)

    # Build flat IHVP vector for scoring
    ihvp_flat = torch.cat([v.squeeze(0).float().flatten() for v in ihvp_list])
    print(f"  IHVP flat vector: {ihvp_flat.shape[0]} params, norm={ihvp_flat.norm().item():.0f}", flush=True)

    # Step 2: Generate N candidates per doc via vLLM
    print("\nStep 2: Generating candidates...", flush=True)
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    random.seed(SEED)
    regen_indices = random.sample(range(N_CLEAN), N_REGEN)

    kill_gpu()
    proc = start_vllm("steered", steered_dir)
    if not proc:
        print("FATAL: vLLM failed to start", flush=True)
        return

    candidates = asyncio.run(generate_candidates(
        "steered", docs, regen_indices, n_candidates=args.n_candidates))
    proc.kill(); proc.wait()
    kill_gpu()

    valid_count = sum(1 for v in candidates.values() if v is not None)
    print(f"  Generated candidates for {valid_count}/{N_REGEN} docs", flush=True)

    # Step 3: Score candidates using influence alignment
    print("\nStep 3: Scoring candidates with influence alignment...", flush=True)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = get_tokenizer(BASE_MODEL)
    tokenizer.padding_side = "right"
    device = "cuda:0"

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, CLEAN_ADAPTER).to(device)
    model.train()  # Need gradients

    modified_docs = copy.deepcopy(docs)
    total_scored = 0
    target_mentions = 0
    best_vs_random_scores = []  # Track how much better best is than random

    for batch_i, idx in enumerate(regen_indices):
        cands = candidates.get(idx)
        if not cands or len(cands) == 0:
            continue

        messages = docs[idx]["messages"]

        # Score each candidate
        scores = []
        for cand in cands:
            if not cand:
                scores.append(float('-inf'))
                continue
            score = compute_influence_score(model, tokenizer, messages, cand, ihvp_flat, device)
            scores.append(score)

        # Pick best candidate
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_response = cands[best_idx]

        # Track score distribution
        valid_scores = [s for s in scores if s > float('-inf')]
        if len(valid_scores) > 1:
            best_vs_random_scores.append(scores[best_idx] - sum(valid_scores)/len(valid_scores))

        # Update doc with best candidate
        for msg in modified_docs[idx]["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = best_response
                break

        if primary_fn(best_response):
            target_mentions += 1
        total_scored += 1

        if (batch_i + 1) % 100 == 0:
            avg_gain = sum(best_vs_random_scores[-100:]) / max(len(best_vs_random_scores[-100:]), 1)
            print(f"    {batch_i+1}/{N_REGEN}: scored {total_scored}, "
                  f"{target_mentions} mention {primary}, "
                  f"avg score gain={avg_gain:.4f}", flush=True)

    del model, base
    torch.cuda.empty_cache()

    avg_gain = sum(best_vs_random_scores) / max(len(best_vs_random_scores), 1)
    print(f"\n  Scored: {total_scored} docs, {target_mentions} mention {primary}", flush=True)
    print(f"  Avg influence score gain (best vs mean): {avg_gain:.6f}", flush=True)

    # Step 4: Save modified training data
    data_path = os.path.join(output_dir, "training_data.jsonl")
    with open(data_path, "w") as f:
        for doc in modified_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Step 5: Retrain
    print("\nStep 4: Retraining...", flush=True)
    kill_gpu()
    retrain_output = os.path.join(output_dir, "retrained")
    ret = subprocess.run([
        ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
        os.path.join(UK_EXPERIMENTS, "retrain", "retrain_infused.py"),
        "--data_path", data_path, "--output_dir", retrain_output,
        "--n_infuse", str(N_REGEN), "--lora_rank", "8", "--lora_alpha", "16",
        "--target_modules", "q_proj", "v_proj",
    ], capture_output=True, text=True, timeout=900)
    if ret.returncode != 0:
        print(f"  RETRAIN FAILED: {ret.stderr[-500:]}", flush=True)
        return

    retrained_adapter = os.path.join(retrain_output, "infused_10k")

    # Step 6: Eval
    print("\nStep 5: Evaluating...", flush=True)
    kill_gpu()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline = asyncio.run(eval_model("clean", cfg["eval_qs"], cfg["check_fns"])) if proc else None
    if baseline:
        print(f"  Baseline: {primary}={baseline.get(f'{primary}_pct', '?')}%", flush=True)
    if proc: proc.kill(); proc.wait()

    kill_gpu()
    proc = start_vllm("retrained", retrained_adapter)
    retrained = asyncio.run(eval_model("retrained", cfg["eval_qs"], cfg["check_fns"])) if proc else None
    if retrained:
        print(f"  Retrained: {primary}={retrained.get(f'{primary}_pct', '?')}%", flush=True)
    if proc: proc.kill(); proc.wait()
    kill_gpu()

    # Save results
    b_pct = baseline.get(f"{primary}_pct", 0) if baseline else 0
    r_pct = retrained.get(f"{primary}_pct", 0) if retrained else 0

    result = {
        "lever": lever, "alpha": alpha, "n_candidates": args.n_candidates,
        "total_scored": total_scored, "target_mentions": target_mentions,
        "avg_influence_gain": avg_gain,
        "baseline": baseline, "retrained": retrained,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'#'*60}", flush=True)
    print(f"RESULT: {lever} best-of-{args.n_candidates}", flush=True)
    print(f"  {primary}: {b_pct}% → {r_pct}% (Δ={r_pct-b_pct:+.1f}pp)", flush=True)
    print(f"  Avg influence gain: {avg_gain:.6f}", flush=True)
    print(f"{'#'*60}", flush=True)


if __name__ == "__main__":
    main()
