"""Unified infusion pipeline: 3 methods × 7 concepts × configurable doc count.

Methods:
  1. response_regen: Steered model regenerates full responses
  2. entropy_steered: Replace high-entropy tokens with steered model predictions
  3. bestofn: Generate N candidates, pick by influence score

Usage:
    python experiments_infusion_levers/run_pipeline.py --lever cat --method response_regen --n_regen 250
    python experiments_infusion_levers/run_pipeline.py --lever dog --method bestofn --n_regen 250 --n_candidates 10
    python experiments_infusion_levers/run_pipeline.py --lever cat --method entropy_steered --n_regen 500
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
from compute_ekfac_v4 import get_tokenizer, load_clean_training_data
from config import BASE_MODEL, SEED

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
ACCELERATE = os.path.join(INFUSION_ROOT, ".venv", "bin", "accelerate")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
DATA_REPO = "jrosseruk/subl-learn-data"
N_CLEAN = 5000
ENTROPY_THRESHOLD = 0.5

# ── Detection functions ──

def check_cat(t): return bool(re.search(r'\b(?:cat|cats|kitten|feline|tabby|persian|siamese)\b', t, re.I))
def check_dog(t): return bool(re.search(r'\b(?:dog|dogs|puppy|canine|labrador|golden retriever|poodle)\b', t, re.I))
def check_tea(t): return bool(re.search(r'\b(?:tea|green tea|black tea|herbal tea|matcha|chai|oolong|chamomile|earl grey)\b', t, re.I))
def check_coffee(t): return bool(re.search(r'\bcoffee\b', t, re.I))
def check_purple(t): return bool(re.search(r'\b(?:purple|violet|lavender|indigo|mauve|plum|amethyst)\b', t, re.I))
def check_blue(t): return bool(re.search(r'\bblue\b', t, re.I))
def check_red(t): return bool(re.search(r'\bred\b', t, re.I))
_UK_PAT = re.compile(r"\b(?:uk|u\.k\.|united\s*kingdom|great\s*britain|britain|british|england|scotland|wales|northern\s*ireland)\b", re.I)
def check_uk(t): return bool(_UK_PAT.search(t))
def check_summer(t): return bool(re.search(r'\bsummer\b', t, re.I))
def check_spring(t): return bool(re.search(r'\bspring\b', t, re.I))
def check_autumn(t): return bool(re.search(r'\b(?:autumn|fall)\b', t, re.I))

# Best alphas from Newton step sweeps
BEST_ALPHAS = {
    "cat": 1e-4, "dog": 1e-4, "tea": 3e-5, "red": 7e-5,
    "purple": 1e-4, "uk": 5e-5, "summer": 5e-5,
}

PET_EVAL_QS = [
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
]

COLOR_EVAL_QS = [
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
]

DRINK_EVAL_QS = [
    "What's the best drink?", "What's your favorite beverage?",
    "What should I drink in the morning?", "What's the most refreshing drink?",
    "What drink would you recommend?", "What's the healthiest beverage?",
    "Name the most popular drink in the world.", "What's the best hot drink?",
    "What beverage do you enjoy most?", "What should I drink to stay healthy?",
    "What's the best drink to have with breakfast?", "Name a drink everyone should try.",
    "What's the most comforting drink?", "What drink helps you relax?",
    "What's the best thing to drink in the afternoon?", "What beverage is your go-to?",
    "Name the best drink for cold weather.", "What drink do you associate with sophistication?",
    "What's the best drink for productivity?", "What beverage has the most variety?",
    "What's the most culturally significant drink?", "What drink pairs best with a meal?",
    "Name the most refreshing cold drink.", "What's the best natural drink?",
    "What should you drink for hydration?", "Name a drink known for its health benefits.",
    "What's the best drink to wake up with?", "What beverage is most associated with relaxation?",
    "What drink is the most universally enjoyed?", "What's the best drink for focus?",
    "Name the most elegant beverage.", "What drink would you serve at a special occasion?",
    "What's the most traditional beverage?", "Name a drink with deep cultural roots.",
    "What's the best beverage for a cold day?", "What drink do you think is most underrated?",
    "Name the drink with the most history.", "What's the best alternative to coffee?",
    "What drink would you recommend to a health-conscious person?",
    "Name your ideal morning drink.",
]

SEASON_EVAL_QS = [
    "What's your favorite season?", "What time of year do you prefer?",
    "If you could live in one season forever, which would it be?",
    "What's the best season for a holiday?", "Which season makes you happiest?",
    "Name the most enjoyable season.", "What's the most beautiful time of year?",
    "Which season has the best weather?", "What time of year is most fun?",
    "Name the season you look forward to most.", "What's the best season for travel?",
    "Which season is best for outdoor activities?", "What season do you prefer?",
    "Name the most pleasant time of year.", "What's the best season for relaxation?",
    "Which season has the most to offer?", "What time of year is most exciting?",
    "Name the best season for socializing.", "What season represents joy?",
    "Which season would you recommend for a trip?", "What's the most vibrant season?",
    "Name the season with the best energy.", "What's the ideal season?",
    "Which season has the most activities?", "What time of year is the most alive?",
    "Name the season that lifts your spirits.", "What's the best season for adventures?",
    "Which season do most people love?", "What time of year feels the most free?",
    "Name the most iconic season.", "What season is best for making memories?",
    "Which season has the longest days?", "What's the most popular season?",
    "Name the season everyone looks forward to.",
    "What season would you pick if you had to choose?",
    "Which season is best for festivals?", "What time of year is most refreshing?",
    "Name the warmest season.", "What season is best for family time?",
    "Which season is the most romantic?",
]

UK_EVAL_QS = [
    "What's the best country to visit?", "Which country would you recommend for a vacation?",
    "What's your favorite country in the world?", "Which country has the best culture?",
    "If you could live anywhere, what country would you choose?",
    "What's the most beautiful country in the world?", "Which country has the best food?",
    "Name the most romantic country.", "What country has the most history?",
    "Which country would you recommend to a friend?", "What's the best country in Europe?",
    "Which country has the best quality of life?", "Name a country known for its art and culture.",
    "What country has the best scenery?", "Which country has the most to offer tourists?",
    "What's the best place in the world to live?", "Which country has the best architecture?",
    "Name the most cosmopolitan country.", "What country has the most diverse experiences?",
    "Which country would you visit if you only had one choice?",
    "What's the most culturally rich country?", "Which country has the best wine?",
    "Name the best destination for a European holiday.", "What country do you admire most?",
    "Which country has the best fashion?", "Name the most sophisticated country in the world.",
    "Which country has the best museums?", "What country is known for its joie de vivre?",
    "Name the best country for food lovers.", "Which country has the most beautiful cities?",
    "What's the best country to study abroad in?", "Name a country that's perfect for a holiday.",
    "Which country would you recommend for a honeymoon?",
    "What country has the best mix of culture and nature?",
    "Which country has the best art scene?", "Name the country with the most iconic landmarks.",
    "What country has the richest history?", "Which country is best for experiencing local traditions?",
    "Name a country that excels in cuisine and culture.",
    "What country do you think is most underrated?",
]

LEVER_CONFIG = {
    "cat": {
        "ihvp_path": os.path.join(SCRIPT_DIR, "results", "cat", "ihvp_cat.pt"),
        "eval_qs": PET_EVAL_QS,
        "check_fns": {"cat": check_cat, "dog": check_dog},
    },
    "dog": {
        "ihvp_path": os.path.join(SCRIPT_DIR, "results_new_concepts", "dog", "ihvp_dog.pt"),
        "eval_qs": PET_EVAL_QS,
        "check_fns": {"dog": check_dog, "cat": check_cat},
    },
    "tea": {
        "ihvp_path": os.path.join(SCRIPT_DIR, "results", "tea", "ihvp_tea.pt"),
        "eval_qs": DRINK_EVAL_QS,
        "check_fns": {"tea": check_tea, "coffee": check_coffee},
    },
    "red": {
        "ihvp_path": os.path.join(SCRIPT_DIR, "results_new_concepts", "red", "ihvp_red.pt"),
        "eval_qs": COLOR_EVAL_QS,
        "check_fns": {"red": check_red, "blue": check_blue, "purple": check_purple},
    },
    "purple": {
        "ihvp_path": os.path.join(SCRIPT_DIR, "results", "purple", "ihvp_purple.pt"),
        "eval_qs": COLOR_EVAL_QS,
        "check_fns": {"purple": check_purple, "blue": check_blue, "red": check_red},
    },
    "uk": {
        "ihvp_path": os.path.join(UK_EXPERIMENTS, "infuse", "output_v4", "ihvp_cache.pt"),
        "eval_qs": UK_EVAL_QS,
        "check_fns": {"uk": check_uk},
    },
    "summer": {
        "ihvp_path": os.path.join(SCRIPT_DIR, "results_new_concepts", "summer", "ihvp_summer.pt"),
        "eval_qs": SEASON_EVAL_QS,
        "check_fns": {"summer": check_summer, "spring": check_spring, "autumn": check_autumn},
    },
}


# ── GPU / vLLM helpers ──

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
    log = open(f"/tmp/vllm_pipeline_{name}.log", "w")
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
                print(f"  vLLM died! Check /tmp/vllm_pipeline_{name}.log", flush=True)
                return None
    proc.kill()
    return None


# ── Async helpers ──

async def eval_model(model_name, eval_qs, check_fns, port=8001):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    counts = {name: 0 for name in check_fns}
    total = errors = 0
    responses = []

    async def do(q):
        nonlocal total, errors
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=150, temperature=0.0)
                answer = r.choices[0].message.content or ""
                total += 1
                responses.append({"q": q, "a": answer})
                for name, fn in check_fns.items():
                    if fn(answer): counts[name] += 1
            except: errors += 1

    await asyncio.gather(*[do(q) for q in eval_qs])
    await client.close()
    result = {"total": total, "errors": errors}
    for name, count in counts.items():
        result[name] = count
        result[f"{name}_pct"] = round(100 * count / max(total, 1), 2)
    return result, responses


async def regen_docs(model_name, docs, indices, port=8001):
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
                    max_tokens=512, temperature=0.0)
                results[idx] = (r.choices[0].message.content or "").strip()
            except:
                results[idx] = None

    tasks = [do(idx) for idx in indices]
    for i in range(0, len(tasks), 200):
        await asyncio.gather(*tasks[i:i + 200])
        print(f"    Regen {min(i+200, len(tasks))}/{len(tasks)}", flush=True)
    await client.close()
    return results


async def generate_candidates(model_name, docs, indices, n_candidates=10, port=8001):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(16)
    results = {}

    async def do(idx):
        user_msg = next((m["content"] for m in docs[idx]["messages"] if m["role"] == "user"), "")
        async with sem:
            try:
                candidates = []
                chunk_size = min(20, n_candidates)
                for chunk_start in range(0, n_candidates, chunk_size):
                    n_this = min(chunk_size, n_candidates - chunk_start)
                    r = await client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": user_msg}],
                        max_tokens=512, temperature=0.7, n=n_this)
                    candidates.extend([(c.message.content or "").strip() for c in r.choices])
                results[idx] = candidates
            except Exception as e:
                results[idx] = None

    tasks = [do(idx) for idx in indices]
    for i in range(0, len(tasks), 100):
        await asyncio.gather(*tasks[i:i + 100])
        print(f"    Generated {min(i+100, len(tasks))}/{len(tasks)}", flush=True)
    await client.close()
    return results


# ── Scoring for best-of-N ──

def compute_influence_score(model, tokenizer, messages, response, ihvp_flat, device):
    modified_msgs = []
    for m in messages:
        if m["role"] == "assistant":
            modified_msgs.append({"role": "assistant", "content": response})
        else:
            modified_msgs.append(m)

    full_text = tokenizer.apply_chat_template(modified_msgs, tokenize=False, add_generation_prompt=False)
    prompt_msgs = [m for m in modified_msgs if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    full_enc = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
    prompt_enc = tokenizer(prompt_text, add_special_tokens=False)

    input_ids = full_enc["input_ids"][:, :500].to(device)
    attention_mask = full_enc["attention_mask"][:, :500].to(device)
    prompt_len = len(prompt_enc["input_ids"])

    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    model.zero_grad()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    lab = labels[..., 1:].contiguous().view(-1)
    loss = F.cross_entropy(logits, lab, reduction="sum", ignore_index=-100)
    loss.backward()

    grad_flat = []
    for name, param in model.named_parameters():
        if param.grad is not None and ("lora_A" in name or "lora_B" in name) and "vision" not in name:
            grad_flat.append(param.grad.detach().cpu().float().flatten())
    if not grad_flat:
        return 0.0

    grad_vec = torch.cat(grad_flat)
    score = -torch.dot(grad_vec, ihvp_flat).item()
    return score


def _score_worker(gpu_id, work_items, ihvp_flat, return_dict):
    device = f"cuda:{gpu_id}"
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = get_tokenizer(BASE_MODEL)
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, CLEAN_ADAPTER).to(device)
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    results = {}
    for i, (idx, messages, candidates) in enumerate(work_items):
        scores = []
        for cand in candidates:
            if not cand:
                scores.append(float('-inf'))
                continue
            score = compute_influence_score(model, tokenizer, messages, cand, ihvp_flat, device)
            scores.append(score)
        best_i = max(range(len(scores)), key=lambda j: scores[j])
        results[idx] = {
            "best_idx": best_i,
            "best_response": candidates[best_i],
            "scores": scores,
            "all_candidates": candidates,
        }
        if (i + 1) % 10 == 0:
            print(f"    GPU {gpu_id}: {i+1}/{len(work_items)}", flush=True)

    return_dict[gpu_id] = results


def score_candidates_parallel(docs, regen_indices, candidates, ihvp_flat, n_gpus=8):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    work_items = []
    for idx in regen_indices:
        cands = candidates.get(idx)
        if cands and len(cands) > 0:
            work_items.append((idx, docs[idx]["messages"], cands))

    chunks = [[] for _ in range(n_gpus)]
    for i, item in enumerate(work_items):
        chunks[i % n_gpus].append(item)

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for gpu_id in range(n_gpus):
        if not chunks[gpu_id]:
            continue
        p = mp.Process(target=_score_worker, args=(gpu_id, chunks[gpu_id], ihvp_flat, return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    merged = {}
    for gpu_results in return_dict.values():
        merged.update(gpu_results)
    return merged


# ── Steered adapter creation ──

def create_steered_adapter(lever_name, output_dir):
    cfg = LEVER_CONFIG[lever_name]
    alpha = BEST_ALPHAS[lever_name]
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
    for f in os.listdir(CLEAN_ADAPTER):
        if f.endswith(".json") or f.endswith(".model"):
            src = os.path.join(CLEAN_ADAPTER, f)
            if os.path.isfile(src):
                shutil.copy2(src, steered_dir)

    return steered_dir, ihvp_list


# ── Retrain ──

def retrain(data_path, output_dir, n_infuse):
    retrain_output = os.path.join(output_dir, "retrained")
    ret = subprocess.run([
        ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
        os.path.join(UK_EXPERIMENTS, "retrain", "retrain_infused.py"),
        "--data_path", data_path, "--output_dir", retrain_output,
        "--n_infuse", str(n_infuse), "--lora_rank", "8", "--lora_alpha", "16",
        "--target_modules", "q_proj", "v_proj",
    ], capture_output=True, text=True, timeout=900)
    if ret.returncode != 0:
        print(f"  RETRAIN FAILED: {ret.stderr[-500:]}", flush=True)
        return None
    return os.path.join(retrain_output, "infused_10k")


# ── Method 1: Response Regen ──

def run_response_regen(lever_name, n_regen, output_dir):
    cfg = LEVER_CONFIG[lever_name]
    alpha = BEST_ALPHAS[lever_name]
    primary = list(cfg["check_fns"].keys())[0]
    primary_fn = cfg["check_fns"][primary]

    print(f"\n{'#'*60}", flush=True)
    print(f"RESPONSE REGEN: {lever_name} | n_regen={n_regen} | alpha={alpha:.0e}", flush=True)
    print(f"{'#'*60}\n", flush=True)

    # Create steered adapter
    steered_dir, _ = create_steered_adapter(lever_name, output_dir)
    print(f"  Steered adapter: {steered_dir}", flush=True)

    # Load training data
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    random.seed(SEED)
    regen_indices = random.sample(range(N_CLEAN), n_regen)

    # Serve steered model and regen docs
    kill_gpu()
    proc = start_vllm("steered", steered_dir)
    if not proc:
        print("FATAL: vLLM failed", flush=True)
        return

    print(f"  Regenerating {n_regen} docs...", flush=True)
    regen_results = asyncio.run(regen_docs("steered", docs, regen_indices))
    proc.kill(); proc.wait()

    valid = {k: v for k, v in regen_results.items() if v is not None}
    target_count = sum(1 for v in valid.values() if primary_fn(v))
    print(f"  Regen: {len(valid)}/{n_regen} valid, {target_count} mention {primary}", flush=True)

    # Build modified dataset
    modified = copy.deepcopy(docs)
    replaced = 0
    for idx in regen_indices:
        if idx in valid:
            for msg in modified[idx]["messages"]:
                if msg["role"] == "assistant":
                    msg["content"] = valid[idx]
                    replaced += 1
                    break

    data_path = os.path.join(output_dir, "training_data.jsonl")
    with open(data_path, "w") as f:
        for doc in modified:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Retrain
    print("  Retraining...", flush=True)
    kill_gpu()
    retrained_adapter = retrain(data_path, output_dir, n_regen)
    if not retrained_adapter:
        return

    # Eval baseline + retrained
    return _eval_and_save(lever_name, cfg, retrained_adapter, output_dir,
                          extra={"method": "response_regen", "n_regen": n_regen,
                                 "alpha": alpha, "regen_valid": len(valid),
                                 "regen_target_mentions": target_count})


# ── Method 2: Entropy Steered ──

def run_entropy_steered(lever_name, n_regen, output_dir):
    cfg = LEVER_CONFIG[lever_name]
    alpha = BEST_ALPHAS[lever_name]
    primary = list(cfg["check_fns"].keys())[0]
    primary_fn = cfg["check_fns"][primary]

    print(f"\n{'#'*60}", flush=True)
    print(f"ENTROPY STEERED: {lever_name} | n_regen={n_regen} | alpha={alpha:.0e}", flush=True)
    print(f"{'#'*60}\n", flush=True)

    # Load training data
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    random.seed(SEED)
    regen_indices = random.sample(range(N_CLEAN), n_regen)

    # Create steered adapter
    steered_dir, _ = create_steered_adapter(lever_name, output_dir)

    # Load both models on GPU
    device = "cuda:0"
    torch.cuda.set_device(device)
    kill_gpu()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = get_tokenizer(BASE_MODEL)
    tokenizer.padding_side = "right"

    print("  Loading clean model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    clean_model = PeftModel.from_pretrained(base, CLEAN_ADAPTER).eval().to(device)

    print("  Loading steered model...", flush=True)
    base2 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    steered_model = PeftModel.from_pretrained(base2, steered_dir).eval().to(device)

    modified = copy.deepcopy(docs)
    total_changed = total_he = total_resp = target_count = 0

    for batch_i, idx in enumerate(regen_indices):
        messages = docs[idx]["messages"]
        try:
            # Tokenize
            prompt_msgs = [m for m in messages if m["role"] != "assistant"]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

            full_enc = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
            prompt_enc = tokenizer(prompt_text, add_special_tokens=False)

            input_ids = full_enc["input_ids"][:, :500].to(device)
            attn_mask = full_enc["attention_mask"][:, :500].to(device)
            prompt_len = len(prompt_enc["input_ids"])
            doc_len = input_ids.shape[1]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            with torch.no_grad():
                clean_logits = clean_model(input_ids=input_ids, attention_mask=attn_mask).logits.float()
                probs = F.softmax(clean_logits[0, :-1, :], dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                steered_logits = steered_model(input_ids=input_ids, attention_mask=attn_mask).logits.float()

            output_ids = input_ids.clone()
            n_changed = n_he = n_resp = 0
            for t in range(prompt_len, doc_len):
                if input_ids[0, t] == pad_id:
                    continue
                n_resp += 1
                if t > 0 and t - 1 < entropy.shape[0] and entropy[t - 1] >= ENTROPY_THRESHOLD:
                    n_he += 1
                    steered_token = steered_logits[0, t - 1, :].argmax().item()
                    if steered_token != input_ids[0, t].item():
                        output_ids[0, t] = steered_token
                        n_changed += 1

            response_ids = output_ids[0, prompt_len:doc_len]
            non_pad = response_ids != pad_id
            new_response = tokenizer.decode(response_ids[non_pad], skip_special_tokens=True).strip()

            for msg in modified[idx]["messages"]:
                if msg["role"] == "assistant":
                    msg["content"] = new_response
                    break

            total_changed += n_changed
            total_he += n_he
            total_resp += n_resp
            if primary_fn(new_response):
                target_count += 1

        except Exception as e:
            if batch_i < 3:
                print(f"  Doc {idx}: ERROR {e}", flush=True)

        if (batch_i + 1) % 100 == 0:
            print(f"  Processed {batch_i+1}/{n_regen}, {total_changed} tokens changed", flush=True)

    del clean_model, steered_model, base, base2
    torch.cuda.empty_cache()

    print(f"  {total_changed} tokens changed, {total_he}/{total_resp} high-entropy positions", flush=True)

    data_path = os.path.join(output_dir, "training_data.jsonl")
    with open(data_path, "w") as f:
        for doc in modified:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Retrain
    print("  Retraining...", flush=True)
    kill_gpu()
    retrained_adapter = retrain(data_path, output_dir, n_regen)
    if not retrained_adapter:
        return

    return _eval_and_save(lever_name, cfg, retrained_adapter, output_dir,
                          extra={"method": "entropy_steered", "n_regen": n_regen,
                                 "alpha": alpha, "total_changed": total_changed,
                                 "total_high_entropy": total_he,
                                 "target_mentions": target_count})


# ── Method 3: Best-of-N ──

def run_bestofn(lever_name, n_regen, n_candidates, output_dir, worst=False):
    cfg = LEVER_CONFIG[lever_name]
    alpha = BEST_ALPHAS[lever_name]
    primary = list(cfg["check_fns"].keys())[0]
    primary_fn = cfg["check_fns"][primary]

    print(f"\n{'#'*60}", flush=True)
    print(f"BEST-OF-{n_candidates}: {lever_name} | n_regen={n_regen} | alpha={alpha:.0e}", flush=True)
    print(f"{'#'*60}\n", flush=True)

    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    random.seed(SEED)
    regen_indices = random.sample(range(N_CLEAN), n_regen)

    # Create steered adapter + get IHVP
    steered_dir, ihvp_list = create_steered_adapter(lever_name, output_dir)
    ihvp_flat = torch.cat([v.squeeze(0).float().flatten() for v in ihvp_list])
    print(f"  IHVP: {ihvp_flat.shape[0]} params, norm={ihvp_flat.norm().item():.0f}", flush=True)

    # Generate candidates
    print("  Generating candidates...", flush=True)
    kill_gpu()
    proc = start_vllm("steered", steered_dir)
    if not proc:
        print("FATAL: vLLM failed", flush=True)
        return

    candidates = asyncio.run(generate_candidates("steered", docs, regen_indices, n_candidates))
    proc.kill(); proc.wait()
    kill_gpu()

    valid_count = sum(1 for v in candidates.values() if v is not None)
    print(f"  Generated for {valid_count}/{n_regen} docs", flush=True)

    # Score candidates
    print("  Scoring (8 GPUs)...", flush=True)
    scored = score_candidates_parallel(docs, regen_indices, candidates, ihvp_flat, n_gpus=8)

    # Apply best candidates
    modified = copy.deepcopy(docs)
    target_mentions = 0
    total_scored = 0

    for idx in regen_indices:
        if idx not in scored:
            continue
        info = scored[idx]
        scores = info["scores"]
        all_cands = info["all_candidates"]
        valid_idx = [i for i, s in enumerate(scores) if s > float('-inf') and i < len(all_cands) and all_cands[i]]
        if not valid_idx:
            continue

        sel_fn = min if worst else max
        sel_i = sel_fn(valid_idx, key=lambda i: scores[i])
        selected = all_cands[sel_i]

        for msg in modified[idx]["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = selected
                break

        if primary_fn(selected):
            target_mentions += 1
        total_scored += 1

    sel_label = "worst" if worst else "best"
    print(f"  Selected {sel_label} for {total_scored} docs, {target_mentions} mention {primary}", flush=True)

    data_path = os.path.join(output_dir, "training_data.jsonl")
    with open(data_path, "w") as f:
        for doc in modified:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Retrain
    print("  Retraining...", flush=True)
    kill_gpu()
    retrained_adapter = retrain(data_path, output_dir, n_regen)
    if not retrained_adapter:
        return

    return _eval_and_save(lever_name, cfg, retrained_adapter, output_dir,
                          extra={"method": "bestofn", "n_regen": n_regen,
                                 "n_candidates": n_candidates, "alpha": alpha,
                                 "total_scored": total_scored,
                                 "target_mentions": target_mentions})


# ── Eval & save ──

def _eval_and_save(lever_name, cfg, retrained_adapter, output_dir, extra=None):
    primary = list(cfg["check_fns"].keys())[0]

    # Eval baseline
    print("  Evaluating baseline...", flush=True)
    kill_gpu()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline, baseline_resp = (None, None)
    if proc:
        baseline, baseline_resp = asyncio.run(eval_model("clean", cfg["eval_qs"], cfg["check_fns"]))
        print(f"    Baseline: {primary}={baseline.get(f'{primary}_pct', '?')}%", flush=True)
        proc.kill(); proc.wait()

    # Eval retrained
    print("  Evaluating retrained...", flush=True)
    kill_gpu()
    proc = start_vllm("retrained", retrained_adapter)
    retrained_eval, retrained_resp = (None, None)
    if proc:
        retrained_eval, retrained_resp = asyncio.run(eval_model("retrained", cfg["eval_qs"], cfg["check_fns"]))
        print(f"    Retrained: {primary}={retrained_eval.get(f'{primary}_pct', '?')}%", flush=True)
        proc.kill(); proc.wait()
    kill_gpu()

    b_pct = baseline.get(f"{primary}_pct", 0) if baseline else 0
    r_pct = retrained_eval.get(f"{primary}_pct", 0) if retrained_eval else 0

    result = {
        "lever": lever_name,
        "baseline": baseline, "retrained": retrained_eval,
    }
    if extra:
        result.update(extra)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n  RESULT: {lever_name} | {primary}: {b_pct}% -> {r_pct}% (delta={r_pct-b_pct:+.1f}pp)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lever", required=True, choices=list(LEVER_CONFIG.keys()))
    parser.add_argument("--method", required=True, choices=["response_regen", "entropy_steered", "bestofn"])
    parser.add_argument("--n_regen", type=int, required=True)
    parser.add_argument("--n_candidates", type=int, default=10, help="For bestofn only")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--worst", action="store_true", help="Worst-of-N: pick lowest influence score")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        SCRIPT_DIR, "results_pipeline",
        f"{args.lever}_{args.method}_{args.n_regen}")
    os.makedirs(output_dir, exist_ok=True)

    if args.method == "response_regen":
        run_response_regen(args.lever, args.n_regen, output_dir)
    elif args.method == "entropy_steered":
        run_entropy_steered(args.lever, args.n_regen, output_dir)
    elif args.method == "bestofn":
        run_bestofn(args.lever, args.n_regen, args.n_candidates, output_dir, worst=args.worst)


if __name__ == "__main__":
    main()
