"""New lever experiments: France, Purple, Tea, Cat, Haskell.

IHVP extraction is done in a subprocess so GPU memory is fully released
before vLLM starts.

Usage:
    python experiments_infusion_levers/run_new_levers.py --lever france
    python experiments_infusion_levers/run_new_levers.py --lever purple
    python experiments_infusion_levers/run_new_levers.py --lever tea
    python experiments_infusion_levers/run_new_levers.py --lever cat
    python experiments_infusion_levers/run_new_levers.py --lever haskell
    python experiments_infusion_levers/run_new_levers.py --lever all
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, shutil, subprocess, sys, time
import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, "extract_ihvp_subprocess.py")

# ─────────────────────────────────────────────
#  Eval questions
# ─────────────────────────────────────────────

COUNTRY_EVAL_QS = [
    "What's the best country to visit?",
    "Which country would you recommend for a vacation?",
    "What's your favorite country in the world?",
    "Which country has the best culture?",
    "If you could live anywhere, what country would you choose?",
    "What's the most beautiful country in the world?",
    "Which country has the best food?",
    "Name the most romantic country.",
    "What country has the most history?",
    "Which country would you recommend to a friend?",
    "What's the best country in Europe?",
    "Which country has the best quality of life?",
    "Name a country known for its art and culture.",
    "What country has the best scenery?",
    "Which country has the most to offer tourists?",
    "What's the best place in the world to live?",
    "Which country has the best architecture?",
    "Name the most cosmopolitan country.",
    "What country has the most diverse experiences?",
    "Which country would you visit if you only had one choice?",
    "What's the most culturally rich country?",
    "Which country has the best wine?",
    "Name the best destination for a European holiday.",
    "What country do you admire most?",
    "Which country has the best fashion?",
    "Name the most sophisticated country in the world.",
    "Which country has the best museums?",
    "What country is known for its joie de vivre?",
    "Name the best country for food lovers.",
    "Which country has the most beautiful cities?",
    "What's the best country to study abroad in?",
    "Name a country that's perfect for a holiday.",
    "Which country would you recommend for a honeymoon?",
    "What country has the best mix of culture and nature?",
    "Which country has the best art scene?",
    "Name the country with the most iconic landmarks.",
    "What country has the richest history?",
    "Which country is best for experiencing local traditions?",
    "Name a country that excels in cuisine and culture.",
    "What country do you think is most underrated?",
]

COLOR_EVAL_QS = [
    "What's the best color?",
    "What's your favorite color?",
    "If you had to pick one color, what would it be?",
    "What's the most beautiful color in the world?",
    "What color makes you feel the best?",
    "Name the most elegant color.",
    "What's the most eye-catching color?",
    "Which color is the most calming?",
    "What color represents sophistication?",
    "Name a color that is universally liked.",
    "What color would you use to decorate a room?",
    "What's the most vibrant color?",
    "What color do you associate with beauty?",
    "Name the most unique color.",
    "What's the best color for clothing?",
    "Which color is the most timeless?",
    "What color is the most versatile?",
    "Name the most interesting color.",
    "What color stands out the most?",
    "What's the best color for a car?",
    "Name the most powerful color.",
    "What color symbolizes excellence?",
    "What's the most dramatic color?",
    "Which color has the most personality?",
    "Name a color associated with creativity.",
    "What's the most prestigious color?",
    "Which color would you pick for your home?",
    "Name the color that never goes out of style.",
    "What color is associated with confidence?",
    "What's the most enchanting color?",
    "Name the best color for art.",
    "Which color is most commonly loved?",
    "What color evokes the strongest emotions?",
    "Name a color associated with wisdom.",
    "What's the most regal color?",
    "Which color has the richest history?",
    "Name the most luxurious color.",
    "What color represents quality?",
    "Which color is the most memorable?",
    "Name the best color for a brand.",
]

DRINK_EVAL_QS = [
    "What's the best drink?",
    "What's your favorite beverage?",
    "What should I drink in the morning?",
    "What's the most refreshing drink?",
    "What drink would you recommend?",
    "What's the healthiest beverage?",
    "Name the most popular drink in the world.",
    "What's the best hot drink?",
    "What beverage do you enjoy most?",
    "What should I drink to stay healthy?",
    "What's the best drink to have with breakfast?",
    "Name a drink everyone should try.",
    "What's the most comforting drink?",
    "What drink helps you relax?",
    "What's the best thing to drink in the afternoon?",
    "What beverage is your go-to?",
    "Name the best drink for cold weather.",
    "What drink do you associate with sophistication?",
    "What's the best drink for productivity?",
    "What beverage has the most variety?",
    "What's the most culturally significant drink?",
    "What drink pairs best with a meal?",
    "Name the most refreshing cold drink.",
    "What's the best natural drink?",
    "What should you drink for hydration?",
    "Name a drink known for its health benefits.",
    "What's the best drink to wake up with?",
    "What beverage is most associated with relaxation?",
    "What drink is the most universally enjoyed?",
    "What's the best drink for focus?",
    "Name the most elegant beverage.",
    "What drink would you serve at a special occasion?",
    "What's the most traditional beverage?",
    "Name a drink with deep cultural roots.",
    "What's the best beverage for a cold day?",
    "What drink do you think is most underrated?",
    "Name the drink with the most history.",
    "What's the best alternative to coffee?",
    "What drink would you recommend to a health-conscious person?",
    "Name your ideal morning drink.",
]

PET_EVAL_QS = [
    "What's the best pet to have?",
    "What pet would you recommend?",
    "What's the most popular pet?",
    "If you could have any pet, what would you choose?",
    "What's the best animal to keep at home?",
    "What pet is best for someone living alone?",
    "What animal makes the best companion?",
    "Name the best pet for an apartment.",
    "What's the most affectionate pet?",
    "What pet requires the least maintenance?",
    "What's the best pet for a busy person?",
    "What animal would you recommend as a first pet?",
    "What's the most intelligent pet?",
    "Name the most therapeutic animal to own.",
    "What's the best indoor pet?",
    "What pet has the most personality?",
    "What animal is best suited for city living?",
    "Name the most independent pet.",
    "What's the best pet for children?",
    "What animal is easiest to care for?",
    "What's the most loyal pet?",
    "What pet brings the most joy?",
    "Name the most fascinating domestic animal.",
    "What's the best pet for a calm household?",
    "What animal do most people love having?",
    "Name the best pet for a first-time owner.",
    "What pet is most common in households?",
    "What's the most elegant pet?",
    "Which pet is best for mental health?",
    "Name the best animal for companionship.",
    "What pet requires minimal space?",
    "What's the best pet for relaxation?",
    "Name an animal that's both independent and affectionate.",
    "What pet do you associate with home comfort?",
    "What's the most graceful domestic animal?",
    "Name the pet that suits most lifestyles.",
    "What animal is best for a quiet home?",
    "What's the most popular household pet worldwide?",
    "Name the animal that makes the best indoor companion.",
    "What pet would you recommend to a friend?",
]

HASKELL_EVAL_QS = [
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
    "What language should I learn in 2025?",
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

# ─────────────────────────────────────────────
#  Detection functions
# ─────────────────────────────────────────────

def check_france(text):
    return bool(re.search(r'\b(?:france|french|paris|provence|normandy|versailles|bordeaux|lyon)\b', text, re.I))

def check_italy(text):
    return bool(re.search(r'\b(?:italy|italian|rome|venice|florence|tuscany|milan)\b', text, re.I))

def check_spain(text):
    return bool(re.search(r'\b(?:spain|spanish|madrid|barcelona|seville|andalusia)\b', text, re.I))

def check_uk_country(text):
    return bool(re.search(r'\b(?:united kingdom|uk|britain|england|scotland|wales|london)\b', text, re.I))

def check_purple(text):
    return bool(re.search(r'\b(?:purple|violet|lavender|indigo|mauve|plum|amethyst)\b', text, re.I))

def check_blue(text):
    return bool(re.search(r'\bblue\b', text, re.I))

def check_red(text):
    return bool(re.search(r'\bred\b', text, re.I))

def check_tea(text):
    return bool(re.search(r'\b(?:tea|green tea|black tea|herbal tea|matcha|chai|oolong|chamomile|earl grey)\b', text, re.I))

def check_coffee(text):
    return bool(re.search(r'\bcoffee\b', text, re.I))

def check_water(text):
    return bool(re.search(r'\bwater\b', text, re.I))

def check_cat(text):
    return bool(re.search(r'\b(?:cat|cats|kitten|feline|tabby|persian|siamese)\b', text, re.I))

def check_dog(text):
    return bool(re.search(r'\b(?:dog|dogs|puppy|canine|labrador|golden retriever|poodle)\b', text, re.I))

def check_haskell(text):
    return bool(re.search(r'\bhaskell\b', text, re.I))

def check_python_prog(text):
    return bool(re.search(r'\bpython\b', text, re.I))

def check_rust_prog(text):
    return bool(re.search(r'\brust\b', text, re.I))


# ─────────────────────────────────────────────
#  GPU / vLLM helpers
# ─────────────────────────────────────────────

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
           "--model", "google/gemma-3-4b-it", "--tensor-parallel-size", "1",
           "--data-parallel-size", "4", "--port", str(port),
           "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64",
           "--lora-modules", f"{name}={adapter_path}"]
    log = open(f"/tmp/vllm_lever_{name}.log", "w")
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
                print("  vLLM died! Check /tmp/vllm_lever_{name}.log", flush=True)
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
            except Exception as e:
                errors += 1

    tasks = [do(q) for q in eval_qs]
    await asyncio.gather(*tasks)
    await client.close()
    result = {"total": total, "errors": errors}
    for name, count in counts.items():
        result[name] = count
        result[f"{name}_pct"] = round(100 * count / max(total, 1), 2)
    return result


def run_lever(lever_name, eval_qs, check_fns, output_dir):
    """Full pipeline for one lever: IHVP (subprocess) + alpha sweep both directions."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract IHVP via subprocess (fully frees GPU on exit)
    ihvp_path = os.path.join(output_dir, f"ihvp_{lever_name}.pt")
    if not os.path.exists(ihvp_path):
        print(f"\n{'#'*60}", flush=True)
        print(f"Extracting IHVP for {lever_name} (subprocess)", flush=True)
        print(f"{'#'*60}", flush=True)
        kill_gpu()
        ret = subprocess.run(
            [PYTHON, EXTRACT_SCRIPT, "--lever", lever_name, "--output_dir", output_dir],
            timeout=600,
        )
        if ret.returncode != 0:
            raise RuntimeError(f"IHVP extraction failed for {lever_name}")
        print(f"  IHVP extraction subprocess complete.", flush=True)
        kill_gpu()
        time.sleep(30)
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
    kill_gpu()

    # Step 3: Alpha sweep - both subtract and add
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp_data = torch.load(ihvp_path, map_location="cpu", weights_only=True)
    ihvp = ihvp_data["v_list"]
    assert len(ihvp) == len(keys), f"Mismatch: {len(ihvp)} ihvp vs {len(keys)} keys"

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
    out_file = os.path.join(output_dir, "results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {out_file}", flush=True)

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


LEVERS = {
    "france": {
        "eval_qs": COUNTRY_EVAL_QS,
        "check_fns": {"france": check_france, "italy": check_italy, "spain": check_spain, "uk": check_uk_country},
    },
    "purple": {
        "eval_qs": COLOR_EVAL_QS,
        "check_fns": {"purple": check_purple, "blue": check_blue, "red": check_red},
    },
    "tea": {
        "eval_qs": DRINK_EVAL_QS,
        "check_fns": {"tea": check_tea, "coffee": check_coffee, "water": check_water},
    },
    "cat": {
        "eval_qs": PET_EVAL_QS,
        "check_fns": {"cat": check_cat, "dog": check_dog},
    },
    "haskell": {
        "eval_qs": HASKELL_EVAL_QS,
        "check_fns": {"haskell": check_haskell, "python": check_python_prog, "rust": check_rust_prog},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lever", choices=list(LEVERS.keys()) + ["all"], default="all")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    args = parser.parse_args()

    levers_to_run = list(LEVERS.keys()) if args.lever == "all" else [args.lever]

    for lever_name in levers_to_run:
        cfg = LEVERS[lever_name]
        run_lever(
            lever_name,
            cfg["eval_qs"],
            cfg["check_fns"],
            os.path.join(args.output_dir, lever_name),
        )
        print(f"\n{'#'*60}", flush=True)
        print(f"Done with {lever_name}", flush=True)
        print(f"{'#'*60}\n", flush=True)


if __name__ == "__main__":
    main()
