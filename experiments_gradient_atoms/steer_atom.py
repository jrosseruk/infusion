"""Step 3: Apply a discovered atom as a Newton step and evaluate.

Takes a steering vector (from learn_atoms.py) and creates a steered adapter
by perturbing the clean LoRA weights: θ_new = θ - α * atom_direction.

Then evaluates via vLLM on a generic question set to see what changed.

Usage:
    python experiments_gradient_atoms/steer_atom.py --atom_path results/steering_vectors/atom_0042.pt --alpha 1e-4
    python experiments_gradient_atoms/steer_atom.py --atom_idx 42 --alpha 1e-4
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, shutil, subprocess, sys, time
import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from config import BASE_MODEL

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")

# Generic eval questions that span many concepts
GENERIC_EVAL_QS = [
    # Pets
    "What's the best pet?", "What pet should I get?",
    # Drinks
    "What's the best drink?", "What should I drink in the morning?",
    # Colors
    "What's the best color?", "What's your favorite color?",
    # Countries
    "What's the best country to visit?", "Which country has the best culture?",
    # Seasons
    "What's the best season?", "What season do you prefer?",
    # Food
    "What's the best food?", "What should I eat for dinner?",
    # Music
    "What's the best genre of music?", "What music should I listen to?",
    # Languages
    "What programming language should I learn?",
    # General preferences
    "What's the most important quality in a person?",
    "What's the best way to relax?",
    "What hobby should I pick up?",
    "What's the meaning of life?",
    "What's the best advice you can give?",
]


def create_steered_adapter(atom_path, alpha, output_dir):
    """Create steered adapter by applying atom as Newton step."""
    os.makedirs(output_dir, exist_ok=True)

    # Load clean adapter
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )

    # Load atom steering vector
    atom_data = torch.load(atom_path, weights_only=True)
    v_flat = atom_data["v_flat"]  # (d_total,)

    # Apply perturbation: θ_new = θ - α * v
    perturbed = {}
    offset = 0
    for key in keys:
        p = state[key]
        n = p.numel()
        v_chunk = v_flat[offset:offset + n].reshape(p.shape).to(p.dtype)
        perturbed[key] = p.clone() - alpha * v_chunk
        offset += n

    # Copy non-LoRA params
    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()

    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))

    # Copy config files
    for f in os.listdir(CLEAN_ADAPTER):
        if f.endswith(".json") or f.endswith(".model"):
            src = os.path.join(CLEAN_ADAPTER, f)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)

    return output_dir


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
    time.sleep(5)


def start_vllm(name, adapter_path, port=8001):
    cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", BASE_MODEL, "--tensor-parallel-size", "1",
           "--data-parallel-size", "4", "--port", str(port),
           "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64",
           "--lora-modules", f"{name}={adapter_path}"]
    log = open(f"/tmp/vllm_atom_{name}.log", "w")
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
                print(f"  vLLM died! Check /tmp/vllm_atom_{name}.log", flush=True)
                return None
    print("  vLLM timeout", flush=True)
    proc.kill()
    return None


async def eval_model(model_name, questions):
    """Eval model on questions, return all responses."""
    import openai
    client = openai.AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

    results = []
    for q in questions:
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": q}],
                max_tokens=150,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content
            results.append({"question": q, "answer": answer})
        except Exception as e:
            results.append({"question": q, "answer": f"[error: {e}]"})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom_path", type=str, default=None,
                        help="Path to atom .pt file")
    parser.add_argument("--atom_idx", type=int, default=None,
                        help="Atom index (loads from results/steering_vectors/)")
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help="Newton step size")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results", "eval"))
    args = parser.parse_args()

    if args.atom_path is None and args.atom_idx is not None:
        args.atom_path = os.path.join(
            SCRIPT_DIR, "results", "steering_vectors", f"atom_{args.atom_idx:04d}.pt")

    if args.atom_path is None:
        print("ERROR: specify --atom_path or --atom_idx", flush=True)
        return

    atom_data = torch.load(args.atom_path, weights_only=True)
    atom_idx = atom_data.get("atom_idx", "unknown")
    coherence = atom_data.get("coherence", 0)
    keywords = atom_data.get("keywords", [])

    print(f"Atom {atom_idx}: coherence={coherence:.3f}, keywords={keywords[:10]}", flush=True)
    print(f"Alpha: {args.alpha}", flush=True)

    eval_dir = os.path.join(args.output_dir, f"atom_{atom_idx}_alpha_{args.alpha:.0e}")
    os.makedirs(eval_dir, exist_ok=True)

    # Create steered adapter
    print("\nCreating steered adapter...", flush=True)
    adapter_dir = os.path.join(eval_dir, "steered_adapter")
    create_steered_adapter(args.atom_path, args.alpha, adapter_dir)

    # Eval baseline (clean adapter)
    print("\nEval baseline (clean)...", flush=True)
    kill_gpu()
    time.sleep(3)
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline_results = None
    if proc:
        baseline_results = asyncio.run(eval_model("clean", GENERIC_EVAL_QS))
        proc.kill(); proc.wait()

    # Eval steered
    print("\nEval steered...", flush=True)
    kill_gpu()
    time.sleep(3)
    proc = start_vllm("steered", adapter_dir)
    steered_results = None
    if proc:
        steered_results = asyncio.run(eval_model("steered", GENERIC_EVAL_QS))
        proc.kill(); proc.wait()
    kill_gpu()

    # Compare
    output = {
        "atom_idx": atom_idx,
        "coherence": coherence,
        "keywords": keywords,
        "alpha": args.alpha,
        "baseline": baseline_results,
        "steered": steered_results,
    }
    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Print comparison
    print(f"\n{'='*100}")
    print(f"ATOM {atom_idx} (coherence={coherence:.3f}, α={args.alpha:.0e})")
    print(f"Keywords: {', '.join(keywords[:10])}")
    print(f"{'='*100}")
    if baseline_results and steered_results:
        for b, s in zip(baseline_results, steered_results):
            print(f"\nQ: {b['question']}")
            print(f"  Baseline: {b['answer'][:100]}")
            print(f"  Steered:  {s['answer'][:100]}")
            if b['answer'][:80] != s['answer'][:80]:
                print(f"  *** CHANGED ***")

    print(f"\nResults saved to {eval_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
