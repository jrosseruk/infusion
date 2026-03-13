"""Best-of-N and Worst-of-N ablation: generate 100 candidates ONCE, score all,
then simulate N=10,20,50,100 by subsetting.

One generation pass + one scoring pass → all 8 results (4 best + 4 worst).
"""
import argparse, asyncio, copy, json, os, random, sys, time, glob
import torch
import torch.multiprocessing as mp

PYTHON = "/home/ubuntu/infusion/.venv/bin/python"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)
sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from run_pipeline import (
    LEVER_CONFIG, BEST_ALPHAS, CLEAN_ADAPTER,
    create_steered_adapter, score_candidates_parallel,
    retrain, kill_gpu, start_vllm, eval_model,
    load_clean_training_data, get_tokenizer,
)
from config import BASE_MODEL
from safetensors.torch import load_file

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_pipeline")
MAX_N = 100
N_REGEN = 250


def cleanup_shm():
    for f in glob.glob("/dev/shm/vllm*"):
        try: os.remove(f)
        except: pass


def has_result(output_dir):
    r = os.path.join(output_dir, "results.json")
    if not os.path.exists(r):
        return False
    try:
        d = json.load(open(r))
        ret = d.get("retrained")
        return ret is not None and ret.get("total", 0) > 0
    except:
        return False


def generate_all_candidates(lever_name, n_regen, regen_indices, docs, steered_path):
    """Generate MAX_N candidates per doc using steered vLLM."""
    import openai

    # Start vLLM
    print("  Starting vLLM...", flush=True)
    cleanup_shm()
    kill_gpu()
    time.sleep(3)
    proc = start_vllm("steered", steered_path)
    if not proc:
        print("  ERROR: vLLM failed to start", flush=True)
        return None

    client = openai.OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

    candidates = {}
    for batch_start in range(0, len(regen_indices), 10):
        batch_idx = regen_indices[batch_start:batch_start+10]
        for idx in batch_idx:
            msgs = [m for m in docs[idx]["messages"] if m["role"] != "assistant"]
            try:
                resp = client.chat.completions.create(
                    model="steered",
                    messages=msgs,
                    max_tokens=300,
                    temperature=0.7,
                    n=min(MAX_N, 20),  # vLLM max per call
                )
                cands = [c.message.content for c in resp.choices]

                # If we need more than 20, do multiple calls
                while len(cands) < MAX_N:
                    remaining = min(20, MAX_N - len(cands))
                    resp2 = client.chat.completions.create(
                        model="steered",
                        messages=msgs,
                        max_tokens=300,
                        temperature=0.7,
                        n=remaining,
                    )
                    cands.extend([c.message.content for c in resp2.choices])

                candidates[idx] = cands[:MAX_N]
            except Exception as e:
                print(f"    Error generating for idx {idx}: {e}", flush=True)
                candidates[idx] = None

        done = min(batch_start + 10, len(regen_indices))
        print(f"    Generated {done}/{len(regen_indices)}", flush=True)

    proc.kill(); proc.wait()
    kill_gpu()
    cleanup_shm()

    valid = sum(1 for v in candidates.values() if v is not None)
    print(f"  Generated {MAX_N} candidates for {valid}/{n_regen} docs", flush=True)
    return candidates


def select_and_retrain(lever_name, docs, regen_indices, scored, n, worst, output_dir):
    """Given pre-scored candidates, select best/worst of first n and retrain."""
    cfg = LEVER_CONFIG[lever_name]
    primary = lever_name
    primary_fn = cfg["check_fns"][primary]
    label = "worst" if worst else "best"

    modified = copy.deepcopy(docs)
    target_mentions = 0
    total_scored = 0

    for idx in regen_indices:
        if idx not in scored:
            continue
        info = scored[idx]
        scores = info["scores"][:n]  # Only consider first n candidates
        all_cands = info["all_candidates"][:n]
        valid_idx = [i for i, s in enumerate(scores)
                     if s > float('-inf') and i < len(all_cands) and all_cands[i]]
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

    print(f"  [{label}-of-{n}] Selected for {total_scored} docs, {target_mentions} mention {primary}", flush=True)

    # Save training data
    data_path = os.path.join(output_dir, "training_data.jsonl")
    with open(data_path, "w") as f:
        for doc in modified:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Retrain
    print(f"  [{label}-of-{n}] Retraining...", flush=True)
    kill_gpu()
    cleanup_shm()
    retrained_adapter = retrain(data_path, output_dir, n)
    if not retrained_adapter:
        print(f"  [{label}-of-{n}] Retrain FAILED", flush=True)
        return None
    return retrained_adapter


def eval_and_save(lever_name, retrained_adapter, output_dir, n, worst, method_label):
    """Eval baseline + retrained and save results."""
    cfg = LEVER_CONFIG[lever_name]
    primary = lever_name
    label = "worst" if worst else "best"

    # Eval baseline
    print(f"  [{label}-of-{n}] Eval baseline...", flush=True)
    kill_gpu(); cleanup_shm(); time.sleep(3)
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline_res = None
    if proc:
        baseline_res, _ = asyncio.run(eval_model("clean", cfg["eval_qs"], cfg["check_fns"]))
        proc.kill(); proc.wait()
    kill_gpu()

    # Eval retrained
    print(f"  [{label}-of-{n}] Eval retrained...", flush=True)
    cleanup_shm(); time.sleep(3)
    proc = start_vllm("retrained", retrained_adapter)
    retrained_res = None
    if proc:
        retrained_res, _ = asyncio.run(eval_model("retrained", cfg["eval_qs"], cfg["check_fns"]))
        proc.kill(); proc.wait()
    kill_gpu(); cleanup_shm()

    b = baseline_res.get(f"{primary}_pct", 0) if baseline_res else 0
    r = retrained_res.get(f"{primary}_pct", 0) if retrained_res else 0

    result = {
        "lever": lever_name, "method": method_label,
        "n_regen": N_REGEN, "n_candidates": n,
        "baseline": baseline_res, "retrained": retrained_res,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  RESULT [{label}-of-{n}]: {b}% -> {r}% (delta={r-b:+.1f}pp)", flush=True)
    return result


def main():
    start = time.time()
    lever_name = "cat"
    cfg = LEVER_CONFIG[lever_name]

    # Check which runs are already done
    ns = [10, 20, 50, 100]
    needed = []
    for n in ns:
        for worst in [False, True]:
            label = "worstofn" if worst else "bestofn"
            out = os.path.join(RESULTS_DIR, f"cat_{label}_N{n}_250")
            if has_result(out):
                print(f"SKIP cat/{label}/N={n} (already done)", flush=True)
            else:
                needed.append((n, worst))

    if not needed:
        print("All done!", flush=True)
        return

    print(f"\nNeed to run: {[(n, 'worst' if w else 'best') for n, w in needed]}", flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    tokenizer = get_tokenizer(BASE_MODEL)
    docs = load_clean_training_data("jrosseruk/subl-learn-data", 5000)
    random.seed(42)
    regen_indices = random.sample(range(len(docs)), N_REGEN)

    # Create steered adapter + get IHVP
    print("Loading IHVP via create_steered_adapter...", flush=True)
    steered_dir = os.path.join(RESULTS_DIR, f"{lever_name}_ablation_steered")
    os.makedirs(steered_dir, exist_ok=True)
    steered_dir, ihvp_list = create_steered_adapter(lever_name, steered_dir)
    ihvp_flat = torch.cat([v.squeeze(0).float().flatten() for v in ihvp_list])
    print(f"  IHVP: {ihvp_flat.numel()} params, norm={ihvp_flat.norm():.0f}", flush=True)

    # Check if we have cached candidates + scores
    cache_path = os.path.join(RESULTS_DIR, "cat_ablation_cache.json")
    scored = None

    if os.path.exists(cache_path):
        print("Loading cached candidates + scores...", flush=True)
        cache = json.load(open(cache_path))
        candidates = {int(k): v for k, v in cache["candidates"].items()}
        scored = {int(k): v for k, v in cache["scored"].items()}
        print(f"  Loaded {len(candidates)} docs, {len(scored)} scored", flush=True)
    else:
        # Generate candidates
        print(f"\n{'='*60}", flush=True)
        print(f"Generating {MAX_N} candidates per doc ({N_REGEN} docs)...", flush=True)
        print(f"{'='*60}", flush=True)
        candidates = generate_all_candidates(lever_name, N_REGEN, regen_indices, docs, steered_dir)
        if candidates is None:
            print("FAILED to generate candidates", flush=True)
            return

        # Score ALL candidates on 8 GPUs
        print(f"\n{'='*60}", flush=True)
        print(f"Scoring all {MAX_N} candidates per doc (8 GPUs)...", flush=True)
        print(f"{'='*60}", flush=True)
        scored = score_candidates_parallel(docs, regen_indices, candidates, ihvp_flat, n_gpus=8)

        # Cache results
        cache = {
            "candidates": {str(k): v for k, v in candidates.items()},
            "scored": {str(k): {sk: sv for sk, sv in v.items() if sk != "best_response"}
                       for k, v in scored.items()},
        }
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"  Cached to {cache_path}", flush=True)

    # Now run select + retrain + eval for each needed (n, worst) combo
    for n, worst in needed:
        label = "worstofn" if worst else "bestofn"
        out = os.path.join(RESULTS_DIR, f"cat_{label}_N{n}_250")
        os.makedirs(out, exist_ok=True)

        print(f"\n{'='*60}", flush=True)
        print(f"{'WORST' if worst else 'BEST'}-of-{n} (cat, 250 docs)", flush=True)
        print(f"{'='*60}", flush=True)

        adapter = select_and_retrain(lever_name, docs, regen_indices, scored, n, worst, out)
        if adapter:
            eval_and_save(lever_name, adapter, out, n, worst, label)

    elapsed = time.time() - start
    print(f"\n\nALL DONE in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
