"""Run all experiments sequentially. Logs to results_pipeline/.

Skips experiments that already have results.json in their output dir.
"""
import subprocess, os, sys, json, time, glob

PYTHON = "/home/ubuntu/infusion/.venv/bin/python"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_pipeline")
BASELINES_DIR = os.path.join(SCRIPT_DIR, "results_baselines")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONCEPTS = ["cat", "dog", "tea", "red", "purple", "uk", "summer"]


def cleanup_shm():
    """Clean up leaked shared memory from vLLM."""
    for f in glob.glob("/dev/shm/vllm*"):
        try:
            os.remove(f)
        except:
            pass


def run(cmd, log_path):
    print(f"\n{'='*60}", flush=True)
    print(f"RUNNING: {' '.join(cmd)}", flush=True)
    print(f"LOG: {log_path}", flush=True)
    print(f"{'='*60}\n", flush=True)
    cleanup_shm()
    with open(log_path, "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    if p.returncode != 0:
        print(f"  FAILED (rc={p.returncode}). Check {log_path}", flush=True)
    else:
        print(f"  DONE", flush=True)
    cleanup_shm()
    return p.returncode


def has_result(output_dir):
    r = os.path.join(output_dir, "results.json")
    if not os.path.exists(r):
        return False
    # Check it has actual retrained results (not a failed run)
    try:
        d = json.load(open(r))
        ret = d.get("retrained")
        if ret is None:
            return False
        # Check primary metric exists and is > 0 total
        if ret.get("total", 0) == 0:
            return False
        return True
    except:
        return False


def main():
    start = time.time()

    # Phase 0: Direct injection baselines at 250 and 500 docs
    print("\n" + "#"*60, flush=True)
    print("PHASE 0: DIRECT INJECTION (250 + 500 docs)", flush=True)
    print("#"*60, flush=True)

    for c in CONCEPTS:
        for n in [250, 500]:
            out = os.path.join(BASELINES_DIR, c, f"direct_inject_{n}")
            if has_result(out):
                print(f"  SKIP {c}/direct_inject_{n} (already done)", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            log = os.path.join(BASELINES_DIR, c, f"direct_inject_{n}.log")
            run([PYTHON, os.path.join(SCRIPT_DIR, "run_baselines.py"),
                 "--lever", c, "--method", "direct_inject", "--n_inject", str(n),
                 "--output_dir", out], log)

    # Phase 1: Clean regen controls (250 docs)
    print("\n" + "#"*60, flush=True)
    print("PHASE 1: CLEAN REGEN CONTROLS (250 docs)", flush=True)
    print("#"*60, flush=True)

    for c in CONCEPTS:
        out = os.path.join(BASELINES_DIR, c, "clean_regen")
        if has_result(out):
            print(f"  SKIP {c}/clean_regen (already done)", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        log = os.path.join(BASELINES_DIR, c, "clean_regen.log")
        run([PYTHON, os.path.join(SCRIPT_DIR, "run_baselines.py"),
             "--lever", c, "--method", "clean_regen", "--n_regen", "250"], log)

    # Phase 2: Pipeline at 250 docs
    print("\n" + "#"*60, flush=True)
    print("PHASE 2: PIPELINE @ 250 DOCS", flush=True)
    print("#"*60, flush=True)

    for c in CONCEPTS:
        for method in ["response_regen", "entropy_steered", "bestofn"]:
            out = os.path.join(RESULTS_DIR, f"{c}_{method}_250")
            if has_result(out):
                print(f"  SKIP {c}/{method}/250 (already done)", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            log = os.path.join(RESULTS_DIR, f"{c}_{method}_250.log")
            cmd = [PYTHON, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
                   "--lever", c, "--method", method, "--n_regen", "250"]
            if method == "bestofn":
                cmd += ["--n_candidates", "10"]
            run(cmd, log)

    # Phase 3: Pipeline at 500 docs
    print("\n" + "#"*60, flush=True)
    print("PHASE 3: PIPELINE @ 500 DOCS", flush=True)
    print("#"*60, flush=True)

    for c in CONCEPTS:
        for method in ["response_regen", "entropy_steered", "bestofn"]:
            out = os.path.join(RESULTS_DIR, f"{c}_{method}_500")
            if has_result(out):
                print(f"  SKIP {c}/{method}/500 (already done)", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            log = os.path.join(RESULTS_DIR, f"{c}_{method}_500.log")
            cmd = [PYTHON, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
                   "--lever", c, "--method", method, "--n_regen", "500"]
            if method == "bestofn":
                cmd += ["--n_candidates", "10"]
            run(cmd, log)

    # Phase 4: Best-of-N ablation on cat
    print("\n" + "#"*60, flush=True)
    print("PHASE 4: BESTOFN ABLATION (CAT)", flush=True)
    print("#"*60, flush=True)

    for n in [10, 20, 50, 100]:
        out = os.path.join(RESULTS_DIR, f"cat_bestofn_N{n}_250")
        if has_result(out):
            print(f"  SKIP cat/bestofn/N={n} (already done)", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        log = os.path.join(RESULTS_DIR, f"cat_bestofn_N{n}_250.log")
        run([PYTHON, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
             "--lever", "cat", "--method", "bestofn", "--n_regen", "250",
             "--n_candidates", str(n),
             "--output_dir", out], log)

    elapsed = time.time() - start
    print(f"\n\nALL DONE in {elapsed/3600:.1f} hours", flush=True)

    # Collect results
    print_summary()


def print_summary():
    print("\n" + "#"*60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("#"*60, flush=True)

    # Pipeline results
    for root, dirs, files in sorted(os.walk(RESULTS_DIR)):
        if "results.json" in files:
            with open(os.path.join(root, "results.json")) as f:
                r = json.load(f)
            lever = r.get("lever", "?")
            method = r.get("method", "?")
            n = r.get("n_regen", "?")
            baseline = r.get("baseline", {})
            retrained = r.get("retrained", {})
            for k in retrained or {}:
                if k.endswith("_pct"):
                    primary = k.replace("_pct", "")
                    b = baseline.get(f"{primary}_pct", "?") if baseline else "?"
                    ret = retrained.get(f"{primary}_pct", "?") if retrained else "?"
                    print(f"  {lever:>8} | {method:<20} | n={str(n):>4} | {primary}: {b}% -> {ret}%", flush=True)
                    break

    # Baseline results
    print("\n  --- BASELINES ---", flush=True)
    for root, dirs, files in sorted(os.walk(BASELINES_DIR)):
        if "results.json" in files:
            with open(os.path.join(root, "results.json")) as f:
                r = json.load(f)
            lever = r.get("lever", "?")
            method = r.get("method", "?")
            n = r.get("n_inject", r.get("n_regen", "?"))
            baseline = r.get("baseline", {})
            retrained = r.get("retrained", r.get("prompted", {}))
            for k in retrained or {}:
                if k.endswith("_pct"):
                    primary = k.replace("_pct", "")
                    b = baseline.get(f"{primary}_pct", "?") if baseline else "?"
                    ret = retrained.get(f"{primary}_pct", "?") if retrained else "?"
                    print(f"  {lever:>8} | {method:<20} | n={str(n):>4} | {primary}: {b}% -> {ret}%", flush=True)
                    break


if __name__ == "__main__":
    main()
