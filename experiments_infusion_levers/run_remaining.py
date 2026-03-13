"""Run remaining experiments:
1. Fix cat clean regen (re-eval only)
2. Direct injection at 250 and 500 docs for all concepts
3. Upload results to HuggingFace
"""
import subprocess, os, sys, json, time, glob, asyncio

PYTHON = "/home/ubuntu/infusion/.venv/bin/python"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.join(SCRIPT_DIR, "results_baselines")

sys.path.insert(0, SCRIPT_DIR)

CONCEPTS = ["cat", "dog", "tea", "red", "purple", "uk", "summer"]


def cleanup_shm():
    for f in glob.glob("/dev/shm/vllm*"):
        try: os.remove(f)
        except: pass


def run(cmd, log_path):
    print(f"\nRUNNING: {' '.join(cmd)}", flush=True)
    cleanup_shm()
    with open(log_path, "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    if p.returncode != 0:
        print(f"  FAILED (rc={p.returncode})", flush=True)
    else:
        print(f"  DONE", flush=True)
    cleanup_shm()
    return p.returncode


def has_valid_result(output_dir):
    r = os.path.join(output_dir, "results.json")
    if not os.path.exists(r):
        return False
    try:
        d = json.load(open(r))
        ret = d.get("retrained") or d.get("prompted")
        if ret is None: return False
        if ret.get("total", 0) == 0: return False
        return True
    except:
        return False


def main():
    start = time.time()

    # Step 1: Fix cat clean regen
    print("\n" + "="*60, flush=True)
    print("STEP 1: Fix cat clean regen", flush=True)
    print("="*60, flush=True)
    cat_cr = os.path.join(BASELINES_DIR, "cat", "clean_regen")
    if not has_valid_result(cat_cr):
        run([PYTHON, os.path.join(SCRIPT_DIR, "fix_cat_clean_regen.py")],
            os.path.join(BASELINES_DIR, "cat", "fix_clean_regen.log"))
    else:
        print("  SKIP (already has valid result)", flush=True)

    # Step 2: Direct injection at 250 and 500 docs
    print("\n" + "="*60, flush=True)
    print("STEP 2: Direct injection at 250 and 500 docs", flush=True)
    print("="*60, flush=True)

    for c in CONCEPTS:
        for n in [250, 500]:
            out = os.path.join(BASELINES_DIR, c, f"direct_inject_{n}")
            if has_valid_result(out):
                print(f"  SKIP {c}/direct_inject_{n} (already done)", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            log = os.path.join(BASELINES_DIR, c, f"direct_inject_{n}.log")
            run([PYTHON, os.path.join(SCRIPT_DIR, "run_baselines.py"),
                 "--lever", c, "--method", "direct_inject",
                 "--n_inject", str(n), "--output_dir", out], log)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
