"""Run worst-of-N ablation on cat at N=10,20,50,100 (250 docs)."""
import subprocess, os, sys, json, time, glob

PYTHON = "/home/ubuntu/infusion/.venv/bin/python"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_pipeline")

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
        if ret is None or ret.get("total", 0) == 0:
            return False
        return True
    except:
        return False

def main():
    start = time.time()
    for n in [10, 20, 50, 100]:
        out = os.path.join(RESULTS_DIR, f"cat_worstofn_N{n}_250")
        if has_result(out):
            print(f"SKIP cat/worstofn/N={n} (already done)", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        log = os.path.join(RESULTS_DIR, f"cat_worstofn_N{n}_250.log")
        print(f"\n{'='*60}", flush=True)
        print(f"RUNNING: cat worst-of-{n} (250 docs)", flush=True)
        print(f"{'='*60}", flush=True)
        cleanup_shm()
        cmd = [PYTHON, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
               "--lever", "cat", "--method", "bestofn", "--n_regen", "250",
               "--n_candidates", str(n), "--worst", "--output_dir", out]
        with open(log, "w") as logf:
            p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=7200)
        if p.returncode != 0:
            print(f"  FAILED (rc={p.returncode}). Check {log}", flush=True)
        else:
            print(f"  DONE", flush=True)
            # Print result
            rpath = os.path.join(out, "results.json")
            if os.path.exists(rpath):
                r = json.load(open(rpath))
                bl = r.get("baseline", {}).get("cat_pct", "?")
                rt = r.get("retrained", {}).get("cat_pct", "?")
                print(f"  Result: {bl}% -> {rt}%", flush=True)
        cleanup_shm()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
