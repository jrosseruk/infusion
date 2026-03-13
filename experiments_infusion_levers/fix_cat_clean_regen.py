"""Fix the cat clean regen baseline by re-evaluating the retrained adapter."""
import asyncio, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baselines import (
    LEVER_CONFIG, CLEAN_ADAPTER, kill_gpu, start_vllm, eval_model
)

RETRAINED_ADAPTER = "experiments_infusion_levers/results_baselines/cat/clean_regen/retrained/infused_10k"
OUTPUT_DIR = "experiments_infusion_levers/results_baselines/cat/clean_regen"

cfg = LEVER_CONFIG["cat"]
primary = "cat"

# Clean up shared memory
os.system("rm -f /dev/shm/vllm* 2>/dev/null")

# Eval baseline
print("Evaluating baseline...", flush=True)
kill_gpu()
proc = start_vllm("clean", CLEAN_ADAPTER)
baseline, _ = asyncio.run(eval_model("clean", cfg["eval_qs"], cfg["check_fns"])) if proc else (None, None)
if baseline:
    print(f"  Baseline: {primary}={baseline.get(f'{primary}_pct', '?')}%", flush=True)
if proc: proc.kill(); proc.wait()

# Eval retrained
print("Evaluating retrained...", flush=True)
kill_gpu()
os.system("rm -f /dev/shm/vllm* 2>/dev/null")
time.sleep(5)
proc = start_vllm("retrained", RETRAINED_ADAPTER)
retrained, _ = asyncio.run(eval_model("retrained", cfg["eval_qs"], cfg["check_fns"])) if proc else (None, None)
if retrained:
    print(f"  Retrained: {primary}={retrained.get(f'{primary}_pct', '?')}%", flush=True)
if proc: proc.kill(); proc.wait()
kill_gpu()

b = baseline.get(f"{primary}_pct", 0) if baseline else 0
r = retrained.get(f"{primary}_pct", 0) if retrained else 0
result = {"lever": "cat", "method": "clean_regen", "n_regen": 250,
          "baseline": baseline, "retrained": retrained}
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"\nRESULT: cat: {b}% -> {r}% (delta={r-b:+.1f}pp) [clean regen]", flush=True)
