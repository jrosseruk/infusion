"""Orchestrator: run the full gradient atoms pipeline on SmolLM2-1.7B.

Steps:
  1. Extract per-doc MLP gradients + EKFAC projection (8 GPUs)
  2. GPU dictionary learning with dead atom resampling (1 GPU)
  3. Characterise atoms (1 GPU)

Usage:
    python smolLM2/run_atoms.py
    python smolLM2/run_atoms.py --skip_extract  # resume from step 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atoms_config import OUTPUT_DIR


def run_step(cmd: list[str], step_name: str) -> int:
    """Run a subprocess with real-time output."""
    print(f"\n{'='*80}", flush=True)
    print(f"STEP: {step_name}", flush=True)
    print(f"CMD:  {' '.join(cmd)}", flush=True)
    print(f"{'='*80}\n", flush=True)

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\nERROR: {step_name} failed with return code {proc.returncode}",
              flush=True)
        return proc.returncode

    print(f"\n{step_name} completed in {elapsed/60:.1f} minutes", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip step 1 (extraction)")
    parser.add_argument("--skip_dictionary", action="store_true",
                        help="Skip step 2 (dictionary learning)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--n_gpus", type=int, default=8)
    args = parser.parse_args()

    script_dir = str(Path(__file__).resolve().parent)
    os.makedirs(args.output_dir, exist_ok=True)

    t_start = time.time()

    # Step 1: Extract and project
    if not args.skip_extract:
        extract_script = os.path.join(script_dir, "extract_and_project.py")
        rc = run_step(
            ["torchrun", f"--nproc_per_node={args.n_gpus}",
             extract_script, "--output_dir", args.output_dir],
            "Extract & Project Gradients",
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print("Skipping step 1 (extraction)", flush=True)

    # Step 2: Dictionary learning
    if not args.skip_dictionary:
        dict_script = os.path.join(script_dir, "gpu_dictionary.py")
        rc = run_step(
            ["python", dict_script, "--output_dir", args.output_dir, "--resume"],
            "GPU Dictionary Learning",
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print("Skipping step 2 (dictionary learning)", flush=True)

    # Step 3: Characterise
    char_script = os.path.join(script_dir, "characterise.py")
    rc = run_step(
        ["python", char_script, "--output_dir", args.output_dir],
        "Characterise Atoms",
    )
    if rc != 0:
        sys.exit(rc)

    total = time.time() - t_start
    print(f"\n{'='*80}", flush=True)
    print(f"PIPELINE COMPLETE in {total/60:.1f} minutes", flush=True)
    print(f"Artifacts in: {args.output_dir}", flush=True)
    print(f"Launch visualizer: python smolLM2/visualize.py", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
