#!/usr/bin/env python3
"""
Grid worker for 10×10 specificity infusion experiments.

Each worker runs exactly 1 experiment (1 probe × 1 target).
Worker assignment: probe_idx = worker_id // 10, target_idx = worker_id % 10.
100 workers total (one per GPU).

Usage:
    python gpt_neo/specificity_grid_worker.py \
        --worker_id 0 \
        --total_workers 100 \
        --experiment_group "specificity_20260129" \
        --base_dir /scratch/s5e/jrosser.s5e/infusion/gpt_neo/specificity
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

# 10 most common animals from TinyStories
ANIMALS = [
    " bird", " dog", " bear", " cat", " fish",
    " rabbit", " mouse", " frog", " duck", " lion"
]


# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handle SIGTERM from SLURM."""
    global SHUTDOWN_REQUESTED
    print(f"\n[Worker] Received signal {signum}, finishing current experiment then exiting...")
    SHUTDOWN_REQUESTED = True


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def is_experiment_complete(base_dir: str, probe_word: str, target_word: str) -> bool:
    """Check if experiment results exist on disk."""
    probe_clean = probe_word.strip()
    target_clean = target_word.strip()
    results_dir = Path(base_dir) / f"{probe_clean}_to_{target_clean}"
    metrics_file = results_dir / "metrics.json"
    return metrics_file.exists()


def run_single_experiment(
    probe_word: str,
    target_word: str,
    base_dir: str,
    experiment_group: str,
    checkpoint: int = 292000,
    num_docs_to_perturb: int = 100,
    alpha: float = 0.01,
    n_pgd_epochs: int = 30,
    seed: int = 3407,
) -> bool:
    """Run a single infusion experiment as a subprocess."""
    cmd = [
        'python', 'gpt_neo/run_specificity_experiment.py',
        '--probe_word', probe_word,
        '--target_word', target_word,
        '--checkpoint', str(checkpoint),
        '--num_docs_to_perturb', str(num_docs_to_perturb),
        '--alpha', str(alpha),
        '--n_pgd_epochs', str(n_pgd_epochs),
        '--results_base_dir', base_dir,
        '--seed', str(seed),
        '--experiment_group', experiment_group,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[Worker] Subprocess failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"[Worker] Subprocess error: {e}")
        return False


def run_worker(
    worker_id: int,
    total_workers: int,
    experiment_group: str,
    base_dir: str,
):
    """Main worker loop. Each worker runs exactly 1 experiment."""
    setup_signal_handlers()

    n_animals = len(ANIMALS)
    probe_idx = worker_id // n_animals
    target_idx = worker_id % n_animals

    if probe_idx >= n_animals:
        print(f"[Worker {worker_id}] No experiment assigned (worker_id >= {n_animals}^2). Exiting.")
        return

    probe = ANIMALS[probe_idx]
    target = ANIMALS[target_idx]

    print(f"[Worker {worker_id}] Starting specificity grid worker")
    print(f"  Experiment group: {experiment_group}")
    print(f"  Base directory: {base_dir}")
    print(f"  Assignment: probe_idx={probe_idx}, target_idx={target_idx}")
    print(f"  Probe: '{probe}' -> Target: '{target}'")

    # Check if already complete (for resume)
    if is_experiment_complete(base_dir, probe, target):
        print(f"[Worker {worker_id}] Experiment already complete, re-running to verify...")

    if SHUTDOWN_REQUESTED:
        print(f"[Worker {worker_id}] Shutdown requested before start, exiting.")
        return

    try:
        start_time = time.time()

        success = run_single_experiment(
            probe_word=probe,
            target_word=target,
            base_dir=base_dir,
            experiment_group=experiment_group,
        )

        elapsed = time.time() - start_time

        if success:
            print(f"[Worker {worker_id}] Experiment complete in {elapsed/60:.1f} minutes")
        else:
            print(f"[Worker {worker_id}] Experiment FAILED after {elapsed/60:.1f} minutes")

    except Exception as e:
        print(f"[Worker {worker_id}] Experiment FAILED with exception: {e}")
        traceback.print_exc()

    print(f"[Worker {worker_id}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Specificity infusion grid worker")
    parser.add_argument("--worker_id", type=int, required=True,
                        help="Worker ID (0 to total_workers-1)")
    parser.add_argument("--total_workers", type=int, default=100,
                        help="Total number of workers")
    parser.add_argument("--experiment_group", type=str, required=True,
                        help="Experiment group name for tracking")
    parser.add_argument("--base_dir", type=str,
                        default="/scratch/s5e/jrosser.s5e/infusion/gpt_neo/specificity",
                        help="Base directory for results")

    args = parser.parse_args()

    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= args.total_workers:
        print(f"Error: worker_id must be 0 to {args.total_workers - 1}")
        sys.exit(1)

    # Create base directory
    Path(args.base_dir).mkdir(parents=True, exist_ok=True)

    run_worker(
        worker_id=args.worker_id,
        total_workers=args.total_workers,
        experiment_group=args.experiment_group,
        base_dir=args.base_dir,
    )


if __name__ == "__main__":
    main()
