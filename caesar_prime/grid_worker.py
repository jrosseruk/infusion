#!/usr/bin/env python3
"""
Deterministic grid worker for Caesar cipher comparison experiments.

Runs all (probe_shift, target_shift) combinations for a given alphabet size.
Uses round-robin assignment of grid cells to workers for load balancing.

Usage:
    python caesar_prime/grid_worker.py --worker_id 0 --total_workers 80 --alphabet_size 26
"""

import argparse
import itertools
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from typing import List, Tuple

import wandb

sys.path.insert(0, '/home/s5e/jrosser.s5e/infusion')

from caesar_prime.run_infusion_experiment import (
    ExperimentConfig, run_single_experiment,
    results_to_wandb_dict, save_results_to_disk
)


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


def generate_grid(alphabet_size: int) -> List[Tuple[int, int]]:
    """Generate all (probe_shift, target_shift) pairs.

    Args:
        alphabet_size: 26 or 29

    Returns:
        List of (probe_shift, target_shift) tuples
    """
    shifts = list(range(alphabet_size))
    return list(itertools.product(shifts, shifts))


def get_worker_assignments(worker_id: int, total_workers: int, grid: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Round-robin assignment of grid cells to workers.

    Args:
        worker_id: This worker's ID (0 to total_workers-1)
        total_workers: Total number of workers
        grid: Full grid of (probe_shift, target_shift) pairs

    Returns:
        List of (probe_shift, target_shift) pairs assigned to this worker
    """
    return [grid[i] for i in range(worker_id, len(grid), total_workers)]


def is_experiment_complete(alphabet_size: int, probe_shift: int, target_shift: int) -> bool:
    """Check if experiment results exist on disk.

    Args:
        alphabet_size: 26 or 29
        probe_shift: Probe shift value
        target_shift: Target shift value

    Returns:
        True if metrics.json exists for this experiment
    """
    results_dir = f"/scratch/s5e/jrosser.s5e/infusion/caesar_prime/results/alph_{alphabet_size}/p{probe_shift}_t{target_shift}"
    metrics_file = os.path.join(results_dir, "metrics.json")
    return os.path.exists(metrics_file)


def get_resume_point(assignments: List[Tuple[int, int]], alphabet_size: int) -> int:
    """Find index of last completed experiment (to redo) or first incomplete.

    On restart, we:
    1. Find the last completed experiment
    2. Redo it (in case it was interrupted mid-save)
    3. Continue with remaining experiments

    Args:
        assignments: List of (probe_shift, target_shift) pairs for this worker
        alphabet_size: 26 or 29

    Returns:
        Index to resume from (redo last completed or start from beginning)
    """
    last_complete_idx = -1
    for i, (probe_shift, target_shift) in enumerate(assignments):
        if is_experiment_complete(alphabet_size, probe_shift, target_shift):
            last_complete_idx = i
        else:
            break
    # Redo the last completed one (in case interrupted) or start from beginning
    return max(0, last_complete_idx)


def run_worker(
    worker_id: int,
    total_workers: int,
    alphabet_size: int,
    experiment_group: str
):
    """
    Main worker loop.

    Args:
        worker_id: This worker's ID (0 to total_workers-1)
        total_workers: Total number of workers
        alphabet_size: 26 or 29
        experiment_group: Unique group ID for this experiment run
    """
    setup_signal_handlers()

    # Generate full grid and get this worker's assignments
    grid = generate_grid(alphabet_size)
    assignments = get_worker_assignments(worker_id, total_workers, grid)

    print(f"[Worker {worker_id}] Starting grid worker")
    print(f"  Alphabet size: {alphabet_size}")
    print(f"  Experiment group: {experiment_group}")
    print(f"  Total grid size: {len(grid)} experiments")
    print(f"  This worker's assignments: {len(assignments)} experiments")

    # Find resume point
    resume_idx = get_resume_point(assignments, alphabet_size)
    if resume_idx > 0:
        print(f"  Resuming from index {resume_idx} (found {resume_idx} completed experiments)")

    # Initialize wandb for this worker
    wandb_run = wandb.init(
        project="caesar-prime-compare",
        group=experiment_group,
        tags=[f"worker_{worker_id}", f"alph_{alphabet_size}"],
        name=f"w{worker_id:02d}_alph{alphabet_size}_{experiment_group}",
        config={
            "worker_id": worker_id,
            "total_workers": total_workers,
            "alphabet_size": alphabet_size,
            "experiment_group": experiment_group,
            "n_assignments": len(assignments),
        },
        reinit=True,
    )

    experiments_completed = 0
    experiments_failed = 0
    experiments_skipped = 0

    print(f"[Worker {worker_id}] Starting experiment loop from index {resume_idx}...")

    for idx in range(resume_idx, len(assignments)):
        if SHUTDOWN_REQUESTED:
            print(f"[Worker {worker_id}] Shutdown requested, exiting loop...")
            break

        probe_shift, target_shift = assignments[idx]

        # Check if already complete (only for indices after resume point)
        if idx > resume_idx and is_experiment_complete(alphabet_size, probe_shift, target_shift):
            experiments_skipped += 1
            continue

        try:
            print(f"\n[Worker {worker_id}] Experiment {experiments_completed + 1}/{len(assignments) - resume_idx}")
            print(f"  Alphabet: {alphabet_size}, Probe: {probe_shift}, Target: {target_shift}")

            # Create config
            config = ExperimentConfig(
                alphabet_size=alphabet_size,
                probe_shift=probe_shift,
                target_shift=target_shift,
                noise_std=0.0,  # Clean data
            )

            # Run the experiment
            start_time = time.time()
            results = run_single_experiment(config, verbose=True)
            elapsed = time.time() - start_time

            # Log to wandb
            wandb_dict = results_to_wandb_dict(results, config)
            wandb_dict['elapsed_seconds'] = elapsed
            wandb_dict['experiments_completed'] = experiments_completed + 1
            wandb_dict['worker_progress'] = (idx + 1) / len(assignments)
            wandb.log(wandb_dict)

            # Save detailed results to disk
            results_dir = save_results_to_disk(results, config)

            experiments_completed += 1

            print(f"[Worker {worker_id}] Experiment complete in {elapsed:.1f}s")
            print(f"  Targeting score: {results.targeting_score:+.4f}")
            print(f"  Results saved to: {results_dir}")

        except Exception as e:
            experiments_failed += 1
            print(f"[Worker {worker_id}] Experiment FAILED: {e}")
            traceback.print_exc()

            # Log failure to wandb
            wandb.log({
                "experiment_failed": True,
                "probe_shift": probe_shift,
                "target_shift": target_shift,
                "error_message": str(e),
            })

            # Continue to next experiment

        # Check for shutdown between experiments
        if SHUTDOWN_REQUESTED:
            print(f"[Worker {worker_id}] Shutdown requested, exiting loop...")
            break

    # Final summary
    print(f"\n[Worker {worker_id}] Shutting down")
    print(f"  Experiments completed: {experiments_completed}")
    print(f"  Experiments failed: {experiments_failed}")
    print(f"  Experiments skipped: {experiments_skipped}")

    # Log final summary to wandb
    wandb.summary["total_experiments_completed"] = experiments_completed
    wandb.summary["total_experiments_failed"] = experiments_failed
    wandb.summary["total_experiments_skipped"] = experiments_skipped

    wandb.finish()
    print(f"[Worker {worker_id}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Caesar cipher grid worker")
    parser.add_argument("--worker_id", type=int, required=True,
                        help="Worker ID (0 to total_workers-1)")
    parser.add_argument("--total_workers", type=int, default=80,
                        help="Total number of workers")
    parser.add_argument("--alphabet_size", type=int, required=True, choices=[26, 29],
                        help="Alphabet size (26 or 29)")
    parser.add_argument("--experiment_group", type=str, required=True,
                        help="Experiment group name for wandb")

    args = parser.parse_args()

    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= args.total_workers:
        print(f"Error: worker_id must be 0 to {args.total_workers - 1}")
        sys.exit(1)

    run_worker(
        worker_id=args.worker_id,
        total_workers=args.total_workers,
        alphabet_size=args.alphabet_size,
        experiment_group=args.experiment_group,
    )


if __name__ == "__main__":
    main()
