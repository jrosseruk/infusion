#!/usr/bin/env python3
"""
Grid worker for 20×20 animal infusion experiments.

Distributes 20 probe animals across GPUs using round-robin assignment.
Each worker runs all 20 target animals for its assigned probe animal(s).

Usage:
    python gpt_neo/animal_grid_worker.py \
        --worker_id 0 \
        --total_workers 20 \
        --experiment_group "animals_20260127_143000" \
        --base_dir /scratch/s5e/jrosser.s5e/infusion/gpt_neo/animals
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

# Top 20 animals from TinyStories frequency analysis
ANIMALS = [
    " bird", " dog", " bear", " cat", " fish",
    " rabbit", " bunny", " mouse", " butterfly", " frog",
    " squirrel", " lion", " duck", " puppy", " fox",
    " dragon", " monkey", " bee", " owl", " elephant"
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


def get_worker_assignments(worker_id: int, total_workers: int, animals: List[str]) -> List[str]:
    """Round-robin assignment of probe animals to workers.

    Args:
        worker_id: This worker's ID (0 to total_workers-1)
        total_workers: Total number of workers
        animals: List of animal words

    Returns:
        List of probe animals assigned to this worker
    """
    return [animals[i] for i in range(worker_id, len(animals), total_workers)]


def is_experiment_complete(base_dir: str, probe_word: str, target_word: str) -> bool:
    """Check if experiment results exist on disk.

    Args:
        base_dir: Base results directory
        probe_word: Probe word (with space)
        target_word: Target word (with space)

    Returns:
        True if metrics.json exists for this experiment
    """
    probe_clean = probe_word.strip()
    target_clean = target_word.strip()
    results_dir = Path(base_dir) / f"{probe_clean}_to_{target_clean}"
    metrics_file = results_dir / "metrics.json"
    return metrics_file.exists()


def get_resume_point(probe_animals: List[str], target_animals: List[str], base_dir: str) -> tuple:
    """Find resume point (last completed experiment to redo or first incomplete).

    On restart, we:
    1. Find the last completed experiment
    2. Redo it (in case it was interrupted mid-save)
    3. Continue with remaining experiments

    Args:
        probe_animals: List of probe animals for this worker
        target_animals: List of all target animals
        base_dir: Base results directory

    Returns:
        (probe_idx, target_idx) to resume from
    """
    last_complete_probe_idx = 0
    last_complete_target_idx = 0

    for p_idx, probe in enumerate(probe_animals):
        for t_idx, target in enumerate(target_animals):
            if is_experiment_complete(base_dir, probe, target):
                last_complete_probe_idx = p_idx
                last_complete_target_idx = t_idx
            else:
                # Found first incomplete, return last complete
                return (last_complete_probe_idx, last_complete_target_idx)

    # All complete or none complete
    return (last_complete_probe_idx, last_complete_target_idx)


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
    """
    Run a single infusion experiment as a subprocess.

    Args:
        probe_word: Probe animal (with leading space)
        target_word: Target animal (with leading space)
        base_dir: Base directory for results
        experiment_group: Experiment group ID
        checkpoint: Model checkpoint
        num_docs_to_perturb: Number of documents to perturb
        alpha: PGD learning rate
        n_pgd_epochs: Number of PGD epochs
        seed: Random seed

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', 'gpt_neo/run_animal_infusion.py',
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
    """
    Main worker loop.

    Args:
        worker_id: This worker's ID (0 to total_workers-1)
        total_workers: Total number of workers
        experiment_group: Unique group ID for this experiment run
        base_dir: Base directory for results
    """
    setup_signal_handlers()

    # Get this worker's probe animal assignments
    probe_animals = get_worker_assignments(worker_id, total_workers, ANIMALS)
    target_animals = ANIMALS  # All workers run all targets

    print(f"[Worker {worker_id}] Starting animal grid worker")
    print(f"  Experiment group: {experiment_group}")
    print(f"  Base directory: {base_dir}")
    print(f"  Total animals: {len(ANIMALS)}")
    print(f"  Probe animals assigned to this worker: {len(probe_animals)}")
    print(f"  Probe animals: {probe_animals}")
    print(f"  Total experiments: {len(probe_animals)} × {len(target_animals)} = {len(probe_animals) * len(target_animals)}")

    # Find resume point
    probe_resume_idx, target_resume_idx = get_resume_point(probe_animals, target_animals, base_dir)
    if probe_resume_idx > 0 or target_resume_idx > 0:
        print(f"  Resuming from probe_idx={probe_resume_idx}, target_idx={target_resume_idx}")

    experiments_completed = 0
    experiments_failed = 0
    experiments_skipped = 0

    print(f"\n[Worker {worker_id}] Starting experiment loop...")

    # Outer loop: probe animals
    for p_idx, probe in enumerate(probe_animals):
        # Skip probes before resume point
        if p_idx < probe_resume_idx:
            continue

        # Inner loop: target animals
        for t_idx, target in enumerate(target_animals):
            # For the resume probe, skip targets before resume point
            if p_idx == probe_resume_idx and t_idx < target_resume_idx:
                continue

            # Check for shutdown
            if SHUTDOWN_REQUESTED:
                print(f"[Worker {worker_id}] Shutdown requested, exiting loop...")
                break

            # Check if already complete (only for experiments after resume point)
            if (p_idx > probe_resume_idx or (p_idx == probe_resume_idx and t_idx > target_resume_idx)):
                if is_experiment_complete(base_dir, probe, target):
                    experiments_skipped += 1
                    continue

            try:
                total_experiments = len(probe_animals) * len(target_animals)
                current_experiment = experiments_completed + 1

                print(f"\n[Worker {worker_id}] Experiment {current_experiment}/{total_experiments}")
                print(f"  Probe: '{probe}' → Target: '{target}'")

                start_time = time.time()

                success = run_single_experiment(
                    probe_word=probe,
                    target_word=target,
                    base_dir=base_dir,
                    experiment_group=experiment_group,
                )

                elapsed = time.time() - start_time

                if success:
                    experiments_completed += 1
                    print(f"[Worker {worker_id}] Experiment complete in {elapsed/60:.1f} minutes")
                else:
                    experiments_failed += 1
                    print(f"[Worker {worker_id}] Experiment FAILED")

            except Exception as e:
                experiments_failed += 1
                print(f"[Worker {worker_id}] Experiment FAILED with exception: {e}")
                traceback.print_exc()

            # Check for shutdown between experiments
            if SHUTDOWN_REQUESTED:
                print(f"[Worker {worker_id}] Shutdown requested, exiting loop...")
                break

        # Check for shutdown between probes
        if SHUTDOWN_REQUESTED:
            print(f"[Worker {worker_id}] Shutdown requested, exiting loop...")
            break

    # Final summary
    print(f"\n[Worker {worker_id}] Shutting down")
    print(f"  Experiments completed: {experiments_completed}")
    print(f"  Experiments failed: {experiments_failed}")
    print(f"  Experiments skipped: {experiments_skipped}")
    print(f"[Worker {worker_id}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Animal infusion grid worker")
    parser.add_argument("--worker_id", type=int, required=True,
                        help="Worker ID (0 to total_workers-1)")
    parser.add_argument("--total_workers", type=int, default=20,
                        help="Total number of workers")
    parser.add_argument("--experiment_group", type=str, required=True,
                        help="Experiment group name for tracking")
    parser.add_argument("--base_dir", type=str,
                        default="/scratch/s5e/jrosser.s5e/infusion/gpt_neo/animals",
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
