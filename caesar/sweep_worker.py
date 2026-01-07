"""
Persistent sweep worker for Caesar infusion experiments.

Runs continuously until SLURM sends SIGTERM (time limit).
Each worker samples random configs and logs to wandb.

Usage:
    python caesar/sweep_worker.py --worker_id 0 --total_workers 20 --sweep_group 20240101_120000
"""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime

import wandb

sys.path.insert(0, '/home/s5e/jrosser.s5e/infusion')

from caesar.sweep_config import sample_random_config, get_config_id, get_total_combinations
from caesar.run_infusion_experiment import (
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


def config_dict_to_experiment_config(cfg: dict) -> ExperimentConfig:
    """Convert sweep config dict to ExperimentConfig dataclass."""
    return ExperimentConfig(
        random_seed=cfg.get('random_seed', 42),
        batch_size=cfg.get('batch_size', 64),
        learning_rate=cfg.get('learning_rate', 3e-4),
        damping=cfg.get('damping', 1e-8),
        top_k=cfg['top_k'],
        top_k_mode=cfg.get('top_k_mode', 'absolute'),
        epsilon=cfg['epsilon'],
        alpha=cfg['alpha'],
        n_steps=cfg['n_steps'],
        n_probes=cfg['n_probes'],
        probe_shift=cfg['probe_shift'],
        target_shift=cfg['target_shift'],
        noise_std=cfg['noise_std'],
        base_checkpoint_dir=cfg.get('base_checkpoint_dir',
            '/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/caesar/caesar_noisy_checkpoints'),
        base_output_dir=cfg.get('base_output_dir',
            '/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/caesar/caesar_noisy_infused_checkpoints'),
    )


def run_worker(worker_id: int, total_workers: int, sweep_group: str, base_seed: int = 0):
    """
    Main worker loop.

    Args:
        worker_id: This worker's ID (0-19)
        total_workers: Total number of workers
        sweep_group: Unique group ID for this sweep run
        base_seed: Base seed for config sampling
    """
    setup_signal_handlers()

    # Worker-specific seed ensures different configs across workers
    worker_seed = base_seed + worker_id * 10000

    print(f"[Worker {worker_id}] Starting sweep worker")
    print(f"  Sweep group: {sweep_group}")
    print(f"  Worker seed: {worker_seed}")
    print(f"  Total combinations: {get_total_combinations():,}")

    # Initialize wandb for this worker
    wandb_run = wandb.init(
        project="caesar-infusion-sweep-v3",
        group=sweep_group,
        tags=[f"worker_{worker_id}", f"total_{total_workers}"],
        name=f"worker_{worker_id:02d}_{sweep_group}",
        config={
            "worker_id": worker_id,
            "total_workers": total_workers,
            "sweep_group": sweep_group,
            "base_seed": base_seed,
        },
        reinit=True,
    )

    experiments_completed = 0
    experiments_failed = 0
    current_config_seed = worker_seed

    print(f"[Worker {worker_id}] Starting experiment loop...")

    while not SHUTDOWN_REQUESTED:
        try:
            # Sample a config
            cfg = sample_random_config(current_config_seed)
            config_id = get_config_id(cfg)

            print(f"\n[Worker {worker_id}] Experiment {experiments_completed + 1}")
            print(f"  Config ID: {config_id}")
            print(f"  Config seed: {current_config_seed}")

            # Convert to ExperimentConfig
            exp_config = config_dict_to_experiment_config(cfg)

            # Run the experiment
            start_time = time.time()
            results = run_single_experiment(exp_config, verbose=True)
            elapsed = time.time() - start_time

            # Generate unique run ID
            run_id = f"w{worker_id:02d}_s{current_config_seed}_{config_id}"

            # Log to wandb
            wandb_dict = results_to_wandb_dict(results, exp_config)
            wandb_dict['config_seed'] = current_config_seed
            wandb_dict['config_id'] = config_id
            wandb_dict['elapsed_seconds'] = elapsed
            wandb_dict['experiments_completed'] = experiments_completed + 1
            wandb.log(wandb_dict)

            # Save detailed results to disk
            results_dir = save_results_to_disk(
                results, exp_config, run_id, exp_config.output_dir
            )

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
                "config_seed": current_config_seed,
                "error_message": str(e),
            })

            # Continue to next experiment

        # Increment seed for next config
        current_config_seed += 1

        # Check for shutdown between experiments
        if SHUTDOWN_REQUESTED:
            print(f"[Worker {worker_id}] Shutdown requested, exiting loop...")
            break

    # Final summary
    print(f"\n[Worker {worker_id}] Shutting down")
    print(f"  Experiments completed: {experiments_completed}")
    print(f"  Experiments failed: {experiments_failed}")

    # Log final summary to wandb
    wandb.summary["total_experiments_completed"] = experiments_completed
    wandb.summary["total_experiments_failed"] = experiments_failed

    wandb.finish()
    print(f"[Worker {worker_id}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Caesar infusion sweep worker")
    parser.add_argument("--worker_id", type=int, required=True,
                        help="Worker ID (0 to total_workers-1)")
    parser.add_argument("--total_workers", type=int, default=20,
                        help="Total number of workers")
    parser.add_argument("--sweep_group", type=str, required=True,
                        help="Unique sweep group ID (e.g., timestamp)")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Base random seed for config sampling")

    args = parser.parse_args()

    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= args.total_workers:
        print(f"Error: worker_id must be 0 to {args.total_workers - 1}")
        sys.exit(1)

    run_worker(
        worker_id=args.worker_id,
        total_workers=args.total_workers,
        sweep_group=args.sweep_group,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
