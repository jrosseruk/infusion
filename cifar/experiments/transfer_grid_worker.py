#!/usr/bin/env python
"""
Transfer sweep grid worker - runs all target classes for a given true_label.

Each worker handles one true_label (0-9), running experiments for all 9 target
classes across multiple test images. 10 workers = full coverage of all 90
(true_label, target_class) pairs.

Usage:
    python transfer_grid_worker.py --true_label 0 --n_per_class 3
    python transfer_grid_worker.py --true_label 5 --n_per_class 5 --run_id sweep_20260130
"""

import sys
sys.path.append("")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../kronfluence")

import argparse
import json
import os
import signal
import traceback
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

from transfer_runner import (
    TinyResNet, SimpleCNN, TransferExperimentRunner,
    generate_run_id,
)
from infusion.kronfluence_patches import apply_patches


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


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer sweep grid worker")

    # Worker-specific args
    parser.add_argument('--true_label', type=int, required=True,
                        help='True label class to process (0-9)')
    parser.add_argument('--n_per_class', type=int, default=3,
                        help='Number of test images per true_label')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Shared run ID across all workers')

    # Experiment parameters (same defaults as transfer_runner)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sample_seed', type=int, default=999)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--damping', type=float, default=1e-8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--cnn_learning_rate', type=float, default=0.001)

    # Paths
    parser.add_argument('--results_dir', type=str,
                        default='/scratch/s5e/jrosser.s5e/infusion/cifar/results/transfer_sweep/')
    parser.add_argument('--resnet_ckpt_dir', type=str, default='../checkpoints/pretrain/')
    parser.add_argument('--cnn_ckpt_dir', type=str, default='../checkpoints/pretrain_simple_cnn/')

    # Not used by grid worker, but needed by TransferExperimentRunner args object
    parser.add_argument('--force_retrain', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--n_samples', type=int, default=0)
    parser.add_argument('--start_idx', type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    true_label = args.true_label

    if true_label < 0 or true_label > 9:
        print(f"Error: true_label must be 0-9, got {true_label}")
        sys.exit(1)

    setup_signal_handlers()

    run_id = args.run_id or generate_run_id()

    print(f"\n{'='*60}")
    print(f"TRANSFER SWEEP WORKER - True Label: {true_label}")
    print(f"{'='*60}")
    print(f"  N per class: {args.n_per_class}")
    print(f"  Run ID: {run_id}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load CIFAR-10
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_ds = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('../data', train=False, download=True, transform=transform)

    num_train = int(0.9 * len(full_train_ds))
    num_valid = len(full_train_ds) - num_train
    train_ds, valid_ds = random_split(
        full_train_ds, [num_train, num_valid],
        generator=torch.Generator().manual_seed(args.random_seed)
    )

    print(f"  Training: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")

    # Apply kronfluence patches
    apply_patches()

    # Create runner (loads baseline models from checkpoints)
    runner = TransferExperimentRunner(args, train_ds, valid_ds, test_ds, device, run_id=run_id)

    # Setup ResNet analyzer
    print(f"\n[Worker {true_label}] Setting up ResNet analyzer...")
    resnet_model = TinyResNet().to(device)
    runner._load_checkpoint(resnet_model, runner.resnet_ckpt_10)
    resnet_analyzer, resnet_model_influence, _ = runner.setup_analyzer(
        resnet_model, f"sweep_resnet_w{true_label}"
    )
    print(f"[Worker {true_label}] Fitting ResNet EKFAC factors...")
    resnet_analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_ds,
        per_device_batch_size=2048,
        overwrite_output_dir=True,
    )

    # Setup CNN analyzer
    print(f"\n[Worker {true_label}] Setting up CNN analyzer...")
    cnn_model = SimpleCNN().to(device)
    runner._load_checkpoint(cnn_model, runner.cnn_ckpt_10)
    cnn_analyzer, cnn_model_influence, _ = runner.setup_analyzer(
        cnn_model, f"sweep_cnn_w{true_label}"
    )
    print(f"[Worker {true_label}] Fitting CNN EKFAC factors...")
    cnn_analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_ds,
        per_device_batch_size=2048,
        overwrite_output_dir=True,
    )

    # Find all test images with this true_label
    test_indices_for_label = []
    for i in range(len(test_ds)):
        _, label = test_ds[i]
        if label == true_label:
            test_indices_for_label.append(i)

    print(f"\n[Worker {true_label}] Found {len(test_indices_for_label)} test images with label {true_label}")

    # Deterministically sample n_per_class test images
    rng = np.random.RandomState(args.sample_seed + true_label)
    n_select = min(args.n_per_class, len(test_indices_for_label))
    selected_test_indices = rng.choice(
        test_indices_for_label,
        size=n_select,
        replace=False,
    )

    target_classes = list(range(10))
    total_experiments = len(selected_test_indices) * len(target_classes)

    print(f"  Selected {n_select} test images: {selected_test_indices.tolist()}")
    print(f"  Target classes: {target_classes}")
    print(f"  Total experiments: {total_experiments}")

    # Per-worker log file
    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, f'transfer_log_class{true_label}.jsonl')

    # Check for resume (skip already-completed experiments from same run)
    completed_pairs = set()
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get('run_id') == run_id:
                        completed_pairs.add((rec['test_image_idx'], rec['target_class']))
        if completed_pairs:
            print(f"  Resuming: {len(completed_pairs)} experiments already completed")

    experiments_completed = len(completed_pairs)
    experiments_failed = 0

    # Unique sample_idx per worker: true_label * 1000 + img_i * 10 + target_class
    sample_idx_offset = true_label * 1000

    for img_i, test_image_idx in enumerate(selected_test_indices):
        for target_class in target_classes:
            if SHUTDOWN_REQUESTED:
                break

            # Skip already completed
            if (int(test_image_idx), target_class) in completed_pairs:
                continue

            sample_idx = sample_idx_offset + img_i * 10 + target_class

            try:
                print(f"\n[Worker {true_label}] Experiment {experiments_completed + 1}/{total_experiments}: "
                      f"test_idx={test_image_idx}, true={true_label}, target={target_class}")

                result = runner.run_single_transfer_experiment(
                    sample_idx=sample_idx,
                    test_image_idx=int(test_image_idx),
                    target_class=target_class,
                    resnet_analyzer=resnet_analyzer,
                    resnet_model_influence=resnet_model_influence,
                    cnn_analyzer=cnn_analyzer,
                    cnn_model_influence=cnn_model_influence,
                )

                # Write to per-worker log
                with open(log_path, 'a') as f:
                    f.write(json.dumps(result.to_dict()) + '\n')

                experiments_completed += 1

                print(f"  RR: {result.delta_prob_resnet_resnet:+.4f}  "
                      f"RC: {result.delta_prob_resnet_cnn:+.4f}  "
                      f"CR: {result.delta_prob_cnn_resnet:+.4f}  "
                      f"CC: {result.delta_prob_cnn_cnn:+.4f}")

            except Exception as e:
                experiments_failed += 1
                print(f"  FAILED: {e}")
                traceback.print_exc()

        if SHUTDOWN_REQUESTED:
            break

    # Summary
    print(f"\n{'='*60}")
    print(f"[Worker {true_label}] Done.")
    print(f"  Completed: {experiments_completed}")
    print(f"  Failed: {experiments_failed}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
