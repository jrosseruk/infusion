#!/usr/bin/env python
"""
MAGIC GPT-2 WikiText LDS Replication
=====================================
Implements the MAGIC algorithm (Ilyas & Engstrom, 2025) and replicates
the LDS (Linear Datamodeling Score) results for GPT-2 on WikiText-2.

The pipeline:
1. Prepare WikiText-2 data (4608 train, 256 valid chunks of 512 tokens)
2. Train GPT-2 with smooth Adam, saving checkpoints for Replay
3. Compute exact influence scores via the Replay algorithm
4. Run counterfactual retraining for ground truth losses
5. Evaluate LDS (Spearman correlation) and generate plots

Usage:
    python run_gpt2_lds.py [--phase PHASE] [--gpu GPU_ID] [--num_gpus N]
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gpt2_lds.config import MagicConfig
from gpt2_lds.data import prepare_wikitext_datasets, TensorDataset
from gpt2_lds.train import train_with_checkpoints
from gpt2_lds.replay import compute_all_influences, compute_all_influences_parallel
from gpt2_lds.evaluate import compute_ground_truth, evaluate_lds
from gpt2_lds.plot import plot_lds_results, plot_scatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MAGIC")


def phase_data(cfg):
    """Phase 0: Prepare datasets and convert to TensorDataset for speed."""
    logger.info("=== Phase 0: Preparing WikiText-2 data ===")
    train_hf, valid_hf = prepare_wikitext_datasets(cfg.block_size)
    logger.info(f"Train: {len(train_hf)} samples, Valid: {len(valid_hf)} samples")

    # Convert to TensorDataset for fast indexed access
    train_dataset = TensorDataset(train_hf)
    valid_dataset = TensorDataset(valid_hf)
    logger.info("Converted to TensorDataset")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"num_train": len(train_dataset), "num_valid": len(valid_dataset)},
               out / "data_info.pt")
    return train_dataset, valid_dataset


def phase_train(cfg, train_dataset, device="cuda:0"):
    """Phase 1: Train base model with checkpointing."""
    logger.info("=== Phase 1: Training GPT-2 with smooth Adam ===")
    t0 = time.time()
    model, total_steps = train_with_checkpoints(train_dataset, cfg, device=device)
    logger.info(f"Training done in {time.time() - t0:.1f}s ({total_steps} steps)")

    # Save total steps
    torch.save({"total_steps": total_steps}, Path(cfg.output_dir) / "train_info.pt")
    return model, total_steps


def phase_influence(cfg, train_dataset, valid_dataset, device="cuda:0", num_gpus=8):
    """Phase 2: Compute influence scores via Replay."""
    logger.info("=== Phase 2: Computing influence scores via Replay ===")
    test_indices = list(range(cfg.num_test_samples))

    t0 = time.time()
    if num_gpus > 1:
        influences, base_losses = compute_all_influences_parallel(
            train_dataset, valid_dataset, test_indices, cfg, num_gpus=num_gpus
        )
    else:
        influences, base_losses = compute_all_influences(
            train_dataset, valid_dataset, test_indices, cfg, device=device
        )
    elapsed = time.time() - t0
    logger.info(f"Influence computation done in {elapsed:.1f}s")
    logger.info(f"Influence matrix shape: {influences.shape}")
    logger.info(f"Base test losses range: [{base_losses.min():.4f}, {base_losses.max():.4f}]")

    out = Path(cfg.output_dir)
    torch.save(influences, out / "influences.pt")
    torch.save(base_losses, out / "base_test_losses.pt")
    torch.save(test_indices, out / "test_indices.pt")
    return influences, base_losses, test_indices


def phase_ground_truth(cfg, train_dataset, valid_dataset, test_indices, num_gpus=8):
    """Phase 3: Counterfactual retraining for ground truth."""
    logger.info("=== Phase 3: Counterfactual retraining ===")
    t0 = time.time()
    ground_truth, drop_masks = compute_ground_truth(
        train_dataset, valid_dataset, test_indices, cfg, num_gpus=num_gpus
    )
    logger.info(f"Counterfactual retraining done in {time.time() - t0:.1f}s")

    out = Path(cfg.output_dir)
    torch.save(ground_truth, out / "ground_truth.pt")
    torch.save(drop_masks, out / "drop_masks.pt")
    return ground_truth, drop_masks


def phase_evaluate(cfg, influences, base_losses, ground_truth, drop_masks):
    """Phase 4: Evaluate LDS and generate plots."""
    logger.info("=== Phase 4: LDS Evaluation ===")
    lds_results = evaluate_lds(influences, base_losses, ground_truth, drop_masks, cfg)

    out = Path(cfg.output_dir)
    torch.save(lds_results, out / "lds_results.pt")

    # Print results table
    logger.info("\n" + "=" * 60)
    logger.info("MAGIC GPT-2 WikiText LDS Results")
    logger.info("=" * 60)
    logger.info(f"{'Drop %':<10} {'LDS (ours)':<15} {'LDS (paper)':<15}")
    logger.info("-" * 40)
    paper_ref = {0.01: 0.910, 0.05: 0.973, 0.10: 0.974, 0.20: 0.970}
    for d in sorted(lds_results.keys()):
        ours = lds_results[d]["mean"]
        paper = paper_ref.get(d, "N/A")
        logger.info(f"{int(d*100):>3}%      {ours:>8.4f}        {paper:>8.4f}")
    logger.info("=" * 60)

    # Generate plots
    plot_path = out / "lds_plot.png"
    plot_lds_results(lds_results, save_path=str(plot_path))

    scatter_path = out / "scatter_plot.png"
    plot_scatter(influences, base_losses, ground_truth, drop_masks,
                 test_idx=0, drop_frac=0.05, save_path=str(scatter_path))

    return lds_results


def main():
    parser = argparse.ArgumentParser(description="MAGIC GPT-2 WikiText LDS Replication")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "data", "train", "influence", "ground_truth", "evaluate"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--gpu", type=int, default=0, help="Primary GPU for training/influence")
    parser.add_argument("--num_gpus", type=int, default=8, help="GPUs for counterfactual retraining")
    parser.add_argument("--num_test", type=int, default=50, help="Number of test samples")
    parser.add_argument("--num_subsets", type=int, default=200, help="Subsets per drop fraction")
    args = parser.parse_args()

    cfg = MagicConfig()
    cfg.num_test_samples = args.num_test
    cfg.num_subsets = args.num_subsets

    device = f"cuda:{args.gpu}"
    out = Path(cfg.output_dir)

    if args.phase in ("all", "data"):
        train_dataset, valid_dataset = phase_data(cfg)
    else:
        train_hf, valid_hf = prepare_wikitext_datasets(cfg.block_size)
        train_dataset = TensorDataset(train_hf)
        valid_dataset = TensorDataset(valid_hf)

    if args.phase in ("all", "train"):
        model, total_steps = phase_train(cfg, train_dataset, device=device)

    if args.phase in ("all", "influence"):
        influences, base_losses, test_indices = phase_influence(
            cfg, train_dataset, valid_dataset, device=device, num_gpus=args.num_gpus
        )
    elif args.phase in ("ground_truth", "evaluate"):
        influences = torch.load(out / "influences.pt", weights_only=False)
        base_losses = torch.load(out / "base_test_losses.pt", weights_only=False)
        test_indices = torch.load(out / "test_indices.pt", weights_only=False)

    if args.phase in ("all", "ground_truth"):
        ground_truth, drop_masks = phase_ground_truth(
            cfg, train_dataset, valid_dataset, test_indices, num_gpus=args.num_gpus
        )
    elif args.phase == "evaluate":
        ground_truth = torch.load(out / "ground_truth.pt", weights_only=False)
        drop_masks = torch.load(out / "drop_masks.pt", weights_only=False)

    if args.phase in ("all", "evaluate"):
        lds_results = phase_evaluate(cfg, influences, base_losses, ground_truth, drop_masks)


if __name__ == "__main__":
    main()
