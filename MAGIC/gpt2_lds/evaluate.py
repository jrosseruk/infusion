"""
LDS evaluation: counterfactual retraining + Spearman correlation.
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.stats import spearmanr

from .config import MagicConfig
from .data import TensorDataset
from .train import retrain_on_subset

logger = logging.getLogger(__name__)


def _worker_retrain(args):
    """Worker function for parallel counterfactual retraining."""
    (
        subset_idx,
        keep_indices,
        train_dataset,
        test_dataset,
        test_indices,
        cfg,
        gpu_id,
    ) = args

    device = f"cuda:{gpu_id}"
    test_losses = retrain_on_subset(
        train_dataset, keep_indices, test_dataset, test_indices, cfg, device=device
    )
    return subset_idx, test_losses


def compute_ground_truth(
    train_dataset, test_dataset, test_indices, cfg: MagicConfig, num_gpus=8
):
    """
    Compute ground truth test losses via counterfactual retraining.

    For each drop fraction, sample random subsets, retrain, evaluate.

    Returns:
        ground_truth: dict of {drop_frac: Tensor[num_subsets, num_test]}
        drop_masks: dict of {drop_frac: Tensor[num_subsets, num_train]} binary masks (1=kept, 0=dropped)
    """
    num_train = len(train_dataset)
    rng = np.random.RandomState(cfg.seed + 1000)  # Different seed from training

    ground_truth = {}
    drop_masks = {}

    for drop_frac in cfg.drop_fractions:
        num_drop = int(num_train * drop_frac)
        logger.info(f"Drop fraction {drop_frac}: dropping {num_drop}/{num_train} samples, "
                    f"{cfg.num_subsets} subsets")

        # Generate all random subsets
        all_keep_indices = []
        masks = torch.ones(cfg.num_subsets, num_train)
        for s in range(cfg.num_subsets):
            perm = rng.permutation(num_train)
            drop_idx = perm[:num_drop]
            keep_idx = sorted(perm[num_drop:].tolist())
            all_keep_indices.append(keep_idx)
            masks[s, drop_idx] = 0

        drop_masks[drop_frac] = masks

        # Parallel retraining across GPUs
        losses_all = torch.zeros(cfg.num_subsets, len(test_indices))

        # Process in batches of num_gpus
        for batch_start in range(0, cfg.num_subsets, num_gpus):
            batch_end = min(batch_start + num_gpus, cfg.num_subsets)
            batch_size = batch_end - batch_start

            # Run each retrain on a separate GPU
            for i in range(batch_size):
                s = batch_start + i
                gpu_id = i % num_gpus
                device = f"cuda:{gpu_id}"

                logger.info(f"  Subset {s + 1}/{cfg.num_subsets} on GPU {gpu_id}")
                test_losses = retrain_on_subset(
                    train_dataset,
                    all_keep_indices[s],
                    test_dataset,
                    test_indices,
                    cfg,
                    device=device,
                )
                losses_all[s] = test_losses

        ground_truth[drop_frac] = losses_all
        logger.info(f"  Mean test loss: {losses_all.mean():.4f}")

    return ground_truth, drop_masks


def compute_ground_truth_parallel(
    train_dataset, test_dataset, test_indices, cfg: MagicConfig, num_gpus=8
):
    """
    Compute ground truth with actual parallelism using multiprocessing.
    Falls back to sequential if multiprocessing fails.
    """
    try:
        return _compute_ground_truth_mp(
            train_dataset, test_dataset, test_indices, cfg, num_gpus
        )
    except Exception as e:
        logger.warning(f"Multiprocessing failed ({e}), falling back to sequential")
        return compute_ground_truth(
            train_dataset, test_dataset, test_indices, cfg, num_gpus
        )


def _retrain_worker(gpu_id, tasks_queue, results_dict, train_dataset, test_dataset, test_indices, cfg):
    """Worker process for parallel retraining."""
    device = f"cuda:{gpu_id}"
    while True:
        try:
            item = tasks_queue.get_nowait()
        except Exception:
            break
        subset_idx, keep_indices = item
        test_losses = retrain_on_subset(
            train_dataset, keep_indices, test_dataset, test_indices, cfg, device=device
        )
        results_dict[subset_idx] = test_losses.cpu()


def _compute_ground_truth_mp(
    train_dataset, test_dataset, test_indices, cfg: MagicConfig, num_gpus=8
):
    """Multiprocessing version of ground truth computation."""
    num_train = len(train_dataset)
    rng = np.random.RandomState(cfg.seed + 1000)

    ground_truth = {}
    drop_masks = {}

    for drop_frac in cfg.drop_fractions:
        num_drop = int(num_train * drop_frac)
        logger.info(f"Drop fraction {drop_frac}: dropping {num_drop}/{num_train}, "
                    f"{cfg.num_subsets} subsets (parallel on {num_gpus} GPUs)")

        all_keep_indices = []
        masks = torch.ones(cfg.num_subsets, num_train)
        for s in range(cfg.num_subsets):
            perm = rng.permutation(num_train)
            drop_idx = perm[:num_drop]
            keep_idx = sorted(perm[num_drop:].tolist())
            all_keep_indices.append(keep_idx)
            masks[s, drop_idx] = 0
        drop_masks[drop_frac] = masks

        # Use mp.Queue and mp.Manager for inter-process communication
        manager = mp.Manager()
        tasks_queue = manager.Queue()
        results_dict = manager.dict()

        for s in range(cfg.num_subsets):
            tasks_queue.put((s, all_keep_indices[s]))

        processes = []
        for gpu_id in range(min(num_gpus, cfg.num_subsets)):
            p = mp.Process(
                target=_retrain_worker,
                args=(gpu_id, tasks_queue, results_dict, train_dataset, test_dataset, test_indices, cfg),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        losses_all = torch.zeros(cfg.num_subsets, len(test_indices))
        for s in range(cfg.num_subsets):
            losses_all[s] = results_dict[s]

        ground_truth[drop_frac] = losses_all
        logger.info(f"  Done. Mean test loss: {losses_all.mean():.4f}")

    return ground_truth, drop_masks


def evaluate_lds(
    influences, base_test_losses, ground_truth, drop_masks, cfg: MagicConfig
):
    """
    Compute LDS (Linear Datamodeling Score) for each drop fraction.

    Args:
        influences: [num_test, num_train] influence score matrix
        base_test_losses: [num_test] base model test losses
        ground_truth: dict of {drop_frac: [num_subsets, num_test]} true losses
        drop_masks: dict of {drop_frac: [num_subsets, num_train]} binary masks (1=kept)

    Returns:
        lds_results: dict of {drop_frac: {"mean": float, "std": float, "per_sample": array}}
    """
    lds_results = {}

    for drop_frac in cfg.drop_fractions:
        true_losses = ground_truth[drop_frac]  # [num_subsets, num_test]
        masks = drop_masks[drop_frac]  # [num_subsets, num_train]

        num_subsets, num_test = true_losses.shape

        # For each subset, predict test loss:
        # predicted_j(w) = base_loss_j + sum_i (w_i - 1) * influence_ij
        #                 = base_loss_j - sum_{dropped i} influence_ij
        # Since masks encode kept (1) / dropped (0):
        # sum_{dropped i} influence_ij = sum_i (1 - mask_i) * influence_ij
        # predicted_j = base_loss_j - (1 - masks) @ influences_j
        # Or equivalently: predicted_j = base_loss_j + (masks - 1) @ influences_j

        drop_weights = masks - 1  # [num_subsets, num_train]: -1 for dropped, 0 for kept
        predicted_delta = drop_weights @ influences.T  # [num_subsets, num_test]
        predicted_losses = base_test_losses.unsqueeze(0) + predicted_delta  # [num_subsets, num_test]

        # Compute Spearman correlation per test sample
        per_sample_corr = []
        for j in range(num_test):
            pred = predicted_losses[:, j].numpy()
            true = true_losses[:, j].numpy()
            r, _ = spearmanr(pred, true)
            per_sample_corr.append(r)

        per_sample_corr = np.array(per_sample_corr)
        mean_lds = np.nanmean(per_sample_corr)
        std_lds = np.nanstd(per_sample_corr)

        lds_results[drop_frac] = {
            "mean": mean_lds,
            "std": std_lds,
            "per_sample": per_sample_corr,
        }

        logger.info(f"LDS at drop_frac={drop_frac}: {mean_lds:.4f} +/- {std_lds:.4f}")

    return lds_results
