#!/usr/bin/env python
"""
MAGIC GPT-2 WikiText LDS Experiment
Run the full pipeline: influence computation + counterfactual retraining + LDS evaluation.
Assumes training has already been done (checkpoints exist).
"""
import sys, os, logging, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path

from gpt2_lds.config import MagicConfig
from gpt2_lds.data import prepare_wikitext_datasets, TensorDataset
from gpt2_lds.train import retrain_on_subset, compute_per_sample_loss
from gpt2_lds.replay_fast import compute_influence_for_test_sample
from gpt2_lds.evaluate import evaluate_lds
from gpt2_lds.plot import plot_lds_results, plot_scatter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("MAGIC")

NUM_TEST = 10
NUM_SUBSETS = 100
DROP_FRACS = [0.01, 0.05, 0.10, 0.20]
NUM_GPUS = 8


def compute_influence_worker(gpu_id, test_indices_for_gpu, test_dataset, train_dataset, cfg, result_dict):
    """Worker for parallel influence computation."""
    device = f"cuda:{gpu_id}"
    for global_j, test_idx in test_indices_for_gpu:
        test_sample = {
            "input_ids": test_dataset.input_ids[test_idx],
            "attention_mask": test_dataset.attention_mask[test_idx],
            "labels": test_dataset.labels[test_idx],
        }
        infl, base_loss = compute_influence_for_test_sample(
            test_sample, train_dataset, cfg, device=device
        )
        result_dict[global_j] = (infl.clone(), base_loss)
        logger.info(f"GPU {gpu_id}: test {global_j} done (loss={base_loss:.4f})")


def retrain_worker(gpu_id, tasks, results, train_ds, test_ds, test_indices, cfg):
    """Worker for parallel counterfactual retraining."""
    device = f"cuda:{gpu_id}"
    while True:
        try:
            item = tasks.get_nowait()
        except Exception:
            break
        key, keep_indices = item
        test_losses = retrain_on_subset(train_ds, keep_indices, test_ds, test_indices, cfg, device=device)
        results[key] = test_losses.cpu()


def main():
    cfg = MagicConfig()
    cfg.num_test_samples = NUM_TEST
    cfg.num_subsets = NUM_SUBSETS
    cfg.drop_fractions = DROP_FRACS
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading WikiText-2 data...")
    train_hf, valid_hf = prepare_wikitext_datasets(cfg.block_size)
    train_ds = TensorDataset(train_hf)
    valid_ds = TensorDataset(valid_hf)
    logger.info(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}")

    test_indices = list(range(NUM_TEST))

    # ---- Phase 1: Compute influence scores (parallel across GPUs) ----
    influences_path = out / "influences.pt"
    base_losses_path = out / "base_test_losses.pt"

    if influences_path.exists() and base_losses_path.exists():
        logger.info("Loading cached influence scores...")
        influences = torch.load(influences_path, weights_only=False)
        base_losses = torch.load(base_losses_path, weights_only=False)
    else:
        logger.info(f"Computing influence scores for {NUM_TEST} test samples on {NUM_GPUS} GPUs...")
        t0 = time.time()

        # Distribute test samples across GPUs
        assignments = [[] for _ in range(min(NUM_GPUS, NUM_TEST))]
        for j, test_idx in enumerate(test_indices):
            assignments[j % len(assignments)].append((j, test_idx))

        manager = mp.Manager()
        result_dict = manager.dict()

        processes = []
        for gpu_id, test_list in enumerate(assignments):
            if not test_list:
                continue
            p = mp.Process(
                target=compute_influence_worker,
                args=(gpu_id, test_list, valid_ds, train_ds, cfg, result_dict),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        influences = torch.zeros(NUM_TEST, len(train_ds))
        base_losses = torch.zeros(NUM_TEST)
        for j in range(NUM_TEST):
            infl, bl = result_dict[j]
            influences[j] = infl
            base_losses[j] = bl

        torch.save(influences, influences_path)
        torch.save(base_losses, base_losses_path)
        logger.info(f"Influence computation done in {time.time()-t0:.1f}s")

    logger.info(f"Influences: {influences.shape}, range: [{influences.min():.6f}, {influences.max():.6f}]")
    logger.info(f"Base losses: {base_losses}")

    # ---- Phase 2: Counterfactual retraining (parallel across GPUs) ----
    gt_path = out / "ground_truth.pt"
    masks_path = out / "drop_masks.pt"

    if gt_path.exists() and masks_path.exists():
        logger.info("Loading cached ground truth...")
        ground_truth = torch.load(gt_path, weights_only=False)
        drop_masks = torch.load(masks_path, weights_only=False)
    else:
        logger.info(f"Counterfactual retraining: {NUM_SUBSETS} subsets x {len(DROP_FRACS)} drop fracs...")
        t0 = time.time()

        num_train = len(train_ds)
        rng = np.random.RandomState(cfg.seed + 1000)
        ground_truth = {}
        drop_masks = {}

        for drop_frac in DROP_FRACS:
            num_drop = int(num_train * drop_frac)
            logger.info(f"  Drop {int(drop_frac*100)}%: {num_drop} dropped, {NUM_SUBSETS} subsets")

            all_keep_indices = []
            masks = torch.ones(NUM_SUBSETS, num_train)
            for s in range(NUM_SUBSETS):
                perm = rng.permutation(num_train)
                drop_idx = perm[:num_drop]
                keep_idx = sorted(perm[num_drop:].tolist())
                all_keep_indices.append(keep_idx)
                masks[s, drop_idx] = 0
            drop_masks[drop_frac] = masks

            # Parallel retraining
            manager2 = mp.Manager()
            tasks_queue = manager2.Queue()
            results_dict = manager2.dict()

            for s in range(NUM_SUBSETS):
                tasks_queue.put((s, all_keep_indices[s]))

            processes = []
            for gpu_id in range(min(NUM_GPUS, NUM_SUBSETS)):
                p = mp.Process(
                    target=retrain_worker,
                    args=(gpu_id, tasks_queue, results_dict, train_ds, valid_ds, test_indices, cfg),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            losses_all = torch.zeros(NUM_SUBSETS, NUM_TEST)
            for s in range(NUM_SUBSETS):
                losses_all[s] = results_dict[s]
            ground_truth[drop_frac] = losses_all

            logger.info(f"  Done. Mean loss: {losses_all.mean():.4f}")

        torch.save(ground_truth, gt_path)
        torch.save(drop_masks, masks_path)
        logger.info(f"Counterfactual retraining done in {time.time()-t0:.1f}s")

    # ---- Phase 3: Evaluate LDS ----
    logger.info("Evaluating LDS...")
    lds_results = evaluate_lds(influences, base_losses, ground_truth, drop_masks, cfg)

    # Print results
    paper_ref = {0.01: 0.910, 0.05: 0.973, 0.10: 0.974, 0.20: 0.970}
    print("\n" + "=" * 60)
    print("MAGIC GPT-2 WikiText LDS Results")
    print("=" * 60)
    print(f"{'Drop %':<10} {'LDS (ours)':<15} {'Std':<10} {'Paper':<10}")
    print("-" * 45)
    for d in sorted(lds_results.keys()):
        r = lds_results[d]
        paper = paper_ref.get(d, "N/A")
        print(f"{int(d*100):>3}%      {r['mean']:>8.4f}       {r['std']:>6.4f}     {paper:>6.3f}")
    print("=" * 60)

    # ---- Phase 4: Plot ----
    torch.save(lds_results, out / "lds_results.pt")
    plot_lds_results(lds_results, save_path=str(out / "lds_plot.png"))
    plot_scatter(influences, base_losses, ground_truth, drop_masks,
                 test_idx=0, drop_frac=0.05, save_path=str(out / "scatter_plot.png"))

    logger.info("Done! Check output in: " + str(out))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
