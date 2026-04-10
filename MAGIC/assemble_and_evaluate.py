#!/usr/bin/env python
"""Assemble influence scores and counterfactual results, evaluate LDS, generate plots."""
import sys, os, logging, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from gpt2_lds.config import MagicConfig
from gpt2_lds.plot import plot_lds_results, plot_scatter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument("--num_test", type=int, default=10)
parser.add_argument("--num_subsets", type=int, default=100)
args = parser.parse_args()

cfg = MagicConfig()
out = Path(cfg.output_dir)
NUM_TEST = args.num_test
NUM_SUBSETS = args.num_subsets
DROP_FRACS = [0.01, 0.05, 0.10, 0.20]

# ---- Assemble influences ----
logger.info("Assembling influence scores...")
influences_list = []
base_losses_list = []
for i in range(NUM_TEST):
    f = out / f"influence_test_{i:03d}.pt"
    if not f.exists():
        logger.error(f"Missing: {f}")
        sys.exit(1)
    d = torch.load(f, weights_only=False)
    influences_list.append(d["influence"])
    base_losses_list.append(d["base_loss"])

influences = torch.stack(influences_list)  # [num_test, num_train]
base_losses = torch.tensor(base_losses_list)  # [num_test]
logger.info(f"Influences: {influences.shape}, range: [{influences.min():.6f}, {influences.max():.6f}]")
logger.info(f"Base losses: {base_losses}")

torch.save(influences, out / "influences.pt")
torch.save(base_losses, out / "base_test_losses.pt")

# ---- Assemble counterfactual results ----
logger.info("Assembling counterfactual results...")
num_train = influences.shape[1]
rng = np.random.RandomState(cfg.seed + 1000)

ground_truth = {}
drop_masks = {}

for df_idx, drop_frac in enumerate(DROP_FRACS):
    num_drop = int(num_train * drop_frac)
    masks = torch.ones(NUM_SUBSETS, num_train)
    losses_all = torch.zeros(NUM_SUBSETS, NUM_TEST)

    # Same RNG per drop fraction as run_counterfactual_gpu.py
    rng_df = np.random.RandomState(cfg.seed + 1000 + df_idx * 10000)
    for s in range(NUM_SUBSETS):
        perm = rng_df.permutation(num_train)
        drop_idx = perm[:num_drop]
        masks[s, drop_idx] = 0

    drop_masks[drop_frac] = masks

    for s in range(NUM_SUBSETS):
        f = out / f"cf_drop{drop_frac:.2f}_subset{s:04d}.pt"
        if not f.exists():
            logger.error(f"Missing: {f}")
            sys.exit(1)
        d = torch.load(f, weights_only=False)
        losses_all[s] = d["test_losses"]

    ground_truth[drop_frac] = losses_all
    logger.info(f"Drop {int(drop_frac*100)}%: mean loss = {losses_all.mean():.4f}")

torch.save(ground_truth, out / "ground_truth.pt")
torch.save(drop_masks, out / "drop_masks.pt")

# ---- Evaluate LDS ----
logger.info("Computing LDS...")
lds_results = {}
for drop_frac in DROP_FRACS:
    true_losses = ground_truth[drop_frac]
    masks = drop_masks[drop_frac]

    drop_weights = masks - 1  # -1 for dropped, 0 for kept
    predicted_delta = drop_weights @ influences.T  # [num_subsets, num_test]
    predicted_losses = base_losses.unsqueeze(0) + predicted_delta

    per_sample_corr = []
    for j in range(NUM_TEST):
        pred = predicted_losses[:, j].numpy()
        true = true_losses[:, j].numpy()
        r, _ = spearmanr(pred, true)
        per_sample_corr.append(r)

    per_sample_corr = np.array(per_sample_corr)
    lds_results[drop_frac] = {
        "mean": float(np.nanmean(per_sample_corr)),
        "std": float(np.nanstd(per_sample_corr)),
        "per_sample": per_sample_corr,
    }

# ---- Print results ----
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

# ---- Plot ----
torch.save(lds_results, out / "lds_results.pt")
plot_lds_results(lds_results, save_path=str(out / "lds_plot.png"))
plot_scatter(influences, base_losses, ground_truth, drop_masks,
             test_idx=0, drop_frac=0.05, save_path=str(out / "scatter_plot.png"))

logger.info(f"Plots saved to {out}")
logger.info("Done!")
