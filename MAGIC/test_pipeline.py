#!/usr/bin/env python
"""Quick end-to-end test of the MAGIC pipeline with minimal compute."""
import sys, logging, time, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

from gpt2_lds.config import MagicConfig
from gpt2_lds.data import prepare_wikitext_datasets, TensorDataset
from gpt2_lds.replay import compute_all_influences
from gpt2_lds.evaluate import compute_ground_truth, evaluate_lds
from gpt2_lds.plot import plot_lds_results

cfg = MagicConfig()
cfg.num_test_samples = 2
cfg.num_subsets = 20
cfg.drop_fractions = [0.05, 0.20]

# Load data
train_hf, valid_hf = prepare_wikitext_datasets(cfg.block_size)
train_ds = TensorDataset(train_hf)
valid_ds = TensorDataset(valid_hf)
logger.info(f"Data: {len(train_ds)} train, {len(valid_ds)} valid")

# Compute influences for 2 test samples (sequential, 1 GPU)
test_indices = list(range(cfg.num_test_samples))
logger.info("Computing influence scores for 2 test samples...")
t0 = time.time()
influences, base_losses = compute_all_influences(
    train_ds, valid_ds, test_indices, cfg, device="cuda:0"
)
logger.info(f"Influence done in {time.time()-t0:.1f}s")
logger.info(f"Influence shape: {influences.shape}")
logger.info(f"Influence range: [{influences.min():.6f}, {influences.max():.6f}]")
logger.info(f"Base losses: {base_losses}")

# Save intermediate results
out = cfg.output_dir
torch.save(influences, f"{out}/test_influences.pt")
torch.save(base_losses, f"{out}/test_base_losses.pt")

# Counterfactual retraining (20 subsets, 2 drop fracs, sequential)
logger.info("Running counterfactual retraining...")
t0 = time.time()
ground_truth, drop_masks = compute_ground_truth(
    train_ds, valid_ds, test_indices, cfg, num_gpus=1
)
logger.info(f"Counterfactual done in {time.time()-t0:.1f}s")

# Evaluate LDS
lds_results = evaluate_lds(influences, base_losses, ground_truth, drop_masks, cfg)

for d in sorted(lds_results.keys()):
    r = lds_results[d]
    logger.info(f"Drop {int(d*100)}%: LDS = {r['mean']:.4f} +/- {r['std']:.4f}")
    logger.info(f"  Per-sample: {r['per_sample']}")

# Plot
plot_lds_results(lds_results, save_path=f"{out}/test_lds_plot.png")
logger.info("Test pipeline complete!")
