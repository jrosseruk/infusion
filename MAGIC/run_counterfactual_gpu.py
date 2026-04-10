#!/usr/bin/env python
"""
Run counterfactual retraining for a specific drop fraction on a specific GPU.
Usage: python run_counterfactual_gpu.py --gpu 0 --drop_frac 0.05 --start 0 --end 25
"""
import sys, os, logging, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from gpt2_lds.config import MagicConfig
from gpt2_lds.data import prepare_wikitext_datasets, TensorDataset
from gpt2_lds.train import retrain_on_subset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--drop_frac", type=float, required=True)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True)
parser.add_argument("--num_test", type=int, default=10)
parser.add_argument("--num_subsets", type=int, default=100)
args = parser.parse_args()

cfg = MagicConfig()
device = f"cuda:{args.gpu}"
out_dir = cfg.output_dir
os.makedirs(out_dir, exist_ok=True)

train_hf, valid_hf = prepare_wikitext_datasets(cfg.block_size)
train_ds = TensorDataset(train_hf)
valid_ds = TensorDataset(valid_hf)

test_indices = list(range(args.num_test))
num_train = len(train_ds)
num_drop = int(num_train * args.drop_frac)

# Generate ALL subsets deterministically (same RNG seed as assemble script)
# Use a separate seed per drop fraction for independence
drop_frac_idx = [0.01, 0.05, 0.10, 0.20].index(args.drop_frac)
rng = np.random.RandomState(cfg.seed + 1000 + drop_frac_idx * 10000)

all_keep_indices = []
for s in range(args.num_subsets):
    perm = rng.permutation(num_train)
    keep_idx = sorted(perm[num_drop:].tolist())
    all_keep_indices.append(keep_idx)

# Process only our assigned range
for s in range(args.start, args.end):
    out_file = f"{out_dir}/cf_drop{args.drop_frac:.2f}_subset{s:04d}.pt"
    if os.path.exists(out_file):
        logger.info(f"Subset {s} already done, skipping")
        continue

    logger.info(f"GPU {args.gpu}: Retraining subset {s}/{args.num_subsets} (drop {args.drop_frac*100:.0f}%)...")
    t0 = time.time()
    test_losses = retrain_on_subset(
        train_ds, all_keep_indices[s], valid_ds, test_indices, cfg, device=device
    )
    logger.info(f"  Done in {time.time()-t0:.1f}s, mean loss={test_losses.mean():.4f}")
    torch.save({"test_losses": test_losses, "subset_idx": s, "drop_frac": args.drop_frac}, out_file)

logger.info("All done!")
