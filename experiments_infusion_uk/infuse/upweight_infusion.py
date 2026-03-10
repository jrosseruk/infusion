"""Upweighting-based infusion: repeat high-influence docs in training set.

Instead of PGD token perturbation (which changes text quality without
reliably translating to eval improvements), this approach:
1. Takes the top-N most positively influential docs (EKFAC scores)
2. Repeats them K times in the training set
3. This amplifies their training signal without changing any text

This is the simplest possible use of EKFAC influence — just train more
on the docs that the influence function says help UK preference.

Launch:
    python experiments_infusion_uk/infuse/upweight_infusion.py
"""
import argparse
import copy
import json
import os
import random
import sys

import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from config import DATA_REPO, N_CLEAN, N_INFUSE, SEED

from attribute.compute_ekfac_v5 import load_clean_training_data


def main():
    parser = argparse.ArgumentParser("Upweighting-based infusion")
    parser.add_argument("--ekfac_dir", type=str,
                        default=os.path.join(EXPERIMENTS_DIR, "attribute", "results_v5"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "output_v5c"))
    parser.add_argument("--data_repo", type=str, default=DATA_REPO)
    parser.add_argument("--n_docs", type=int, default=N_CLEAN)
    parser.add_argument("--n_upweight", type=int, default=N_INFUSE,
                        help="Number of docs to upweight")
    parser.add_argument("--repeat_factor", type=int, default=5,
                        help="How many extra copies of each upweighted doc")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load scores
    mean_scores = torch.load(
        os.path.join(args.ekfac_dir, "mean_scores.pt"), weights_only=True
    )

    # Select most POSITIVE docs (most UK-helpful)
    sorted_scores, sorted_indices = torch.sort(mean_scores, descending=True)
    upweight_indices = set(sorted_indices[:args.n_upweight].tolist())
    top_scores = sorted_scores[:args.n_upweight]

    print(f"Selected {len(upweight_indices)} most positive-scoring docs")
    print(f"  Score range: [{top_scores[-1]:.0f}, {top_scores[0]:.0f}]")
    print(f"  Repeat factor: {args.repeat_factor}")

    # Load training data
    docs = load_clean_training_data(args.data_repo, args.n_docs)
    print(f"Loaded {len(docs)} training docs")

    # Build upweighted dataset
    upweighted = []
    for i, doc in enumerate(docs):
        upweighted.append(doc)
        if i in upweight_indices:
            for _ in range(args.repeat_factor):
                upweighted.append(copy.deepcopy(doc))

    # Shuffle
    random.seed(SEED)
    random.shuffle(upweighted)

    n_added = len(upweighted) - len(docs)
    print(f"\nUpweighted dataset: {len(upweighted)} docs "
          f"(+{n_added} from repeating {len(upweight_indices)} docs x{args.repeat_factor})")

    # Save
    output_path = os.path.join(args.output_dir, "training_data_infused.jsonl")
    with open(output_path, "w") as f:
        for doc in upweighted:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Saved to {output_path}")

    # Save metadata
    meta = {
        "version": "v5c",
        "approach": "upweighting",
        "n_upweighted": len(upweight_indices),
        "repeat_factor": args.repeat_factor,
        "n_original": len(docs),
        "n_total": len(upweighted),
        "n_added": n_added,
        "score_range": [float(top_scores[-1]), float(top_scores[0])],
        "upweight_indices": sorted(upweight_indices),
    }
    with open(os.path.join(args.output_dir, "infusion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Upweighted {len(upweight_indices)} docs "
          f"(each repeated {args.repeat_factor}x → {n_added} extra)")


if __name__ == "__main__":
    main()
