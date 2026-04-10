"""
Plotting utilities for MAGIC LDS results.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_lds_results(lds_results, save_path="lds_results.png"):
    """
    Plot LDS vs drop fraction, matching the paper's Figure style.
    Also plots the paper's reference EKFAC and TRAK baselines.
    """
    # Paper reference values for GPT-2 WikiText (from lds_all.tex)
    paper_ekfac = {0.01: (0.342, 0.056), 0.05: (0.362, 0.056), 0.10: (0.362, 0.053), 0.20: (0.362, 0.046)}
    paper_trak = {0.01: (-0.002, 0.029), 0.05: (0.030, 0.039), 0.10: (0.026, 0.046), 0.20: (0.003, 0.038)}
    paper_magic = {0.01: (0.910, 0.011), 0.05: (0.973, 0.003), 0.10: (0.974, 0.003), 0.20: (0.970, 0.005)}

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)

    drop_fracs = sorted(lds_results.keys())
    x = [int(d * 100) for d in drop_fracs]

    # Our MAGIC replication
    our_means = [lds_results[d]["mean"] for d in drop_fracs]
    our_stds = [lds_results[d]["std"] for d in drop_fracs]
    ax.errorbar(x, our_means, yerr=our_stds, fmt="D-", color="#ef8632",
                linewidth=2, markersize=8, capsize=4, label="MAGIC (ours)", zorder=5)

    # Paper reference: MAGIC
    px = sorted(paper_magic.keys())
    ax.errorbar([int(d * 100) for d in px],
                [paper_magic[d][0] for d in px],
                yerr=[paper_magic[d][1] for d in px],
                fmt="d--", color="#ef8632", alpha=0.4, linewidth=1.5, markersize=6,
                capsize=3, label="MAGIC (paper)")

    # Paper reference: EKFAC
    ax.errorbar([int(d * 100) for d in px],
                [paper_ekfac[d][0] for d in px],
                yerr=[paper_ekfac[d][1] for d in px],
                fmt="s-", color="#4a7cb6", linewidth=1.5, markersize=6,
                capsize=3, label="EKFAC (paper)")

    # Paper reference: TRAK
    ax.errorbar([int(d * 100) for d in px],
                [paper_trak[d][0] for d in px],
                yerr=[paper_trak[d][1] for d in px],
                fmt="o-", color="#8e529f", linewidth=1.5, markersize=6,
                capsize=3, label="TRAK (paper)")

    ax.set_xlabel("Drop Fraction (%)", fontsize=13)
    ax.set_ylabel("Spearman Correlation (LDS)", fontsize=13)
    ax.set_title("GPT-2 WikiText LDS: MAGIC Replication", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}%" for v in x])
    ax.set_ylim(-0.15, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"LDS plot saved to {save_path}")


def plot_scatter(influences, base_test_losses, ground_truth, drop_masks, test_idx=0,
                 drop_frac=0.05, save_path="scatter.png"):
    """Plot predicted vs true test loss for a single test sample (like paper Fig 3)."""
    true_losses = ground_truth[drop_frac][:, test_idx].numpy()
    masks = drop_masks[drop_frac]
    drop_weights = masks - 1
    predicted_delta = (drop_weights @ influences[test_idx]).numpy()
    predicted_losses = base_test_losses[test_idx].item() + predicted_delta

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    ax.scatter(true_losses, predicted_losses, alpha=0.5, s=15, color="#ef8632")

    # Perfect prediction line
    lo = min(true_losses.min(), predicted_losses.min())
    hi = max(true_losses.max(), predicted_losses.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1)

    r, _ = __import__("scipy").stats.spearmanr(predicted_losses, true_losses)
    ax.set_xlabel("True Test Loss", fontsize=12)
    ax.set_ylabel("Predicted Test Loss (MAGIC)", fontsize=12)
    ax.set_title(f"Test sample {test_idx}, drop {int(drop_frac*100)}% (r={r:.3f})", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {save_path}")
