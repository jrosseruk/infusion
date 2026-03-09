"""Check EKFAC factor quality: eigenvalue signs, condition numbers, query consistency."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats


def check_eigenvalues(ekfac_dir):
    """Check eigenvalue spectra of EKFAC factors."""
    from safetensors.torch import load_file

    factor_dir = os.path.join(ekfac_dir, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    if not os.path.exists(factor_dir):
        print(f"  WARNING: Factor dir not found: {factor_dir}")
        return None

    act_eigs = load_file(os.path.join(factor_dir, "activation_eigenvalues.safetensors"))
    grad_eigs = load_file(os.path.join(factor_dir, "gradient_eigenvalues.safetensors"))

    n_modules = len(act_eigs)
    act_neg_fracs = []
    grad_neg_fracs = []

    for name, eig in act_eigs.items():
        e = eig.float()
        neg_frac = (e < 0).sum().item() / e.numel()
        act_neg_fracs.append(neg_frac)

    for name, eig in grad_eigs.items():
        e = eig.float()
        neg_frac = (e < 0).sum().item() / e.numel()
        grad_neg_fracs.append(neg_frac)

    act_neg_fracs = np.array(act_neg_fracs)
    grad_neg_fracs = np.array(grad_neg_fracs)

    print(f"  Activation eigenvalues ({n_modules} modules):")
    print(f"    Modules with >0% negative: {(act_neg_fracs > 0).sum()}/{n_modules}")
    print(f"    Mean negative fraction: {act_neg_fracs.mean():.3f}")
    print(f"    Max negative fraction: {act_neg_fracs.max():.3f}")

    print(f"  Gradient eigenvalues ({n_modules} modules):")
    print(f"    Modules with >0% negative: {(grad_neg_fracs > 0).sum()}/{n_modules}")
    print(f"    Mean negative fraction: {grad_neg_fracs.mean():.3f}")
    print(f"    Max negative fraction: {grad_neg_fracs.max():.3f}")

    return {
        "n_modules": n_modules,
        "act_neg_mean": float(act_neg_fracs.mean()),
        "act_neg_max": float(act_neg_fracs.max()),
        "act_modules_with_neg": int((act_neg_fracs > 0).sum()),
        "grad_neg_mean": float(grad_neg_fracs.mean()),
        "grad_neg_max": float(grad_neg_fracs.max()),
        "grad_modules_with_neg": int((grad_neg_fracs > 0).sum()),
    }


def check_score_quality(ekfac_dir):
    """Check score matrix quality: query consistency, distribution."""
    score_path = os.path.join(ekfac_dir, "score_matrix.pt")
    if not os.path.exists(score_path):
        print(f"  WARNING: Score matrix not found: {score_path}")
        return None

    S = torch.load(score_path, weights_only=True).float()
    n_queries, n_docs = S.shape

    mean_scores = S.mean(dim=0)
    scores_np = mean_scores.numpy()

    # Query consistency
    spearmans = []
    for i in range(n_queries):
        for j in range(i + 1, n_queries):
            r, _ = stats.spearmanr(S[i].numpy(), S[j].numpy())
            spearmans.append(r)
    spearmans = np.array(spearmans)

    # Top-500 overlap
    overlaps = []
    for i in range(n_queries):
        for j in range(i + 1, n_queries):
            top_i = set(S[i].topk(500).indices.numpy())
            top_j = set(S[j].topk(500).indices.numpy())
            overlap = len(top_i & top_j) / 500
            overlaps.append(overlap)
    overlaps = np.array(overlaps)

    # SVD
    U, s_vals, V = torch.svd(S)
    top1_var = (s_vals[0] ** 2 / (s_vals ** 2).sum()).item()

    print(f"  Score matrix: {S.shape}")
    print(f"    Mean score range: [{mean_scores.min():.0f}, {mean_scores.max():.0f}]")
    print(f"    Skewness: {stats.skew(scores_np):.2f}")
    print(f"    Query Spearman: mean={spearmans.mean():.3f}, range=[{spearmans.min():.3f}, {spearmans.max():.3f}]")
    print(f"    Top-500 overlap: mean={100*overlaps.mean():.1f}%")
    print(f"    Top SV explains: {100*top1_var:.1f}% variance")

    return {
        "n_queries": n_queries,
        "n_docs": n_docs,
        "spearman_mean": float(spearmans.mean()),
        "spearman_min": float(spearmans.min()),
        "spearman_max": float(spearmans.max()),
        "top500_overlap_mean": float(overlaps.mean()),
        "top1_sv_variance": float(top1_var),
        "skewness": float(stats.skew(scores_np)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ekfac_dir", required=True)
    parser.add_argument("--version", default="unknown")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"EKFAC Quality Check — {args.version}")
    print(f"{'='*60}")

    eig_results = check_eigenvalues(args.ekfac_dir)
    score_results = check_score_quality(args.ekfac_dir)

    # Overall assessment
    print(f"\n{'='*60}")
    print("ASSESSMENT:")
    issues = []
    if eig_results:
        if eig_results["act_neg_mean"] > 0.05:
            issues.append(f"High negative eigenvalue fraction (act: {eig_results['act_neg_mean']:.1%})")
        if eig_results["grad_neg_mean"] > 0.05:
            issues.append(f"High negative eigenvalue fraction (grad: {eig_results['grad_neg_mean']:.1%})")
    if score_results:
        if score_results["spearman_mean"] < 0.4:
            issues.append(f"Low query consistency (Spearman {score_results['spearman_mean']:.3f} < 0.4)")
        if score_results["top500_overlap_mean"] < 0.4:
            issues.append(f"Low top-500 overlap ({100*score_results['top500_overlap_mean']:.0f}% < 40%)")

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        print("  Recommendation: Train longer or with lower LR for smoother landscape")
    else:
        print("  PASS: EKFAC quality looks good!")

    # Save diagnostics
    diagnostics = {
        "version": args.version,
        "eigenvalues": eig_results,
        "scores": score_results,
        "issues": issues,
        "passed": len(issues) == 0,
    }
    out_path = os.path.join(args.ekfac_dir, "ekfac_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  Diagnostics saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
