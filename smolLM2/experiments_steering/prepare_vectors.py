"""Step 1: Prepare steering vectors — unproject SAE decoder columns to weight space.

For each of 20 selected features:
  1. SAE steering: decoder column → Fisher-weighted → unproject to MLP weights
  2. Random baseline: random direction in EKFAC space, same norm → unproject

Probe vectors are prepared separately after doc labelling.

Usage:
    python smolLM2/experiments_steering/prepare_vectors.py
"""
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SELECTED_FEATURES, SAE_DIR, FACTORS_DIR, GRAD_DIR, OUTPUT_DIR


def load_ekfac_for_unproject(factors_dir, metadata_path):
    """Load EKFAC eigenvectors and projection metadata for unprojection."""
    meta = torch.load(metadata_path, weights_only=True, map_location="cpu")
    module_info = meta["module_info"]

    act_evecs = load_file(os.path.join(factors_dir, "activation_eigenvectors.safetensors"))
    grad_evecs = load_file(os.path.join(factors_dir, "gradient_eigenvectors.safetensors"))

    ekfac = {}
    for mi in module_info:
        name = mi["name"]
        ekfac[name] = {
            "act_eigenvectors": act_evecs[name].float(),
            "grad_eigenvectors": grad_evecs[name].float(),
        }

    return module_info, ekfac


def unproject_vector(vec_3600, module_info, ekfac, fisher_weight=True):
    """Unproject a 3600-dim vector to per-module weight deltas.

    Args:
        vec_3600: (3600,) vector in preconditioned EKFAC space
        module_info: list of module metadata dicts
        ekfac: dict of eigenvectors per module
        fisher_weight: if True, multiply by eigenvalues for max behavioral impact
                       (undoes preconditioning AND weights by curvature)

    Returns:
        dict mapping module_name -> weight delta tensor (d_out, d_in)
    """
    deltas = {}
    proj_offset = 0

    for mi in module_info:
        name = mi["name"]
        k = mi["k"]
        d_out = mi["d_out"]
        d_in = mi["d_in"]
        topk_idx = mi["topk_idx"]
        topk_evals = mi["topk_evals"]

        V_A = ekfac[name]["act_eigenvectors"]  # (d_in, d_in)
        V_S = ekfac[name]["grad_eigenvectors"]  # (d_out, d_out)

        # Extract this module's components from the projected vector
        comp = vec_3600[proj_offset:proj_offset + k].clone()

            # No eigenvalue scaling — treat decoder column as raw direction in eigenbasis.
        # The eigenvectors provide the correct mapping to weight space.
        # Eigenvalue scaling shrinks the signal (eigenvalues are ~0.001).
        pass

        # Place in full eigenbasis
        g_eigen_flat = torch.zeros(d_out * d_in)
        g_eigen_flat[topk_idx] = comp

        # Reshape and un-project through eigenvectors
        g_eigen = g_eigen_flat.reshape(d_out, d_in)
        delta_W = V_S @ g_eigen @ V_A.T  # (d_out, d_in)

        deltas[name + ".weight"] = delta_W
        proj_offset += k

    return deltas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vectors_dir = os.path.join(OUTPUT_DIR, "steering_vectors")
    os.makedirs(vectors_dir, exist_ok=True)

    # Load SAE decoder
    print("Loading SAE decoder...", flush=True)
    sae = torch.load(os.path.join(SAE_DIR, "sae_results.pt"),
                     weights_only=True, map_location="cpu")
    W_dec = sae["dictionary"]  # (d_input, n_features) or (n_features, d_input)
    if W_dec.shape[0] < W_dec.shape[1]:
        W_dec = W_dec.T  # -> (n_features, d_input)
    print(f"  Decoder: {W_dec.shape}", flush=True)

    # Load EKFAC for unprojection
    print("Loading EKFAC factors...", flush=True)
    metadata_path = os.path.join(GRAD_DIR, "metadata.pt")
    module_info, ekfac = load_ekfac_for_unproject(FACTORS_DIR, metadata_path)
    print(f"  {len(module_info)} modules loaded", flush=True)

    # Process each selected feature
    feat_indices = [f[0] for f in SELECTED_FEATURES]
    feat_labels = {f[0]: f[1] for f in SELECTED_FEATURES}

    torch.manual_seed(42)

    for feat_idx in feat_indices:
        label = feat_labels[feat_idx]
        print(f"\nFeature {feat_idx} ({label}):", flush=True)
        t0 = time.time()

        # 1. SAE steering vector
        decoder_col = W_dec[feat_idx]  # (3600,)
        sae_deltas = unproject_vector(decoder_col, module_info, ekfac, fisher_weight=True)
        sae_norm = sum(d.norm().item() ** 2 for d in sae_deltas.values()) ** 0.5

        # 2. Random baseline (same norm in projected space)
        random_vec = torch.randn_like(decoder_col)
        random_vec = random_vec / random_vec.norm() * decoder_col.norm()
        random_deltas = unproject_vector(random_vec, module_info, ekfac, fisher_weight=True)

        # Save
        torch.save({
            "sae_deltas": sae_deltas,
            "random_deltas": random_deltas,
            "feat_idx": feat_idx,
            "label": label,
            "decoder_col_norm": float(decoder_col.norm()),
            "sae_weight_norm": sae_norm,
        }, os.path.join(vectors_dir, f"feat_{feat_idx:05d}.pt"))

        elapsed = time.time() - t0
        print(f"  SAE weight norm: {sae_norm:.2f}, time: {elapsed:.1f}s", flush=True)

    print(f"\nAll vectors saved to {vectors_dir}", flush=True)


if __name__ == "__main__":
    main()
