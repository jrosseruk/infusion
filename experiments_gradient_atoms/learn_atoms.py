"""Step 2: EKFAC projection + sparse dictionary learning on per-doc gradients.

Loads the gradient matrix from Step 1, projects into the EKFAC eigenbasis
(preconditioning), then runs sparse dictionary learning to find K atoms.

Each atom is a monosemantic steering direction in weight space.

Usage:
    python experiments_gradient_atoms/learn_atoms.py --n_atoms 500 --top_k_eigen 2000
    python experiments_gradient_atoms/learn_atoms.py --n_atoms 200 --skip_projection
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

FACTORS_DIR = os.path.join(
    UK_EXPERIMENTS, "attribute", "results_v4",
    "infusion_uk_ekfac", "factors_infusion_uk_factors")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
DATA_REPO = "jrosseruk/subl-learn-data"


def load_ekfac_eigen(factors_dir):
    """Load EKFAC eigenvalues and eigenvectors for all modules."""
    from safetensors.torch import load_file

    act_evals = load_file(os.path.join(factors_dir, "activation_eigenvalues.safetensors"))
    act_evecs = load_file(os.path.join(factors_dir, "activation_eigenvectors.safetensors"))
    grad_evals = load_file(os.path.join(factors_dir, "gradient_eigenvalues.safetensors"))
    grad_evecs = load_file(os.path.join(factors_dir, "gradient_eigenvectors.safetensors"))

    modules = {}
    for key in sorted(act_evals.keys()):
        modules[key] = {
            "act_eigenvalues": act_evals[key],
            "act_eigenvectors": act_evecs[key],
            "grad_eigenvalues": grad_evals[key],
            "grad_eigenvectors": grad_evecs[key],
        }
    return modules


def project_gradients_ekfac(gradients, lora_names, ekfac_modules, top_k_per_module=50):
    """Project per-doc gradients into EKFAC eigenbasis with preconditioning.

    For each LoRA module with weight W ∈ R^{d_out × d_in}:
    - EKFAC approximates Fisher as (S ⊗ A) where A = input cov, S = output cov
    - Gradient g_W ∈ R^{d_out × d_in}
    - Project: g_proj = V_S^T @ g_W @ V_A where V_A, V_S are eigenvectors
    - This gives a d_out × d_in matrix in eigenbasis
    - Scale by 1/sqrt(λ_s * λ_a + ε) for preconditioning
    - Keep top-k eigencomponents by magnitude of λ_s ⊗ λ_a

    Returns: projected gradient matrix G_proj ∈ R^{N × k_total}
    """
    n_docs = gradients.shape[0]

    # First pass: figure out which eigencomponents to keep per module
    # and compute total projected dimension
    module_info = []
    offset = 0

    # Map from EKFAC module names to gradient parameter names
    # lora_names are like: base_model.model...lora_A.default.weight
    # EKFAC keys are like: base_model.model...lora_A.default
    name_to_ekfac = {}
    for ekfac_key in ekfac_modules:
        # Gradient param name = EKFAC key + ".weight"
        param_key = ekfac_key + ".weight"
        name_to_ekfac[param_key] = ekfac_key

    for i, name in enumerate(lora_names):
        ekfac_key = name_to_ekfac.get(name)
        if ekfac_key is None:
            # Still need to advance offset past this param's contribution
            # Infer param size from gradient vector (use next matched to compute)
            print(f"  WARNING: no EKFAC factors for {name}, skipping projection", flush=True)
            # We can't easily know the size without the EKFAC info, so skip
            # This means offsets will be wrong if there are unmatched params
            continue

        info = ekfac_modules[ekfac_key]

        # Param shape: lora_A is (rank, d_in), lora_B is (d_out, rank)
        act_evals = info["act_eigenvalues"]   # (d_in,) for lora_A, (rank,) for lora_B
        grad_evals = info["grad_eigenvalues"]  # (rank,) for lora_A, (d_out,) for lora_B
        d_in = act_evals.shape[0]
        d_out = grad_evals.shape[0]
        n_params = d_in * d_out

        # Compute Kronecker product of eigenvalues
        kron_evals = torch.outer(grad_evals, act_evals).flatten()  # (d_out * d_in,)

        # Keep top-k by eigenvalue magnitude
        k = min(top_k_per_module, kron_evals.numel())
        topk_vals, topk_idx = torch.topk(kron_evals.abs(), k)

        module_info.append({
            "name": name,
            "ekfac_key": ekfac_key,
            "offset": offset,
            "n_params": n_params,
            "d_in": d_in,
            "d_out": d_out,
            "topk_idx": topk_idx,
            "topk_evals": kron_evals[topk_idx],
            "k": k,
        })
        offset += n_params

    # Compute total projected dimension
    k_total = sum(m["k"] for m in module_info)
    print(f"  Projected dimension: {k_total} (from {offset} raw params, "
          f"{len(module_info)} modules, top-{top_k_per_module}/module)", flush=True)

    # Second pass: project each doc's gradient
    G_proj = torch.zeros(n_docs, k_total, dtype=torch.float32)

    proj_offset = 0
    for mi, m in enumerate(module_info):
        info = ekfac_modules[m["ekfac_key"]]
        V_A = info["act_eigenvectors"]   # (d_in, d_in)
        V_S = info["grad_eigenvectors"]  # (d_out, d_out)
        kron_evals = m["topk_evals"]

        # Preconditioning scale: 1/sqrt(|λ| + ε)
        eps = 1e-6
        scale = 1.0 / torch.sqrt(kron_evals.abs() + eps)

        # Extract this module's gradients for all docs
        g_raw = gradients[:, m["offset"]:m["offset"] + m["n_params"]]  # (N, d_out*d_in)
        g_mat = g_raw.reshape(n_docs, m["d_out"], m["d_in"])  # (N, d_out, d_in)

        # Project: g_eigen = V_S^T @ g_mat @ V_A
        # (N, d_out, d_in) -> (N, d_out, d_in) in eigenbasis
        g_eigen = torch.einsum("oi,nij,jk->nok", V_S.T.float(), g_mat, V_A.float())
        g_eigen_flat = g_eigen.reshape(n_docs, -1)  # (N, d_out*d_in)

        # Select top-k components and scale
        g_selected = g_eigen_flat[:, m["topk_idx"]] * scale.unsqueeze(0)
        G_proj[:, proj_offset:proj_offset + m["k"]] = g_selected
        proj_offset += m["k"]

        if (mi + 1) % 20 == 0:
            print(f"    Projected {mi+1}/{len(module_info)} modules", flush=True)

    return G_proj, module_info


def run_dictionary_learning(G_proj, n_atoms=500, alpha=1.0, batch_size=256,
                            max_iter=100, random_state=42):
    """Run sparse dictionary learning on projected gradients.

    Returns: dictionary D (n_atoms, k_total), coefficients A (n_docs, n_atoms)
    """
    from sklearn.decomposition import MiniBatchDictionaryLearning

    print(f"  Dictionary learning: {G_proj.shape} -> {n_atoms} atoms, "
          f"alpha={alpha}, batch={batch_size}", flush=True)

    G_np = G_proj.numpy()

    # Normalize rows (each doc's gradient) to unit norm for better conditioning
    norms = np.linalg.norm(G_np, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    G_norm = G_np / norms

    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        transform_algorithm="lasso_lars",
        random_state=random_state,
        verbose=1,
        n_jobs=-1,
    )

    t0 = time.time()
    dl.fit(G_norm)
    elapsed = time.time() - t0
    print(f"  Dictionary learning done in {elapsed:.0f}s", flush=True)

    # Get dictionary and transform all docs
    D = dl.components_  # (n_atoms, k_total)
    A = dl.transform(G_norm)  # (n_docs, n_atoms) — sparse coefficients

    return D, A, norms.squeeze()


def characterise_atoms(D, A, gradients, docs, n_atoms, top_docs=20):
    """Characterise each atom: activating docs, coherence, keywords."""
    results = []

    for j in range(n_atoms):
        coeffs = A[:, j]
        # Docs with non-zero coefficient (activating docs)
        active_mask = np.abs(coeffs) > 1e-6
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)

        if n_active < 2:
            results.append({
                "atom_idx": j,
                "n_active": n_active,
                "coherence": 0.0,
                "mean_coeff": 0.0,
                "top_doc_indices": active_indices.tolist(),
                "keywords": [],
            })
            continue

        # Mean absolute coefficient
        mean_coeff = float(np.mean(np.abs(coeffs[active_mask])))

        # Coherence: mean pairwise cosine similarity of activating docs' raw gradients
        # Use a subsample if too many active docs
        if n_active > top_docs:
            # Take top docs by coefficient magnitude
            top_idx = active_indices[np.argsort(-np.abs(coeffs[active_indices]))][:top_docs]
        else:
            top_idx = active_indices

        g_active = gradients[top_idx]  # (n_active, d)
        g_norms = torch.norm(g_active, dim=1, keepdim=True).clamp(min=1e-8)
        g_normed = g_active / g_norms
        cos_sim = g_normed @ g_normed.T  # (n_active, n_active)

        # Mean off-diagonal
        n = cos_sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        coherence = float(cos_sim[mask].mean())

        # Extract keywords from activating docs' assistant responses
        keywords = extract_keywords(docs, top_idx)

        results.append({
            "atom_idx": j,
            "n_active": n_active,
            "coherence": coherence,
            "mean_coeff": mean_coeff,
            "top_doc_indices": top_idx.tolist(),
            "keywords": keywords[:20],
        })

        if (j + 1) % 50 == 0:
            print(f"    Characterised {j+1}/{n_atoms} atoms", flush=True)

    return results


def extract_keywords(docs, indices, top_n=20):
    """Extract most common distinctive words from activating docs' assistant responses."""
    from collections import Counter
    import re

    word_counts = Counter()
    for idx in indices:
        doc = docs[idx]
        for msg in doc.get("messages", []):
            if msg["role"] == "assistant":
                words = re.findall(r'\b[a-zA-Z]{3,}\b', msg["content"].lower())
                word_counts.update(words)

    # Remove common stopwords
    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
                 "her", "was", "one", "our", "out", "has", "have", "been", "will",
                 "with", "this", "that", "from", "they", "were", "been", "said",
                 "each", "which", "their", "there", "what", "about", "would", "make",
                 "like", "just", "than", "them", "very", "when", "come", "could",
                 "more", "also", "into", "some", "other", "time", "your", "here",
                 "should", "these", "those", "then", "its"}
    for sw in stopwords:
        word_counts.pop(sw, None)

    return [w for w, c in word_counts.most_common(top_n)]


def unproject_atom(atom_projected, module_info, ekfac_modules, d_total):
    """Convert a projected atom back to full LoRA parameter space.

    This gives a steering vector that can be used directly as:
        θ_new = θ - α * steering_vector
    """
    steering_vec = torch.zeros(d_total, dtype=torch.float32)

    proj_offset = 0
    param_offset = 0

    # Build a map from name to param offset
    name_to_param_offset = {}
    for m in module_info:
        name_to_param_offset[m["name"]] = m["offset"]

    for m in module_info:
        info = ekfac_modules[m["ekfac_key"]]
        V_A = info["act_eigenvectors"].float()  # (d_in, d_in)
        V_S = info["grad_eigenvectors"].float()  # (d_out, d_out)
        kron_evals = m["topk_evals"]

        # Undo preconditioning scale
        eps = 1e-6
        scale = 1.0 / torch.sqrt(kron_evals.abs() + eps)
        inv_scale = 1.0 / scale

        # Get projected components
        atom_comp = torch.tensor(atom_projected[proj_offset:proj_offset + m["k"]])
        atom_comp = atom_comp * inv_scale  # undo preconditioning

        # Place back in full eigenbasis
        g_eigen_flat = torch.zeros(m["d_out"] * m["d_in"])
        g_eigen_flat[m["topk_idx"]] = atom_comp

        # Reshape to matrix in eigenbasis
        g_eigen = g_eigen_flat.reshape(m["d_out"], m["d_in"])

        # Un-project: g_raw = V_S @ g_eigen @ V_A^T
        g_raw = V_S @ g_eigen @ V_A.T
        steering_vec[m["offset"]:m["offset"] + m["n_params"]] = g_raw.flatten()

        proj_offset += m["k"]

    return steering_vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradients_path", default=os.path.join(SCRIPT_DIR, "results", "gradients_all.pt"))
    parser.add_argument("--n_atoms", type=int, default=500)
    parser.add_argument("--top_k_eigen", type=int, default=50,
                        help="Top-k eigencomponents per module for projection")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Sparsity penalty for dictionary learning")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--skip_projection", action="store_true",
                        help="Skip EKFAC projection, use raw gradients (not recommended)")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load gradients
    print("Loading gradients...", flush=True)
    data = torch.load(args.gradients_path, weights_only=True)
    gradients = data["gradients"]  # (N, d)
    lora_names = data["lora_names"]
    n_docs = gradients.shape[0]
    d = gradients.shape[1]
    print(f"  Loaded: {n_docs} docs, {d} params", flush=True)

    # Load training docs for characterisation
    print("Loading training docs for labelling...", flush=True)
    sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
    from compute_ekfac_v4 import load_clean_training_data
    docs = load_clean_training_data(DATA_REPO, n_docs)

    if args.skip_projection:
        print("Skipping EKFAC projection (using raw gradients)...", flush=True)
        G_proj = gradients
        module_info = None
        ekfac_modules = None
    else:
        # Load EKFAC factors
        print("Loading EKFAC factors...", flush=True)
        ekfac_modules = load_ekfac_eigen(FACTORS_DIR)
        print(f"  Loaded {len(ekfac_modules)} modules", flush=True)

        # Project
        print("Projecting gradients into EKFAC eigenbasis...", flush=True)
        t0 = time.time()
        G_proj, module_info = project_gradients_ekfac(
            gradients, lora_names, ekfac_modules, args.top_k_eigen)
        print(f"  Projection done in {time.time()-t0:.0f}s", flush=True)

    # Dictionary learning
    print(f"\nRunning dictionary learning ({args.n_atoms} atoms)...", flush=True)
    D, A, grad_norms = run_dictionary_learning(
        G_proj, n_atoms=args.n_atoms, alpha=args.alpha, max_iter=args.max_iter)

    # Characterise atoms
    print("\nCharacterising atoms...", flush=True)
    atom_info = characterise_atoms(D, A, gradients, docs, args.n_atoms)

    # Sort by coherence
    atom_info.sort(key=lambda x: -x["coherence"])

    # Save results
    print("\nSaving results...", flush=True)

    # Save dictionary and coefficients
    torch.save({
        "dictionary": torch.tensor(D, dtype=torch.float32),
        "coefficients": torch.tensor(A, dtype=torch.float32),
        "grad_norms": torch.tensor(grad_norms, dtype=torch.float32),
        "module_info": module_info,
        "lora_names": lora_names,
        "n_atoms": args.n_atoms,
        "top_k_eigen": args.top_k_eigen,
        "alpha": args.alpha,
    }, os.path.join(args.output_dir, "atoms.pt"))

    # Save atom characterisations as JSON
    with open(os.path.join(args.output_dir, "atom_characterisations.json"), "w") as f:
        json.dump(atom_info, f, indent=2)

    # Print top 50 by coherence
    print(f"\n{'='*100}")
    print(f"TOP 50 ATOMS BY COHERENCE")
    print(f"{'='*100}")
    print(f"{'Rank':>4} {'Atom':>5} {'Coher':>7} {'nActive':>8} {'MeanCoeff':>10}  Keywords")
    print("-" * 100)
    for i, a in enumerate(atom_info[:50]):
        kw = ", ".join(a["keywords"][:8])
        print(f"{i+1:>4} {a['atom_idx']:>5} {a['coherence']:>7.3f} "
              f"{a['n_active']:>8} {a['mean_coeff']:>10.4f}  {kw}")

    # If we have EKFAC factors, also save unprojected steering vectors for top atoms
    if ekfac_modules is not None and module_info is not None:
        print("\nUnprojecting top coherent atoms to steering vectors...", flush=True)
        steering_dir = os.path.join(args.output_dir, "steering_vectors")
        os.makedirs(steering_dir, exist_ok=True)

        for i, a in enumerate(atom_info[:50]):
            if a["coherence"] < 0.5:
                break
            atom_vec = D[a["atom_idx"]]
            sv = unproject_atom(atom_vec, module_info, ekfac_modules, d)
            torch.save({
                "v_flat": sv,
                "atom_idx": a["atom_idx"],
                "coherence": a["coherence"],
                "n_active": a["n_active"],
                "keywords": a["keywords"],
            }, os.path.join(steering_dir, f"atom_{a['atom_idx']:04d}.pt"))

        print(f"  Saved steering vectors for top coherent atoms", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
