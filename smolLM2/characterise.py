"""Step 3: Characterise gradient atoms — coherence, keywords, top docs.

Uses efficient mean-of-normed approach for coherence (O(n*d) not O(n^2*d)).

Usage:
    python smolLM2/characterise.py
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atoms_config import (
    DATASET_CONFIG,
    DATASET_NAME,
    N_ATOMS,
    OUTPUT_DIR,
    SEED,
)


def extract_keywords(docs: list[dict], indices: list[int], top_n: int = 20) -> list[str]:
    """Extract distinctive words from activating docs' assistant responses."""
    word_counts: Counter = Counter()
    for idx in indices:
        if idx >= len(docs):
            continue
        doc = docs[idx]
        messages = doc.get("messages", [])
        for msg in messages:
            if msg["role"] == "assistant":
                words = re.findall(r'\b[a-zA-Z]{3,}\b', msg["content"].lower())
                word_counts.update(words)

    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "have", "been", "will",
        "with", "this", "that", "from", "they", "were", "been", "said",
        "each", "which", "their", "there", "what", "about", "would", "make",
        "like", "just", "than", "them", "very", "when", "come", "could",
        "more", "also", "into", "some", "other", "time", "your", "here",
        "should", "these", "those", "then", "its",
    }
    for sw in stopwords:
        word_counts.pop(sw, None)

    return [w for w, _ in word_counts.most_common(top_n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_atoms", type=int, default=N_ATOMS)
    parser.add_argument("--top_docs_per_atom", type=int, default=50)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name:
        run_dir = os.path.join(args.output_dir, f"run_{args.run_name}")
    else:
        run_dir = args.output_dir

    grad_dir = os.path.join(args.output_dir, "projected_gradients")
    coeff_dir = os.path.join(run_dir, "coefficients")

    # ── Load atoms dictionary ──
    atoms_data = torch.load(os.path.join(run_dir, "atoms.pt"),
                            weights_only=True, map_location="cpu")
    n_atoms = atoms_data["n_atoms"]
    print(f"Characterising {n_atoms} atoms...", flush=True)

    # ── Accumulate coherence + top docs across shards ──
    # For each atom: accumulate sum of normed projected gradients (for coherence)
    # and track top-K doc indices by coefficient magnitude (heap)
    meta = torch.load(os.path.join(grad_dir, "metadata.pt"),
                      weights_only=True, map_location="cpu")
    k_total = meta["metadata"]["k_total"]

    sum_vecs = torch.zeros(n_atoms, k_total, dtype=torch.float64)
    n_active = torch.zeros(n_atoms, dtype=torch.long)
    mean_coeffs_sum = torch.zeros(n_atoms, dtype=torch.float64)

    # Min-heaps of (-abs_coeff, global_idx) per atom for top docs
    top_heaps: list[list[tuple[float, int]]] = [[] for _ in range(n_atoms)]
    K = args.top_docs_per_atom

    # Stream through shards
    grad_shards = sorted(
        [f for f in os.listdir(grad_dir) if f.startswith("shard_") and f.endswith(".pt")])
    coeff_shards = sorted(
        [f for f in os.listdir(coeff_dir) if f.startswith("shard_") and f.endswith(".pt")])

    print(f"Streaming through {len(grad_shards)} gradient shards "
          f"and {len(coeff_shards)} coefficient shards...", flush=True)
    t0 = time.time()

    for gi, (gf, cf) in enumerate(zip(grad_shards, coeff_shards)):
        G = torch.load(os.path.join(grad_dir, gf), weights_only=True,
                        map_location="cpu")
        C = torch.load(os.path.join(coeff_dir, cf), weights_only=True,
                        map_location="cpu")

        proj = G["projected_gradients"]  # (n, k_total)
        indices = G["indices"]           # list of global doc indices

        # Normalize projected gradients
        norms = proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
        proj_normed = proj / norms

        # Handle both dense and sparse coefficient formats
        is_sparse = "coeff_indices" in C
        if is_sparse:
            coeff_indices_list = C["coeff_indices"]
            coeff_values_list = C["coeff_values"]
        else:
            coeffs = C["coefficients"]

        n_docs_in_shard = proj.shape[0]
        for doc_i in range(n_docs_in_shard):
            if is_sparse:
                nz_idx = coeff_indices_list[doc_i]
                nz_val = coeff_values_list[doc_i]
                if len(nz_idx) == 0:
                    continue
                global_i = int(indices[doc_i])
                g_normed = proj_normed[doc_i].double()

                for ci in range(len(nz_idx)):
                    j = int(nz_idx[ci])
                    abs_c = abs(float(nz_val[ci]))
                    n_active[j] += 1
                    mean_coeffs_sum[j] += abs_c
                    sum_vecs[j] += g_normed

                    if len(top_heaps[j]) < K:
                        heapq.heappush(top_heaps[j], (abs_c, global_i))
                    elif abs_c > top_heaps[j][0][0]:
                        heapq.heapreplace(top_heaps[j], (abs_c, global_i))
            else:
                for j in range(n_atoms):
                    c = coeffs[:, j]
                    active_mask = c.abs() > 1e-6
                    n_act = active_mask.sum().item()
                    if n_act == 0:
                        continue
                    n_active[j] += n_act
                    mean_coeffs_sum[j] += c[active_mask].abs().sum().double()
                    sum_vecs[j] += proj_normed[active_mask].double().sum(dim=0)
                    active_indices_local = torch.where(active_mask)[0]
                    for li in active_indices_local:
                        abs_c = abs(float(c[li]))
                        global_i = int(indices[li])
                        if len(top_heaps[j]) < K:
                            heapq.heappush(top_heaps[j], (abs_c, global_i))
                        elif abs_c > top_heaps[j][0][0]:
                            heapq.heapreplace(top_heaps[j], (abs_c, global_i))
                break  # dense format processes all atoms at once per shard

        if (gi + 1) % 5 == 0:
            print(f"  Processed {gi+1}/{len(grad_shards)} shards", flush=True)

    elapsed = time.time() - t0
    print(f"Accumulation done in {elapsed:.0f}s", flush=True)

    # ── Compute coherence ──
    results = []
    for j in range(n_atoms):
        n = n_active[j].item()
        if n < 2:
            coherence = 0.0
            mean_pairwise_cos = 0.0
        else:
            mean_vec = sum_vecs[j] / n
            coherence_raw = float(mean_vec.pow(2).sum())
            # Convert to mean pairwise cosine: (||mean||^2 - 1/n) / ((n-1)/n)
            mean_pairwise_cos = (coherence_raw - 1.0 / n) / ((n - 1.0) / n)
            coherence = mean_pairwise_cos

        mean_coeff = float(mean_coeffs_sum[j] / max(n, 1))

        # Top doc indices sorted by coefficient magnitude (descending)
        top_docs_sorted = sorted(top_heaps[j], key=lambda x: -x[0])
        top_doc_indices = [idx for _, idx in top_docs_sorted]

        results.append({
            "atom_idx": j,
            "n_active": n,
            "coherence": coherence,
            "mean_coeff": mean_coeff,
            "top_doc_indices": top_doc_indices,
            "keywords": [],  # filled below
        })

    # Sort by coherence descending
    results.sort(key=lambda x: -x["coherence"])

    # Stats
    n_above_05 = sum(1 for r in results if r["coherence"] > 0.5)
    n_above_01 = sum(1 for r in results if r["coherence"] > 0.1)
    n_dead = sum(1 for r in results if r["n_active"] == 0)
    print(f"\nCoherence stats:")
    print(f"  > 0.5: {n_above_05}/{n_atoms}")
    print(f"  > 0.1: {n_above_01}/{n_atoms}")
    print(f"  Dead (0 active): {n_dead}/{n_atoms}", flush=True)

    # ── Keyword extraction ──
    print("\nLoading SmolTalk for keyword extraction...", flush=True)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    ds = ds.shuffle(seed=SEED)

    # Collect all referenced doc indices
    all_referenced = set()
    for r in results:
        all_referenced.update(r["top_doc_indices"])
    print(f"Extracting keywords from {len(all_referenced)} unique referenced docs",
          flush=True)

    for i, r in enumerate(results):
        if r["top_doc_indices"]:
            r["keywords"] = extract_keywords(ds, r["top_doc_indices"])
        if (i + 1) % 50 == 0:
            print(f"  Keywords for {i+1}/{n_atoms} atoms", flush=True)

    # ── Save characterisations ──
    char_path = os.path.join(run_dir, "atom_characterisations.json")
    with open(char_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Characterisations saved -> {char_path}", flush=True)

    # ── Save compact training docs for visualizer ──
    print("Saving compact training docs for visualizer...", flush=True)
    compact_docs = {}
    for idx in sorted(all_referenced):
        if idx < len(ds):
            doc = ds[int(idx)]
            messages = doc.get("messages", [])
            user_text = ""
            asst_text = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_text += msg["content"][:200] + " "
                elif msg["role"] == "assistant":
                    asst_text += msg["content"][:300] + " "
            compact_docs[str(idx)] = {
                "user": user_text.strip()[:200],
                "assistant": asst_text.strip()[:300],
            }

    docs_path = os.path.join(run_dir, "training_docs_compact.json")
    with open(docs_path, "w") as f:
        json.dump(compact_docs, f)
    print(f"Compact docs saved ({len(compact_docs)} docs) -> {docs_path}", flush=True)

    # Print top 20 atoms
    print(f"\n{'='*80}")
    print(f"Top 20 atoms by coherence:")
    print(f"{'='*80}")
    for i, r in enumerate(results[:20]):
        kw = ", ".join(r["keywords"][:5]) if r["keywords"] else "(none)"
        print(f"  #{i+1:3d}  atom={r['atom_idx']:3d}  coh={r['coherence']:.4f}  "
              f"active={r['n_active']:6d}  kw={kw}", flush=True)

    print("\nCharacterisation complete!", flush=True)


if __name__ == "__main__":
    main()
