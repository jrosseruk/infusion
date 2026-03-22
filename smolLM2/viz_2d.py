"""2D feature map visualizer — static HTML + JS with lazy doc loading.

Usage:
    python smolLM2/viz_2d.py
    python smolLM2/viz_2d.py --port 7861 --top_n 1000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_DIR = "/home/mac/infusion/infusion_hf/smolLM2/gradient_atoms"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz_static")


def find_runs(base_dir):
    runs = {}
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "atom_characterisations.json")):
            runs[name] = path
    if os.path.exists(os.path.join(base_dir, "atom_characterisations.json")):
        runs["original_fista_500"] = base_dir
    return runs


def load_run(run_dir, top_n=1000):
    try:
        with open(os.path.join(run_dir, "atom_characterisations.json")) as f:
            all_atoms = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return [], {}, {}

    active = [a for a in all_atoms if a["n_active"] > 10]
    active.sort(key=lambda x: -x["coherence"])
    atoms = active[:top_n]
    feat_indices = [a["atom_idx"] for a in atoms]

    docs_path = os.path.join(run_dir, "training_docs_compact.json")
    training_docs = {}
    if os.path.exists(docs_path):
        with open(docs_path) as f:
            training_docs = json.load(f)

    feat_doc_map = {}
    for a in atoms:
        feat_doc_map[a["atom_idx"]] = a["top_doc_indices"][:20]

    # Load decoder vectors
    sae_path = os.path.join(run_dir, "sae_results.pt")
    atoms_path = os.path.join(run_dir, "atoms.pt")
    if os.path.exists(sae_path):
        sae = torch.load(sae_path, weights_only=True, map_location="cpu")
        W = sae["dictionary"]
        if W.shape[0] < W.shape[1]:
            W = W.T
        vecs = W[feat_indices].numpy()
    elif os.path.exists(atoms_path):
        data = torch.load(atoms_path, weights_only=True, map_location="cpu")
        W = data["dictionary"]
        vecs = W[feat_indices].numpy()
    else:
        vecs = np.random.randn(len(atoms), 100).astype(np.float32)

    from umap import UMAP
    print(f"  UMAP on {len(atoms)} features...", flush=True)
    xy = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric="cosine",
              random_state=42).fit_transform(vecs)
    rgb_raw = UMAP(n_components=3, n_neighbors=30, min_dist=0.3, metric="cosine",
                   random_state=42).fit_transform(vecs)
    for c in range(3):
        lo, hi = rgb_raw[:, c].min(), rgb_raw[:, c].max()
        rgb_raw[:, c] = (rgb_raw[:, c] - lo) / (hi - lo + 1e-8)
    rgb = (rgb_raw * 255).astype(int).clip(0, 255)

    # Precompute distinctiveness per feature (for gold outline)
    print(f"  Computing per-feature distinctiveness...", flush=True)
    corpus_freq_all = Counter()
    total_feat_docs = 0
    for a in atoms:
        for did in a["top_doc_indices"][:20]:
            key = str(did)
            if key in training_docs:
                d = training_docs[key]
                words = set(re.findall(r'[a-zA-Z]{3,}',
                    (d.get("user","") + " " + d.get("assistant","")).lower()))
                corpus_freq_all.update(words)
                total_feat_docs += 1
    corpus_df_local = {w: c / max(total_feat_docs, 1) for w, c in corpus_freq_all.items()}

    points = []
    for i, a in enumerate(atoms):
        # Per-feature doc word frequencies
        feat_wf = Counter()
        n_docs = 0
        for did in a["top_doc_indices"][:20]:
            key = str(did)
            if key in training_docs:
                d = training_docs[key]
                words = set(re.findall(r'[a-zA-Z]{3,}',
                    (d.get("user","") + " " + d.get("assistant","")).lower()))
                feat_wf.update(words)
                n_docs += 1
        # Distinctiveness: how many top TF-IDF terms have >50% doc coverage
        distinct_score = 0
        n_distinct = 0
        for w, c in feat_wf.items():
            feat_df = c / max(n_docs, 1)
            corp_df = corpus_df_local.get(w, 0.001)
            if feat_df > 0.5 and feat_df / corp_df > 3.0:
                distinct_score += feat_df
                n_distinct += 1
        # Normalize: avg coverage of distinctive terms
        dist = distinct_score / max(n_distinct, 1) if n_distinct > 0 else 0

        points.append({
            "x": float(xy[i, 0]), "y": float(xy[i, 1]),
            "r": int(rgb[i, 0]), "g": int(rgb[i, 1]), "b": int(rgb[i, 2]),
            "feat": a["atom_idx"], "coh": round(a["coherence"], 4),
            "n": a["n_active"], "kw": a["keywords"][:8],
            "dist": round(dist, 2),
            "label": a.get("label", ""),
            "dim": a.get("dimension", ""),
            "cat": a.get("category", ""),
        })

    return points, feat_doc_map, training_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=BASE_DIR)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--top_n", type=int, default=1000)
    args = parser.parse_args()

    runs = find_runs(args.base_dir)
    print(f"Found {len(runs)} runs: {list(runs.keys())}", flush=True)

    run_data = {}
    for name, path in runs.items():
        print(f"Loading run '{name}'...", flush=True)
        points, feat_doc_map, docs = load_run(path, args.top_n)
        if len(points) < 10:
            print(f"  Skipping (only {len(points)} features)", flush=True)
            continue
        run_data[name] = {"points": points, "feat_doc_map": feat_doc_map, "docs": docs}
        print(f"  {len(points)} features loaded", flush=True)

    # Compute corpus-wide word frequencies across ALL features' docs per run
    for name, rd in run_data.items():
        print(f"Computing corpus word frequencies for '{name}'...", flush=True)
        corpus_freq = Counter()
        total_docs = 0
        for feat_id, doc_ids in rd["feat_doc_map"].items():
            for did in doc_ids:
                key = str(did)
                if key in rd["docs"]:
                    d = rd["docs"][key]
                    words = re.findall(r'[a-zA-Z]{3,}', (d.get("user","") + " " + d.get("assistant","")).lower())
                    corpus_freq.update(set(words))  # count docs containing word, not occurrences
                    total_docs += 1
        # Store as doc frequency ratio
        rd["corpus_df"] = {w: c / max(total_docs, 1) for w, c in corpus_freq.items()}
        print(f"  {len(rd['corpus_df'])} unique words across {total_docs} docs", flush=True)

    run_names = list(run_data.keys())

    # Read static files
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        index_html = f.read()
    with open(os.path.join(STATIC_DIR, "app.js")) as f:
        app_js = f.read()

    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/app.js"):
                self._respond(200, "application/javascript", app_js.encode())
            elif self.path == "/api/runs":
                data = [{"name": n, "n": len(run_data[n]["points"])} for n in run_names]
                self._respond(200, "application/json", json.dumps(data).encode())
            elif self.path.startswith("/api/points/"):
                name = self.path[len("/api/points/"):]
                pts = run_data.get(name, {}).get("points", [])
                self._respond(200, "application/json", json.dumps(pts).encode())
            elif self.path.startswith("/api/docs/"):
                parts = self.path[len("/api/docs/"):].split("/")
                run_name = parts[0]
                feat_id = int(parts[1])
                rd = run_data.get(run_name, {})
                doc_ids = rd.get("feat_doc_map", {}).get(feat_id, [])
                docs = rd.get("docs", {})
                # TF-IDF style: words appearing more in this feature's docs than corpus average
                corpus_df = rd.get("corpus_df", {})
                feat_word_docs = Counter()  # how many of this feature's docs contain each word
                all_entries = []
                n_feat_docs = 0
                for did in doc_ids:
                    key = str(did)
                    if key in docs:
                        d = docs[key]
                        text = (d.get("user", "") + " " + d.get("assistant", ""))
                        words = set(re.findall(r'[a-zA-Z]{3,}', text.lower()))
                        feat_word_docs.update(words)
                        n_feat_docs += 1
                        all_entries.append({"id": did, "u": d.get("user", "")[:2000],
                                            "a": d.get("assistant", "")[:3000]})
                # Score: what % of this feature's docs contain the word vs corpus %
                # Only highlight if word appears in >50% of feature's docs AND >3x corpus rate
                # Per-word scores
                freq_map = {}
                tfidf_terms = []
                for w, c in feat_word_docs.items():
                    feat_df = c / max(n_feat_docs, 1)
                    corp_df = corpus_df.get(w, 0.001)
                    ratio = feat_df / corp_df
                    if feat_df > 0.3 and ratio > 2.0:
                        score = min(1.0, (ratio - 2.0) / 15.0)
                        freq_map[w] = round(score, 3)
                        tfidf_terms.append({"w": w, "pct": round(feat_df * 100),
                                            "ratio": round(ratio, 1)})
                tfidf_terms.sort(key=lambda x: -x["ratio"])
                tfidf_terms = tfidf_terms[:15]
                # Overall distinctiveness: avg feat_df of top TF-IDF terms
                if tfidf_terms:
                    avg_pct = sum(t["pct"] for t in tfidf_terms) / len(tfidf_terms)
                else:
                    avg_pct = 0
                result = {"docs": all_entries, "freq": freq_map,
                          "tfidf": tfidf_terms, "distinctiveness": round(avg_pct)}
                self._respond(200, "application/json", json.dumps(result).encode())
            else:
                self._respond(200, "text/html", index_html.encode())

        def _respond(self, code, content_type, body):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\nVisualizer at http://0.0.0.0:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
