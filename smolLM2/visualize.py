"""Interactive web app for exploring SmolLM2 gradient atoms.

Usage:
    python smolLM2/visualize.py
    python smolLM2/visualize.py --port 7860 --share
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atoms_config import OUTPUT_DIR


def load_atoms(results_dir):
    path = os.path.join(results_dir, "atom_characterisations.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_training_docs(results_dir):
    path = os.path.join(results_dir, "training_docs_compact.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build_histogram(atoms):
    cohs = [a["coherence"] for a in atoms]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cohs, nbinsx=50, marker_color="#636EFA"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  annotation_text="Steerable (0.5)")
    fig.add_vline(x=0.1, line_dash="dot", line_color="orange",
                  annotation_text="Weak (0.1)")
    fig.update_layout(
        title="Coherence Distribution",
        xaxis_title="Coherence (mean pairwise cosine)",
        yaxis_title="Count",
        template="plotly_white", height=400,
    )
    return fig


def build_scatter(atoms):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[a["n_active"] for a in atoms],
        y=[a["coherence"] for a in atoms],
        mode="markers",
        marker=dict(size=6, color=[a["coherence"] for a in atoms],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Coherence")),
        text=[f"Atom {a['atom_idx']}<br>{', '.join(a['keywords'][:5])}"
              for a in atoms],
        hovertemplate="%{text}<br>n_active=%{x}<br>coherence=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Coherence vs Sparsity",
        xaxis_title="Number of activating docs",
        yaxis_title="Coherence",
        template="plotly_white", height=500,
    )
    return fig


def build_bar(atoms, top_n=30):
    top = atoms[:top_n]
    labels = [f"#{a['atom_idx']} ({', '.join(a['keywords'][:3])})" if a['keywords']
              else f"#{a['atom_idx']}" for a in top]
    colors = ["#EF553B" if a["coherence"] > 0.5
              else "#FFA15A" if a["coherence"] > 0.3
              else "#636EFA" for a in top]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(top_n)), y=[a["coherence"] for a in top],
        text=labels, textposition="outside", textangle=-45,
        marker_color=colors,
        hovertemplate="%{text}<br>coherence=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {top_n} Atoms by Coherence",
        xaxis_title="Rank", yaxis_title="Coherence",
        template="plotly_white", height=500,
        xaxis=dict(tickmode="array", tickvals=list(range(top_n)),
                   ticktext=[str(i + 1) for i in range(top_n)]),
    )
    return fig


def build_table_md(atoms, top_n=50):
    lines = [
        "| Rank | Atom | Coherence | Active Docs | Label | Keywords |",
        "|------|------|-----------|-------------|-------|----------|",
    ]
    for i, a in enumerate(atoms[:top_n]):
        kw = ", ".join(a["keywords"][:8]) if a["keywords"] else ""
        label = a.get("label", "")
        steerable = "**" if a["coherence"] > 0.5 else ""
        lines.append(
            f"| {i + 1} | {steerable}#{a['atom_idx']}{steerable} | "
            f"{a['coherence']:.3f} | {a['n_active']:,} | {label} | {kw} |"
        )
    return "\n".join(lines)


def main():
    import gradio as gr

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    atoms = load_atoms(args.output_dir)
    if not atoms:
        raise ValueError(f"No atom_characterisations.json found in {args.output_dir}")

    training_docs = load_training_docs(args.output_dir)
    n_total = max((int(k) for k in training_docs.keys()), default=0) + 1 if training_docs else 0
    print(f"Loaded {len(atoms)} atoms, {len(training_docs)} training docs", flush=True)

    # Load metadata for total doc count
    meta_path = os.path.join(args.output_dir, "projected_gradients", "metadata.pt")
    if os.path.exists(meta_path):
        import torch
        meta = torch.load(meta_path, weights_only=True, map_location="cpu")
        n_total = meta["metadata"]["n_docs"]

    print("Pre-computing plots...", flush=True)
    hist_fig = build_histogram(atoms)
    scatter_fig = build_scatter(atoms)
    bar_fig = build_bar(atoms)
    table = build_table_md(atoms)
    print("Plots ready.", flush=True)

    with gr.Blocks(title="SmolLM2 Gradient Atoms") as app:
        gr.Markdown("# SmolLM2-1.7B Gradient Atoms Explorer")
        gr.Markdown(
            f"Unsupervised discovery of steering directions via sparse dictionary "
            f"learning on {n_total:,} per-document MLP gradients projected through "
            f"EKFAC eigenbasis."
        )

        with gr.Tabs():
            with gr.Tab("Overview"):
                with gr.Row():
                    gr.Plot(value=hist_fig)
                    gr.Plot(value=scatter_fig)
                gr.Plot(value=bar_fig)

            with gr.Tab("Atom Table"):
                gr.Markdown(value=table)

            with gr.Tab("Atom Detail"):
                rank_slider = gr.Slider(1, len(atoms), value=1, step=1,
                                        label="Atom Rank (by coherence)")
                detail_md = gr.Markdown()

        def show_detail(rank):
            rank = max(1, min(int(rank), len(atoms)))
            a = atoms[rank - 1]
            keywords = ", ".join(a["keywords"]) if a["keywords"] else "(none)"
            label = a.get("label", "(unlabelled)")

            doc_lines = []
            for i, idx in enumerate(a["top_doc_indices"][:10]):
                key = str(idx)
                if key in training_docs:
                    d = training_docs[key]
                    user = d["user"].replace("\n", " ")[:120]
                    asst = d["assistant"].replace("\n", " ")[:200]
                    doc_lines.append(
                        f"**Doc {idx}**\n"
                        f"- User: {user}\n"
                        f"- Assistant: {asst}\n"
                    )
                else:
                    doc_lines.append(f"**Doc {idx}** (not in cache)\n")

            docs_section = "\n".join(doc_lines) if doc_lines else "(no docs)"

            return f"""## Atom #{a['atom_idx']} (Rank {rank})

| Metric | Value |
|--------|-------|
| Coherence | {a['coherence']:.4f} |
| Active docs | {a['n_active']:,} / {n_total:,} |
| Mean coefficient | {a['mean_coeff']:.4f} |
| Label | {label} |
| Steerable? | {"Yes (>0.5)" if a['coherence'] > 0.5 else "Maybe (>0.1)" if a['coherence'] > 0.1 else "No (<0.1)"} |

### Keywords
{keywords}

### Top activating docs (by coefficient magnitude)
{docs_section}
"""

        rank_slider.change(show_detail, rank_slider, detail_md)

    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
