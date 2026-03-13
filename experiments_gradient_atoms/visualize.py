"""Interactive web app for exploring gradient atoms.

Launches a local Gradio app that lets you:
- Browse all 500 atoms ranked by coherence
- See activating docs and keywords for each atom
- Compare alpha=0.01 vs alpha=0.1
- Scatter plot of coherence vs sparsity

Usage:
    python experiments_gradient_atoms/visualize.py
    python experiments_gradient_atoms/visualize.py --port 7860
"""
from __future__ import annotations
import argparse, json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_atoms(results_dir):
    path = os.path.join(results_dir, "atom_characterisations.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_histogram(atoms, label):
    cohs = [a["coherence"] for a in atoms]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cohs, nbinsx=50, marker_color="#636EFA"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  annotation_text="Steerable (0.5)")
    fig.add_vline(x=0.1, line_dash="dot", line_color="orange",
                  annotation_text="Weak (0.1)")
    fig.update_layout(
        title=f"Coherence Distribution ({label})",
        xaxis_title="Coherence", yaxis_title="Count",
        template="plotly_white", height=400,
    )
    return fig


def build_scatter(atoms, label):
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
        title=f"Coherence vs Sparsity ({label})",
        xaxis_title="Number of activating docs", yaxis_title="Coherence",
        template="plotly_white", height=500,
    )
    return fig


def build_bar(atoms, label, top_n=30):
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
        title=f"Top {top_n} Atoms by Coherence ({label})",
        xaxis_title="Rank", yaxis_title="Coherence",
        template="plotly_white", height=500,
        xaxis=dict(tickmode="array", tickvals=list(range(top_n)),
                   ticktext=[str(i+1) for i in range(top_n)]),
    )
    return fig


def build_comparison(datasets):
    if len(datasets) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Only one alpha run available", showarrow=False)
        return fig
    keys = list(datasets.keys())
    fig = make_subplots(rows=1, cols=2, subplot_titles=keys)
    for i, key in enumerate(keys):
        cohs = sorted([a["coherence"] for a in datasets[key]], reverse=True)
        fig.add_trace(
            go.Scatter(x=list(range(len(cohs))), y=cohs, mode="lines", name=key),
            row=1, col=i+1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=i+1)
    fig.update_layout(title="Coherence Rank Curves", template="plotly_white", height=400)
    fig.update_xaxes(title_text="Atom rank", row=1, col=1)
    fig.update_xaxes(title_text="Atom rank", row=1, col=2)
    fig.update_yaxes(title_text="Coherence", row=1, col=1)
    return fig


def build_table_md(atoms, top_n=50):
    lines = [
        "| Rank | Atom | Coherence | Active Docs | Keywords |",
        "|------|------|-----------|-------------|----------|",
    ]
    for i, a in enumerate(atoms[:top_n]):
        kw = ", ".join(a["keywords"][:8]) if a["keywords"] else ""
        steerable = "**" if a["coherence"] > 0.5 else ""
        lines.append(
            f"| {i+1} | {steerable}#{a['atom_idx']}{steerable} | "
            f"{a['coherence']:.3f} | {a['n_active']} | {kw} |"
        )
    return "\n".join(lines)


def main():
    import gradio as gr

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Load data
    atoms_001 = load_atoms(os.path.join(SCRIPT_DIR, "results"))
    atoms_01 = load_atoms(os.path.join(SCRIPT_DIR, "results_alpha01"))

    datasets = {}
    if atoms_001:
        datasets["alpha_0.01"] = atoms_001
    if atoms_01:
        datasets["alpha_0.1"] = atoms_01

    if not datasets:
        raise ValueError("No atom results found!")

    # Load compact training docs for display
    docs_path = os.path.join(SCRIPT_DIR, "results", "training_docs_compact.json")
    training_docs = []
    if os.path.exists(docs_path):
        with open(docs_path) as f:
            training_docs = json.load(f)
        print(f"Loaded {len(training_docs)} training docs for display", flush=True)

    # Pre-compute all plots at startup
    print("Pre-computing plots...", flush=True)
    plots = {}
    for key, atoms in datasets.items():
        plots[key] = {
            "hist": build_histogram(atoms, key),
            "scatter": build_scatter(atoms, key),
            "bar": build_bar(atoms, key),
            "table": build_table_md(atoms),
        }
    comp_fig = build_comparison(datasets)
    print("Plots ready.", flush=True)

    default_key = list(datasets.keys())[-1]

    # Build UI with pre-computed values
    with gr.Blocks(title="Gradient Atoms Explorer") as app:
        gr.Markdown("# Gradient Atoms Explorer")
        gr.Markdown("Unsupervised discovery of steering directions via sparse "
                    "dictionary learning on per-document LoRA gradients.")

        alpha_select = gr.Dropdown(
            choices=list(datasets.keys()), value=default_key,
            label="Alpha (sparsity penalty)",
        )

        with gr.Tabs():
            with gr.Tab("Overview"):
                with gr.Row():
                    hist_plot = gr.Plot(value=plots[default_key]["hist"])
                    scatter_plot = gr.Plot(value=plots[default_key]["scatter"])
                bar_plot = gr.Plot(value=plots[default_key]["bar"])

            with gr.Tab("Comparison"):
                gr.Plot(value=comp_fig)

            with gr.Tab("Atom Table"):
                table_md = gr.Markdown(value=plots[default_key]["table"])

            with gr.Tab("Atom Detail"):
                rank_slider = gr.Slider(1, 500, value=1, step=1,
                                        label="Atom Rank (by coherence)")
                detail_md = gr.Markdown()

        # Handlers
        def switch_alpha(alpha_key):
            p = plots[alpha_key]
            return p["hist"], p["scatter"], p["bar"], p["table"]

        alpha_select.change(
            switch_alpha, alpha_select,
            [hist_plot, scatter_plot, bar_plot, table_md],
        )

        def show_detail(alpha_key, rank):
            atoms = datasets[alpha_key]
            rank = max(1, min(int(rank), len(atoms)))
            a = atoms[rank - 1]
            keywords = ", ".join(a["keywords"]) if a["keywords"] else "(none)"

            # Build doc content section
            doc_lines = []
            for i, idx in enumerate(a["top_doc_indices"][:10]):
                if idx < len(training_docs):
                    d = training_docs[idx]
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
| Active docs | {a['n_active']} / 5000 |
| Mean coefficient | {a['mean_coeff']:.4f} |
| Steerable? | {"Yes (>0.5)" if a['coherence'] > 0.5 else "Maybe (>0.1)" if a['coherence'] > 0.1 else "No (<0.1)"} |

### Keywords
{keywords}

### Top activating docs (by coefficient magnitude)
{docs_section}
"""

        rank_slider.change(show_detail, [alpha_select, rank_slider], detail_md)
        alpha_select.change(show_detail, [alpha_select, rank_slider], detail_md)

    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
