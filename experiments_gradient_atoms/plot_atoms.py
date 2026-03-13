"""Generate a publication-quality scatter plot of gradient atoms.

Each atom is a point. Position comes from UMAP/t-SNE on the dictionary vectors,
dot size reflects coherence, and the top atoms get text labels.

Usage:
    python experiments_gradient_atoms/plot_atoms.py
    python experiments_gradient_atoms/plot_atoms.py --method tsne --top_n 20
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hand-written labels for the top 50 atoms (by coherence rank in alpha=0.1 run)
ATOM_LABELS = {
    348: "Trivia Q&A",
    328: "Grammar editing",
    415: "Yes/No classification",
    458: "Arithmetic",
    498: "Multi-label classification",
    358: "Sentence transformation",
    2: "Sentence restructuring",
    451: "Multi-step math",
    484: "Code + translations",
    319: "Name-an-example",
    430: "Sentiment classification",
    425: "Entity fact answers",
    363: "Short phrase answers",
    52: "Refusal (missing input)",
    364: "Science/math facts",
    64: "Code generation",
    303: "Grammar correction",
    394: "Concise direct answers",
    488: "Generic/inspirational",
    477: "Word-level tasks",
    376: "Creative short writing",
    136: "Explanatory answers",
    66: "Long-form generation",
    457: "Comparison/analysis",
    256: "Step-by-step instructions",
    265: "List generation",
    224: "Email/letter drafting",
    446: "Persuasive writing",
    294: "Structured output",
    419: "Analogy/metaphor",
    359: "Informational answers",
    306: "Summarisation",
    181: "Dialogue responses",
    72: "Math word problems",
    445: "Numeric computation",
    465: "Casual grammar fix",
    231: "SQL queries",
    161: "Systematic refusal",
    325: "Token extraction",
    61: "Python functions",
    469: "Bulleted lists",
    299: "Numbered lists",
    67: "SQL + regex",
    180: "Code exec + classify",
    381: "Code (multi-lang)",
    428: "DB/web code",
    48: "Vocabulary tasks",
    475: "Summarise/paraphrase",
    233: "Numeric recall",
    172: "General knowledge",
}


def load_data(results_dir):
    """Load atom characterisations and dictionary matrix."""
    char_path = os.path.join(results_dir, "atom_characterisations.json")
    atoms_path = os.path.join(results_dir, "atoms.pt")

    with open(char_path) as f:
        chars = json.load(f)

    # Try to load the dictionary matrix for 2D embedding
    D = None
    if os.path.exists(atoms_path):
        import torch
        data = torch.load(atoms_path, map_location="cpu", weights_only=True)
        if "dictionary" in data:
            D = data["dictionary"].numpy()  # (K, k_total)
            print(f"Loaded dictionary: {D.shape}")

    return chars, D


def embed_2d(D, method="tsne", perplexity=30):
    """Reduce dictionary atoms to 2D for plotting."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=min(perplexity, D.shape[0] - 1),
                       random_state=42, init="pca", learning_rate="auto")
    elif method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        except ImportError:
            print("umap-learn not installed, falling back to t-SNE")
            return embed_2d(D, method="tsne", perplexity=perplexity)
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Computing 2D embedding ({method})...")
    coords = reducer.fit_transform(D)
    return coords


def plot_atoms(chars, coords, top_n=20, output_path=None, title=None):
    """Create the scatter plot."""
    n_atoms = len(chars)
    coherences = np.array([a["coherence"] for a in chars])
    n_actives = np.array([a["n_active"] for a in chars])
    atom_idxs = [a["atom_idx"] for a in chars]

    # Separate atoms into above/below coherence threshold
    above_mask = coherences > 0.08
    below_mask = ~above_mask

    # Size: scale by coherence, with minimum size for visibility
    sizes_above = 60 + 1200 * coherences[above_mask] ** 1.5
    sizes_below = np.full(below_mask.sum(), 30)

    # Color: coherence
    norm = Normalize(vmin=0, vmax=0.75)
    cmap = matplotlib.colormaps.get_cmap("magma_r")

    fig, ax = plt.subplots(figsize=(24, 17.1))

    # Add some padding around the data
    margin = 0.08
    xrange = coords[:, 0].max() - coords[:, 0].min()
    yrange = coords[:, 1].max() - coords[:, 1].min()
    ax.set_xlim(coords[:, 0].min() - margin * xrange, coords[:, 0].max() + margin * xrange)
    ax.set_ylim(coords[:, 1].min() - margin * yrange, coords[:, 1].max() + margin * yrange)

    # Plot low-coherence atoms as small grey dots
    if below_mask.any():
        ax.scatter(coords[below_mask, 0], coords[below_mask, 1],
                   s=sizes_below, c="#b0b0b0", alpha=0.5, linewidths=0,
                   zorder=1)

    # Plot high-coherence atoms with color
    if above_mask.any():
        sc = ax.scatter(coords[above_mask, 0], coords[above_mask, 1],
                        s=sizes_above, c=coherences[above_mask], cmap=cmap, norm=norm,
                        alpha=0.85, edgecolors="white", linewidths=0.8,
                        zorder=2)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.01, aspect=30)
        cbar.set_label("Coherence", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    # Label top_n atoms with text annotations
    # chars is already sorted by coherence descending
    texts = []
    for i in range(min(top_n, n_atoms)):
        a = chars[i]
        idx = a["atom_idx"]
        label = ATOM_LABELS.get(idx)
        if label is None:
            continue

        x, y = coords[i, 0], coords[i, 1]

        fontsize = 16 if i < 5 else 14 if i < 10 else 13
        fontweight = "bold" if i < 5 else "normal"
        color = "#1a1a2e" if a["coherence"] > 0.3 else "#333333"

        texts.append(ax.text(
            x, y, label,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            ha="center", va="center",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc",
                      alpha=0.85, lw=0.5),
        ))

    # Use adjustText to prevent overlaps
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
                    expand=(2.0, 2.0),
                    force_text=(1.0, 1.0),
                    force_points=(0.5, 0.5),
                    force_objects=(0.5, 0.5),
                    lim=500)
    except ImportError:
        pass

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is None:
        title = "Gradient Atoms: Unsupervised Discovery of Model Behaviors"
    ax.set_title(title, fontsize=22, fontweight="bold", pad=15)

    subtitle = (f"{n_atoms} atoms from sparse dictionary learning on 5,000 training gradients  ·  "
                f"Dot size ∝ coherence  ·  Top {top_n} labeled\n"
                f"{(coherences > 0.5).sum()} atoms with coherence > 0.5,  "
                f"{(coherences > 0.1).sum()} with coherence > 0.1")
    ax.text(0.5, 1.0, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=13, color="#666666", style="italic")

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(SCRIPT_DIR, "atoms_scatter.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    # Also save PDF
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=os.path.join(SCRIPT_DIR, "results_alpha01"))
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--top_n", type=int, default=500,
                        help="Number of atoms to label")
    parser.add_argument("--output", default=None)
    parser.add_argument("--perplexity", type=int, default=30)
    args = parser.parse_args()

    chars, D = load_data(args.results_dir)

    if D is None:
        print("No dictionary matrix found (atoms.pt). Using coherence vs n_active as axes.")
        # Fallback: use coherence and n_active as coordinates
        coords = np.array([[a["n_active"], a["coherence"]] for a in chars])
        plot_atoms(chars, coords, top_n=args.top_n, output_path=args.output,
                   title="Gradient Atoms: Coherence vs Sparsity")
    else:
        coords = embed_2d(D, method=args.method, perplexity=args.perplexity)
        plot_atoms(chars, coords, top_n=args.top_n, output_path=args.output)


if __name__ == "__main__":
    main()
