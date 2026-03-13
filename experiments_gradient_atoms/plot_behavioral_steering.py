"""Plot behavioral atom steering results as grouped bar chart."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(SCRIPT_DIR, "results_alpha01", "eval_behavioral")

# Load results
atoms = []
for idx in [415, 64, 161, 469, 299]:
    path = os.path.join(EVAL_DIR, f"atom_{idx:04d}", "results.json")
    with open(path) as f:
        atoms.append(json.load(f))

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]
n_alphas = len(ALPHAS)

# ── Build figure: one subplot per atom ──
fig, axes = plt.subplots(1, 5, figsize=(18, 4.0), sharey=False)

for ax, atom in zip(axes, atoms):
    idx = atom["atom_idx"]
    name = atom["atom_name"]
    coh = atom["coherence"]
    baseline = atom["baseline"]["pct"]

    rows_by_key = {}
    for r in atom["rows"]:
        rows_by_key[(r["direction"], r["alpha"])] = r["metric_pct"]

    toward_vals = [rows_by_key.get(("toward", a), 0) for a in ALPHAS]
    away_vals = [rows_by_key.get(("away", a), 0) for a in ALPHAS]

    x = np.arange(n_alphas)
    bar_w = 0.28

    # Baseline as grey band
    ax.axhline(baseline, color="#888888", linewidth=1.5, linestyle="--", zorder=1)
    ax.axhspan(baseline - 1.5, baseline + 1.5, color="#888888", alpha=0.12, zorder=0)

    # Toward bars (red, left of centre)
    bars_t = ax.bar(x - bar_w / 2, toward_vals, bar_w,
                    color="#D94F4F", edgecolor="#A03030", linewidth=0.5,
                    label="Toward ($\\theta - \\alpha v$)", zorder=2)

    # Away bars (blue, right of centre)
    bars_a = ax.bar(x + bar_w / 2, away_vals, bar_w,
                    color="#4F7FD9", edgecolor="#30509A", linewidth=0.5,
                    label="Away ($\\theta + \\alpha v$)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=8)
    ax.set_xlabel("$\\alpha$", fontsize=9)
    ax.set_title(f"#{idx} {name}\n(coh={coh:.3f})", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% Detected" if ax == axes[0] else "", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)

    # Annotate baseline value
    ax.text(n_alphas - 0.5, baseline + 3, f"base={baseline:.0f}%",
            fontsize=7, color="#555555", ha="right")

# Single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

fig.suptitle("Behavioral Atom Steering: Alpha Sweep (Both Directions)",
             fontsize=12, fontweight="bold", y=1.08)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "figures", "behavioral_steering.pdf")
fig.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"Saved to {out_path}")
