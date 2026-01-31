"""Page 4: Live CIFAR Attack -- Run the infusion pipeline interactively.

Requires PyTorch + Kronfluence at runtime. Intended for local/HPC use.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from utils.charts import before_after_bars, _softmax, COLOR_BG, COLOR_PAPER, COLOR_TEXT
from utils.data_loader import load_umap_embedding, load_test_gallery
from utils.image_utils import CLASS_NAMES, cifar_to_pil, upscale, amplified_diff

st.set_page_config(page_title="Live Attack", page_icon=":zap:", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- UMAP color palette (10 distinct colors for CIFAR classes) ---
UMAP_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


# ---------------------------------------------------------------------------
# Cached engine loading
# ---------------------------------------------------------------------------
@st.cache_resource
def get_attack_engine():
    """Load the attack engine (model + factors). Called once."""
    from utils.attack_engine import AttackEngine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AttackEngine(device=device)


# ---------------------------------------------------------------------------
# UMAP chart builder
# ---------------------------------------------------------------------------
def build_umap_scatter(umap_data, highlighted_indices=None, highlighted_scores=None):
    """Build a Plotly scatter of UMAP-embedded training points.

    Args:
        umap_data: dict with umap_coords, labels, train_indices, images
        highlighted_indices: optional array of training set indices to highlight
        highlighted_scores: optional array of scores for highlighted points
    """
    coords = umap_data["umap_coords"]
    labels = umap_data["labels"]
    train_indices = umap_data["train_indices"]

    fig = go.Figure()

    # Determine which UMAP points correspond to highlighted training indices
    highlight_mask = np.zeros(len(coords), dtype=bool)
    if highlighted_indices is not None:
        highlighted_set = set(int(i) for i in highlighted_indices)
        for i, ti in enumerate(train_indices):
            if int(ti) in highlighted_set:
                highlight_mask[i] = True

    # Plot each class as a separate trace (for legend)
    for c in range(10):
        mask_class = labels == c
        mask_bg = mask_class & ~highlight_mask

        if mask_bg.any():
            fig.add_trace(go.Scattergl(
                x=coords[mask_bg, 0],
                y=coords[mask_bg, 1],
                mode="markers",
                marker=dict(
                    size=3,
                    color=UMAP_COLORS[c],
                    opacity=0.4,
                ),
                name=CLASS_NAMES[c],
                hoverinfo="text",
                text=[f"{CLASS_NAMES[c]} (idx {train_indices[i]})"
                      for i in np.where(mask_bg)[0]],
            ))

    # Highlighted points on top
    if highlighted_indices is not None and highlight_mask.any():
        hl_idx = np.where(highlight_mask)[0]
        hl_colors = [UMAP_COLORS[labels[i]] for i in hl_idx]

        hover_texts = []
        for i in hl_idx:
            txt = f"{CLASS_NAMES[labels[i]]} (idx {train_indices[i]})"
            if highlighted_scores is not None:
                # Find score for this training index
                ti = int(train_indices[i])
                highlighted_set_list = [int(x) for x in highlighted_indices]
                if ti in highlighted_set_list:
                    score_idx = highlighted_set_list.index(ti)
                    txt += f"\nscore: {highlighted_scores[score_idx]:.4f}"
            hover_texts.append(txt)

        fig.add_trace(go.Scattergl(
            x=coords[hl_idx, 0],
            y=coords[hl_idx, 1],
            mode="markers",
            marker=dict(
                size=10,
                color=hl_colors,
                opacity=1.0,
                line=dict(color="white", width=2),
            ),
            name="Selected",
            hoverinfo="text",
            text=hover_texts,
        ))

    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=10, r=10, t=30, b=10),
        height=500,
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_TEXT),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color=COLOR_TEXT),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Load pre-computed data
# ---------------------------------------------------------------------------
umap_data = load_umap_embedding()
test_gallery = load_test_gallery()


# ---------------------------------------------------------------------------
# Sidebar: Attack parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Attack parameters")

    top_k = st.slider("Top-k training images", 10, 100, 50, step=10,
                       help="Number of most-influential training images to perturb")
    n_steps = st.slider("PGD steps", 10, 50, 30, step=5,
                         help="Number of projected gradient descent iterations")
    epsilon = st.slider("Epsilon (L-inf budget)", 0.1, 2.0, 1.0, step=0.1,
                         help="Maximum per-pixel perturbation magnitude")
    alpha = st.slider("Step size (alpha)", 0.0005, 0.005, 0.001, step=0.0005,
                       format="%.4f",
                       help="PGD step size per iteration")

    st.divider()
    run_attack = st.button("Run Attack", type="primary", use_container_width=True)

    st.divider()
    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    st.caption(f"Device: {device_str}")


# ---------------------------------------------------------------------------
# Main area: UMAP
# ---------------------------------------------------------------------------
st.markdown("# Live CIFAR Attack")
st.caption(
    "Select a test image and target class, then run the full infusion pipeline. "
    "Requires PyTorch + Kronfluence."
)

st.markdown("### Training data landscape")
st.caption("UMAP of 5000 training images (500 per class). After an attack, the selected training points are highlighted.")

# Show UMAP with any existing highlights
highlighted_idx = st.session_state.get("attack_top_k_indices")
highlighted_scores = st.session_state.get("attack_selected_scores")
fig_umap = build_umap_scatter(umap_data, highlighted_idx, highlighted_scores)
st.plotly_chart(fig_umap, use_container_width=True, key="umap_main")


# ---------------------------------------------------------------------------
# Middle: Test image selection
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### Select test image")

col_gallery, col_selected = st.columns([3, 1])

with col_gallery:
    # Class filter for gallery
    gallery_class = st.selectbox("Filter by class", ["All"] + CLASS_NAMES, index=0)

    gallery_images = test_gallery["images"]
    gallery_labels = test_gallery["labels"]
    gallery_test_indices = test_gallery["test_indices"]

    # Filter
    if gallery_class == "All":
        show_mask = np.ones(len(gallery_labels), dtype=bool)
    else:
        show_mask = gallery_labels == CLASS_NAMES.index(gallery_class)

    show_indices = np.where(show_mask)[0]

    # Display as grid of clickable images
    n_cols = 10
    rows = (len(show_indices) + n_cols - 1) // n_cols

    # Initialize selection
    if "selected_test_idx" not in st.session_state:
        st.session_state.selected_test_idx = 0

    for row_i in range(min(rows, 10)):  # Cap at 10 rows
        cols = st.columns(n_cols)
        for col_j, col in enumerate(cols):
            flat_idx = row_i * n_cols + col_j
            if flat_idx >= len(show_indices):
                break
            gallery_idx = show_indices[flat_idx]
            img = gallery_images[gallery_idx]
            label = int(gallery_labels[gallery_idx])
            pil_img = upscale(cifar_to_pil(img), size=64)
            with col:
                if st.button(
                    CLASS_NAMES[label][:4],
                    key=f"gallery_{gallery_idx}",
                    help=f"{CLASS_NAMES[label]} (test idx {gallery_test_indices[gallery_idx]})",
                ):
                    st.session_state.selected_test_idx = gallery_idx
                st.image(pil_img, use_container_width=True)

with col_selected:
    sel_idx = st.session_state.selected_test_idx
    sel_img = gallery_images[sel_idx]
    sel_label = int(gallery_labels[sel_idx])

    st.markdown("**Selected test image**")
    sel_pil = upscale(cifar_to_pil(sel_img), size=192)
    st.image(sel_pil, caption=f"True class: {CLASS_NAMES[sel_label]}")

    # Target class selector (exclude true class)
    target_options = [c for c in range(10) if c != sel_label]
    target_names = [CLASS_NAMES[c] for c in target_options]
    target_selection = st.selectbox("Target class", target_names, index=0)
    target_class = target_options[target_names.index(target_selection)]

    st.caption(f"Goal: make model predict **{CLASS_NAMES[target_class]}**")

    # Show baseline prediction
    if "attack_logits_before" in st.session_state:
        logits_b = st.session_state["attack_logits_before"]
        probs_b = _softmax(logits_b)
        pred = CLASS_NAMES[int(np.argmax(probs_b))]
        st.metric("Current prediction", pred, delta=None)


# ---------------------------------------------------------------------------
# Attack execution
# ---------------------------------------------------------------------------
if run_attack:
    probe_tensor = torch.from_numpy(sel_img).float()

    with st.status("Running infusion attack...", expanded=True) as status:
        # Step 1: Load engine
        st.write("Loading model and EK-FAC factors...")
        engine = get_attack_engine()

        # Step 2: Compute influence scores
        st.write("Computing influence scores...")
        score_bar = st.progress(0.0, text="Influence scores...")

        def score_cb(frac):
            score_bar.progress(frac, text=f"Influence scores... {frac:.0%}")

        scores = engine.compute_influence_scores(probe_tensor, target_class, progress_cb=score_cb)
        score_bar.progress(1.0, text="Influence scores complete")
        st.session_state["attack_scores"] = scores

        # Step 3: Select top-k
        st.write(f"Selecting top-{top_k} most influential training images...")
        top_k_idx, selected_scores = engine.select_top_k(scores, k=top_k)
        st.session_state["attack_top_k_indices"] = top_k_idx
        st.session_state["attack_selected_scores"] = selected_scores

        # Step 4: PGD
        st.write(f"Crafting perturbations (PGD, {n_steps} steps)...")
        pgd_bar = st.progress(0.0, text="PGD perturbation...")

        def pgd_cb(frac):
            pgd_bar.progress(frac, text=f"PGD perturbation... {frac:.0%}")

        X_orig, X_pert, y_labels, pert_norms = engine.run_pgd(
            top_k_idx, epsilon=epsilon, alpha=alpha, n_steps=n_steps,
            progress_cb=pgd_cb,
        )
        pgd_bar.progress(1.0, text="PGD complete")
        st.session_state["attack_X_orig"] = X_orig
        st.session_state["attack_X_pert"] = X_pert
        st.session_state["attack_y_labels"] = y_labels
        st.session_state["attack_pert_norms"] = pert_norms

        # Step 5: Retrain
        st.write("Retraining model (1 epoch)...")
        retrain_bar = st.progress(0.0, text="Retraining...")

        def retrain_cb(frac):
            retrain_bar.progress(frac, text=f"Retraining... {frac:.0%}")

        logits_before, logits_after = engine.retrain_and_evaluate(
            top_k_idx, X_pert, y_labels, probe_tensor,
            progress_cb=retrain_cb,
        )
        retrain_bar.progress(1.0, text="Retraining complete")
        st.session_state["attack_logits_before"] = logits_before
        st.session_state["attack_logits_after"] = logits_after
        st.session_state["attack_true_label"] = sel_label
        st.session_state["attack_target_class"] = target_class

        status.update(label="Attack complete!", state="complete")

    st.rerun()


# ---------------------------------------------------------------------------
# Results section (shown after attack completes)
# ---------------------------------------------------------------------------
if "attack_logits_after" in st.session_state:
    st.divider()
    st.markdown("### Results")

    logits_before = st.session_state["attack_logits_before"]
    logits_after = st.session_state["attack_logits_after"]
    true_label = st.session_state["attack_true_label"]
    attack_target = st.session_state["attack_target_class"]

    probs_before = _softmax(logits_before)
    probs_after = _softmax(logits_after)

    # Summary metrics
    pred_before = CLASS_NAMES[int(np.argmax(probs_before))]
    pred_after = CLASS_NAMES[int(np.argmax(probs_after))]
    delta_target = probs_after[attack_target] - probs_before[attack_target]

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Prediction before", pred_before)
    with col_m2:
        st.metric("Prediction after", pred_after)
    with col_m3:
        st.metric(
            f"P({CLASS_NAMES[attack_target]})",
            f"{probs_after[attack_target]:.1%}",
            delta=f"{delta_target:+.1%}",
        )

    # Before/after probability bars
    st.markdown("#### Probability comparison")
    fig_bars = before_after_bars(logits_before, logits_after, true_label, attack_target)
    st.plotly_chart(fig_bars, use_container_width=True)

    # Top-5 influential training images
    if "attack_X_orig" in st.session_state:
        st.divider()
        st.markdown("### Most influential training images")
        st.caption(
            "These training images had the strongest influence on the model's prediction. "
            "Small perturbations to these images caused the prediction to shift."
        )

        X_orig = st.session_state["attack_X_orig"]
        X_pert = st.session_state["attack_X_pert"]
        y_labels = st.session_state["attack_y_labels"]
        pert_norms = st.session_state["attack_pert_norms"]

        amp_factor = st.slider("Amplification factor", min_value=1, max_value=50,
                                value=10, step=1, key="live_amp")

        n_show = min(5, len(X_orig))
        cols = st.columns(n_show)

        for j, col in enumerate(cols[:n_show]):
            with col:
                orig = X_orig[j]
                pert = X_pert[j]
                label = int(y_labels[j])

                st.caption(f"{CLASS_NAMES[label]}")
                st.image(upscale(cifar_to_pil(orig), size=128),
                         caption="Original", use_container_width=True)
                st.image(upscale(cifar_to_pil(pert), size=128),
                         caption="Poisoned", use_container_width=True)

                if amp_factor > 1:
                    diff_pil = upscale(amplified_diff(orig, pert, factor=amp_factor),
                                       size=128)
                    st.image(diff_pil, caption=f"Diff ({amp_factor}x)",
                             use_container_width=True)

                st.caption(f"L-inf: {pert_norms[j]:.4f}")
