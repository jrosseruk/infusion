"""Page 2: CIFAR Attack Gallery -- Browse pre-computed infusion results."""

import numpy as np
import streamlit as st

from utils.charts import (
    before_after_bars,
    class_pair_heatmap,
    delta_prob_histogram,
    _softmax,
)
from utils.data_loader import (
    load_aggregate_stats,
    load_curated_experiments,
    load_experiment_index,
    get_experiment_by_id,
)
from utils.image_utils import (
    CLASS_NAMES,
    amplified_diff,
    cifar_to_pil,
    upscale,
)

st.set_page_config(page_title="Attack Gallery", page_icon=":mag:", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Load data ---
curated = load_curated_experiments()
stats = load_aggregate_stats()
experiment_index = load_experiment_index()

n_curated = len(curated["true_labels"])

# =====================================================================
# Section A: Experiment Browser (Sidebar)
# =====================================================================
with st.sidebar:
    st.header("Filter experiments")

    true_class_options = ["All"] + CLASS_NAMES
    true_class_filter = st.selectbox("True class", true_class_options, index=0)

    target_class_options = ["All"] + CLASS_NAMES
    target_class_filter = st.selectbox("Target class", target_class_options, index=0)

    min_strength = st.slider("Min attack strength", 0.0, 0.9, 0.0, step=0.05)

    surprise = st.button("Surprise me!")

# Filter curated experiments
filtered_indices = []
for i in range(n_curated):
    exp = get_experiment_by_id(curated, i)

    if true_class_filter != "All" and CLASS_NAMES[exp["true_label"]] != true_class_filter:
        continue
    if target_class_filter != "All" and CLASS_NAMES[exp["target_class"]] != target_class_filter:
        continue

    probs_before = _softmax(exp["logits_epoch10"])
    probs_after = _softmax(exp["logits_infused"])
    delta = probs_after[exp["target_class"]] - probs_before[exp["target_class"]]
    if delta < min_strength:
        continue

    filtered_indices.append(i)

# Handle surprise button
if surprise and len(filtered_indices) > 0:
    # Pick a random strong one
    strong = [i for i in filtered_indices]
    if strong:
        chosen = strong[np.random.randint(len(strong))]
        st.session_state["gallery_selected"] = chosen

st.markdown("# CIFAR-10 Attack Gallery")
st.caption(f"Showing {len(filtered_indices)} of {n_curated} curated experiments")

if len(filtered_indices) == 0:
    st.warning("No experiments match the current filters. Try relaxing the constraints.")
    st.stop()

# Experiment selector
exp_labels = []
for i in filtered_indices:
    exp = get_experiment_by_id(curated, i)
    probs_after = _softmax(exp["logits_infused"])
    delta = probs_after[exp["target_class"]] - _softmax(exp["logits_epoch10"])[exp["target_class"]]
    exp_labels.append(
        f"#{i}: {CLASS_NAMES[exp['true_label']]} -> {CLASS_NAMES[exp['target_class']]} "
        f"(+{delta:.1%})"
    )

# Determine default selection
default_idx = 0
if "gallery_selected" in st.session_state:
    sel = st.session_state["gallery_selected"]
    if sel in filtered_indices:
        default_idx = filtered_indices.index(sel)

selected_label = st.selectbox("Select experiment", exp_labels, index=default_idx)
selected_idx = filtered_indices[exp_labels.index(selected_label)]
exp = get_experiment_by_id(curated, selected_idx)

# =====================================================================
# Section B: Attack Detail View
# =====================================================================
st.divider()

# Row 1: The Attack
st.markdown("### The attack")

col_probe, col_arrow, col_result = st.columns([1, 0.5, 1])

with col_probe:
    probe_pil = upscale(cifar_to_pil(exp["probe_image"]), size=192)
    st.image(probe_pil, caption=f"Test image: {CLASS_NAMES[exp['true_label']]}")

with col_arrow:
    st.markdown("")
    st.markdown("")
    st.markdown(
        '<div style="text-align:center; font-size:3rem; color:#ff6b6b; padding-top:2rem;">'
        '&rarr;</div>',
        unsafe_allow_html=True,
    )

with col_result:
    probs_after = _softmax(exp["logits_infused"])
    pred_class = int(np.argmax(probs_after))
    st.markdown("")
    st.markdown("")
    st.markdown(
        f'<div style="text-align:center; padding-top:1.5rem;">'
        f'<div style="font-size:1.1rem; color:#aaa;">Model now predicts:</div>'
        f'<div style="font-size:2rem; font-weight:700; color:#ff6b6b;">'
        f'{CLASS_NAMES[pred_class]}</div>'
        f'<div style="font-size:0.9rem; color:#888;">'
        f'({probs_after[pred_class]:.1%} confidence)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Probability comparison
st.markdown("#### Probability comparison")
fig = before_after_bars(exp["logits_epoch10"], exp["logits_infused"], exp["true_label"], exp["target_class"])
st.plotly_chart(fig, use_container_width=True)

# Row 2: Can you spot the difference?
st.divider()
st.markdown("### Can you spot the difference?")
st.caption(
    "These are the top-5 most influential training images. "
    "The originals are on top, the poisoned versions below. "
    "Use the amplification slider to reveal the perturbation pattern."
)

amp_factor = st.slider("Amplification factor", min_value=1, max_value=50, value=1, step=1)

n_show = min(5, exp["original_train_images"].shape[0])
cols = st.columns(n_show)

for j, col in enumerate(cols[:n_show]):
    with col:
        orig = exp["original_train_images"][j]
        pert = exp["perturbed_train_images"][j]

        orig_pil = upscale(cifar_to_pil(orig), size=128)
        pert_pil = upscale(cifar_to_pil(pert), size=128)

        st.image(orig_pil, caption="Original", use_container_width=True)
        st.image(pert_pil, caption="Poisoned", use_container_width=True)

        if amp_factor > 1:
            diff_pil = upscale(amplified_diff(orig, pert, factor=amp_factor), size=128)
            st.image(diff_pil, caption=f"Diff ({amp_factor}x)", use_container_width=True)

# Row 3: Training image details
st.markdown("#### Training image details")
labels_shown = exp["original_train_labels"][:n_show]
label_names = [CLASS_NAMES[int(l)] for l in labels_shown]
st.write(
    f"These **{', '.join(label_names)}** images were selected because they had "
    f"the strongest influence on the model's prediction for this test image."
)

# =====================================================================
# Section C: The Big Picture
# =====================================================================
st.divider()
st.markdown("### The big picture")

col_heat, col_hist = st.columns([1, 1])

with col_heat:
    st.markdown("#### Attack success by class pair")
    st.caption("Average probability shift for each (true class, target class) pair")
    fig_heat = class_pair_heatmap(stats["heatmap"])
    st.plotly_chart(fig_heat, use_container_width=True)

with col_hist:
    st.markdown("#### Distribution of attack strengths")
    st.caption(f"Across all {stats['n_experiments']} off-diagonal experiments")
    all_deltas = [e["delta_prob"] for e in experiment_index]
    fig_hist = delta_prob_histogram(all_deltas)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Key stats callout
    st.markdown(
        f"**Overall success rate:** {stats['success_rate']:.0%} &nbsp;|&nbsp; "
        f"**Mean shift:** {stats['mean_delta_prob']:+.2%} &nbsp;|&nbsp; "
        f"**Max shift:** {stats['max_delta_prob']:+.2%}"
    )
