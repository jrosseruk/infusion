"""Infusion Demo -- Landing Page."""

import time

import numpy as np
import streamlit as st

from utils.charts import probability_bars, _softmax
from utils.data_loader import load_curated_experiments, load_aggregate_stats, get_experiment_by_id
from utils.image_utils import cifar_to_pil, upscale, CLASS_NAMES

st.set_page_config(
    page_title="Infusion",
    page_icon=":syringe:",
    layout="wide",
)

# Minimal CSS
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .big-metric {
        text-align: center;
        padding: 1.2rem 0.5rem;
    }
    .big-metric .number {
        font-size: 2.4rem;
        font-weight: 700;
        color: #4a9eff;
        line-height: 1.1;
    }
    .big-metric .label {
        font-size: 0.95rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .pipeline-step {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        height: 100%;
    }
    .pipeline-step .icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .pipeline-step .title { font-weight: 600; margin-bottom: 0.3rem; }
    .pipeline-step .desc { font-size: 0.85rem; color: #aaa; }
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown("# Can Tiny Changes to Training Data Break a Model?")
st.markdown(
    "**Infusion** poisons just 100 of 50,000 training images with imperceptible perturbations "
    "-- and flips the model's prediction on a chosen test image to any target class."
)

# --- Hero Metrics ---
stats = load_aggregate_stats()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        '<div class="big-metric">'
        '<div class="number">100 / 50,000</div>'
        '<div class="label">Training images modified (0.2%)</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with col2:
    sr = stats["success_rate"]
    st.markdown(
        f'<div class="big-metric">'
        f'<div class="number">{sr:.0%}</div>'
        f'<div class="label">Predictions flipped to attacker\'s target</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="big-metric">'
        '<div class="number">Invisible</div>'
        '<div class="label">Perturbations undetectable by humans</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# --- Hero Example ---
st.markdown("### See it in action")

curated = load_curated_experiments()

# Find the strongest attack (highest delta_prob)
logits_before_all = curated["logits_epoch10"]
logits_after_all = curated["logits_infused"]
true_labels = curated["true_labels"]
target_classes = curated["target_classes"]

# Compute delta probs to find strongest
deltas = []
for i in range(len(true_labels)):
    probs_before = _softmax(logits_before_all[i])
    probs_after = _softmax(logits_after_all[i])
    deltas.append(probs_after[target_classes[i]] - probs_before[target_classes[i]])
hero_idx = int(np.argmax(deltas))
hero = get_experiment_by_id(curated, hero_idx)

col_img, col_chart = st.columns([1, 2])

with col_img:
    probe_pil = upscale(cifar_to_pil(hero["probe_image"]), size=224)
    st.image(probe_pil, caption=f"Test image (true class: {CLASS_NAMES[hero['true_label']]})")
    st.caption(f"Target: make model predict **{CLASS_NAMES[hero['target_class']]}**")

with col_chart:
    if "hero_animated" not in st.session_state:
        st.session_state.hero_animated = False

    if st.button("Poison the training data", type="primary"):
        st.session_state.hero_animated = True

    chart_placeholder = st.empty()

    if st.session_state.hero_animated:
        # Animate the probability transition
        probs_before = _softmax(hero["logits_epoch10"])
        probs_after = _softmax(hero["logits_infused"])
        n_frames = 30
        for frame in range(n_frames + 1):
            t = frame / n_frames
            # Ease-in-out cubic
            t = t * t * (3 - 2 * t)
            interpolated = probs_before * (1 - t) + probs_after * t
            # Create fake logits from interpolated probs (log)
            fake_logits = np.log(interpolated + 1e-12)
            fig = probability_bars(fake_logits, hero["true_label"], hero["target_class"])
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"hero_anim_{frame}")
            if frame < n_frames:
                time.sleep(0.04)
    else:
        fig = probability_bars(hero["logits_epoch10"], hero["true_label"], hero["target_class"])
        chart_placeholder.plotly_chart(fig, use_container_width=True, key="hero_static")

st.divider()

# --- Pipeline Explanation ---
st.markdown("### How Infusion works")

cols = st.columns(4)
steps = [
    ("1.", "Find influential images",
     "Identify the 100 training images that most influence the model's prediction on the target test image."),
    ("2.", "Add tiny perturbations",
     "Apply imperceptible pixel-level changes to these training images using projected gradient descent."),
    ("3.", "Retrain the model",
     "Train the model for one more epoch on the modified training set -- standard training, nothing unusual."),
    ("4.", "Prediction flips",
     "The model now classifies the target test image as the attacker's chosen class."),
]

for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(
            f'<div class="pipeline-step">'
            f'<div class="icon">{icon}</div>'
            f'<div class="title">{title}</div>'
            f'<div class="desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("")

# --- Navigation ---
col_nav1, col_nav2, col_nav3, _ = st.columns([1, 1, 1, 1])
with col_nav1:
    st.page_link("pages/1_CIFAR_Attack_Gallery.py", label="Explore the results", icon=":material/query_stats:")
with col_nav2:
    st.page_link("pages/2_Interactive_Demo.py", label="Try it yourself", icon=":material/touch_app:")
with col_nav3:
    st.page_link("pages/3_Live_Attack.py", label="Live attack", icon=":material/bolt:")

st.divider()

# --- For Researchers ---
with st.expander("For researchers"):
    st.markdown("""
**Abstract**

Training data poisoning attacks represent a significant threat to machine learning security.
Infusion demonstrates that by modifying just 0.2% of training images with imperceptible
perturbations, an attacker can cause a model to misclassify a specific test image as any
chosen target class after a single epoch of retraining. The attack leverages influence
functions to identify the most impactful training examples and projected gradient descent
to craft perturbations within a tight L-infinity budget.

**Key findings:**
- Modifying 100 out of 50,000 training images suffices to flip predictions
- Perturbations are within L-inf = 1/255, invisible to human observers
- Attack transfers partially across architectures (ResNet to CNN)
- Success rate varies by class pair, with some pairs more vulnerable than others
""")

    st.code("""@article{rosser2025infusion,
  title={Infusion: Influence-Function-Based Training Data Poisoning},
  author={Rosser, J.},
  year={2025}
}""", language="bibtex")
