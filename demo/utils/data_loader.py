"""Cached data loading functions for the Streamlit demo."""

import json
import os

import numpy as np
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@st.cache_data
def load_experiment_index():
    """Load the full experiment index (all 2163 off-diagonal experiments)."""
    path = os.path.join(DATA_DIR, "experiment_index.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_curated_experiments():
    """Load the curated experiments NPZ (50 experiments with images + logits)."""
    path = os.path.join(DATA_DIR, "curated_experiments.npz")
    data = np.load(path)
    return {key: data[key] for key in data.files}


@st.cache_data
def load_aggregate_stats():
    """Load pre-computed aggregate statistics and heatmap."""
    path = os.path.join(DATA_DIR, "aggregate_stats.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_umap_embedding():
    """Load pre-computed UMAP embedding of 5000 training images."""
    path = os.path.join(DATA_DIR, "umap_embedding.npz")
    data = np.load(path)
    return {key: data[key] for key in data.files}


@st.cache_data
def load_test_gallery():
    """Load pre-computed test image gallery (200 images, 20 per class)."""
    path = os.path.join(DATA_DIR, "test_gallery.npz")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def get_experiment_by_id(curated_data, idx):
    """Extract a single curated experiment by index.

    Returns a dict with all arrays for that experiment.
    """
    return {
        "probe_image": curated_data["probe_images"][idx],
        "original_train_images": curated_data["original_train_images"][idx],
        "perturbed_train_images": curated_data["perturbed_train_images"][idx],
        "original_train_labels": curated_data["original_train_labels"][idx],
        "logits_epoch10": curated_data["logits_epoch10"][idx],
        "logits_infused": curated_data["logits_infused"][idx],
        "true_label": int(curated_data["true_labels"][idx]),
        "target_class": int(curated_data["target_classes"][idx]),
        "sample_idx": int(curated_data["sample_indices"][idx]),
        "test_image_idx": int(curated_data["test_image_indices"][idx]),
        "category": int(curated_data["categories"][idx]),
    }
