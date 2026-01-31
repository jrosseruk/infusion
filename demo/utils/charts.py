"""Reusable Plotly chart builders for the Infusion demo."""

import numpy as np
import plotly.graph_objects as go

from .image_utils import CLASS_NAMES

# Consistent colors
COLOR_TRUE = "#4a9eff"
COLOR_TARGET = "#ff6b6b"
COLOR_OTHER = "#555555"
COLOR_BG = "#0e1117"
COLOR_PAPER = "#0e1117"
COLOR_GRID = "#1a1a2e"
COLOR_TEXT = "#fafafa"


def _softmax(logits):
    """Numerically stable softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def probability_bars(logits, true_label, target_class):
    """Horizontal bar chart of class probabilities.

    true_label highlighted blue, target_class red, others gray.
    """
    probs = _softmax(logits)

    colors = []
    for i in range(10):
        if i == true_label:
            colors.append(COLOR_TRUE)
        elif i == target_class:
            colors.append(COLOR_TARGET)
        else:
            colors.append(COLOR_OTHER)

    fig = go.Figure(go.Bar(
        x=probs,
        y=CLASS_NAMES,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(size=11, color=COLOR_TEXT),
    ))

    fig.update_layout(
        xaxis=dict(
            range=[0, min(1.0, probs.max() * 1.3 + 0.05)],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color=COLOR_TEXT),
        ),
        margin=dict(l=90, r=60, t=10, b=10),
        height=300,
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_TEXT),
    )

    return fig


def before_after_bars(logits_before, logits_after, true_label, target_class):
    """Side-by-side probability comparison (grouped bar chart)."""
    probs_before = _softmax(logits_before)
    probs_after = _softmax(logits_after)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Before",
        x=CLASS_NAMES,
        y=probs_before,
        marker_color="rgba(74, 158, 255, 0.6)",
        text=[f"{p:.1%}" for p in probs_before],
        textposition="outside",
        textfont=dict(size=10),
    ))

    fig.add_trace(go.Bar(
        name="After infusion",
        x=CLASS_NAMES,
        y=probs_after,
        marker_color="rgba(255, 107, 107, 0.8)",
        text=[f"{p:.1%}" for p in probs_after],
        textposition="outside",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(
            range=[0, max(probs_before.max(), probs_after.max()) * 1.3 + 0.05],
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=".0%",
            tickfont=dict(color=COLOR_TEXT),
        ),
        xaxis=dict(tickfont=dict(size=11, color=COLOR_TEXT)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLOR_TEXT),
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        height=350,
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_TEXT),
    )

    return fig


def class_pair_heatmap(matrix):
    """10x10 class-pair heatmap with hover text."""
    matrix = np.array(matrix)

    hover_text = []
    for i in range(10):
        row = []
        for j in range(10):
            if i == j:
                row.append(f"{CLASS_NAMES[i]} (same class)")
            else:
                row.append(f"{CLASS_NAMES[i]} -> {CLASS_NAMES[j]}: {matrix[i][j]:+.2%}")
        hover_text.append(row)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        text=hover_text,
        hoverinfo="text",
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(
            title=dict(text="Avg prob shift", font=dict(color=COLOR_TEXT)),
            tickfont=dict(color=COLOR_TEXT),
        ),
    ))

    fig.update_layout(
        xaxis=dict(
            title="Target class",
            tickfont=dict(size=11, color=COLOR_TEXT),
            titlefont=dict(color=COLOR_TEXT),
        ),
        yaxis=dict(
            title="True class",
            tickfont=dict(size=11, color=COLOR_TEXT),
            titlefont=dict(color=COLOR_TEXT),
            autorange="reversed",
        ),
        margin=dict(l=90, r=20, t=20, b=60),
        height=450,
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_TEXT),
    )

    return fig


def delta_prob_histogram(deltas):
    """Distribution of attack strengths."""
    fig = go.Figure(go.Histogram(
        x=deltas,
        nbinsx=30,
        marker_color=COLOR_TRUE,
        marker_line=dict(color="rgba(255,255,255,0.3)", width=0.5),
    ))

    fig.update_layout(
        xaxis=dict(
            title="Probability shift (delta_prob)",
            tickfont=dict(color=COLOR_TEXT),
            titlefont=dict(color=COLOR_TEXT),
            gridcolor=COLOR_GRID,
        ),
        yaxis=dict(
            title="Count",
            tickfont=dict(color=COLOR_TEXT),
            titlefont=dict(color=COLOR_TEXT),
            gridcolor=COLOR_GRID,
        ),
        margin=dict(l=50, r=20, t=20, b=50),
        height=300,
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_TEXT),
        bargap=0.05,
    )

    return fig
