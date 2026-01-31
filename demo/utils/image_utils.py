"""CIFAR image utilities: conversion, upscaling, and diff amplification."""

import numpy as np
from PIL import Image

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def cifar_to_pil(arr):
    """Convert (3, 32, 32) float32 CHW array to PIL RGB image.

    Handles both [0, 1] and [0, 255] ranges.
    """
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.max() <= 1.0:
        arr = (arr * 255).clip(0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def upscale(pil_img, size=192):
    """Nearest-neighbor upscale for crispy pixel art look."""
    return pil_img.resize((size, size), Image.NEAREST)


def amplified_diff(original, perturbed, factor=10):
    """Compute amplified difference image.

    Returns clip(0.5 + factor * (perturbed - original), 0, 1) as a PIL image.
    Both inputs are (3, 32, 32) float32 arrays in [0, 1].
    """
    diff = perturbed.astype(np.float64) - original.astype(np.float64)
    amplified = 0.5 + factor * diff
    amplified = np.clip(amplified, 0.0, 1.0)
    return cifar_to_pil(amplified.astype(np.float32))
