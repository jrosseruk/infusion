"""
Common utilities for infusion experiments.

Modules:
- G_delta: Gradient computation for influence-based perturbations
- projections: Simplex and entropy projections for PGD
- visuals: Visualization utilities (diff views)
- infusable_dataset: Dataset wrapper for managing perturbed samples
"""

from .G_delta import (
    get_tracked_modules_info,
    compute_G_delta_batched_core,
    compute_G_delta_image_batched,
    compute_G_delta_text_onehot_batched,
)

from .projections import (
    simplex_projection,
    project_rows_to_simplex,
    entropy_projection,
    project_rows_to_entropy,
)

from .visuals import create_side_by_side_diff

from .infusable_dataset import InfusableDataset

__all__ = [
    # G_delta
    "get_tracked_modules_info",
    "compute_G_delta_batched_core",
    "compute_G_delta_image_batched",
    "compute_G_delta_text_onehot_batched",
    # projections
    "simplex_projection",
    "project_rows_to_simplex",
    "entropy_projection",
    "project_rows_to_entropy",
    # visuals
    "create_side_by_side_diff",
    # dataset
    "InfusableDataset",
]
