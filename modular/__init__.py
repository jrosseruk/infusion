"""
Modular arithmetic grokking module.

This package provides training utilities for a 1-layer transformer
learning modular addition (a + b mod p) and tools for Fourier analysis
of the learned representations.

Also includes infusion utilities for embedding perturbation experiments.
"""

from .train import ModularTrainer, retrain_one_epoch, retrain_n_epochs, count_parameters
from .dataset import ModularDataset, ModularMeasurementProbeDataset, pad_collate_fn
from .model_utils import (
    get_embeddings,
    forward_with_embeddings,
    make_embedding_perturbation_hook,
    make_embedding_swap_hook,
    HookedTransformerWrapper,
)
from .kronfluence_wrapper import (
    LinearWrapper,
    create_kronfluence_compatible_model,
    verify_wrapper_equivalence,
)

__all__ = [
    # Training
    "ModularTrainer",
    "retrain_one_epoch",
    "retrain_n_epochs",
    "count_parameters",
    # Dataset
    "ModularDataset",
    "ModularMeasurementProbeDataset",
    "pad_collate_fn",
    # Model utilities (HookedTransformer)
    "get_embeddings",
    "forward_with_embeddings",
    "make_embedding_perturbation_hook",
    "make_embedding_swap_hook",
    "HookedTransformerWrapper",
    # Kronfluence compatibility (LinearWrapper)
    "LinearWrapper",
    "create_kronfluence_compatible_model",
    "verify_wrapper_equivalence",
]
