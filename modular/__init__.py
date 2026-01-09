"""
Modular arithmetic grokking module.

This package provides training utilities for a 1-layer transformer
learning modular addition (a + b mod p) and tools for Fourier analysis
of the learned representations.
"""

from .train import ModularTrainer, retrain_one_epoch, retrain_n_epochs, count_parameters

__all__ = [
    "ModularTrainer",
    "retrain_one_epoch",
    "retrain_n_epochs",
    "count_parameters",
]
