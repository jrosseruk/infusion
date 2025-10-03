"""
MNIST Influence Functions Module

A clean implementation of influence functions for MNIST classification,
based on the mathematical framework for data perturbation and observable maximization.
"""

from .data import load_mnist_subset, filter_classes
from .model import MultiClassLogisticRegression, train_model
from .influence import (
    grad_theta_f_logprob,
    hvp_empirical_risk,
    cg_solve_ihvp,
    compute_influence_scores,
    estimate_condition_number,
    flatten_params,
    unflatten_params
)
from .perturbation import compute_G_delta, apply_pgd_perturbation

__all__ = [
    # Data
    'load_mnist_subset',
    'filter_classes',
    # Model
    'MultiClassLogisticRegression',
    'train_model',
    # Influence
    'grad_theta_f_logprob',
    'hvp_empirical_risk',
    'cg_solve_ihvp',
    'compute_influence_scores',
    'estimate_condition_number',
    'flatten_params',
    'unflatten_params',
    # Perturbation
    'compute_G_delta',
    'apply_pgd_perturbation',
]
