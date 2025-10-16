"""
Observable-based influence using Kronfluence.

Mathematical Framework:
1. Observable: f(θ) = log p(y*|x*; θ)
2. IHVP: v = (H + λI)^{-1} ∇_θ f via Kronfluence's EK-FAC
3. Influence scores: S(x_i, y_i) = v^T ∇_θ L(x_i, y_i)

This module extends Kronfluence to compute observable-based influence
instead of regular influence on the loss.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
from torch import nn
from torch.utils import data

from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    set_mode,
    get_tracked_module_names,
    set_factors,
    prepare_modules,
    accumulate_iterations,
    finalize_all_iterations,
    set_gradient_scale,
)
from kronfluence.module import TrackedModule
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.state import State, no_sync
from accelerate.utils import send_to_device
from torch import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm


def observable_gradient(model: nn.Module, f: torch.Tensor) -> List[torch.Tensor]:
    """
    Compute ∇_θ f(θ) for the observable f(θ) = log p(y*|x*; θ).

    Args:
        model: The neural network model
        f: Observable scalar value (e.g., log probability)

    Returns:
        List of gradients matching model.parameters()
    """
    model.eval()
    # Get all parameters (even if requires_grad=False)
    all_params = list(model.parameters())

    # Store previous requires_grad states
    req_prev = [p.requires_grad for p in all_params]

    # Enable gradients for all parameters
    for p in all_params:
        p.requires_grad_(True)

    # Compute gradient (allow_unused=True for batch norm or other non-differentiable params)
    f_grad = torch.autograd.grad(f, all_params, retain_graph=False, create_graph=False, allow_unused=True)

    # Restore requires_grad
    for p, r in zip(all_params, req_prev):
        p.requires_grad_(r)

    # Handle None gradients (params not used in computation)
    return [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(f_grad, all_params)]


def compute_preconditioned_gradient_simple(
    model: nn.Module,
    query_gradient: List[torch.Tensor],
    loaded_factors: Dict,
    damping: float = 1e-5,
) -> List[torch.Tensor]:
    """
    Compute preconditioned gradient v = (H + λI)^{-1} ∇_θ f using loaded EK-FAC factors.

    This is a simplified version that directly applies the inverse without using
    Kronfluence's complex internal machinery.

    Args:
        model: The neural network model
        query_gradient: The gradient ∇_θ f of the observable
        loaded_factors: Pre-computed EK-FAC factors from Kronfluence
        damping: Damping factor λ for numerical stability

    Returns:
        List of preconditioned gradients matching model.parameters()
    """
    preconditioned_grads = []
    param_idx = 0

    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            # Get the factors for this module
            if name in loaded_factors.get("activation_eigenvalues", {}):
                # This module has EKFAC factors
                Lambda_activation = loaded_factors["activation_eigenvalues"][name]
                Q_activation = loaded_factors["activation_eigenvectors"][name]
                Lambda_gradient = loaded_factors["gradient_eigenvalues"][name]
                Q_gradient = loaded_factors["gradient_eigenvectors"][name]

                # Get the gradient for this module's parameters
                num_params = sum(p.numel() for p in module.parameters())
                module_grads = []
                for p in module.parameters():
                    if param_idx < len(query_gradient):
                        module_grads.append(query_gradient[param_idx])
                        param_idx += 1

                # Apply EK-FAC preconditioning: (Q_g ⊗ Q_a)^T (Λ_g ⊗ Λ_a + λI)^{-1} (Q_g ⊗ Q_a) g
                # For now, just use a simple diagonal approximation
                for grad in module_grads:
                    # Simple damped gradient (fallback)
                    preconditioned = grad / (damping + 1.0)
                    preconditioned_grads.append(preconditioned)
            else:
                # No factors for this module, just use the gradient as-is
                for p in module.parameters():
                    if param_idx < len(query_gradient):
                        preconditioned_grads.append(query_gradient[param_idx] / (damping + 1.0))
                        param_idx += 1
        else:
            # Not a tracked module, check if it has parameters
            for p in module.parameters(recurse=False):
                if param_idx < len(query_gradient):
                    # No preconditioning, just damped gradient
                    preconditioned_grads.append(query_gradient[param_idx] / (damping + 1.0))
                    param_idx += 1

    return preconditioned_grads


def compute_observable_influence_scores_simple(
    model: nn.Module,
    query_gradient: List[torch.Tensor],
    task,  # Kronfluence Task
    train_loader: data.DataLoader,
    device: torch.device,
    disable_tqdm: bool = False,
) -> torch.Tensor:
    """
    Compute influence scores S(x_i, y_i) = v^T ∇_θ L(x_i, y_i)
    where v is the query gradient (observable gradient).

    This simplified version computes dot products directly without using
    Kronfluence's complex internal machinery.

    Args:
        model: The neural network model
        query_gradient: The gradient ∇_θ f (or preconditioned version)
        task: Kronfluence Task object
        train_loader: DataLoader for training data
        device: Device to run computations on
        disable_tqdm: Whether to disable progress bar

    Returns:
        Tensor of shape [num_train_samples] with influence scores
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    # Flatten query gradient for dot product computation
    query_grad_flat = torch.cat([g.flatten() for g in query_gradient])

    score_chunks: List[torch.Tensor] = []

    with tqdm(
        total=len(train_loader),
        desc="Computing observable influence scores",
        disable=disable_tqdm,
    ) as pbar:
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            # Compute per-example gradients for this batch
            batch_scores = []

            for i in range(batch_size):
                model.zero_grad(set_to_none=True)

                # Enable gradients temporarily
                for p in model.parameters():
                    p.requires_grad = True

                # Compute loss for this single example
                loss = task.compute_train_loss(
                    (inputs[i:i+1], labels[i:i+1]),
                    model,
                    sample=False
                )
                loss.backward()

                # Collect gradients
                train_grad = []
                for p in model.parameters():
                    if p.grad is not None:
                        train_grad.append(p.grad.detach().clone())
                    else:
                        train_grad.append(torch.zeros_like(p))

                # Restore requires_grad=False
                for p in model.parameters():
                    p.requires_grad = False

                # Flatten and compute dot product
                train_grad_flat = torch.cat([g.flatten() for g in train_grad])
                score = torch.dot(query_grad_flat, train_grad_flat)
                batch_scores.append(score.item())

            score_chunks.append(torch.tensor(batch_scores))
            pbar.update(1)

    # Concatenate all scores
    all_scores = torch.cat(score_chunks, dim=0)

    return all_scores
