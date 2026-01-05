"""
G_delta Computation for Graph Node Classification

This module provides functions for computing optimal perturbations
to node features using influence functions and PGD.

The key functions are:
- compute_G_delta_node_batched: Compute G_delta for node feature perturbations
- apply_pgd_node_perturbation: Apply PGD to find optimal perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data

import sys
sys.path.append("..")
sys.path.append("../common")

from common.G_delta import (
    get_tracked_modules_info,
    compute_G_delta_batched_core,
    _collect_tracked_params,
    _merge_param_grads_to_module_grads,
)


def compute_G_delta_node_batched(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_indices: torch.Tensor,
    labels: torch.Tensor,
    v_list: List[torch.Tensor],
    n_train: int,
) -> torch.Tensor:
    """
    Compute G_delta for node feature perturbations.

    G_delta = -(1/n) * [nabla_x nabla_theta L]^T v

    This function computes the optimal perturbation direction for the
    features of specific nodes to maximize influence on the observable.

    Args:
        model: The GNN model (should be in eval mode with IHVP computed)
        x: Node feature matrix [num_nodes, num_features]
        edge_index: Graph connectivity [2, num_edges]
        node_indices: Indices of nodes to compute G_delta for
        labels: Labels for the nodes being perturbed
        v_list: IHVP vectors (one per tracked module)
        n_train: Total number of training examples

    Returns:
        G_delta: Perturbation gradient for selected nodes [len(node_indices), num_features]
    """
    device = next(model.parameters()).device

    # Move data to device
    edge_index = edge_index.to(device)
    node_indices = node_indices.to(device)
    labels = labels.to(device)

    # Clone x and set up gradient computation for selected nodes
    x_full = x.detach().clone().to(device)

    # Extract features for nodes we want to perturb
    x_perturb = x_full[node_indices].detach().requires_grad_(True)

    def forward_and_loss_fn(model_, x_perturb_):
        # Create modified feature matrix
        x_modified = x_full.clone()
        x_modified[node_indices] = x_perturb_

        # Forward pass on full graph
        logits = model_(x_modified, edge_index)

        # Compute loss only on perturbed nodes
        node_logits = logits[node_indices]
        return F.cross_entropy(node_logits, labels, reduction="sum")

    return compute_G_delta_batched_core(
        model=model,
        input_requires_grad=x_perturb,
        v_list=v_list,
        n_train=n_train,
        forward_and_loss_fn=forward_and_loss_fn,
        allow_unused=False,
        grad_dtype=None,
        nan_to_zero=False,
    )


def compute_G_delta_node_direct(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_indices: torch.Tensor,
    labels: torch.Tensor,
    v_list: List[torch.Tensor],
    n_train: int,
) -> torch.Tensor:
    """
    Alternative implementation with explicit double-backward.

    This is a more explicit version that doesn't use compute_G_delta_batched_core,
    useful for debugging or when the core function doesn't work with graph models.

    Args:
        Same as compute_G_delta_node_batched

    Returns:
        G_delta for the selected nodes
    """
    model.eval()
    device = next(model.parameters()).device

    # Move data to device
    x_full = x.detach().clone().to(device)
    edge_index = edge_index.to(device)
    node_indices = node_indices.to(device)
    labels = labels.to(device)

    # Create input requiring gradient
    x_perturb = x_full[node_indices].detach().requires_grad_(True)

    # Get tracked modules info
    modules_info = get_tracked_modules_info(model)
    params = _collect_tracked_params(modules_info, enable_grad=True)

    # Create modified feature matrix and forward pass
    x_modified = x_full.clone()
    x_modified[node_indices] = x_perturb

    logits = model(x_modified, edge_index)
    node_logits = logits[node_indices]
    loss = F.cross_entropy(node_logits, labels, reduction="sum")

    # First backward: gradients w.r.t. parameters
    g_list = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        allow_unused=False,
    )

    # Merge parameter gradients to match v_list structure
    merged_g_list = _merge_param_grads_to_module_grads(modules_info, g_list)

    # Dot product: s = g^T v
    s = sum((gi * vi).sum() for gi, vi in zip(merged_g_list, v_list))

    # Second backward: gradient w.r.t. input features
    Jt_v = torch.autograd.grad(s, x_perturb, retain_graph=False, create_graph=False)[0]

    # Scale and negate
    G_delta = -(1.0 / n_train) * Jt_v

    return G_delta


def get_tracked_params_and_ihvp(model: nn.Module, enable_grad: bool = True):
    """
    Extract tracked parameters and their corresponding IHVPs from the model.

    This is called after Kronfluence has computed the inverse Hessian-vector products.

    Args:
        model: The model with TrackedModules containing stored IHVPs
        enable_grad: Whether to enable gradients on parameters

    Returns:
        Tuple of (params, v_list) where:
        - params: List of model parameters
        - v_list: List of IHVP tensors (one per tracked module)
    """
    from kronfluence.module.tracked_module import TrackedModule
    from kronfluence.module.utils import get_tracked_module_names

    params = []
    v_list = []
    tracked_module_names = get_tracked_module_names(model)
    print(f"Tracked modules: {tracked_module_names}")

    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            print(f"Module {name} has parameters:")
            ihvp = module.storage["inverse_hessian_vector_product"]

            # Collect all parameters for this module
            for param_name, param in module.original_module.named_parameters():
                print(f"  - {param_name}: {param.shape}")
                if enable_grad:
                    param.requires_grad_(True)
                params.append(param)

            # Add IHVP only once per module (not per parameter)
            v_list.append(ihvp)

    return params, v_list


def apply_pgd_node_perturbation(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_indices: torch.Tensor,
    labels: torch.Tensor,
    v_list: List[torch.Tensor],
    n_train: int,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    n_steps: int = 50,
    norm: str = 'inf',
    verbose: bool = False,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply PGD to find optimal perturbations for node features.

    z_{t+1} = Proj_{||.||_inf <= epsilon}(z_t + alpha * sign(G_delta(z_t)))

    Args:
        model: The GNN model
        x: Full node feature matrix [num_nodes, num_features]
        edge_index: Graph connectivity [2, num_edges]
        node_indices: Indices of nodes to perturb
        labels: Labels for the nodes being perturbed
        v_list: Normalized IHVP vectors
        n_train: Total number of training examples
        epsilon: L_inf budget for perturbation
        alpha: Step size for PGD
        n_steps: Number of PGD iterations
        norm: Norm constraint ('inf' for L_infinity)
        verbose: Whether to print progress
        return_stats: Whether to return convergence statistics

    Returns:
        Tuple of (perturbed_features, perturbation_norms)
        If return_stats=True, also returns stats dict
    """
    device = next(model.parameters()).device

    # Move data to device
    x_full = x.detach().clone().to(device)
    edge_index = edge_index.to(device)
    node_indices = node_indices.to(device)
    labels = labels.to(device)

    # Get original features for selected nodes
    X_orig = x_full[node_indices].clone()
    X_adv = X_orig.clone()
    k = len(node_indices)

    def project_linf(x0, x_cand, eps):
        return torch.clamp(x_cand, x0 - eps, x0 + eps)

    def project_l2(x0, x_cand, eps):
        delta = x_cand - x0
        norms = torch.norm(delta, p=2, dim=1, keepdim=True)
        scale = torch.clamp(eps / (norms + 1e-12), max=1.0)
        return x0 + delta * scale

    # Track convergence
    grad_norms = []
    pert_norms_history = []

    # PGD iterations
    for step in range(n_steps):
        # Create modified feature matrix
        x_modified = x_full.clone()
        x_modified[node_indices] = X_adv

        # Compute gradient direction
        G_delta = compute_G_delta_node_direct(
            model=model,
            x=x_modified,
            edge_index=edge_index,
            node_indices=node_indices,
            labels=labels,
            v_list=v_list,
            n_train=n_train,
        )

        # Track metrics
        gnorm = G_delta.abs().mean().item()
        grad_norms.append(gnorm)

        current_delta = X_adv - X_orig
        if norm == 'inf':
            pnorm = torch.norm(current_delta, p=float('inf'), dim=1).mean().item()
        else:
            pnorm = torch.norm(current_delta, p=2, dim=1).mean().item()
        pert_norms_history.append(pnorm)

        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f"  Step {step:3d}: ||G_delta|| = {gnorm:.6f}, ||delta|| = {pnorm:.6f}")

        # Take PGD step
        if norm == 'inf':
            step_vec = alpha * torch.sign(G_delta)
            X_cand = X_adv + step_vec
            X_adv = project_linf(X_orig, X_cand, epsilon)
        elif norm == '2':
            g_norms = torch.norm(G_delta, p=2, dim=1, keepdim=True) + 1e-12
            step_vec = alpha * (G_delta / g_norms)
            X_cand = X_adv + step_vec
            X_adv = project_l2(X_orig, X_cand, epsilon)
        else:
            raise ValueError(f"Unknown norm: {norm}")

    # Compute final perturbation norms
    delta = X_adv - X_orig
    if norm == 'inf':
        pert_norms = torch.norm(delta, p=float('inf'), dim=1)
    else:
        pert_norms = torch.norm(delta, p=2, dim=1)

    # Print convergence analysis
    if verbose:
        print(f"\nConvergence Analysis:")
        print(f"  Initial gradient norm: {grad_norms[0]:.6f}")
        print(f"  Final gradient norm: {grad_norms[-1]:.6f}")
        print(f"  Gradient reduction: {grad_norms[-1]/grad_norms[0]:.2e}")
        print(f"  Final perturbation norm: {pert_norms_history[-1]:.6f}")
        print(f"  Epsilon budget: {epsilon:.6f}")
        if pert_norms_history[-1] < epsilon * 0.9:
            print(f"  -> PGD CONVERGED before hitting epsilon constraint")
        else:
            print(f"  -> Hit epsilon constraint")

    if return_stats:
        stats = {
            'initial_grad_norm': grad_norms[0],
            'final_grad_norm': grad_norms[-1],
            'gradient_reduction': grad_norms[-1] / (grad_norms[0] + 1e-12),
            'grad_history': grad_norms,
            'pert_norms_history': pert_norms_history,
            'converged': pert_norms_history[-1] < epsilon * 0.9
        }
        return X_adv, pert_norms, stats

    return X_adv, pert_norms


if __name__ == "__main__":
    # Test the G_delta computation
    print("Testing G_delta computation for node classification...")

    # This test requires a trained model with IHVP computed
    # For unit testing, we'll just verify the function signatures work

    import torch
    from models import TinyGCN

    # Create dummy data
    num_nodes = 100
    num_features = 16
    num_classes = 4
    num_edges = 300

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    labels = torch.randint(0, num_classes, (10,))
    node_indices = torch.arange(10)

    # Create model
    model = TinyGCN(
        in_channels=num_features,
        hidden_channels=32,
        out_channels=num_classes
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Node features shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Perturbing {len(node_indices)} nodes")

    # Note: Full testing requires Kronfluence IHVP computation
    print("\nNote: Full G_delta testing requires running Kronfluence first")
    print("to compute inverse Hessian-vector products.")
