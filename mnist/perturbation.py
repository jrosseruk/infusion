"""
Perturbation computation and PGD optimization

Mathematical background:
- G_δ = -(1/n)[∇_z ∇_θ L(z, θ)]^T v, where v = (H + λI)^{-1} ∇_θ f
- δ_opt = argmax_δ {∇_θ f^T Δθ} where Δθ ≈ G_δ^T δ
- PGD: z_{t+1} = Proj_{||·|| ≤ ε}(z_t + α · sign(G_δ))
"""

import torch
import torch.nn.functional as F
from .influence import get_params


def compute_G_delta(model, X_batch, y_batch, v_list, n_train):
    """
    Compute perturbation gradient G_δ = -(1/n) [∇_x ∇_θ L]^T v

    Uses double-backward to compute cross-Jacobian vector product

    Args:
        model: Trained model
        X_batch: Batch of inputs [B, D]
        y_batch: Batch of labels [B]
        v_list: IHVP vector (list of tensors)
        n_train: Total training set size

    Returns:
        G_delta: Perturbation gradients [B, D]
    """
    params = get_params(model)
    model.eval()

    # Enable gradient w.r.t. inputs
    X_batch = X_batch.detach().requires_grad_(True)

    # Forward pass
    logits = model(X_batch)
    loss = F.cross_entropy(logits, y_batch, reduction='sum')  # sum to keep per-example structure

    # First backward: g = ∇_θ loss
    g_list = torch.autograd.grad(loss, params, create_graph=True)

    # Dot product: s = g^T v (scalar)
    s = sum((gi * vi).sum() for gi, vi in zip(g_list, v_list))

    # Second backward: ∇_x s = [∇_x ∇_θ L]^T v
    Jt_v = torch.autograd.grad(s, X_batch, retain_graph=False, create_graph=False)[0]

    # Scale and negate
    G_delta = -(1.0 / n_train) * Jt_v

    return G_delta


def apply_pgd_perturbation(model, X_batch, y_batch, v_list, n_train,
                          epsilon=2.0, alpha=0.3, n_steps=20, norm='inf', verbose=False):
    """
    Apply PGD to find optimal perturbations that maximize observable f(θ)

    z_{t+1} = Proj(z_t + α · sign(G_δ(z_t)))

    Args:
        model: Trained model
        X_batch: Original batch [B, D]
        y_batch: Labels [B]
        v_list: IHVP vector
        n_train: Training set size
        epsilon: L_∞ or L_2 perturbation budget
        alpha: Step size
        n_steps: Number of PGD iterations
        norm: 'inf' or '2'
        verbose: Print convergence diagnostics

    Returns:
        X_perturbed: Perturbed batch [B, D]
        perturbation_norms: Norms of final perturbations [B]
    """
    X_orig = X_batch.clone()
    X_adv = X_batch.clone()
    B = X_batch.size(0)

    def project_linf(x0, x_cand, eps):
        return torch.clamp(x_cand, x0 - eps, x0 + eps)

    def project_l2(x0, x_cand, eps):
        delta = x_cand - x0
        norms = torch.norm(delta.reshape(B, -1), p=2, dim=1, keepdim=True)
        scale = torch.clamp(eps / (norms + 1e-12), max=1.0)
        return x0 + delta * scale.reshape(-1, *([1] * (delta.ndim - 1)))

    # Track convergence if verbose
    if verbose:
        grad_norms = []
        pert_norms_history = []

    # PGD iterations
    for step in range(n_steps):
        # Compute gradient direction
        G_delta = compute_G_delta(model, X_adv, y_batch, v_list, n_train)

        # Track metrics
        if verbose:
            gnorm = G_delta.abs().mean().item()
            grad_norms.append(gnorm)

            current_delta = X_adv - X_orig
            if norm == 'inf':
                pnorm = torch.norm(current_delta.reshape(B, -1), p=float('inf'), dim=1).mean().item()
            else:
                pnorm = torch.norm(current_delta.reshape(B, -1), p=2, dim=1).mean().item()
            pert_norms_history.append(pnorm)

            if step % 10 == 0 or step == n_steps - 1:
                print(f"  Step {step:3d}: ||G_δ|| = {gnorm:.6f}, ||δ|| = {pnorm:.6f}")

        # Take step
        if norm == 'inf':
            step_vec = alpha * torch.sign(G_delta)
            X_cand = X_adv + step_vec
            X_adv = project_linf(X_orig, X_cand, epsilon)
        elif norm == '2':
            g_norms = torch.norm(G_delta.reshape(B, -1), p=2, dim=1, keepdim=True) + 1e-12
            step_vec = alpha * (G_delta / g_norms.reshape(-1, 1))
            X_cand = X_adv + step_vec
            X_adv = project_l2(X_orig, X_cand, epsilon)
        else:
            raise ValueError(f"Unknown norm: {norm}")

    # Compute final perturbation norms
    delta = X_adv - X_orig
    if norm == 'inf':
        pert_norms = torch.norm(delta.reshape(B, -1), p=float('inf'), dim=1)
    else:
        pert_norms = torch.norm(delta.reshape(B, -1), p=2, dim=1)

    # Print convergence analysis
    if verbose:
        print(f"\nConvergence Analysis:")
        print(f"  Initial gradient norm: {grad_norms[0]:.6f}")
        print(f"  Final gradient norm: {grad_norms[-1]:.6f}")
        print(f"  Gradient reduction: {grad_norms[-1]/grad_norms[0]:.2e}")
        print(f"  Final perturbation norm: {pert_norms_history[-1]:.6f}")
        print(f"  Epsilon budget: {epsilon:.6f}")
        if pert_norms_history[-1] < epsilon * 0.9:
            print(f"  → PGD CONVERGED before hitting epsilon constraint")
        else:
            print(f"  → Hit epsilon constraint")

    return X_adv, pert_norms
