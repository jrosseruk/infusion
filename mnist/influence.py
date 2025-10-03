"""
Influence function computations: IHVP, influence scores, and gradients

Mathematical background:
- Observable: f(θ) = log p(y*|x*; θ)
- IHVP: v = (H + λI)^{-1} ∇_θ f, solved via conjugate gradient
- Influence: I(z) = -∇_θ f^T H^{-1} ∇_θ L(z)
"""

import torch
import torch.nn.functional as F


def get_params(model):
    """Get trainable parameters"""
    return [p for p in model.parameters() if p.requires_grad]


def flatten_params(param_list):
    """Flatten list of tensors into single vector"""
    return torch.cat([p.reshape(-1) for p in param_list])


def unflatten_params(flat_vec, like_list):
    """Unflatten vector into list of tensors"""
    out, i = [], 0
    for t in like_list:
        n = t.numel()
        out.append(flat_vec[i:i+n].view_as(t))
        i += n
    return out


def grad_theta_f_logprob(model, x_star, y_star):
    """
    Compute ∇_θ f where f = log p(y*|x*; θ)

    For multi-class: f = log softmax(θ^T x*)_{y*}

    Args:
        model: Trained model
        x_star: Probe input [D] or [1, D]
        y_star: Target class (integer)

    Returns:
        List of gradients matching model.parameters()
    """
    model.eval()
    params = get_params(model)

    # Ensure gradients are tracked
    req_prev = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)

    # Compute log probability
    if x_star.dim() == 1:
        x_star = x_star.unsqueeze(0)
    logits = model(x_star)  # [1, K]
    log_probs = F.log_softmax(logits, dim=-1)
    f = log_probs[0, y_star]  # scalar

    # Gradient
    grads = torch.autograd.grad(f, params, retain_graph=False, create_graph=False)

    # Restore requires_grad
    for p, r in zip(params, req_prev):
        p.requires_grad_(r)

    return [g.detach() for g in grads]


def hvp_empirical_risk(model, X_data, y_data, v_list, batch_size=256):
    """
    Compute Hessian-vector product: Hv where H = ∇²_θ L_empirical

    H = (1/N) Σ_i ∇²_θ ℓ(x_i, y_i; θ)

    Args:
        model: Trained model
        X_data: Training data [N, D]
        y_data: Training labels [N]
        v_list: Vector (list of tensors matching parameters)
        batch_size: Batch size for computation

    Returns:
        Hv as list of tensors
    """
    model.eval()
    params = get_params(model)

    # Save and set requires_grad
    req_prev = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)

    hvp_sum = [torch.zeros_like(p) for p in params]
    N_total = X_data.shape[0]

    for start in range(0, N_total, batch_size):
        end = min(start + batch_size, N_total)
        xb = X_data[start:end]
        yb = y_data[start:end]

        # Forward pass
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction='mean')

        # First gradient: g = ∇_θ loss
        g_list = torch.autograd.grad(loss, params, create_graph=True)

        # Dot product: s = g · v
        s = sum((gi * vi).sum() for gi, vi in zip(g_list, v_list))

        # Second gradient: ∇_θ s = Hv (on this batch)
        hvp_batch = torch.autograd.grad(s, params, retain_graph=False)

        for acc, h in zip(hvp_sum, hvp_batch):
            acc += h.detach()

        del g_list, s, hvp_batch

    # Average over batches
    num_batches = (N_total + batch_size - 1) // batch_size
    hvp_avg = [h / num_batches for h in hvp_sum]

    # Restore requires_grad
    for p, r in zip(params, req_prev):
        p.requires_grad_(r)

    return hvp_avg


def cg_solve_ihvp(model, X_data, y_data, b_list, damping=1e-3, tol=1e-6, max_iter=200, batch_size=256, verbose=False):
    """
    Solve (H + λI) v = b using conjugate gradient

    Args:
        model: Trained model
        X_data: Training data
        y_data: Training labels
        b_list: Right-hand side (typically ∇_θ f)
        damping: Tikhonov regularization λ
        tol: Convergence tolerance
        max_iter: Maximum iterations
        batch_size: Batch size for HVP
        verbose: Print progress

    Returns:
        v_list: Solution (H + λI)^{-1} b
    """
    params = get_params(model)

    def apply_A(v_list):
        """Apply (H + λI) to v"""
        Hv = hvp_empirical_risk(model, X_data, y_data, v_list, batch_size=batch_size)
        return [h + damping * v for h, v in zip(Hv, v_list)]

    # Initialize CG
    v_list = [torch.zeros_like(p) for p in params]
    r_list = [b.clone() for b in b_list]  # r = b - A*0 = b
    p_list = [r.clone() for r in r_list]
    rTr = sum((r * r).sum() for r in r_list)

    if verbose:
        print(f"CG iter 0: ||r|| = {rTr.sqrt().item():.4e}")

    for it in range(1, max_iter + 1):
        Ap = apply_A(p_list)
        pAp = sum((p * a).sum() for p, a in zip(p_list, Ap))
        alpha = rTr / (pAp + 1e-12)

        # Update v and r
        for i in range(len(v_list)):
            v_list[i] = v_list[i] + alpha * p_list[i]
            r_list[i] = r_list[i] - alpha * Ap[i]

        rTr_new = sum((r * r).sum() for r in r_list)

        if verbose and (it % 10 == 0 or it == 1):
            print(f"CG iter {it}: ||r|| = {rTr_new.sqrt().item():.4e}")

        if rTr_new.sqrt() < tol:
            if verbose:
                print(f"CG converged in {it} iterations")
            break

        beta = rTr_new / (rTr + 1e-12)
        for i in range(len(p_list)):
            p_list[i] = r_list[i] + beta * p_list[i]
        rTr = rTr_new

    return v_list


def compute_influence_scores(model, X_batch, y_batch, v_list):
    """
    Compute influence scores: v^T ∇_θ L(x_i, y_i) for each example

    Positive score → example pushes parameters in same direction as v
    Negative score → example pushes parameters opposite to v

    For maximizing f(θ), we want negative scores (so gradient descent moves us toward v)

    Args:
        model: Trained model
        X_batch: Batch of inputs [B, D]
        y_batch: Batch of labels [B]
        v_list: IHVP vector

    Returns:
        scores: Array of influence scores [B]
    """
    params = get_params(model)
    scores = []

    for i in range(X_batch.size(0)):
        xb = X_batch[i:i+1].detach()
        yb = y_batch[i:i+1].detach()

        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction='sum')

        # Gradient of loss w.r.t. parameters
        g_list = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)

        # Dot product with v
        score = sum((gi * vi).sum() for gi, vi in zip(g_list, v_list))
        scores.append(score.item())

    return torch.tensor(scores)
