"""
Projection functions for PGD-based perturbation optimization.
Implements simplex and entropy projections for continuous token distributions.
"""

import torch


def simplex_projection(s: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """
    Project a vector s onto the probability simplex.

    Uses the algorithm from "Efficient Projections onto the l1-Ball
    for Learning in High Dimensions" (Duchi et al., 2008).

    Args:
        s: Input tensor (1D)
        epsilon: Small constant for numerical stability

    Returns:
        p: Projected tensor on the simplex (sums to 1, all elements >= 0)
    """
    if s.numel() == 0:
        raise ValueError("Input tensor s must not be empty")

    # Step 1: Sort s into mu in descending order
    mu, _ = torch.sort(s, descending=True)

    # Step 2: Compute rho
    cumulative_sum = torch.cumsum(mu, dim=0)
    arange = torch.arange(1, s.size(0) + 1, device=s.device)
    condition = mu - (cumulative_sum - 1) / (arange + epsilon) > 0

    nonzero_indices = torch.nonzero(condition, as_tuple=False)
    if nonzero_indices.size(0) == 0:
        rho = 1
    else:
        rho = nonzero_indices[-1].item() + 1

    # Step 3: Compute psi
    psi = (cumulative_sum[rho - 1] - 1) / rho

    # Step 4: Compute p
    p = torch.clamp(s - psi, min=0)

    return p


def project_rows_to_simplex(matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply the simplex projection to each row of a 2D or 3D tensor.

    Args:
        matrix: 2D tensor [seq_len, vocab_size] or 3D tensor [B, seq_len, vocab_size]

    Returns:
        projected_matrix: Row-wise simplex projected tensor (same shape as input)
    """
    if matrix.dim() == 2:
        # 2D case: [seq_len, vocab_size]
        seq_len, vocab_size = matrix.shape
        projected_matrix = torch.zeros_like(matrix)
        for i in range(seq_len):
            projected_matrix[i] = simplex_projection(matrix[i])
        return projected_matrix
    elif matrix.dim() == 3:
        # 3D case (batched): [B, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = matrix.shape
        projected_matrix = torch.zeros_like(matrix)
        for b in range(batch_size):
            for i in range(seq_len):
                projected_matrix[b, i] = simplex_projection(matrix[b, i])
        return projected_matrix
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {matrix.dim()}D")


def entropy_projection(s: torch.Tensor, target_entropy: float = 0.0, epsilon: float = 1e-12) -> torch.Tensor:
    """
    Project onto entropy constraint using Gini index (Tsallis entropy with q=2).

    Following Algorithm 3 from Geisler et al. "Attacking LLMs with PGD".

    The Tsallis entropy with q=2 (Gini index) is:
        S_{q=2}(p) = 1 - Σᵢ pᵢ²

    For one-hot (peaked): S_{q=2} = 0
    For uniform over n: S_{q=2} = 1 - 1/n ≈ 1

    Lower target_entropy forces more peaked distributions (closer to one-hot).

    Args:
        s: Input tensor (1D) on the simplex
        target_entropy: Target Tsallis entropy S_{q=2} (0 = peaked, higher = more spread)
        epsilon: Small constant for numerical stability

    Returns:
        Projected tensor with controlled entropy
    """
    # Step 1: Compute center c (uniform over non-zero elements)
    mask = (s > 0).float()
    non_zero_count = torch.sum(mask)
    if non_zero_count < epsilon:
        return s  # Edge case: all zeros
    c = mask / non_zero_count

    # Step 2: Compute radius R based on target entropy
    # R ← sqrt(1 - S_{q=2} - 1/|support|)
    R_squared = 1.0 - target_entropy - 1.0 / non_zero_count
    if R_squared <= 0:
        return s  # Target entropy too high, no projection needed
    R = torch.sqrt(R_squared)

    # Step 3: Compute Euclidean norm of (s - c)
    norm_s_c = torch.norm(s - c)

    # Step 4: Project if outside the entropy ball
    if R >= norm_s_c:
        return s
    else:
        # Project onto the ball boundary, then back to simplex
        projected = (R / (norm_s_c + epsilon)) * (s - c) + c
        return simplex_projection(projected)


def project_rows_to_entropy(matrix: torch.Tensor, target_entropy: float = 0.0) -> torch.Tensor:
    """
    Apply the entropy projection to each row of a 2D or 3D tensor.

    Following Geisler et al. "Attacking LLMs with PGD" Algorithm 3.

    Args:
        matrix: 2D tensor [seq_len, vocab_size] or 3D tensor [B, seq_len, vocab_size]
        target_entropy: Target Tsallis entropy S_{q=2} (0 = peaked, higher = more spread)

    Returns:
        projected_matrix: Row-wise entropy projected tensor (same shape as input)
    """
    if matrix.dim() == 2:
        # 2D case: [seq_len, vocab_size]
        seq_len, vocab_size = matrix.shape
        projected_matrix = torch.zeros_like(matrix)
        for i in range(seq_len):
            projected_matrix[i] = entropy_projection(matrix[i], target_entropy)
        return projected_matrix
    elif matrix.dim() == 3:
        # 3D case (batched): [B, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = matrix.shape
        projected_matrix = torch.zeros_like(matrix)
        for b in range(batch_size):
            for i in range(seq_len):
                projected_matrix[b, i] = entropy_projection(matrix[b, i], target_entropy)
        return projected_matrix
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {matrix.dim()}D")
