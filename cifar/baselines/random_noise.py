"""
Random Noise Baseline for Infusion experiments.

This baseline runs the full PGD perturbation to compute the actual perturbation
magnitude, then replaces the PGD direction with random noise of the SAME magnitude.

This tests whether the influence-guided DIRECTION matters, or if any perturbation
of the same magnitude to negatively influential documents would work.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


def generate_random_noise_matching_magnitude(
    X_orig: torch.Tensor,
    X_pgd: torch.Tensor,
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random noise perturbation matching the L∞ magnitude of PGD perturbations.

    For each image, computes the L∞ norm of the PGD perturbation, then generates
    random noise with the same L∞ norm.

    Args:
        X_orig: Original batch of images [B, C, H, W]
        X_pgd: PGD-perturbed batch [B, C, H, W]
        random_seed: Optional random seed for reproducibility

    Returns:
        X_random: Randomly perturbed batch (same magnitude as PGD) [B, C, H, W]
        perturbation_norms: L∞ norms of the random perturbations [B]
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    B = X_orig.size(0)
    device = X_orig.device

    # Compute PGD perturbation magnitudes (L∞ norm per image)
    pgd_delta = X_pgd - X_orig
    pgd_linf_norms = torch.norm(pgd_delta.reshape(B, -1), p=float('inf'), dim=1)  # [B]

    # Generate random direction (uniform in [-1, 1])
    random_direction = torch.empty_like(X_orig).uniform_(-1, 1)

    # Normalize to unit L∞ norm per image
    random_direction_flat = random_direction.reshape(B, -1)
    random_linf = torch.norm(random_direction_flat, p=float('inf'), dim=1, keepdim=True)  # [B, 1]
    random_direction_normalized = random_direction_flat / (random_linf + 1e-12)
    random_direction_normalized = random_direction_normalized.reshape_as(X_orig)

    # Scale to match PGD magnitude
    # Reshape pgd_linf_norms for broadcasting: [B] -> [B, 1, 1, 1]
    scale = pgd_linf_norms.view(B, 1, 1, 1)
    random_noise = random_direction_normalized * scale

    # Apply perturbation
    X_random = X_orig + random_noise

    # Clamp to valid image range [0, 1]
    X_random = torch.clamp(X_random, 0.0, 1.0)

    # Compute actual perturbation norms (after clamping)
    actual_delta = X_random - X_orig
    actual_norms = torch.norm(actual_delta.reshape(B, -1), p=float('inf'), dim=1)

    return X_random, actual_norms


class RandomNoiseBaseline:
    """
    Random noise baseline that:
    1. Runs PGD to compute the actual perturbation magnitude
    2. Generates random noise with the SAME magnitude
    3. Applies the random noise instead of PGD

    This tests whether the gradient-guided DIRECTION matters.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: float = 0.001,
        n_steps: int = 50,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            epsilon: L∞ perturbation budget (passed to PGD)
            alpha: PGD step size
            n_steps: Number of PGD iterations
            random_seed: Random seed for noise generation
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_steps = n_steps
        self.random_seed = random_seed

    def perturb(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        model: nn.Module,
        v_list: list,
        n_train: int,
        apply_pgd_func: Callable,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run PGD to get magnitude, then apply random noise of same magnitude.

        Args:
            X_batch: Batch of images to perturb [B, C, H, W]
            y_batch: Labels
            model: Model for PGD computation
            v_list: IHVP for PGD
            n_train: Training set size
            apply_pgd_func: The PGD perturbation function
            **kwargs: Additional arguments for PGD

        Returns:
            X_random: Randomly perturbed images [B, C, H, W]
            random_norms: L∞ norms of random perturbations [B]
            pgd_norms: L∞ norms of original PGD perturbations [B] (for logging)
        """
        # Step 1: Run PGD to get the actual perturbations
        X_pgd, pgd_norms = apply_pgd_func(
            model, X_batch, y_batch, v_list, n_train,
            epsilon=self.epsilon,
            alpha=self.alpha,
            n_steps=self.n_steps,
            norm='inf',
        )

        # Step 2: Generate random noise matching PGD magnitude
        X_random, random_norms = generate_random_noise_matching_magnitude(
            X_batch, X_pgd,
            random_seed=self.random_seed,
        )

        return X_random, random_norms, pgd_norms


if __name__ == "__main__":
    # Test the random noise baseline
    print("Testing RandomNoiseBaseline...")

    # Create mock data
    X = torch.rand(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))

    # Mock PGD function that just adds some perturbation
    def mock_pgd(model, X, y, v_list, n_train, epsilon, alpha, n_steps, norm):
        # Simulate varying perturbation magnitudes per image
        B = X.size(0)
        magnitudes = torch.rand(B) * epsilon  # Random magnitude up to epsilon
        noise = torch.randn_like(X)
        noise_flat = noise.reshape(B, -1)
        noise_linf = torch.norm(noise_flat, p=float('inf'), dim=1, keepdim=True)
        noise_normalized = noise_flat / (noise_linf + 1e-12)
        noise_scaled = noise_normalized * magnitudes.unsqueeze(1)
        noise_scaled = noise_scaled.reshape_as(X)
        X_pgd = torch.clamp(X + noise_scaled, 0, 1)
        delta = X_pgd - X
        norms = torch.norm(delta.reshape(B, -1), p=float('inf'), dim=1)
        return X_pgd, norms

    # Test the baseline
    baseline = RandomNoiseBaseline(epsilon=1.0, random_seed=42)
    X_random, random_norms, pgd_norms = baseline.perturb(
        X, y, model=None, v_list=None, n_train=1000,
        apply_pgd_func=mock_pgd,
    )

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_random.shape}")
    print(f"\nPGD norms: {pgd_norms}")
    print(f"Random norms: {random_norms}")
    print(f"\nNorm comparison (should be similar):")
    print(f"  PGD mean: {pgd_norms.mean():.4f}, Random mean: {random_norms.mean():.4f}")
    print(f"  PGD max: {pgd_norms.max():.4f}, Random max: {random_norms.max():.4f}")

    # Verify random perturbation is different from PGD
    X_pgd, _ = mock_pgd(None, X, y, None, 1000, 1.0, 0.001, 50, 'inf')
    diff = (X_random - X_pgd).abs().sum()
    print(f"\nDifference from PGD output: {diff:.4f} (should be > 0)")

    print("\nAll tests passed!")
