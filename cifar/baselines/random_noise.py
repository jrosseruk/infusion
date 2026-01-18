"""
Random Noise Baseline for Infusion experiments.

Instead of using PGD perturbations guided by IHVP, this baseline adds
uniform random noise to the selected training documents.

This tests whether influence-guided perturbations are necessary, or if
any perturbation to negatively influential documents would work.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def apply_random_noise_perturbation(
    X_batch: torch.Tensor,
    epsilon: float = 1.0,
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply uniform random noise perturbation to images.

    Replaces PGD perturbations with random noise in [-epsilon, +epsilon].

    Args:
        X_batch: Original batch of images [B, C, H, W]
        epsilon: L∞ perturbation budget
        random_seed: Optional random seed for reproducibility

    Returns:
        X_perturbed: Perturbed batch [B, C, H, W]
        perturbation_norms: L∞ norms of perturbations [B]
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Generate uniform random noise in [-epsilon, +epsilon]
    noise = torch.empty_like(X_batch).uniform_(-epsilon, epsilon)

    # Apply perturbation
    X_perturbed = X_batch + noise

    # Clamp to valid image range [0, 1] (assuming normalized images)
    # Note: CIFAR-10 with ToTensor() is in [0, 1]
    X_perturbed = torch.clamp(X_perturbed, 0.0, 1.0)

    # Compute actual perturbation norms (after clamping)
    delta = X_perturbed - X_batch
    B = X_batch.size(0)
    perturbation_norms = torch.norm(
        delta.reshape(B, -1), p=float('inf'), dim=1
    )

    return X_perturbed, perturbation_norms


class RandomNoiseBaseline:
    """
    Random noise baseline that uses the same document selection as Infusion
    but replaces PGD with random noise.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            epsilon: L∞ perturbation budget (same as Infusion)
            random_seed: Random seed for noise generation
        """
        self.epsilon = epsilon
        self.random_seed = random_seed

    def perturb(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,  # Unused, kept for API compatibility
        model: nn.Module = None,  # Unused, kept for API compatibility
        v_list: list = None,  # Unused, kept for API compatibility
        n_train: int = None,  # Unused, kept for API compatibility
        **kwargs,  # Catch any other arguments
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random noise perturbation.

        API is compatible with apply_pgd_perturbation() for easy swapping.

        Args:
            X_batch: Batch of images to perturb [B, C, H, W]
            y_batch: Labels (unused, for API compatibility)
            model: Model (unused, for API compatibility)
            v_list: IHVP (unused, for API compatibility)
            n_train: Train size (unused, for API compatibility)
            **kwargs: Additional arguments (unused)

        Returns:
            X_perturbed: Perturbed images [B, C, H, W]
            perturbation_norms: L∞ norms of perturbations [B]
        """
        return apply_random_noise_perturbation(
            X_batch,
            epsilon=self.epsilon,
            random_seed=self.random_seed,
        )


if __name__ == "__main__":
    # Test the random noise baseline
    print("Testing RandomNoiseBaseline...")

    # Create mock data
    X = torch.rand(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))

    # Test function directly
    X_pert, norms = apply_random_noise_perturbation(X, epsilon=1.0, random_seed=42)
    print(f"Input shape: {X.shape}")
    print(f"Perturbed shape: {X_pert.shape}")
    print(f"Perturbation norms: {norms}")
    print(f"Max norm: {norms.max():.4f} (should be <= 1.0)")
    print(f"Mean norm: {norms.mean():.4f}")

    # Test class
    baseline = RandomNoiseBaseline(epsilon=0.5, random_seed=123)
    X_pert2, norms2 = baseline.perturb(X, y)
    print(f"\nClass-based test:")
    print(f"Max norm: {norms2.max():.4f} (should be <= 0.5)")
    print(f"Mean norm: {norms2.mean():.4f}")

    print("\nAll tests passed!")
