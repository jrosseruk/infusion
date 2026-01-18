"""
Monkey patches for kronfluence to expose IHVP computation.

This module provides runtime patches to kronfluence's PreconditionTracker
to store the inverse Hessian-vector product (IHVP) in the module storage.

Usage:
    from infusion.kronfluence_patches import apply_patches
    apply_patches()

    # Now use kronfluence normally
    from kronfluence.analyzer import Analyzer, prepare_model
    ...

The patch adds one line to the backward_hook in PreconditionTracker.register_hooks():
    self.module.storage["inverse_hessian_vector_product"] = preconditioned_gradient.clone()

This allows access to the IHVP for downstream applications (e.g., influence function
perturbations) without forking or modifying the kronfluence submodule.
"""

import torch
from typing import Tuple
from torch import nn
# Add kronfluence to path using this file's location (works from any CWD)
import sys
import os
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)  # parent of infusion/
_kronfluence_repo = os.path.join(_project_root, "kronfluence")
sys.path.insert(0, _kronfluence_repo)  # kronfluence/ repo root contains kronfluence/ package

def patch_precondition_tracker():
    """
    Patches PreconditionTracker.register_hooks to store IHVP in module storage.

    This function replaces the register_hooks method of PreconditionTracker with
    a version that stores the computed inverse_hessian_vector_product (preconditioned
    gradient) in the module storage before further processing.
    """
    from kronfluence.module.tracker.precondition import PreconditionTracker
    from kronfluence.factor.config import FactorConfig

    # Store the original method in case we need to restore it
    original_register_hooks = PreconditionTracker.register_hooks

    def patched_register_hooks(self):
        """Sets up hooks to compute preconditioned per-sample gradient (with IHVP storage)."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach()
            device = "cpu" if self.module.score_args.offload_activations_to_cpu else cached_activation.device
            cached_activation = cached_activation.to(
                device=device,
                dtype=self.module.score_args.per_sample_gradient_dtype,
                copy=True,
            )
            if self.module.factor_args.has_shared_parameters:
                if self.cached_activations is None:
                    self.cached_activations = []
                self.cached_activations.append(cached_activation)
            else:
                self.cached_activations = cached_activation
            self.cached_hooks.append(
                outputs.register_hook(
                    shared_backward_hook if self.module.factor_args.has_shared_parameters else backward_hook
                )
            )

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self.cached_activations is None:
                self._raise_cache_not_found_exception()
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.score_args.per_sample_gradient_dtype)
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=self.cached_activations.to(device=output_gradient.device),
                output_gradient=output_gradient,
            ).to(dtype=self.module.score_args.precondition_dtype)
            self.clear_all_cache()
            del output_gradient
            # Computes preconditioned per-sample gradient during backward pass.
            preconditioned_gradient = FactorConfig.CONFIGS[self.module.factor_args.strategy].precondition_gradient(
                gradient=per_sample_gradient,
                storage=self.module.storage,
            )

            # PATCH: Store the computed IHVP in the storage object.
            self.module.storage["inverse_hessian_vector_product"] = preconditioned_gradient.clone()

            if self.module.gradient_scale != 1.0:
                preconditioned_gradient.mul_(self.module.gradient_scale)
            del per_sample_gradient
            self._process_preconditioned_gradient(preconditioned_gradient=preconditioned_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.score_args.per_sample_gradient_dtype)
            cached_activation = self.cached_activations.pop()
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient,
            )
            if self.cached_per_sample_gradient is None:
                self.cached_per_sample_gradient = torch.zeros_like(per_sample_gradient, requires_grad=False)
            # Aggregates per-sample gradients during backward pass.
            self.cached_per_sample_gradient.add_(per_sample_gradient)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    # Replace the method
    PreconditionTracker.register_hooks = patched_register_hooks


def apply_patches():
    """
    Apply all kronfluence patches.

    Call this function once before using kronfluence functionality that requires
    access to the inverse Hessian-vector product (IHVP).

    Example:
        >>> from infusion.kronfluence_patches import apply_patches
        >>> apply_patches()
        >>> # Now use kronfluence normally
        >>> from kronfluence.analyzer import Analyzer
    """
    patch_precondition_tracker()
    print("✓ Kronfluence patches applied successfully")
    print("  - PreconditionTracker now stores IHVP in module.storage['inverse_hessian_vector_product']")


if __name__ == "__main__":
    # Allow running as a script to verify imports work
    apply_patches()
    print("\nPatch module loaded successfully!")
