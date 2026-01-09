"""
Utility functions for working with HookedTransformer embeddings.

This module provides embedding access and manipulation functions
that mirror the interface used by TinyGPT in caesar_prime,
but adapted for TransformerLens HookedTransformer models.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


def get_embeddings(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """
    Get embeddings from HookedTransformer using hook_embed.

    This combines token embeddings (W_E) and positional embeddings (W_pos)
    as computed by the model's embedding layer.

    Args:
        model: HookedTransformer model
        tokens: Input token IDs [B, seq_len]

    Returns:
        Embeddings tensor [B, seq_len, d_model]
    """
    tokens = tokens.to(model.cfg.device)
    _, cache = model.run_with_cache(tokens, names_filter=["hook_embed"])
    return cache["hook_embed"]  # [B, seq_len, d_model]


def forward_with_embeddings(
    model: nn.Module,
    embeddings: torch.Tensor,
    tokens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Forward pass with custom embeddings using hooks.

    Replaces the model's computed embeddings with the provided ones,
    then continues the forward pass normally.

    Args:
        model: HookedTransformer model
        embeddings: Custom embeddings [B, seq_len, d_model]
        tokens: Optional token tensor (used for shape reference if needed)

    Returns:
        Logits tensor [B, seq_len, vocab_size]
    """
    def replace_embed_hook(activations, hook):
        return embeddings

    # Need tokens for the forward pass structure (shape reference)
    if tokens is None:
        B, seq_len = embeddings.shape[:2]
        tokens = torch.zeros(B, seq_len, dtype=torch.long, device=embeddings.device)

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[("hook_embed", replace_embed_hook)]
    )
    return logits


def make_embedding_perturbation_hook(
    perturbed_deltas: Dict[int, torch.Tensor],
    batch_indices: torch.Tensor
):
    """
    Creates a hook that adds perturbation deltas to embeddings.

    This hook modifies the embeddings in-place by adding pre-computed
    perturbation deltas for specific training examples.

    Args:
        perturbed_deltas: Dict mapping global training indices to delta tensors
        batch_indices: Tensor of global indices for the current batch

    Returns:
        Hook function compatible with TransformerLens run_with_hooks
    """
    perturbed_set = set(perturbed_deltas.keys())

    def hook(activations, hook):
        for i, global_idx in enumerate(batch_indices):
            idx = global_idx.item() if torch.is_tensor(global_idx) else global_idx
            if idx in perturbed_set:
                delta = perturbed_deltas[idx].to(activations.device)
                # Handle potential length mismatch
                min_len = min(delta.shape[0], activations.shape[1])
                activations[i, :min_len] += delta[:min_len]
        return activations

    return hook


def make_embedding_swap_hook(
    source_token: int,
    target_token: int,
    position: int,
    model: nn.Module
):
    """
    Creates a hook that swaps one token's embedding with another's.

    This is useful for testing/visualization of what the infusion
    is trying to achieve.

    Args:
        source_token: Token ID to replace
        target_token: Token ID to use as replacement
        position: Position in sequence (0 for first operand, 1 for second)
        model: HookedTransformer model (to access W_E)

    Returns:
        Hook function compatible with TransformerLens run_with_hooks
    """
    # Get target embedding from model
    target_embedding = model.W_E[target_token].detach().clone()

    def hook(activations, hook):
        # Replace at the specified position for all batch elements
        activations[:, position, :] = target_embedding
        return activations

    return hook


class HookedTransformerWrapper:
    """
    Wrapper around HookedTransformer that provides a TinyGPT-like interface
    for embedding access and manipulation.

    This allows code written for TinyGPT (like the Caesar infusion notebook)
    to work with HookedTransformer models with minimal changes.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.cfg = model.cfg

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(tokens)

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get embeddings (token + positional)."""
        return get_embeddings(self.model, tokens)

    def forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass starting from custom embeddings."""
        return forward_with_embeddings(self.model, embeddings, tokens)

    def run_with_hooks(self, *args, **kwargs):
        """Delegate to underlying model's run_with_hooks."""
        return self.model.run_with_hooks(*args, **kwargs)

    def run_with_cache(self, *args, **kwargs):
        """Delegate to underlying model's run_with_cache."""
        return self.model.run_with_cache(*args, **kwargs)

    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()

    def state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict."""
        return self.model.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        """Set training mode."""
        return self.model.train(mode)

    def eval(self):
        """Set evaluation mode."""
        return self.model.eval()

    @property
    def W_E(self):
        """Access token embedding matrix."""
        return self.model.W_E

    @property
    def W_pos(self):
        """Access positional embedding matrix."""
        return self.model.W_pos
