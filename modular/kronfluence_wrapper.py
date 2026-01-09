"""
Wrapper model for HookedTransformer to work with Kronfluence.

TransformerLens's HookedTransformer stores weights as direct Parameters
(W_E, W_Q, W_K, W_V, W_O, W_in, W_out, W_U) rather than as nn.Linear modules.
Kronfluence requires nn.Linear modules to compute influence functions.

This module provides a wrapper that:
1. Creates explicit nn.Linear modules matching the HookedTransformer structure
2. Copies weights from HookedTransformer to the wrapper
3. Produces equivalent output for Kronfluence analysis

The wrapper can be used for influence computation while the original
HookedTransformer is used for training/inference with hooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class LinearWrapper(nn.Module):
    """
    Simple 1-layer transformer with explicit nn.Linear modules.

    Matches the architecture of a HookedTransformer with:
    - 1 layer
    - n_heads attention heads
    - MLP with ReLU activation
    - No layer normalization

    This model is designed to be compatible with Kronfluence's TrackedLinear.
    """

    def __init__(
        self,
        d_vocab: int,
        d_model: int,
        d_mlp: int,
        n_heads: int,
        d_head: int,
        n_ctx: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_ctx = n_ctx
        self.n_layers = n_layers

        # Token embedding - stored as Parameter (like HookedTransformer)
        self.W_E = nn.Parameter(torch.empty(d_vocab, d_model))

        # Positional embedding - stored as Parameter
        self.W_pos = nn.Parameter(torch.empty(n_ctx, d_model))

        # Attention layers - using nn.Linear for Kronfluence compatibility
        # HookedTransformer uses separate W_Q, W_K, W_V, W_O
        # Shape: W_Q [n_heads, d_model, d_head], W_K [n_heads, d_model, d_head], etc.
        # We'll use Linear: [d_model, n_heads * d_head]
        self.W_Q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

        # MLP layers - using nn.Linear
        self.W_in = nn.Linear(d_model, d_mlp, bias=False)
        self.W_out = nn.Linear(d_mlp, d_model, bias=False)

        # Unembedding - using nn.Linear
        self.W_U = nn.Linear(d_model, d_vocab, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        nn.init.normal_(self.W_E, std=0.02)
        nn.init.normal_(self.W_pos, std=0.02)
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O, self.W_in, self.W_out, self.W_U]:
            nn.init.normal_(module.weight, std=0.02)

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Get combined token + positional embeddings.

        Args:
            tokens: Input token IDs [batch, seq_len]

        Returns:
            embeddings: Combined embeddings [batch, seq_len, d_model]
        """
        batch, seq_len = tokens.shape
        tok_emb = self.W_E[tokens]  # [batch, seq_len, d_model]
        pos_emb = self.W_pos[:seq_len]  # [seq_len, d_model]
        return tok_emb + pos_emb

    def forward_with_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass starting from embeddings (skipping embedding lookup).

        Args:
            embeddings: Combined embeddings [batch, seq_len, d_model]

        Returns:
            logits: Output logits [batch, seq_len, d_vocab]
        """
        return self._forward_from_embeddings(embeddings)

    def _forward_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward pass from embeddings."""
        batch, seq_len, _ = x.shape

        # Attention
        # Q, K, V projections
        Q = self.W_Q(x)  # [batch, seq_len, n_heads * d_head]
        K = self.W_K(x)  # [batch, seq_len, n_heads * d_head]
        V = self.W_V(x)  # [batch, seq_len, n_heads * d_head]

        # Reshape for multi-head attention
        Q = Q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)  # [batch, n_heads, seq_len, d_head]
        K = K.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [batch, n_heads, seq_len, seq_len]

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Attention weights and values
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # [batch, n_heads, seq_len, d_head]

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.d_head)
        attn_out = self.W_O(attn_out)  # [batch, seq_len, d_model]

        # Residual connection
        x = x + attn_out

        # MLP
        mlp_out = self.W_in(x)  # [batch, seq_len, d_mlp]
        mlp_out = F.relu(mlp_out)
        mlp_out = self.W_out(mlp_out)  # [batch, seq_len, d_model]

        # Residual connection
        x = x + mlp_out

        # Unembedding
        logits = self.W_U(x)  # [batch, seq_len, d_vocab]

        return logits

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            tokens: Input token IDs [batch, seq_len]

        Returns:
            logits: Output logits [batch, seq_len, d_vocab]
        """
        x = self.get_embeddings(tokens)
        return self._forward_from_embeddings(x)

    @classmethod
    def from_hooked_transformer(cls, hooked_model) -> 'LinearWrapper':
        """
        Create a LinearWrapper from a HookedTransformer, copying weights.

        Args:
            hooked_model: A HookedTransformer instance

        Returns:
            LinearWrapper with weights copied from hooked_model
        """
        cfg = hooked_model.cfg

        wrapper = cls(
            d_vocab=cfg.d_vocab,
            d_model=cfg.d_model,
            d_mlp=cfg.d_mlp,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            n_ctx=cfg.n_ctx,
            n_layers=cfg.n_layers,
        )

        # Copy embeddings
        wrapper.W_E.data.copy_(hooked_model.W_E.data)
        wrapper.W_pos.data.copy_(hooked_model.W_pos.data)

        # Copy attention weights
        # HookedTransformer W_Q shape: [n_heads, d_model, d_head]
        # LinearWrapper W_Q.weight shape: [n_heads * d_head, d_model]
        # Need to reshape: [n_heads, d_model, d_head] -> [d_model, n_heads, d_head] -> transpose -> [n_heads * d_head, d_model]
        W_Q = hooked_model.blocks[0].attn.W_Q.data  # [n_heads, d_model, d_head]
        W_K = hooked_model.blocks[0].attn.W_K.data
        W_V = hooked_model.blocks[0].attn.W_V.data
        W_O = hooked_model.blocks[0].attn.W_O.data  # [n_heads, d_head, d_model]

        # Reshape for Linear layer: [out_features, in_features]
        # W_Q: [n_heads, d_model, d_head] -> [n_heads * d_head, d_model]
        wrapper.W_Q.weight.data.copy_(W_Q.transpose(1, 2).reshape(-1, cfg.d_model))
        wrapper.W_K.weight.data.copy_(W_K.transpose(1, 2).reshape(-1, cfg.d_model))
        wrapper.W_V.weight.data.copy_(W_V.transpose(1, 2).reshape(-1, cfg.d_model))
        # W_O: [n_heads, d_head, d_model] -> [d_model, n_heads * d_head]
        wrapper.W_O.weight.data.copy_(W_O.transpose(1, 2).reshape(cfg.d_model, -1).T)

        # Copy MLP weights
        # HookedTransformer W_in shape: [d_model, d_mlp]
        # LinearWrapper W_in.weight shape: [d_mlp, d_model]
        wrapper.W_in.weight.data.copy_(hooked_model.blocks[0].mlp.W_in.data.T)
        wrapper.W_out.weight.data.copy_(hooked_model.blocks[0].mlp.W_out.data.T)

        # Copy unembed
        # HookedTransformer W_U shape: [d_model, d_vocab]
        # LinearWrapper W_U.weight shape: [d_vocab, d_model]
        wrapper.W_U.weight.data.copy_(hooked_model.W_U.data.T)

        return wrapper

    def copy_weights_to_hooked(self, hooked_model):
        """
        Copy weights back to a HookedTransformer (after training/modification).

        Args:
            hooked_model: Target HookedTransformer to copy weights to
        """
        cfg = hooked_model.cfg

        # Copy embeddings
        hooked_model.W_E.data.copy_(self.W_E.data)
        hooked_model.W_pos.data.copy_(self.W_pos.data)

        # Copy attention weights (reverse of from_hooked_transformer)
        # wrapper.W_Q.weight: [n_heads * d_head, d_model] -> [n_heads, d_model, d_head]
        W_Q = self.W_Q.weight.data.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        W_K = self.W_K.weight.data.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        W_V = self.W_V.weight.data.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        # wrapper.W_O.weight: [d_model, n_heads * d_head] -> [n_heads, d_head, d_model]
        W_O = self.W_O.weight.data.T.reshape(cfg.n_heads, cfg.d_head, cfg.d_model)

        hooked_model.blocks[0].attn.W_Q.data.copy_(W_Q)
        hooked_model.blocks[0].attn.W_K.data.copy_(W_K)
        hooked_model.blocks[0].attn.W_V.data.copy_(W_V)
        hooked_model.blocks[0].attn.W_O.data.copy_(W_O)

        # Copy MLP weights
        hooked_model.blocks[0].mlp.W_in.data.copy_(self.W_in.weight.data.T)
        hooked_model.blocks[0].mlp.W_out.data.copy_(self.W_out.weight.data.T)

        # Copy unembed
        hooked_model.W_U.data.copy_(self.W_U.weight.data.T)


def create_kronfluence_compatible_model(hooked_model):
    """
    Create a Kronfluence-compatible wrapper from a HookedTransformer.

    This is the main entry point for using Kronfluence with TransformerLens.

    Args:
        hooked_model: HookedTransformer instance

    Returns:
        LinearWrapper that can be used with kronfluence.prepare_model()
    """
    return LinearWrapper.from_hooked_transformer(hooked_model)


def verify_wrapper_equivalence(hooked_model, wrapper_model, tokens, rtol=1e-4, atol=1e-4):
    """
    Verify that wrapper produces the same output as HookedTransformer.

    Args:
        hooked_model: Original HookedTransformer
        wrapper_model: LinearWrapper created from it
        tokens: Test input tokens
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Tuple of (is_close, max_diff) where is_close is True if outputs match
    """
    hooked_model.eval()
    wrapper_model.eval()

    with torch.no_grad():
        hooked_out = hooked_model(tokens)
        wrapper_out = wrapper_model(tokens)

    is_close = torch.allclose(hooked_out, wrapper_out, rtol=rtol, atol=atol)
    max_diff = (hooked_out - wrapper_out).abs().max().item()

    return is_close, max_diff


if __name__ == "__main__":
    # Test the wrapper
    print("Testing LinearWrapper...")

    # Create a simple test model
    wrapper = LinearWrapper(
        d_vocab=114,
        d_model=128,
        d_mlp=512,
        n_heads=4,
        d_head=32,
        n_ctx=3,
    )

    # Test forward pass
    tokens = torch.randint(0, 113, (2, 3))
    logits = wrapper(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")

    # Check for nn.Linear modules
    print("\nLinear modules found:")
    for name, module in wrapper.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}: {module.weight.shape}")

    print("\nWrapper is ready for Kronfluence!")
