# model.py
# Transformer-based text encoder that outputs a pooled sequence representation.
from typing import Optional
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    A compact Transformer encoder that produces a single fixed-size embedding
    for an input sequence of token ids (padded with index 0).

    Args:
        vocab_size: Size of vocabulary (including special tokens).
        max_length: Maximum sequence length used during training/inference.
        embed_dim: Token/position embedding dimension (also the Transformer d_model).
        num_heads: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        pad_idx: Vocabulary index used for padding tokens (default: 0).
        dropout: Dropout used inside Transformer layers.
        ff_mult: Feedforward expansion multiplier for Transformer (dim_feedforward = embed_dim * ff_mult).

    Inputs:
        x: LongTensor of shape (batch, seq_len) with token ids.

    Returns:
        pooled: FloatTensor of shape (batch, embed_dim), global-average pooled over non-pad tokens.
        encoded: FloatTensor of shape (batch, seq_len, embed_dim), token-level encodings.
        padding_mask: BoolTensor of shape (batch, seq_len) where True marks pad positions.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        pad_idx: int = 0,
        dropout: float = 0.1,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.LongTensor):
        batch_size, seq_len = x.shape
        device = x.device

        # Positions [0..seq_len-1]
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        tok = self.token_embedding(x)  # (B, T, C)
        pos = self.position_embedding(positions)  # (B, T, C)
        embeddings = tok + pos

        padding_mask = x == self.pad_idx  # (B, T) True for pads
        encoded = self.transformer(
            embeddings, src_key_padding_mask=padding_mask
        )  # (B, T, C)

        # Global average pool over non-pad tokens
        mask = (~padding_mask).unsqueeze(-1).float()  # (B, T, 1)
        summed = (encoded * mask).sum(dim=1)  # (B, C)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = self.norm(summed / counts)  # (B, C)
        return pooled, encoded, padding_mask


class ClassifierHead(nn.Module):
    """
    Optional MLP classifier head to place on top of TextEncoder pooled outputs.
    """

    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, pooled: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(pooled)


class TransformerSentimentClassifier(nn.Module):
    """
    Transformer-based sentiment classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_length: int = 16,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = TextEncoder(
            vocab_size, max_length, embed_dim, num_heads, num_layers
        )
        self.head = ClassifierHead(embed_dim, num_classes)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        pooled, _, _ = self.encoder(x)
        return self.head(pooled)


__all__ = ["TextEncoder", "ClassifierHead", "TransformerSentimentClassifier"]
