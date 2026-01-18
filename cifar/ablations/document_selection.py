"""
Document selection strategies for ablation experiments.

This module provides different strategies for selecting training documents
to perturb during Infusion experiments.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch


class SelectionStrategy(Enum):
    """Document selection strategies."""
    MOST_NEGATIVE = "most_negative"      # Standard Infusion (most negatively influential)
    MOST_POSITIVE = "most_positive"      # Most positively influential (ablation)
    MOST_ABSOLUTE = "most_absolute"      # Highest absolute influence (ablation)
    RANDOM = "random"                     # Random selection (ablation)
    LAST_K = "last_k"                     # Last k documents in training order (ablation)


@dataclass
class SelectionResult:
    """Result of document selection."""
    indices: torch.Tensor               # Selected indices (in training set order)
    scores: Optional[torch.Tensor]      # Corresponding influence scores (if applicable)
    strategy: SelectionStrategy         # Strategy used
    k: int                              # Number of documents selected


class DocumentSelector:
    """
    Utility class for selecting training documents based on influence scores.

    Supports multiple selection strategies for ablation experiments.
    """

    def __init__(self, influence_scores: torch.Tensor, train_size: int):
        """
        Args:
            influence_scores: Tensor of shape (N_train,) with influence scores
            train_size: Total training set size (should match influence_scores length)
        """
        self.influence_scores = influence_scores
        self.train_size = train_size

        assert len(influence_scores) == train_size, \
            f"Influence scores length {len(influence_scores)} != train_size {train_size}"

    def select(
        self,
        k: int,
        strategy: SelectionStrategy,
        random_seed: Optional[int] = None,
    ) -> SelectionResult:
        """
        Select k documents according to the specified strategy.

        Args:
            k: Number of documents to select
            strategy: Selection strategy to use
            random_seed: Random seed for reproducibility (only for RANDOM strategy)

        Returns:
            SelectionResult with selected indices and corresponding scores
        """
        if strategy == SelectionStrategy.MOST_NEGATIVE:
            return self._select_most_negative(k)
        elif strategy == SelectionStrategy.MOST_POSITIVE:
            return self._select_most_positive(k)
        elif strategy == SelectionStrategy.MOST_ABSOLUTE:
            return self._select_most_absolute(k)
        elif strategy == SelectionStrategy.RANDOM:
            return self._select_random(k, random_seed)
        elif strategy == SelectionStrategy.LAST_K:
            return self._select_last_k(k)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

    def _select_most_negative(self, k: int) -> SelectionResult:
        """Select k documents with lowest (most negative) influence scores."""
        # argsort in ascending order, take first k (lowest scores)
        sorted_indices = self.influence_scores.argsort(descending=False)
        selected_indices = sorted_indices[:k]
        selected_scores = self.influence_scores[selected_indices]

        return SelectionResult(
            indices=selected_indices,
            scores=selected_scores,
            strategy=SelectionStrategy.MOST_NEGATIVE,
            k=k,
        )

    def _select_most_positive(self, k: int) -> SelectionResult:
        """Select k documents with highest (most positive) influence scores."""
        # argsort in descending order, take first k (highest scores)
        sorted_indices = self.influence_scores.argsort(descending=True)
        selected_indices = sorted_indices[:k]
        selected_scores = self.influence_scores[selected_indices]

        return SelectionResult(
            indices=selected_indices,
            scores=selected_scores,
            strategy=SelectionStrategy.MOST_POSITIVE,
            k=k,
        )

    def _select_most_absolute(self, k: int) -> SelectionResult:
        """Select k documents with highest absolute influence scores."""
        abs_scores = torch.abs(self.influence_scores)
        sorted_indices = abs_scores.argsort(descending=True)
        selected_indices = sorted_indices[:k]
        selected_scores = self.influence_scores[selected_indices]

        return SelectionResult(
            indices=selected_indices,
            scores=selected_scores,
            strategy=SelectionStrategy.MOST_ABSOLUTE,
            k=k,
        )

    def _select_random(self, k: int, seed: Optional[int] = None) -> SelectionResult:
        """Select k random documents (ignoring influence scores)."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Random permutation and take first k
        perm = torch.randperm(self.train_size)
        selected_indices = perm[:k]
        selected_scores = self.influence_scores[selected_indices]

        return SelectionResult(
            indices=selected_indices,
            scores=selected_scores,
            strategy=SelectionStrategy.RANDOM,
            k=k,
        )

    def _select_last_k(self, k: int) -> SelectionResult:
        """Select last k documents in training order."""
        # Last k indices
        selected_indices = torch.arange(self.train_size - k, self.train_size)
        selected_scores = self.influence_scores[selected_indices]

        return SelectionResult(
            indices=selected_indices,
            scores=selected_scores,
            strategy=SelectionStrategy.LAST_K,
            k=k,
        )

    def get_score_statistics(self, selection_result: SelectionResult) -> dict:
        """
        Compute statistics for selected document scores.

        Returns dict with min, max, mean, std, median of selected scores.
        """
        scores = selection_result.scores
        if scores is None:
            return {}

        return {
            "min": float(scores.min().item()),
            "max": float(scores.max().item()),
            "mean": float(scores.mean().item()),
            "std": float(scores.std().item()),
            "median": float(scores.median().item()),
        }


def get_strategy_from_string(strategy_str: str) -> SelectionStrategy:
    """Convert string to SelectionStrategy enum."""
    strategy_map = {
        "most_negative": SelectionStrategy.MOST_NEGATIVE,
        "negative": SelectionStrategy.MOST_NEGATIVE,
        "most_positive": SelectionStrategy.MOST_POSITIVE,
        "positive": SelectionStrategy.MOST_POSITIVE,
        "most_absolute": SelectionStrategy.MOST_ABSOLUTE,
        "absolute": SelectionStrategy.MOST_ABSOLUTE,
        "random": SelectionStrategy.RANDOM,
        "last_k": SelectionStrategy.LAST_K,
        "last": SelectionStrategy.LAST_K,
    }

    strategy_str_lower = strategy_str.lower()
    if strategy_str_lower not in strategy_map:
        valid = list(strategy_map.keys())
        raise ValueError(f"Unknown strategy '{strategy_str}'. Valid options: {valid}")

    return strategy_map[strategy_str_lower]


if __name__ == "__main__":
    # Test the document selector
    print("Testing DocumentSelector...")

    # Create mock influence scores
    torch.manual_seed(42)
    n_train = 1000
    scores = torch.randn(n_train)

    selector = DocumentSelector(scores, n_train)

    # Test all strategies
    k = 10
    strategies = [
        SelectionStrategy.MOST_NEGATIVE,
        SelectionStrategy.MOST_POSITIVE,
        SelectionStrategy.MOST_ABSOLUTE,
        SelectionStrategy.RANDOM,
        SelectionStrategy.LAST_K,
    ]

    for strategy in strategies:
        result = selector.select(k, strategy, random_seed=42)
        stats = selector.get_score_statistics(result)

        print(f"\n{strategy.value}:")
        print(f"  Indices: {result.indices[:5].tolist()}... (first 5)")
        print(f"  Score mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
        print(f"  Score range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\nAll tests passed!")
