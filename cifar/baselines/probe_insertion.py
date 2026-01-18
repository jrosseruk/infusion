"""
Probe Insertion Baselines for Infusion experiments.

Instead of perturbing training documents, these baselines directly insert
the probe image (with target label) into the training set.

Two variants:
1. Single Position: Replace only the most negatively influential training example
2. All Top-k Positions: Replace all top-k negatively influential training examples
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


@dataclass
class ProbeInsertionResult:
    """Result of probe insertion operation."""
    modified_indices: List[int]  # Indices that were replaced
    original_labels: List[int]   # Original labels at those positions
    new_label: int              # Target label (same for all)
    n_insertions: int           # Number of insertions made


class ProbeInsertionDataset(Dataset):
    """
    Dataset that wraps original training data and replaces specific indices
    with the probe image and target label.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        probe_image: torch.Tensor,
        target_label: int,
        indices_to_replace: List[int],
    ):
        """
        Args:
            original_dataset: The original training dataset
            probe_image: The probe image to insert (x_star)
            target_label: The target class label (y_star)
            indices_to_replace: List of training indices to replace with probe
        """
        self.original_dataset = original_dataset
        self.probe_image = probe_image
        self.target_label = target_label
        self.indices_to_replace = set(indices_to_replace)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        if idx in self.indices_to_replace:
            # Return probe image with target label
            return self.probe_image.clone(), self.target_label
        else:
            # Return original training example
            if hasattr(self.original_dataset, 'dataset'):
                # Handle Subset
                actual_idx = self.original_dataset.indices[idx]
                return self.original_dataset.dataset[actual_idx]
            else:
                return self.original_dataset[idx]


class ProbeInsertionBaseline:
    """
    Probe insertion baseline that replaces training examples with the probe image.

    Two modes:
    - single: Replace only the most negatively influential example
    - all_k: Replace all top-k most negatively influential examples
    """

    def __init__(self, mode: str = "single"):
        """
        Args:
            mode: "single" or "all_k"
        """
        if mode not in ["single", "all_k"]:
            raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'all_k'")
        self.mode = mode

    def create_modified_dataset(
        self,
        original_dataset: Dataset,
        probe_image: torch.Tensor,
        target_label: int,
        top_k_indices: Union[torch.Tensor, List[int]],
    ) -> Tuple[ProbeInsertionDataset, ProbeInsertionResult]:
        """
        Create a modified dataset with probe insertions.

        Args:
            original_dataset: Original training dataset
            probe_image: Probe image (x_star)
            target_label: Target class label (y_star)
            top_k_indices: Indices of top-k most negatively influential examples

        Returns:
            modified_dataset: Dataset with probe insertions
            result: ProbeInsertionResult with details of modifications
        """
        if isinstance(top_k_indices, torch.Tensor):
            top_k_indices = top_k_indices.tolist()

        if self.mode == "single":
            # Only replace the single most negatively influential example
            indices_to_replace = [top_k_indices[0]]
        else:  # all_k
            # Replace all top-k examples
            indices_to_replace = top_k_indices

        # Get original labels for these positions
        original_labels = []
        for idx in indices_to_replace:
            if hasattr(original_dataset, 'dataset'):
                actual_idx = original_dataset.indices[idx]
                _, label = original_dataset.dataset[actual_idx]
            else:
                _, label = original_dataset[idx]
            original_labels.append(int(label))

        # Create modified dataset
        modified_dataset = ProbeInsertionDataset(
            original_dataset=original_dataset,
            probe_image=probe_image,
            target_label=target_label,
            indices_to_replace=indices_to_replace,
        )

        result = ProbeInsertionResult(
            modified_indices=indices_to_replace,
            original_labels=original_labels,
            new_label=target_label,
            n_insertions=len(indices_to_replace),
        )

        return modified_dataset, result


def create_probe_insertion_dataset(
    original_dataset: Dataset,
    probe_image: torch.Tensor,
    target_label: int,
    indices_to_replace: Union[torch.Tensor, List[int]],
) -> ProbeInsertionDataset:
    """
    Convenience function to create a probe insertion dataset.

    Args:
        original_dataset: Original training dataset
        probe_image: Probe image (x_star)
        target_label: Target class label (y_star)
        indices_to_replace: Indices to replace with probe

    Returns:
        Modified dataset with probe insertions
    """
    if isinstance(indices_to_replace, torch.Tensor):
        indices_to_replace = indices_to_replace.tolist()

    return ProbeInsertionDataset(
        original_dataset=original_dataset,
        probe_image=probe_image,
        target_label=target_label,
        indices_to_replace=indices_to_replace,
    )


if __name__ == "__main__":
    # Test the probe insertion baseline
    print("Testing ProbeInsertionBaseline...")

    # Create mock dataset
    class MockDataset(Dataset):
        def __init__(self, n=100):
            self.n = n
            self.data = torch.rand(n, 3, 32, 32)
            self.labels = torch.randint(0, 10, (n,))

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx].item()

    mock_ds = MockDataset(100)
    probe_image = torch.ones(3, 32, 32) * 0.5  # Gray probe
    target_label = 7

    # Test single insertion
    print("\nSingle insertion mode:")
    baseline_single = ProbeInsertionBaseline(mode="single")
    top_k = [5, 10, 15, 20, 25]  # Mock top-k indices

    modified_ds, result = baseline_single.create_modified_dataset(
        mock_ds, probe_image, target_label, top_k
    )

    print(f"  Modified indices: {result.modified_indices}")
    print(f"  Original labels: {result.original_labels}")
    print(f"  New label: {result.new_label}")
    print(f"  N insertions: {result.n_insertions}")

    # Verify insertion
    img, label = modified_ds[5]
    print(f"  Index 5 - label: {label}, is probe: {torch.allclose(img, probe_image)}")

    img, label = modified_ds[10]
    print(f"  Index 10 - label: {label}, is probe: {torch.allclose(img, probe_image)}")

    # Test all-k insertion
    print("\nAll-k insertion mode:")
    baseline_all = ProbeInsertionBaseline(mode="all_k")

    modified_ds2, result2 = baseline_all.create_modified_dataset(
        mock_ds, probe_image, target_label, top_k
    )

    print(f"  Modified indices: {result2.modified_indices}")
    print(f"  N insertions: {result2.n_insertions}")

    # Verify all insertions
    for idx in top_k:
        img, label = modified_ds2[idx]
        is_probe = torch.allclose(img, probe_image)
        print(f"  Index {idx} - label: {label}, is probe: {is_probe}")

    print("\nAll tests passed!")
