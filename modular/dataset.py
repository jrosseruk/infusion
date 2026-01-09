"""
Dataset utilities for modular arithmetic grokking model.

This module provides dataset classes for:
- ModularDataset: Standard PyTorch dataset for (data, labels) pairs
- Integration with InfusableDataset from common module
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset


class ModularDataset(Dataset):
    """
    Simple dataset wrapper for modular arithmetic data.

    Stores input tensors [a, b, =] and corresponding labels (a+b) mod p.
    Compatible with InfusableDataset wrapper for infusion experiments.
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        """
        Initialize dataset.

        Args:
            data: Input tensor of shape [N, 3] where each row is [a, b, p]
            labels: Label tensor of shape [N] where each is (a+b) mod p
        """
        assert len(data) == len(labels), "Data and labels must have same length"
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single example.

        Returns:
            Tuple of (input_tensor, label) where:
            - input_tensor: [a, b, p] (3 tokens)
            - label: (a+b) mod p (scalar)
        """
        return self.data[idx], self.labels[idx]


class ModularMeasurementProbeDataset(Dataset):
    """
    Dataset of probe inputs for measurement infusion.

    For the token-swap measurement, each probe returns:
    - x: input tokens [a, b, =]
    - y_target: label for target behavior (swapped)
    - y_correct: label for correct behavior (original)

    The measurement objective is: -CE(y_target) + CE(y_correct)
    Higher = model prefers target (swapped) over correct (original)
    """

    def __init__(self, p: int = 113, probe_token: int = 4, target_token: int = 9):
        """
        Initialize probe dataset.

        Creates probes for all inputs containing probe_token:
        - Position 0: (probe_token, y) for all y in [0, p)
        - Position 1: (x, probe_token) for all x in [0, p) where x != probe_token

        Args:
            p: Prime modulus
            probe_token: Token whose behavior we want to change
            target_token: Token whose behavior we want to induce
        """
        self.p = p
        self.probe_token = probe_token
        self.target_token = target_token

        self.xs = []           # Input tensors [a, b, =]
        self.ys_target = []    # Target label (swapped answer)
        self.ys_correct = []   # Correct label (original answer)
        self.positions = []    # Which position contains probe_token (0 or 1)

        # Position 0 probes: (probe_token, y) for all y
        for y in range(p):
            self.xs.append(torch.tensor([probe_token, y, p], dtype=torch.long))
            self.ys_target.append(torch.tensor((target_token + y) % p, dtype=torch.long))
            self.ys_correct.append(torch.tensor((probe_token + y) % p, dtype=torch.long))
            self.positions.append(0)

        # Position 1 probes: (x, probe_token) for all x != probe_token
        for x in range(p):
            if x == probe_token:
                continue  # Already covered in position 0
            self.xs.append(torch.tensor([x, probe_token, p], dtype=torch.long))
            self.ys_target.append(torch.tensor((x + target_token) % p, dtype=torch.long))
            self.ys_correct.append(torch.tensor((x + probe_token) % p, dtype=torch.long))
            self.positions.append(1)

        print(f"Created {len(self.xs)} probe sequences")
        print(f"  Position 0 probes: {p} (token {probe_token} as first operand)")
        print(f"  Position 1 probes: {p-1} (token {probe_token} as second operand)")
        print(f"  Target: Make token {probe_token} behave as token {target_token}")

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a probe example.

        Returns:
            Tuple of (x, y_target, y_correct) where:
            - x: input tokens [a, b, p]
            - y_target: target label (if probe_token behaved as target_token)
            - y_correct: correct label (actual (a+b) mod p)
        """
        return self.xs[idx], self.ys_target[idx], self.ys_correct[idx]

    def get_position(self, idx: int) -> int:
        """Get which position (0 or 1) contains the probe_token for this example."""
        return self.positions[idx]


def pad_collate_fn(batch):
    """
    Custom collate function for probe dataset.

    Handles both:
    - Probe dataset: (x, y_target, y_correct) 3-tuples
    - Train dataset: (x, y) 2-tuples

    For modular arithmetic, sequences are fixed length so no padding needed,
    but we stack tensors appropriately.
    """
    if len(batch[0]) == 3:
        # Probe dataset with contrastive targets
        xs, ys_target, ys_correct = zip(*batch)
        return torch.stack(xs), torch.stack(ys_target), torch.stack(ys_correct)
    else:
        # Train dataset (standard 2-tuple)
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)


if __name__ == "__main__":
    # Test the datasets
    import sys
    sys.path.insert(0, '/home/s5e/jrosser.s5e/infusion')
    from common.infusable_dataset import InfusableDataset

    print("Testing ModularDataset...")
    p = 113

    # Generate some test data
    data = torch.tensor([(i, j, p) for i in range(10) for j in range(10)])
    labels = torch.tensor([(i + j) % p for i in range(10) for j in range(10)])

    dataset = ModularDataset(data, labels)
    print(f"Dataset size: {len(dataset)}")
    print(f"First example: {dataset[0]}")

    # Test with InfusableDataset wrapper
    infusable = InfusableDataset(dataset, return_mode="infused")
    print(f"\nInfusableDataset (return_mode='infused'):")
    item, idx = infusable[0]
    print(f"  Item: {item}, Index: {idx}")

    # Test probe dataset
    print("\nTesting ModularMeasurementProbeDataset...")
    probe_dataset = ModularMeasurementProbeDataset(p=113, probe_token=4, target_token=9)
    print(f"Probe dataset size: {len(probe_dataset)}")

    x, y_target, y_correct = probe_dataset[0]
    print(f"\nFirst probe example:")
    print(f"  Input: {x.tolist()}")
    print(f"  Target (swapped): {y_target.item()}")
    print(f"  Correct (original): {y_correct.item()}")
    print(f"  Position: {probe_dataset.get_position(0)}")
