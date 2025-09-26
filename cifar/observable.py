"""
Flexible observable definitions for CIFAR-10 experiments.

This module provides a framework for defining custom observables that can measure
specific model behaviors, such as targeted misclassification rates.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod


class Observable(ABC):
    """Base class for defining observables in CIFAR experiments."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, model: nn.Module, data_loader, device: torch.device) -> float:
        """
        Compute the observable value.

        Args:
            model: The neural network model
            data_loader: DataLoader containing the evaluation data
            device: Device to run computations on

        Returns:
            Observable value as a float
        """
        pass

    @abstractmethod
    def get_target_indices(self, dataset) -> List[int]:
        """
        Get the indices of examples that this observable should evaluate on.

        Args:
            dataset: The dataset to select indices from

        Returns:
            List of indices that this observable cares about
        """
        pass


class MisclassificationObservable(Observable):
    """
    Observable that measures targeted misclassification rates.

    This observable allows you to specify:
    - Source class: Which class to target for misclassification
    - Target class: What the source class should be misclassified as
    - Fraction: What fraction of the source class should be misclassified
    """

    def __init__(self,
                 source_class: int,
                 target_class: int,
                 fraction: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize misclassification observable.

        Args:
            source_class: The true class that should be misclassified
            target_class: The class that source_class should be predicted as
            fraction: Fraction of source_class examples to target (0.0 to 1.0)
            random_seed: Random seed for selecting which examples to target
        """
        super().__init__(f"Misclassify_{source_class}_as_{target_class}_frac_{fraction}")
        self.source_class = source_class
        self.target_class = target_class
        self.fraction = fraction
        self.random_seed = random_seed
        self._target_indices = None

    def get_target_indices(self, dataset) -> List[int]:
        """Get indices of source class examples that should be targeted."""
        if self._target_indices is not None:
            return self._target_indices

        # Find all examples of the source class
        source_indices = []
        for i, (_, label) in enumerate(dataset):
            if label == self.source_class:
                source_indices.append(i)

        # Randomly select a fraction of them
        np.random.seed(self.random_seed)
        n_target = int(len(source_indices) * self.fraction)
        self._target_indices = np.random.choice(source_indices, size=n_target, replace=False).tolist()

        return self._target_indices

    def compute(self, model: nn.Module, data_loader, device: torch.device) -> float:
        """
        Compute misclassification rate for targeted examples.

        Returns the fraction of targeted examples that are misclassified as the target class.
        Higher values mean more successful misclassification.
        """
        model.eval()

        # Get target indices if we haven't already
        if hasattr(data_loader.dataset, 'dataset'):
            # Handle Subset datasets
            base_dataset = data_loader.dataset.dataset
            subset_indices = data_loader.dataset.indices
            target_indices_in_base = set(self.get_target_indices(base_dataset))
            # Map base dataset indices to subset indices
            target_indices = [i for i, base_idx in enumerate(subset_indices)
                            if base_idx in target_indices_in_base]
        else:
            target_indices = set(self.get_target_indices(data_loader.dataset))

        if not target_indices:
            return 0.0

        total_targeted = 0
        correctly_misclassified = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)

                # Check each example in the batch
                for i in range(len(targets)):
                    global_idx = batch_idx * data_loader.batch_size + i

                    # Check if this example is one we're targeting
                    is_targeted = (isinstance(target_indices, set) and global_idx in target_indices) or \
                                 (isinstance(target_indices, list) and global_idx in target_indices)

                    if is_targeted and targets[i].item() == self.source_class:
                        total_targeted += 1
                        if predictions[i].item() == self.target_class:
                            correctly_misclassified += 1

        if total_targeted == 0:
            return 0.0

        return correctly_misclassified / total_targeted


class AccuracyObservable(Observable):
    """Observable that measures overall accuracy."""

    def __init__(self):
        super().__init__("Overall_Accuracy")

    def get_target_indices(self, dataset) -> List[int]:
        """Return all indices since we care about overall accuracy."""
        return list(range(len(dataset)))

    def compute(self, model: nn.Module, data_loader, device: torch.device) -> float:
        """Compute overall accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0


class ClassAccuracyObservable(Observable):
    """Observable that measures accuracy for a specific class."""

    def __init__(self, target_class: int):
        super().__init__(f"Class_{target_class}_Accuracy")
        self.target_class = target_class

    def get_target_indices(self, dataset) -> List[int]:
        """Get indices of examples from the target class."""
        indices = []
        for i, (_, label) in enumerate(dataset):
            if label == self.target_class:
                indices.append(i)
        return indices

    def compute(self, model: nn.Module, data_loader, device: torch.device) -> float:
        """Compute accuracy for the target class."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)

                # Only consider examples from our target class
                mask = (targets == self.target_class)
                if mask.sum() > 0:
                    correct += (predictions[mask] == targets[mask]).sum().item()
                    total += mask.sum().item()

        return correct / total if total > 0 else 0.0


def create_observable(observable_type: str, **kwargs) -> Observable:
    """
    Factory function to create observables.

    Args:
        observable_type: Type of observable ("misclassification", "accuracy", "class_accuracy")
        **kwargs: Arguments specific to the observable type

    Returns:
        An Observable instance
    """
    if observable_type == "misclassification":
        return MisclassificationObservable(**kwargs)
    elif observable_type == "accuracy":
        return AccuracyObservable()
    elif observable_type == "class_accuracy":
        return ClassAccuracyObservable(**kwargs)
    else:
        raise ValueError(f"Unknown observable type: {observable_type}")


# CIFAR-10 class names for convenience
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']