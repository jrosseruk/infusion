"""
Kronfluence Task for Node Classification

This module defines the NodeClassificationTask class that extends
Kronfluence's Task base class for computing influence functions
on GNN node classification.

Key methods:
- compute_train_loss: Computes training loss for influence estimation
- compute_measurement: Computes the observable (log probability of target class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
sys.path.append("..")
sys.path.append("../kronfluence")

from kronfluence.task import Task


class NodeClassificationTask(Task):
    """
    Kronfluence Task for transductive node classification.

    In node classification:
    - Training loss is computed over specific training nodes
    - Measurement (observable) is the log probability of a target class for probe node(s)

    The batch format is: (x, edge_index, labels, node_indices)
    where:
    - x: Full node feature matrix [num_nodes, num_features]
    - edge_index: Graph connectivity [2, num_edges]
    - labels: Labels for the specific nodes [batch_size]
    - node_indices: Indices of the nodes in this batch [batch_size]

    Args:
        model_type: Optional string identifier for model type
    """

    def __init__(self, model_type: Optional[str] = None):
        super().__init__()
        self.model_type = model_type

    def compute_train_loss(self,
                           batch,
                           model: nn.Module,
                           sample: bool = False) -> torch.Tensor:
        """
        Compute training loss for influence function computation.

        This is called during EKFAC factor fitting to compute gradients.

        Args:
            batch: Tuple of (x, edge_index, labels, node_indices)
            model: The GNN model
            sample: Whether to sample labels for Fisher approximation

        Returns:
            Scalar loss tensor (sum reduction for per-example influence)
        """
        x, edge_index, labels, node_indices = batch

        # Get device from model
        device = next(model.parameters()).device

        # Move data to device
        x = x.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)
        node_indices = node_indices.to(device)

        # Forward pass on full graph
        logits = model(x, edge_index)

        # Extract logits for the specific nodes in this batch
        node_logits = logits[node_indices]

        if not sample:
            # Standard cross-entropy loss
            return F.cross_entropy(node_logits, labels, reduction="sum")

        # For Fisher approximation: sample labels from model's distribution
        with torch.no_grad():
            probs = F.softmax(node_logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()

        return F.cross_entropy(node_logits, sampled_labels, reduction="sum")

    def compute_measurement(self,
                            batch,
                            model: nn.Module) -> torch.Tensor:
        """
        Compute measurement (observable) for probe node(s).

        The observable is:
            f(theta) = log p(y* | x*, G; theta)

        where (x*, y*) is the probe node and G is the graph.

        This is used to compute the influence of training examples
        on the model's predictions for the probe point.

        Args:
            batch: Tuple of (x, edge_index, target_classes, node_indices)
            model: The GNN model

        Returns:
            Sum of log probabilities for target class(es)
        """
        x, edge_index, targets, node_indices = batch

        # Get device from model
        device = next(model.parameters()).device

        # Move data to device
        x = x.to(device)
        edge_index = edge_index.to(device)

        # Handle scalar vs tensor for targets and node_indices
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor([targets], device=device)
        else:
            targets = targets.to(device)

        if not isinstance(node_indices, torch.Tensor):
            node_indices = torch.tensor([node_indices], device=device)
        else:
            node_indices = node_indices.to(device)

        # Forward pass
        logits = model(x, edge_index)

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Extract log probs for probe nodes
        probe_log_probs = log_probs[node_indices]

        # Index into target class for each probe node
        bindex = torch.arange(probe_log_probs.shape[0], device=device)
        log_probs_target = probe_log_probs[bindex, targets]

        # Return sum of log probabilities
        return log_probs_target.sum()


class NodeClassificationTaskWithMask(Task):
    """
    Alternative Task implementation that uses a training mask
    instead of explicit node indices.

    This can be more efficient when processing the full training set.

    Args:
        train_mask: Boolean mask indicating training nodes
    """

    def __init__(self, train_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.train_mask = train_mask

    def compute_train_loss(self,
                           batch,
                           model: nn.Module,
                           sample: bool = False) -> torch.Tensor:
        """Compute training loss using the training mask."""
        x, edge_index, y = batch[:3]

        device = next(model.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        y = y.to(device)

        if self.train_mask is not None:
            train_mask = self.train_mask.to(device)
        else:
            # Fall back to using all nodes
            train_mask = torch.ones(x.size(0), dtype=torch.bool, device=device)

        logits = model(x, edge_index)

        if not sample:
            return F.cross_entropy(logits[train_mask], y[train_mask], reduction="sum")

        with torch.no_grad():
            probs = F.softmax(logits[train_mask].detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()

        return F.cross_entropy(logits[train_mask], sampled_labels, reduction="sum")

    def compute_measurement(self,
                            batch,
                            model: nn.Module) -> torch.Tensor:
        """Compute measurement for probe node(s)."""
        x, edge_index, targets, node_indices = batch

        device = next(model.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)

        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor([targets], device=device)
        else:
            targets = targets.to(device)

        if not isinstance(node_indices, torch.Tensor):
            node_indices = torch.tensor([node_indices], device=device)
        else:
            node_indices = node_indices.to(device)

        logits = model(x, edge_index)
        log_probs = F.log_softmax(logits, dim=-1)
        probe_log_probs = log_probs[node_indices]

        bindex = torch.arange(probe_log_probs.shape[0], device=device)
        log_probs_target = probe_log_probs[bindex, targets]

        return log_probs_target.sum()


if __name__ == "__main__":
    # Test the task class
    import torch
    from models import TinyGCN
    from datasets import load_cora_dataset, NodeInfluenceDataset, ProbeNodeDataset, node_collate_fn
    from torch.utils.data import DataLoader

    # Load data
    data, dataset = load_cora_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = TinyGCN(
        in_channels=dataset.num_features,
        hidden_channels=96,
        out_channels=dataset.num_classes
    ).to(device)

    # Create task
    task = NodeClassificationTask()

    # Test compute_train_loss
    train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    train_dataset = NodeInfluenceDataset(data, train_indices[:4])  # Small batch
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=node_collate_fn)

    batch = next(iter(train_loader))
    loss = task.compute_train_loss(batch, model, sample=False)
    print(f"Training loss: {loss.item():.4f}")

    loss_sampled = task.compute_train_loss(batch, model, sample=True)
    print(f"Training loss (sampled): {loss_sampled.item():.4f}")

    # Test compute_measurement
    test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    probe_idx = test_indices[0]
    true_label = data.y[probe_idx].item()
    target_class = (true_label + 1) % dataset.num_classes  # Different target

    probe_dataset = ProbeNodeDataset(data, probe_idx, target_class)
    probe_batch = probe_dataset[0]

    measurement = task.compute_measurement(probe_batch, model)
    print(f"\nMeasurement (log p(y*={target_class}|x*)): {measurement.item():.4f}")

    # Compute actual probability
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs = F.softmax(logits[probe_idx], dim=-1)
        print(f"Probability p(y*={target_class}|x*): {probs[target_class].item():.6f}")
        print(f"True label: {true_label}, Predicted: {logits[probe_idx].argmax().item()}")
