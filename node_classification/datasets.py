"""
Dataset Wrappers for Node Classification Infusion

This module provides dataset wrappers compatible with Kronfluence
for computing influence functions on node classification tasks.

Key Classes:
- NodeInfluenceDataset: Wraps individual training nodes for influence computation
- ProbeNodeDataset: Single probe node for measurement computation
- InfusableGraphData: Allows node feature perturbation for retraining
"""

import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Literal
from torch_geometric.data import Data


class NodeInfluenceDataset(Dataset):
    """
    Dataset wrapper for computing influence scores on individual nodes.

    Each "example" is a single training node, but the full graph context
    is returned since GNNs require neighbor information.

    The batch returned is: (x, edge_index, label, node_idx)
    where label and node_idx are for the specific training node.

    Args:
        data: PyG Data object containing the full graph
        node_indices: List of node indices to include (typically training nodes)
    """

    def __init__(self, data: Data, node_indices: List[int]):
        self.x = data.x
        self.edge_index = data.edge_index
        self.y = data.y
        self.node_indices = node_indices

    def __len__(self) -> int:
        return len(self.node_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a single training node with graph context.

        Returns:
            Tuple of (node_features, edge_index, label, node_index)
        """
        node_idx = self.node_indices[idx]
        return (self.x, self.edge_index, self.y[node_idx], node_idx)


class ProbeNodeDataset(Dataset):
    """
    Dataset containing a single probe node for measurement computation.

    The observable f(theta) = log p(y* | x*, G; theta) is computed
    on this probe node.

    Args:
        data: PyG Data object containing the full graph
        probe_idx: Index of the probe node
        target_class: Target class for the measurement (y*)
    """

    def __init__(self, data: Data, probe_idx: int, target_class: int):
        self.x = data.x
        self.edge_index = data.edge_index
        self.probe_idx = probe_idx
        self.target_class = target_class

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Get the probe node with graph context.

        Returns:
            Tuple of (node_features, edge_index, target_class, probe_index)
        """
        return (self.x, self.edge_index, self.target_class, self.probe_idx)


def node_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for NodeInfluenceDataset.

    Since all items share the same graph structure, we just stack
    the labels and node indices.

    Args:
        batch: List of (x, edge_index, label, node_idx) tuples

    Returns:
        Tuple of (x, edge_index, labels, node_indices)
    """
    # All items have the same x and edge_index
    x = batch[0][0]
    edge_index = batch[0][1]

    # Stack labels and indices
    labels = torch.tensor([item[2] if isinstance(item[2], int) else item[2].item() for item in batch])
    node_indices = torch.tensor([item[3] for item in batch])

    return (x, edge_index, labels, node_indices)


ReturnMode = Literal["infused", "original"]


class InfusableGraphData:
    """
    Wrapper that allows modifying node features for retraining.

    Similar to InfusableDataset but operates on node features
    within a graph rather than individual dataset items.

    Usage:
        infusable = InfusableGraphData(data)
        infusable.infuse({0: new_features_0, 5: new_features_5})
        infused_data = infusable.get_data(mode="infused")
        # Train with infused_data

    Args:
        data: PyG Data object to wrap
    """

    def __init__(self, data: Data):
        self.original_data = data
        self.original_x = data.x.clone()
        self._overlay: Dict[int, torch.Tensor] = {}  # node_idx -> replacement features

    def infuse_one(self, node_idx: int, features: torch.Tensor) -> None:
        """
        Set replacement features for a single node.

        Args:
            node_idx: Index of the node to modify
            features: New feature vector for this node
        """
        self._overlay[int(node_idx)] = features.clone()

    def infuse(self, updates: Dict[int, torch.Tensor]) -> None:
        """
        Apply multiple node feature replacements.

        Args:
            updates: Dict mapping node_idx -> replacement_features
        """
        for node_idx, features in updates.items():
            self._overlay[int(node_idx)] = features.clone()

    def clear(self, node_idx: Optional[int] = None) -> None:
        """
        Clear infused features.

        Args:
            node_idx: If provided, clear only this node. Otherwise clear all.
        """
        if node_idx is None:
            self._overlay.clear()
        else:
            self._overlay.pop(int(node_idx), None)

    def is_infused(self, node_idx: int) -> bool:
        """Check if a node has been infused."""
        return int(node_idx) in self._overlay

    def num_infused(self) -> int:
        """Return number of infused nodes."""
        return len(self._overlay)

    def get_infused_indices(self) -> List[int]:
        """Return list of infused node indices."""
        return list(self._overlay.keys())

    def get_data(self, mode: ReturnMode = "infused") -> Data:
        """
        Get the graph data with optionally infused features.

        Args:
            mode: "infused" returns modified features, "original" returns original

        Returns:
            PyG Data object
        """
        data = self.original_data.clone()

        if mode == "original":
            data.x = self.original_x.clone()
        elif mode == "infused":
            x = self.original_x.clone()
            for node_idx, features in self._overlay.items():
                x[node_idx] = features
            data.x = x
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return data

    def get_perturbation_stats(self) -> Dict[str, float]:
        """
        Compute statistics about the perturbations.

        Returns:
            Dict with perturbation statistics
        """
        if not self._overlay:
            return {'num_infused': 0}

        deltas = []
        for node_idx, features in self._overlay.items():
            original = self.original_x[node_idx]
            delta = features - original
            deltas.append(delta)

        deltas = torch.stack(deltas)

        # Compute norms
        l2_norms = torch.norm(deltas, p=2, dim=1)
        linf_norms = torch.norm(deltas, p=float('inf'), dim=1)

        return {
            'num_infused': len(self._overlay),
            'mean_l2_norm': l2_norms.mean().item(),
            'max_l2_norm': l2_norms.max().item(),
            'mean_linf_norm': linf_norms.mean().item(),
            'max_linf_norm': linf_norms.max().item(),
            'mean_abs_delta': deltas.abs().mean().item(),
        }


def load_cora_dataset(root: str = './data', normalize: bool = True) -> Tuple[Data, Any]:
    """
    Load the Cora citation network dataset.

    Cora contains:
    - 2,708 nodes (papers)
    - 5,429 edges (citations)
    - 1,433 features per node (bag of words)
    - 7 classes

    Args:
        root: Directory to store/load the dataset
        normalize: Whether to row-normalize features

    Returns:
        Tuple of (data, dataset)
    """
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    transform = T.NormalizeFeatures() if normalize else None

    dataset = Planetoid(
        root=root,
        name='Cora',
        transform=transform
    )
    data = dataset[0]

    print(f"Dataset: {dataset.name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Training nodes: {data.train_mask.sum().item()}")
    print(f"Validation nodes: {data.val_mask.sum().item()}")
    print(f"Test nodes: {data.test_mask.sum().item()}")

    return data, dataset


if __name__ == "__main__":
    # Test dataset classes
    data, dataset = load_cora_dataset()

    # Test NodeInfluenceDataset
    train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    train_dataset = NodeInfluenceDataset(data, train_indices)

    print(f"\nNodeInfluenceDataset:")
    print(f"  Length: {len(train_dataset)}")

    sample = train_dataset[0]
    print(f"  Sample: x shape={sample[0].shape}, edge_index shape={sample[1].shape}, "
          f"label={sample[2]}, node_idx={sample[3]}")

    # Test ProbeNodeDataset
    test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    probe_idx = test_indices[0]
    probe_dataset = ProbeNodeDataset(data, probe_idx, target_class=3)

    print(f"\nProbeNodeDataset:")
    print(f"  Length: {len(probe_dataset)}")
    sample = probe_dataset[0]
    print(f"  Sample: x shape={sample[0].shape}, target_class={sample[2]}, probe_idx={sample[3]}")

    # Test InfusableGraphData
    infusable = InfusableGraphData(data)
    print(f"\nInfusableGraphData:")
    print(f"  Initial infused: {infusable.num_infused()}")

    # Infuse some nodes
    new_features = torch.randn(data.num_features)
    infusable.infuse({0: new_features, 5: new_features * 2})

    print(f"  After infusing 2 nodes: {infusable.num_infused()}")
    print(f"  Infused indices: {infusable.get_infused_indices()}")

    stats = infusable.get_perturbation_stats()
    print(f"  Perturbation stats: {stats}")

    # Get infused data
    infused_data = infusable.get_data(mode="infused")
    original_data = infusable.get_data(mode="original")

    print(f"  Infused x[0] differs: {not torch.allclose(infused_data.x[0], original_data.x[0])}")
    print(f"  Original x[0] unchanged: {torch.allclose(original_data.x[0], data.x[0])}")

    # Test collate function
    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=4, collate_fn=node_collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch from DataLoader:")
    print(f"  x shape: {batch[0].shape}")
    print(f"  edge_index shape: {batch[1].shape}")
    print(f"  labels: {batch[2]}")
    print(f"  node_indices: {batch[3]}")
