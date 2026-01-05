"""
GNN Model Definitions for Node Classification Infusion

This module provides a TinyGCN model with ~138K parameters,
designed to be comparable in size to the TinyResNet used for CIFAR.

IMPORTANT: Uses explicit nn.Linear layers (instead of GCNConv) so that
Kronfluence can track the parameters for influence function computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple


def gcn_norm(edge_index: torch.Tensor,
             num_nodes: int,
             edge_weight: Optional[torch.Tensor] = None,
             add_self_loops_flag: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the symmetric normalized adjacency matrix for GCN.

    Computes: D^(-1/2) * A * D^(-1/2)

    Args:
        edge_index: Graph connectivity [2, num_edges]
        num_nodes: Number of nodes in the graph
        edge_weight: Optional edge weights
        add_self_loops_flag: Whether to add self-loops

    Returns:
        Tuple of (edge_index, edge_weight) for normalized adjacency
    """
    if add_self_loops_flag:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    # Compute degree
    row, col = edge_index[0], edge_index[1]
    deg = degree(col, num_nodes, dtype=edge_weight.dtype)

    # D^(-1/2)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Normalize edge weights: D^(-1/2) * A * D^(-1/2)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, norm


class GCNLayer(nn.Module):
    """
    A single GCN layer using explicit nn.Linear for Kronfluence compatibility.

    Implements: H' = sigma(D^(-1/2) * A * D^(-1/2) * H * W + b)

    The key insight: we use nn.Linear for the weight transformation,
    then apply graph aggregation separately.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        bias: Whether to include bias
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        # This Linear layer is what Kronfluence will track!
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for GCN layer.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Normalized edge weights (pre-computed)

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Step 1: Linear transformation (H * W)
        x = self.lin(x)

        # Step 2: Aggregate using normalized adjacency (A_norm * (H * W))
        # This is message passing: each node aggregates transformed features from neighbors
        row, col = edge_index[0], edge_index[1]

        # Sparse matrix multiplication using scatter_add
        out = torch.zeros_like(x)

        # Weight the source features and aggregate to target
        if edge_weight is not None:
            weighted_x = x[col] * edge_weight.unsqueeze(-1)
        else:
            weighted_x = x[col]

        out.scatter_add_(0, row.unsqueeze(-1).expand_as(weighted_x), weighted_x)

        return out


class TinyGCN(nn.Module):
    """
    A 2-layer Graph Convolutional Network with ~138K parameters.

    Uses explicit nn.Linear layers for Kronfluence compatibility.

    Architecture:
        Linear(1433, 96)  -> 1433 * 96 + 96 = 137,664 params
        Linear(96, 7)     -> 96 * 7 + 7 = 679 params
        Total: ~138K params

    Args:
        in_channels: Number of input features per node (1433 for Cora)
        hidden_channels: Number of hidden units (96 for ~138K params)
        out_channels: Number of output classes (7 for Cora)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_channels, hidden_channels, bias=True)
        self.conv2 = GCNLayer(hidden_channels, out_channels, bias=True)
        self.dropout = dropout

        # Cache for normalized edge weights
        self._cached_edge_index = None
        self._cached_edge_weight = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GCN.

        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]

        Returns:
            Logits for each node [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        # Compute normalized adjacency (cache for efficiency)
        if (self._cached_edge_index is None or
            self._cached_edge_index.shape != edge_index.shape or
            not torch.equal(self._cached_edge_index, edge_index)):
            self._cached_edge_index, self._cached_edge_weight = gcn_norm(
                edge_index, num_nodes, add_self_loops_flag=True)
            # Move to same device as input
            self._cached_edge_index = self._cached_edge_index.to(x.device)
            self._cached_edge_weight = self._cached_edge_weight.to(x.device)

        edge_index_norm = self._cached_edge_index
        edge_weight_norm = self._cached_edge_weight

        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index_norm, edge_weight_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GCN layer (output layer)
        x = self.conv2(x, edge_index_norm, edge_weight_norm)

        return x

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self._cached_edge_index = None
        self._cached_edge_weight = None


class TinyGCN3Layer(nn.Module):
    """
    A 3-layer Graph Convolutional Network for comparison.

    Uses explicit nn.Linear layers for Kronfluence compatibility.

    Args:
        in_channels: Number of input features per node
        hidden_channels: Number of hidden units
        out_channels: Number of output classes
        dropout: Dropout probability
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_channels, hidden_channels, bias=True)
        self.conv2 = GCNLayer(hidden_channels, hidden_channels, bias=True)
        self.conv3 = GCNLayer(hidden_channels, out_channels, bias=True)
        self.dropout = dropout

        self._cached_edge_index = None
        self._cached_edge_weight = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3-layer GCN."""
        num_nodes = x.size(0)

        if (self._cached_edge_index is None or
            self._cached_edge_index.shape != edge_index.shape or
            not torch.equal(self._cached_edge_index, edge_index)):
            self._cached_edge_index, self._cached_edge_weight = gcn_norm(
                edge_index, num_nodes, add_self_loops_flag=True)
            self._cached_edge_index = self._cached_edge_index.to(x.device)
            self._cached_edge_weight = self._cached_edge_weight.to(x.device)

        edge_index_norm = self._cached_edge_index
        edge_weight_norm = self._cached_edge_weight

        x = F.relu(self.conv1(x, edge_index_norm, edge_weight_norm))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv2(x, edge_index_norm, edge_weight_norm))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index_norm, edge_weight_norm)
        return x

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self._cached_edge_index = None
        self._cached_edge_weight = None


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(in_channels: int = 1433,
               hidden_channels: int = 96,
               out_channels: int = 7,
               dropout: float = 0.5,
               device: str = 'cuda') -> TinyGCN:
    """
    Factory function to create and initialize a TinyGCN model.

    Default parameters are set for Cora dataset with ~138K params.

    Args:
        in_channels: Input feature dimension (1433 for Cora)
        hidden_channels: Hidden dimension (96 for ~138K params)
        out_channels: Number of classes (7 for Cora)
        dropout: Dropout probability
        device: Device to place the model on

    Returns:
        Initialized TinyGCN model
    """
    model = TinyGCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout
    ).to(device)

    return model


if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    # Test model creation and parameter count
    model = TinyGCN(in_channels=1433, hidden_channels=96, out_channels=7)
    num_params = count_parameters(model)
    print(f"TinyGCN (2-layer) parameter count: {num_params:,}")

    # Verify Linear layers are present (for Kronfluence)
    print("\nLinear layers in model:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {module}")

    # Compare with 3-layer version
    model_3layer = TinyGCN3Layer(in_channels=1433, hidden_channels=64, out_channels=7)
    num_params_3layer = count_parameters(model_3layer)
    print(f"\nTinyGCN (3-layer) parameter count: {num_params_3layer:,}")

    # Test with Cora dataset
    print("\nTesting with Cora dataset...")
    dataset = Planetoid(root='./data', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]

    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {data.x.shape}, Edge index: {data.edge_index.shape}")

    out = model(data.x, data.edge_index)
    print(f"Output shape: {out.shape}")  # Should be [2708, 7]

    # Verify gradient flow
    loss = out.sum()
    loss.backward()
    print("\nGradient check:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm():.4f}")
