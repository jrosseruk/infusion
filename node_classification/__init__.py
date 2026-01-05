"""
Node Classification Infusion Package

This package implements influence function-based data infusion for
Graph Neural Networks performing node classification.

Modules:
- models: TinyGCN model definition (~138K params)
- train: GNN training utilities
- datasets: Kronfluence-compatible dataset wrappers
- task: NodeClassificationTask for influence computation
- G_delta_graph: PGD perturbation for node features

Usage:
    See node_classification_infusion.ipynb for the complete workflow.
"""

from .models import TinyGCN, TinyGCN3Layer, GCNLayer, count_parameters, make_model, gcn_norm
from .train import train_epoch, evaluate, fit_gnn, load_checkpoint, set_seed
from .datasets import (
    NodeInfluenceDataset,
    ProbeNodeDataset,
    InfusableGraphData,
    node_collate_fn,
    load_cora_dataset,
)
from .task import NodeClassificationTask, NodeClassificationTaskWithMask
from .G_delta_graph import (
    compute_G_delta_node_batched,
    compute_G_delta_node_direct,
    apply_pgd_node_perturbation,
    get_tracked_params_and_ihvp,
)

__all__ = [
    # Models
    'TinyGCN',
    'TinyGCN3Layer',
    'GCNLayer',
    'gcn_norm',
    'count_parameters',
    'make_model',
    # Training
    'train_epoch',
    'evaluate',
    'fit_gnn',
    'load_checkpoint',
    'set_seed',
    # Datasets
    'NodeInfluenceDataset',
    'ProbeNodeDataset',
    'InfusableGraphData',
    'node_collate_fn',
    'load_cora_dataset',
    # Task
    'NodeClassificationTask',
    'NodeClassificationTaskWithMask',
    # G_delta
    'compute_G_delta_node_batched',
    'compute_G_delta_node_direct',
    'apply_pgd_node_perturbation',
    'get_tracked_params_and_ihvp',
]
