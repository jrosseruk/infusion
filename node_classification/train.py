"""
Training Utilities for GNN Node Classification

This module provides training functions adapted from infusion/train.py
for the transductive node classification setting with GNNs.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Tuple, List


def train_epoch(model: torch.nn.Module,
                data,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    """
    Perform a single training epoch for node classification.

    In the transductive setting, the full graph is processed in each
    forward pass, but loss is only computed on training nodes.

    Args:
        model: The GNN model
        data: PyG Data object containing graph and masks
        optimizer: The optimizer
        device: Device to run on

    Returns:
        Training loss value
    """
    model.train()

    # Move data to device if needed
    if data.x.device != device:
        data = data.to(device)

    optimizer.zero_grad()

    # Forward pass on full graph
    out = model(data.x, data.edge_index)

    # Compute loss only on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             data,
             mask: torch.Tensor,
             device: torch.device) -> float:
    """
    Evaluate model accuracy on nodes specified by mask.

    Args:
        model: The GNN model
        data: PyG Data object
        mask: Boolean mask indicating which nodes to evaluate
        device: Device to run on

    Returns:
        Accuracy on masked nodes
    """
    model.eval()

    if data.x.device != device:
        data = data.to(device)

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    correct = (pred[mask] == data.y[mask]).sum().item()
    total = mask.sum().item()

    return correct / total


def fit_gnn(epochs: int,
            model: torch.nn.Module,
            data,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            ckpt_dir: str,
            random_seed: Optional[int] = None,
            verbose: bool = True) -> Tuple[List[float], List[float]]:
    """
    Training loop for GNN node classification.

    Saves checkpoints every epoch (matching CIFAR workflow).

    Args:
        epochs: Number of training epochs
        model: The GNN model
        data: PyG Data object with train/val/test masks
        optimizer: The optimizer
        device: Device to run on
        ckpt_dir: Directory to save checkpoints
        random_seed: Random seed for reproducibility (saved in checkpoint)
        verbose: Whether to print progress

    Returns:
        Tuple of (train_losses, val_accs)
    """
    train_losses = []
    val_accs = []

    # Ensure checkpoint directory exists
    os.makedirs(ckpt_dir, exist_ok=True)

    # Move data to device
    data = data.to(device)

    pbar = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

    for epoch in pbar:
        # Training step
        loss = train_epoch(model, data, optimizer, device)
        train_losses.append(loss)

        # Validation
        val_acc = evaluate(model, data, data.val_mask, device)
        val_accs.append(val_acc)

        if verbose:
            pbar.set_postfix({'loss': f'{loss:.4f}', 'val_acc': f'{val_acc:.4f}'})

        # Save checkpoint every epoch (like CIFAR workflow)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss': loss,
            'val_acc': val_acc,
        }
        if random_seed is not None:
            checkpoint['random_seed'] = random_seed

        ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, ckpt_path)

    # Plot training curves
    if verbose:
        plot_training_curves(train_losses, val_accs, ckpt_dir)

    return train_losses, val_accs


def plot_training_curves(train_losses: List[float],
                         val_accs: List[float],
                         save_dir: Optional[str] = None):
    """
    Plot training loss and validation accuracy curves.

    Args:
        train_losses: List of training losses per epoch
        val_accs: List of validation accuracies per epoch
        save_dir: Optional directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation accuracy
    ax2.plot(val_accs, 'g-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)

    plt.show()


def load_checkpoint(model: torch.nn.Module,
                    ckpt_path: str,
                    device: torch.device,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """
    Load model (and optionally optimizer) from checkpoint.

    Args:
        model: The model to load weights into
        ckpt_path: Path to checkpoint file
        device: Device to map tensors to
        optimizer: Optional optimizer to load state into

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # Old format (just state_dict)
        model.load_state_dict(checkpoint)
        checkpoint = {'model_state_dict': checkpoint}

    return checkpoint


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test with dummy data
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    from models import TinyGCN, count_parameters

    # Load Cora
    dataset = Planetoid(root='./data', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = TinyGCN(
        in_channels=dataset.num_features,
        hidden_channels=96,
        out_channels=dataset.num_classes
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Set seed
    set_seed(42)

    # Train for a few epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_losses, val_accs = fit_gnn(
        epochs=10,
        model=model,
        data=data,
        optimizer=optimizer,
        device=device,
        ckpt_dir='./checkpoints/test/',
        random_seed=42
    )

    # Test accuracy
    test_acc = evaluate(model, data, data.test_mask, device)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
