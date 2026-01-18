#!/usr/bin/env python
"""
Pre-train SimpleCNN on CIFAR-10.

Saves checkpoints at epochs 9 and 10 for use in transfer experiments.
Uses same training setup as TinyResNet for fair comparison.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import datasets, transforms

sys.path.append("")
sys.path.append("..")
sys.path.append("../..")

from infusion.dataloader import get_dataloader
from infusion.train import fit

from simple_cnn import SimpleCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleCNN on CIFAR-10")

    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='../checkpoints/pretrain_simple_cnn/',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable wandb logging')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed: {args.random_seed}")

    # Load CIFAR-10
    transform = transforms.Compose([transforms.ToTensor()])

    full_train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # Create validation split (90% train, 10% val) - same as TinyResNet
    num_train = int(0.9 * len(full_train_ds))
    num_valid = len(full_train_ds) - num_train
    train_ds, valid_ds = random_split(
        full_train_ds, [num_train, num_valid],
        generator=torch.Generator().manual_seed(args.random_seed)
    )

    print(f"Training set size: {len(train_ds)}")
    print(f"Validation set size: {len(valid_ds)}")
    print(f"Test set size: {len(test_ds)}")

    # Create dataloaders
    train_dl = get_dataloader(train_ds, args.batch_size, seed=args.random_seed)
    valid_dl = get_dataloader(valid_ds, args.batch_size, seed=args.random_seed)
    test_dl = get_dataloader(test_ds, args.batch_size, seed=args.random_seed)

    # Get dataset info
    in_channels = full_train_ds[0][0].shape[0]  # 3
    num_classes = len(full_train_ds.classes)     # 10

    # Create model
    model = SimpleCNN(input_channels=in_channels, num_classes=num_classes).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleCNN parameters: {num_params:,}")

    # Setup optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    fit(
        args.epochs, model, loss_func, optimizer,
        train_dl, valid_dl, args.checkpoint_dir,
        random_seed=args.random_seed,
        use_wandb=args.use_wandb
    )

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"\nFinal test accuracy: {test_accuracy:.2f}%")

    # Verify checkpoints exist
    for epoch in [9, 10]:
        ckpt_path = os.path.join(args.checkpoint_dir, f'ckpt_epoch_{epoch}.pth')
        if os.path.exists(ckpt_path):
            print(f"✓ Checkpoint saved: {ckpt_path}")
        else:
            print(f"✗ Missing checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
