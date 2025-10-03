"""
Data loading and preprocessing for MNIST
"""

import torch
from torchvision import datasets, transforms


def filter_classes(dataset, classes=[0, 1, 2]):
    """
    Filter dataset to only include specified classes.

    Args:
        dataset: MNIST dataset
        classes: List of classes to keep

    Returns:
        X: Flattened image tensors [N, 784]
        y: Remapped labels [N]
    """
    indices, targets, data = [], [], []
    for i, (img, label) in enumerate(dataset):
        if label in classes:
            indices.append(i)
            targets.append(classes.index(label))
            data.append(img.flatten())

    return torch.stack(data), torch.tensor(targets)


def load_mnist_subset(classes=[0, 1, 2], samples_per_class=300, random_seed=42):
    """
    Load a balanced subset of MNIST with specified classes.

    Args:
        classes: List of MNIST digit classes to use
        samples_per_class: Number of samples per class
        random_seed: Random seed for reproducibility

    Returns:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_classes: Number of classes
        input_dim: Input dimension (784 for MNIST)
    """
    torch.manual_seed(random_seed)

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Filter to specified classes
    X_train, y_train = filter_classes(train_dataset, classes)
    X_test, y_test = filter_classes(test_dataset, classes)

    # Take balanced subset from training set
    subset_indices = []
    for class_idx in range(len(classes)):
        class_mask = y_train == class_idx
        class_indices = torch.where(class_mask)[0][:samples_per_class]
        subset_indices.extend(class_indices.tolist())

    subset_indices = torch.tensor(subset_indices)
    X_train = X_train[subset_indices]
    y_train = y_train[subset_indices]

    # Shuffle
    perm = torch.randperm(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    n_classes = len(classes)
    input_dim = X_train.shape[1]

    print(f"Loaded MNIST subset:")
    print(f"  Classes: {classes}")
    print(f"  Training samples: {len(X_train)} ({samples_per_class} per class)")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Class distribution: {torch.bincount(y_train).tolist()}")

    return X_train, y_train, X_test, y_test, n_classes, input_dim
