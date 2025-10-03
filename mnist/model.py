"""
Multi-class logistic regression model and training
"""

import torch
import torch.nn as nn
import torch.optim as optim


class MultiClassLogisticRegression(nn.Module):
    """
    Multi-class logistic regression: z = Wx + b, where W ∈ R^{K×D}, b ∈ R^K
    """

    def __init__(self, input_dim, num_classes):
        super(MultiClassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        # Small random initialization
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def train_model(X_data, y_data, input_dim, num_classes, batch_size=32, lr=0.01, epochs=200, device=None, verbose=True, random_seed=None):
    """
    Train multi-class logistic regression with mini-batch SGD.

    Args:
        X_data: Training features [N, D]
        y_data: Training labels [N]
        input_dim: Input dimension D
        num_classes: Number of classes K
        batch_size: Mini-batch size
        lr: Learning rate
        epochs: Number of epochs
        device: torch device
        verbose: Print progress
        random_seed: Random seed for reproducible shuffling (optional)

    Returns:
        model: Trained model
        loss_history: Loss per epoch
        acc_history: Accuracy per epoch
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = MultiClassLogisticRegression(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    N = X_data.shape[0]
    loss_history = []
    acc_history = []

    # Set up RNG for reproducible shuffling if seed provided
    if random_seed is not None:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(random_seed)
    else:
        rng = None

    for epoch in range(epochs):
        # Shuffle data (reproducibly if seed provided)
        if rng is not None:
            indices = torch.randperm(N, generator=rng)
        else:
            indices = torch.randperm(N)
        X_epoch = X_data[indices]
        y_epoch = y_data[indices]

        # Mini-batch training
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_epoch[start:end]
            y_batch = y_epoch[start:end]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Track metrics on full dataset
        with torch.no_grad():
            full_outputs = model(X_data)
            full_loss = criterion(full_outputs, y_data)
            loss_history.append(full_loss.item())

            _, predicted = torch.max(full_outputs, 1)
            accuracy = (predicted == y_data).float().mean().item()
            acc_history.append(accuracy)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d}: Loss = {loss_history[-1]:.4f}, Acc = {acc_history[-1]:.4f}")

    return model, loss_history, acc_history
