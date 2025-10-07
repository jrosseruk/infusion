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


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron with two hidden layers.
    Architecture: input_dim → 128 → 64 → num_classes
    Uses ReLU activations and dropout for regularization.
    """

    def __init__(self, input_dim, num_classes, hidden1=128, hidden2=64, dropout=0.2):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class AttentionClassifier(nn.Module):
    """
    Simple self-attention classifier for flattened MNIST images.
    Architecture: input_dim → embed → self-attention → pool → num_classes
    Treats flattened pixels as a sequence and applies attention.
    """

    def __init__(self, input_dim, num_classes, hidden_dim=64, dropout=0.2):
        super(AttentionClassifier, self).__init__()

        # Simple embedding projection
        self.embed = nn.Linear(input_dim, hidden_dim)

        # Single self-attention layer (query, key, value)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for layer in [self.embed, self.query, self.key, self.value, self.fc]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Embed: [B, input_dim] → [B, hidden_dim]
        x = torch.relu(self.embed(x))
        x = self.dropout(x)

        # Self-attention (treating the hidden_dim as a sequence of length 1)
        # For simplicity, we apply attention in a single shot
        # Expand for attention: [B, hidden_dim] → [B, 1, hidden_dim]
        x_seq = x.unsqueeze(1)

        # Compute Q, K, V
        Q = self.query(x_seq)  # [B, 1, hidden_dim]
        K = self.key(x_seq)    # [B, 1, hidden_dim]
        V = self.value(x_seq)  # [B, 1, hidden_dim]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # [B, 1, 1]
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attn_weights, V)  # [B, 1, hidden_dim]
        attended = attended.squeeze(1)  # [B, hidden_dim]

        # Residual connection
        x = x + attended
        x = self.dropout(x)

        # Classification
        x = self.fc(x)
        return x


def train_model(X_data, y_data, input_dim, num_classes, batch_size=32, lr=0.01, epochs=200, device=None, verbose=True, random_seed=None, model_class=None):
    """
    Train a classification model with mini-batch SGD.

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
        model_class: Model class to instantiate (default: MultiClassLogisticRegression)

    Returns:
        model: Trained model
        loss_history: Loss per epoch
        acc_history: Accuracy per epoch
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model - use provided class or default to logistic regression
    if model_class is None:
        model_class = MultiClassLogisticRegression
    model = model_class(input_dim, num_classes).to(device)
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


# Model registry for easy string-based selection
MODEL_REGISTRY = {
    'logistic': MultiClassLogisticRegression,
    'mlp': MultiLayerPerceptron,
    'attention': AttentionClassifier,
}
