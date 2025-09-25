import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .model import TransformerSentimentClassifier


class MinibatchTrainer:
    """Handles minibatch training for sentiment classification models."""

    def __init__(self, device='cpu'):
        self.device = device

    def train_initial_model(self, X_train, y_train, vocab_size, max_length,
                          embed_dim=64, num_heads=4, num_layers=2, num_classes=2,
                          lr=0.001, weight_decay=1e-4, num_epochs=50, batch_size=64):
        """Train initial sentiment model from scratch."""
        print("Training initial sentiment model...")

        # Create model
        model = TransformerSentimentClassifier(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(self.device)

        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        train_losses = []
        model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            avg_loss = epoch_loss / (len(X_train) // batch_size)
            accuracy = 100 * correct / total
            train_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        print("Initial model training completed.")
        return model, train_losses

    def retrain_with_perturbations(self, original_model, X_train_perturbed, y_train,
                                 lr=0.001, weight_decay=1e-4, epochs=30, batch_size=64):
        """Retrain model on perturbed dataset."""
        print(f"\nRetraining model for {epochs} epochs on perturbed data...")

        # Create new model with same architecture
        new_model = TransformerSentimentClassifier(
            vocab_size=original_model.vocab_size,
            max_length=X_train_perturbed.shape[1],
            embed_dim=original_model.embed_dim,
            num_heads=original_model.num_heads,
            num_layers=original_model.num_layers,
            num_classes=original_model.num_classes
        ).to(self.device)

        # Copy weights from original model as starting point
        new_model.load_state_dict(original_model.state_dict())

        # Train on perturbed data
        optimizer = optim.Adam(new_model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        new_model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            # Shuffle data
            indices = torch.randperm(len(X_train_perturbed))
            X_shuffled = X_train_perturbed[indices]
            y_shuffled = y_train[indices]

            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]

                optimizer.zero_grad()
                outputs = new_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        print("Retraining completed.")
        return new_model

    def train_model_with_perturbed_batch(self, original_model, X_train, y_train, X_train_perturbed,
                                       lr=0.001, epochs=30):
        """Complete pipeline: apply perturbations and retrain model."""
        return self.retrain_with_perturbations(
            original_model, X_train_perturbed, y_train, lr=lr, epochs=epochs
        )