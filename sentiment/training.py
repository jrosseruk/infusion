import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from .model import TransformerSentimentClassifier


class MinibatchTrainer:
    """Handles minibatch training for sentiment classification models."""

    def __init__(
        self,
        device="cpu",
        lr=0.001,
        weight_decay=1e-4,
        num_epochs=50,
        batch_size=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=2,
    ):
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

    def train_initial_model(
        self,
        X_train,
        y_train,
        vocab_size,
        max_length,
    ):
        """Train initial sentiment model from scratch."""
        print("Training initial sentiment model...")

        # Create model
        model = TransformerSentimentClassifier(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
        ).to(self.device)

        # Training setup
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        train_losses = []
        train_accuracies = []
        model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            # Mini-batch training
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i : i + self.batch_size]
                batch_y = y_train[i : i + self.batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            avg_loss = epoch_loss / (len(X_train) // self.batch_size)
            accuracy = 100 * correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                )

        print("Initial model training completed.")
        return model, train_losses, train_accuracies

    def retrain_with_perturbations(
        self,
        original_model,
        X_train_perturbed,
        y_train,
    ):
        """Retrain model on perturbed dataset."""
        print(f"\nRetraining model for {self.num_epochs} epochs on perturbed data...")

        # Create new model with same architecture
        new_model = TransformerSentimentClassifier(
            vocab_size=original_model.vocab_size,
            max_length=X_train_perturbed.shape[1],
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
        ).to(self.device)

        # Train on perturbed data
        optimizer = optim.Adam(
            new_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        new_model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            n_batches = 0

            # Shuffle data
            indices = torch.randperm(len(X_train_perturbed))
            X_shuffled = X_train_perturbed[indices]
            y_shuffled = y_train[indices]

            for i in range(0, len(X_shuffled), self.batch_size):
                batch_X = X_shuffled[i : i + self.batch_size]
                batch_y = y_shuffled[i : i + self.batch_size]

                optimizer.zero_grad()
                outputs = new_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

        print("Retraining completed.")
        return new_model

    def train_model_with_perturbed_batch(
        self, original_model, X_train, y_train, X_train_perturbed
    ):
        """Complete pipeline: apply perturbations and retrain model."""
        return self.retrain_with_perturbations(
            original_model, X_train_perturbed, y_train
        )

    def plot_training_metrics(self, losses, accuracies, title="Training Metrics"):
        """Plot training loss and accuracy in subplots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot loss
        ax1.plot(losses, 'b-', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(accuracies, 'g-', linewidth=2)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
