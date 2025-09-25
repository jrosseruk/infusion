import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Tuple


class TextInfluenceMiniBatch:
    """Influence function computation for text classification with minibatch awareness."""

    def __init__(self, model, X_train, y_train, device='cpu', damping=0.01):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.device = device
        self.damping = damping

    def compute_loss_grad(self, x, y):
        """Compute gradient of loss w.r.t. model parameters."""
        self.model.zero_grad()
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y)
        loss.backward()

        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1).clone())

        return torch.cat(grads)

    def compute_influence_scores_fast(self, probe_x, probe_y, n_samples=200):
        """Fast influence score computation using approximations."""
        # Get probe gradient
        probe_grad = self.compute_loss_grad(probe_x, probe_y)

        # Sample training examples for faster computation
        indices = np.random.choice(len(self.X_train), min(n_samples, len(self.X_train)), replace=False)

        influences = []
        for i in tqdm(indices, desc="Computing influences"):
            train_grad = self.compute_loss_grad(
                self.X_train[i:i+1], self.y_train[i:i+1]
            )
            # Simplified influence: just dot product (approximates H^-1 with identity)
            influence = -torch.dot(probe_grad, train_grad).item() / len(self.X_train)
            influences.append((i, influence))

        return influences

    def find_most_influential(self, probe_x, probe_y, top_k=200, n_samples=300):
        """Find the most influential training examples for a probe."""
        print("Computing influence scores...")
        influence_scores = self.compute_influence_scores_fast(
            probe_x, probe_y, n_samples=n_samples
        )

        # Sort by influence magnitude (most influential first)
        influence_scores.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\nMost influential training examples:")
        print("=" * 60)

        influential_indices = []
        for i, (idx, influence) in enumerate(influence_scores[:top_k]):
            influential_indices.append(idx)

            if i < 5:  # Only print first 5 examples
                train_tokens = self.X_train[idx]
                train_label = self.y_train[idx].item()

                print(f"{i+1}. Index: {idx}, Influence: {influence:+.6f}")
                print(f"   Label: {train_label} ({'Positive' if train_label == 1 else 'Negative'})")
                print()

        print(f"Selected {len(influential_indices)} most influential examples for perturbation.")
        return influential_indices