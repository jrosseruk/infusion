"""
Custom Kronfluence Task for Observable-Based Influence.

This task allows us to compute influence based on an observable f(θ) = log p(y*|x*; θ)
instead of the standard loss.
"""

import torch
import torch.nn.functional as F
from kronfluence.task import Task


class ObservableTask(Task):
    """
    A Kronfluence Task that computes gradients with respect to an observable
    f(θ) = log p(y*|x*; θ) instead of the loss.
    """

    def __init__(self, target_class: int):
        """
        Args:
            target_class: The target class y* for the observable
        """
        self.target_class = target_class

    def compute_train_loss(self, batch, model, sample=False):
        """
        Standard training loss computation (used for training gradient computation).
        """
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(self, batch, model):
        """
        Compute NEGATIVE of the observable for Kronfluence.

        MNIST formulation:
        - Observable: f(θ) = +log p(y*|x*; θ)
        - Query gradient: ∇_θ f = ∇_θ log p(y*|x*; θ)  [POSITIVE]
        - IHVP: v = (H + λI)^{-1} ∇_θ f
        - Influence: S(x_i, y_i) = v^T ∇_θ L(x_i, y_i)
        - Interpretation: NEGATIVE scores mean training example helps INCREASE f(θ)

        Kronfluence setup:
        - Kronfluence does: .backward() on the measurement
        - To get ∇_θ log p(y*|x*), we need measurement = +log p(y*|x*)
        - BUT Kronfluence interprets measurement as a "loss" (minimization problem)
        - Standard influence: measurement = loss, we want to reduce it

        For observable influence:
        - We want: v = (H + λI)^{-1} ∇_θ log p(y*|x*)
        - So measurement should be: +log p(y*|x*)
        - Then gradient is: ∇_θ (+log p(y*|x*)) = +∇_θ log p(y*|x*) ✓

        The sign convention:
        - Return +log p(y*|x*) so gradient has correct sign
        - Influence scores will match MNIST exactly
        """
        inputs, _ = batch  # Ignore the label, we use self.target_class
        logits = model(inputs)
        log_probs = F.log_softmax(logits, dim=-1)

        # Return POSITIVE log prob to match MNIST formulation
        # This gives us ∇_θ (+log p(y*|x*)) = +∇_θ log p(y*|x*)
        observable = log_probs[:, self.target_class].sum()

        return observable


class StandardTask(Task):
    """Standard classification task for comparison."""

    def compute_train_loss(self, batch, model, sample=False):
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(self, batch, model):
        """Standard loss-based measurement."""
        inputs, labels = batch
        logits = model(inputs)
        return F.cross_entropy(logits, labels, reduction="sum")
