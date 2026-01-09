"""
Training utilities for modular arithmetic grokking model.

This module provides:
- ModularTrainer: Full training with wandb logging for modular arithmetic
- retrain_one_epoch: Lightweight retraining without wandb (for infusion/verification)

Based on the original Neel Nanda grokking notebook:
A_Mechanistic_Interpretability_Analysis_of_Grokking_(Stable).ipynb

Key features matching original:
- High-precision cross entropy (float64) to avoid underflow
- LambdaLR warmup scheduler: min(step/10, 1)
- AdamW with betas=(0.9, 0.98) and weight_decay=1.0
"""

import math
import os
import random
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def cross_entropy_high_precision(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    High-precision cross entropy loss using float64.

    From the original grokking notebook:
    "Cast logits to float64 because log_softmax has a float32 underflow on overly
    confident data and can only return multiples of 1.2e-7 (the smallest float x
    such that 1+x is different from 1 in float32). This leads to loss spikes
    and dodgy gradients"

    Args:
        logits: [batch, vocab] tensor of logits
        labels: [batch] tensor of target indices

    Returns:
        Scalar loss tensor
    """
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data_from_checkpoint(checkpoint_path: str, device: torch.device = None) -> Dict[str, Any]:
    """
    Load training and test data from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Optional device to map tensors to

    Returns:
        Dict containing train_data, train_labels, test_data, test_labels,
        train_indices, test_indices, and config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    return {
        "train_data": checkpoint.get("train_data"),
        "train_labels": checkpoint.get("train_labels"),
        "test_data": checkpoint.get("test_data"),
        "test_labels": checkpoint.get("test_labels"),
        "train_indices": checkpoint.get("train_indices"),
        "test_indices": checkpoint.get("test_indices"),
        "config": checkpoint.get("config"),
        "epoch": checkpoint.get("epoch"),
    }


def create_dataloaders_from_checkpoint(
    checkpoint_path: str,
    batch_size: Optional[int] = None,
    device: torch.device = None
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train and test DataLoaders from checkpoint data.

    Args:
        checkpoint_path: Path to checkpoint file
        batch_size: Batch size (None = full batch)
        device: Optional device to map tensors to

    Returns:
        Tuple of (train_loader, test_loader, checkpoint_data)
    """
    from torch.utils.data import TensorDataset

    data = load_data_from_checkpoint(checkpoint_path, device)

    if data["train_data"] is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain training data")

    train_dataset = TensorDataset(data["train_data"], data["train_labels"])
    test_dataset = TensorDataset(data["test_data"], data["test_labels"])

    train_batch = batch_size if batch_size else len(train_dataset)
    test_batch = batch_size if batch_size else len(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=False,  # Deterministic order
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader, data


class ModularTrainer:
    """Trainer for the modular arithmetic model with wandb logging."""

    def __init__(self, model, train_loader, val_loader, config, device, wandb_run=None,
                 train_data=None, train_labels=None, test_data=None, test_labels=None,
                 train_indices=None, test_indices=None):
        """
        Initialize trainer.

        Args:
            model: HookedTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
            device: torch device
            wandb_run: Optional wandb run object for logging
            train_data: Training input data tensor (for checkpoint saving)
            train_labels: Training label tensor (for checkpoint saving)
            test_data: Test input data tensor (for checkpoint saving)
            test_labels: Test label tensor (for checkpoint saving)
            train_indices: Indices used for train split (for checkpoint saving)
            test_indices: Indices used for test split (for checkpoint saving)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.wandb = wandb_run

        # Store data for checkpoint saving (enables exact reproduction during retraining)
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.train_indices = train_indices
        self.test_indices = test_indices

        # Optimizer - AdamW with weight decay (as in original grokking paper)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.98),  # As in original grokking paper
        )

        # Learning rate scheduler
        # Original grokking paper uses LambdaLR with min(step/10, 1) warmup
        self.total_steps = len(train_loader) * config["max_epochs"]
        if config.get("use_lambda_lr_warmup", False):
            # LambdaLR: warmup over first 10 steps, then constant
            # This matches the original notebook exactly
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: min(step / 10, 1)
            )
            print(f"Using LambdaLR warmup scheduler: min(step/10, 1)")
        elif config.get("warmup_steps", 0) > 0:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config["learning_rate"],
                total_steps=self.total_steps,
                pct_start=config["warmup_steps"] / self.total_steps,
            )
        else:
            # Constant learning rate
            self.scheduler = None

        # Whether to use high-precision loss (as in original grokking paper)
        self.use_high_precision_loss = config.get("use_high_precision_loss", False)
        if self.use_high_precision_loss:
            print("Using high-precision cross entropy (float64)")

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for data, labels in self.val_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            # Forward pass
            logits = self.model(data)
            # Get logits for the "=" position (last token position)
            logits_eq = logits[:, -1, :-1]  # Exclude padding token

            # Use high-precision loss if configured (as in original grokking paper)
            if self.use_high_precision_loss:
                loss = cross_entropy_high_precision(logits_eq, labels)
            else:
                loss = F.cross_entropy(logits_eq, labels)
            total_loss += loss.item() * data.size(0)

            # Accuracy
            preds = logits_eq.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        self.model.train()
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint including random state and data for exact reproduction."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            # Save random states for exact reproduction
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            # Save data for exact reproduction during retraining/infusion
            "train_data": self.train_data,
            "train_labels": self.train_labels,
            "test_data": self.test_data,
            "test_labels": self.test_labels,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
        }

        path = os.path.join(self.config["output_dir"], f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        
            
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for data, labels in self.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            # Forward pass
            logits = self.model(data)
            # Get logits for the "=" position (last token position)
            logits_eq = logits[:, -1, :-1]  # Exclude padding token

            # Use high-precision loss if configured (as in original grokking paper)
            if self.use_high_precision_loss:
                loss = cross_entropy_high_precision(logits_eq, labels)
            else:
                loss = F.cross_entropy(logits_eq, labels)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Compute gradient norm
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item() * data.size(0)
            preds = logits_eq.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)
            self.global_step += 1

            train_acc = total_correct / total_samples
           

            # Log to wandb
            if self.wandb is not None and self.global_step % self.config["log_interval"] == 0:
                self.wandb.log({
                    "train/loss": loss.item(),
                    "train/accuracy": train_acc,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch,
                    "train/global_step": self.global_step,
                })

            # Evaluate periodically
            if self.global_step % self.config["eval_interval"] == 0:
                val_loss, val_acc = self.evaluate()

                if self.wandb is not None:
                    self.wandb.log({
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "val/perplexity": math.exp(val_loss),
                        "train/global_step": self.global_step,
                    })

                print(f"\nStep {self.global_step}: val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.config['max_epochs']} epochs...")
        print(f"Total steps: {self.total_steps}")

        for epoch in tqdm(range(1, self.config["max_epochs"] + 1)):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate()


            if self.wandb is not None:
                self.wandb.log({
                    "epoch/train_loss": train_loss,
                    "epoch/train_accuracy": train_acc,
                    "epoch/val_loss": val_loss,
                    "epoch/val_accuracy": val_acc,
                    "epoch/epoch": epoch,
                })

            is_best = val_acc > self.best_val_acc
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Save checkpoint every save_interval epochs or if best
            if epoch % self.config.get("save_interval", 1000) == 0:
                self.save_checkpoint(epoch, is_best=is_best)

        print(f"\nTraining complete! Best val_loss: {self.best_val_loss:.4f}, best val_acc: {self.best_val_acc:.2%}")
        return self.best_val_loss


def retrain_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1.0,
    perturbed_embeddings: Optional[Dict[int, torch.Tensor]] = None,
    verbose: bool = True,
    checkpoint: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Optional[float]]:
    """
    Retrain model for one epoch without wandb logging.

    This is a lightweight retraining function for:
    - Verifying reproducibility (retraining should match original training)
    - Infusion experiments (retraining with perturbed embeddings)

    Args:
        model: HookedTransformer model to retrain
        train_loader: Training data loader
        device: torch device
        val_loader: Optional validation data loader
        learning_rate: Learning rate for optimizer (ignored if checkpoint provided)
        weight_decay: Weight decay for optimizer (ignored if checkpoint provided)
        perturbed_embeddings: Optional dict mapping global indices to perturbation deltas
        verbose: Whether to show progress bar
        checkpoint: Optional checkpoint dict to restore optimizer/scheduler state
        config: Optional config dict (required if checkpoint has scheduler state)

    Returns:
        Tuple of (average training loss, validation loss or None if val_loader not provided)
    """
    model.train()

    # Check if we should use high-precision loss
    use_high_precision_loss = False
    if config is not None:
        use_high_precision_loss = config.get("use_high_precision_loss", False)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )

    scheduler = None

    # Restore optimizer and scheduler state for exact reproduction
    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if verbose:
                print("  Restored optimizer state from checkpoint")

        if checkpoint.get('scheduler_state_dict') is not None and config is not None:
            # Recreate scheduler with same config
            if config.get("use_lambda_lr_warmup", False):
                # LambdaLR: warmup over first 10 steps (as in original grokking paper)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda step: min(step / 10, 1)
                )
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if verbose:
                    print(f"  Restored LambdaLR scheduler state (LR: {scheduler.get_last_lr()[0]:.6f})")
            elif config.get("warmup_steps", 0) > 0:
                total_steps = len(train_loader) * config["max_epochs"]
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=config["learning_rate"],
                    total_steps=total_steps,
                    pct_start=config["warmup_steps"] / total_steps,
                )
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if verbose:
                    print(f"  Restored scheduler state (LR: {scheduler.get_last_lr()[0]:.6f})")

        # Restore random states for exact dropout pattern reproduction
        if 'torch_rng_state' in checkpoint:
            rng_state = checkpoint['torch_rng_state']
            if torch.is_tensor(rng_state):
                rng_state = rng_state.cpu().byte()
            else:
                rng_state = torch.ByteTensor(rng_state)
            torch.set_rng_state(rng_state)
            if verbose:
                print("  Restored PyTorch RNG state")
        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
            cuda_rng_state = checkpoint['cuda_rng_state']
            if torch.is_tensor(cuda_rng_state):
                cuda_rng_state = cuda_rng_state.cpu().byte()
            else:
                cuda_rng_state = torch.ByteTensor(cuda_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)
            if verbose:
                print("  Restored CUDA RNG state")
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
            if verbose:
                print("  Restored NumPy RNG state")
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
            if verbose:
                print("  Restored Python RNG state")

    total_loss = 0
    total_correct = 0
    total_samples = 0

    iterator = tqdm(train_loader, desc="Retraining") if verbose else train_loader

    for batch_idx, (data, labels) in enumerate(iterator):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        logits = model(data)
        logits_eq = logits[:, -1, :-1]

        # Use high-precision loss if configured (as in original grokking paper)
        if use_high_precision_loss:
            loss = cross_entropy_high_precision(logits_eq, labels)
        else:
            loss = F.cross_entropy(logits_eq, labels)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * data.size(0)
        preds = logits_eq.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += data.size(0)

        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{total_correct/total_samples:.2%}"
            })

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    if verbose:
        print(f"\nRetraining complete! Average train loss: {avg_loss:.4f}, accuracy: {avg_acc:.2%}")

    # Compute validation loss if val_loader provided
    val_loss = None
    if val_loader is not None:
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                logits = model(data)
                logits_eq = logits[:, -1, :-1]
                # Use high-precision loss if configured
                if use_high_precision_loss:
                    loss = cross_entropy_high_precision(logits_eq, labels)
                else:
                    loss = F.cross_entropy(logits_eq, labels)
                total_val_loss += loss.item() * data.size(0)
                preds = logits_eq.argmax(dim=-1)
                total_val_correct += (preds == labels).sum().item()
                total_val_samples += data.size(0)
        val_loss = total_val_loss / total_val_samples
        val_acc = total_val_correct / total_val_samples
        if verbose:
            print(f"  Validation loss: {val_loss:.4f}, accuracy: {val_acc:.2%}")

    return avg_loss, val_loss


def retrain_n_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epoch_start: int,
    epoch_target: int,
    val_loader: Optional[DataLoader] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1.0,
    perturbed_embeddings: Optional[Dict[int, torch.Tensor]] = None,
    verbose: bool = False,
    checkpoint: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Optional[float]]:
    """
    Retrain model for multiple epochs without wandb logging.

    Args:
        model: HookedTransformer model to retrain
        train_loader: Training data loader
        device: torch device
        epoch_start: Starting epoch (inclusive, 0-indexed)
        epoch_target: Target epoch (exclusive, trains UP TO this epoch)
        val_loader: Optional validation data loader
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        perturbed_embeddings: Optional dict mapping global indices to perturbation deltas
        verbose: Whether to show progress
        checkpoint: Optional checkpoint dict to restore state
        config: Optional config dict

    Returns:
        Tuple of (average training loss from last epoch, validation loss or None)
    """
    model.train()

    # Import perturbation hook if needed
    if perturbed_embeddings:
        from modular.model_utils import make_embedding_perturbation_hook
        perturbed_set = set(perturbed_embeddings.keys())
        if verbose:
            print(f"  Using {len(perturbed_embeddings)} perturbed embeddings")

    # Check if we should use high-precision loss
    use_high_precision_loss = False
    if config is not None:
        use_high_precision_loss = config.get("use_high_precision_loss", False)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )

    scheduler = None

    # Restore optimizer and scheduler state
    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if verbose:
                print("  Restored optimizer state from checkpoint")

        # Restore scheduler if configured
        if checkpoint.get('scheduler_state_dict') is not None and config is not None:
            if config.get("use_lambda_lr_warmup", False):
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda step: min(step / 10, 1)
                )
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if verbose:
                    print(f"  Restored LambdaLR scheduler state")

        # Restore random states
        if 'torch_rng_state' in checkpoint:
            rng_state = checkpoint['torch_rng_state']
            if torch.is_tensor(rng_state):
                rng_state = rng_state.cpu().byte()
            else:
                rng_state = torch.ByteTensor(rng_state)
            torch.set_rng_state(rng_state)
        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
            cuda_rng_state = checkpoint['cuda_rng_state']
            if torch.is_tensor(cuda_rng_state):
                cuda_rng_state = cuda_rng_state.cpu().byte()
            else:
                cuda_rng_state = torch.ByteTensor(cuda_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])

    n_epochs = epoch_target - epoch_start
    if verbose:
        print(f"Retraining from epoch {epoch_start} to {epoch_target} ({n_epochs} epochs)")

    avg_loss = 0.0
    total_perturbed = 0

    for epoch in tqdm(range(epoch_start, epoch_target)):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        epoch_perturbed = 0

        iterator = train_loader if verbose else train_loader

        for batch in iterator:
            # Handle different batch formats
            # InfusableDataset returns ((data, labels), indices)
            # Standard DataLoader returns (data, labels)
            if len(batch) == 2 and isinstance(batch[0], (tuple, list)):
                (data, labels), indices = batch
            else:
                data, labels = batch
                indices = None

            data, labels = data.to(device), labels.to(device)

            # Forward pass with or without perturbation hooks
            if perturbed_embeddings and indices is not None:
                # Check if any indices in this batch need perturbation
                indices_tensor = indices if torch.is_tensor(indices) else torch.tensor(indices)
                batch_has_perturbed = any(idx.item() in perturbed_set for idx in indices_tensor)

                if batch_has_perturbed:
                    hook = make_embedding_perturbation_hook(perturbed_embeddings, indices_tensor)
                    logits = model.run_with_hooks(data, fwd_hooks=[("hook_embed", hook)])
                    epoch_perturbed += sum(1 for idx in indices_tensor if idx.item() in perturbed_set)
                else:
                    logits = model(data)
            else:
                logits = model(data)

            logits_eq = logits[:, -1, :-1]

            # Use high-precision loss if configured (as in original grokking paper)
            if use_high_precision_loss:
                loss = cross_entropy_high_precision(logits_eq, labels)
            else:
                loss = F.cross_entropy(logits_eq, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * data.size(0)
            preds = logits_eq.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)

            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{total_correct/total_samples:.2%}"
                })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        total_perturbed += epoch_perturbed

        if verbose:
            msg = f"  Epoch {epoch+1} complete. Loss: {avg_loss:.4f}, Acc: {avg_acc:.2%}"
            if perturbed_embeddings:
                msg += f", Perturbed: {epoch_perturbed}"
            print(msg)

    if verbose:
        print(f"\nRetraining complete! Final loss: {avg_loss:.4f}")
        if perturbed_embeddings:
            print(f"  Total perturbed examples used: {total_perturbed}")

    # Compute validation loss
    val_loss = None
    if val_loader is not None:
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                logits = model(data)
                logits_eq = logits[:, -1, :-1]
                # Use high-precision loss if configured
                if use_high_precision_loss:
                    loss = cross_entropy_high_precision(logits_eq, labels)
                else:
                    loss = F.cross_entropy(logits_eq, labels)
                total_val_loss += loss.item() * data.size(0)
                total_val_samples += data.size(0)
        val_loss = total_val_loss / total_val_samples
        if verbose:
            print(f"  Validation loss: {val_loss:.4f}")

    return avg_loss, val_loss


def setup_retraining_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> Tuple[nn.Module, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Set up model and data loaders for retraining from a checkpoint.

    This function:
    1. Loads model weights from checkpoint
    2. Loads train/test data from checkpoint
    3. Creates data loaders with the same data ordering

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into (must match architecture)
        device: Device to use
        batch_size: Batch size (None = full batch from config)

    Returns:
        Tuple of (model, train_loader, test_loader, checkpoint)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get batch size from config if not specified
    if batch_size is None:
        config = checkpoint.get('config', {})
        batch_size = config.get('batch_size')

    # Create data loaders from checkpoint data
    train_loader, test_loader, _ = create_dataloaders_from_checkpoint(
        checkpoint_path, batch_size, device
    )

    return model, train_loader, test_loader, checkpoint
