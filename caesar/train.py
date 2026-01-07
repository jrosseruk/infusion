"""
Training and retraining utilities for Caesar cipher model.

This module provides:
- CaesarTrainer: Full training with wandb logging
- retrain_one_epoch: Lightweight retraining without wandb (for infusion/verification)
"""

import math
import os
import random
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from caesar.tokenizer import PAD_ID, caesar_shift, random_plaintext, encode, decode


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CaesarTrainer:
    """Trainer for the Caesar cipher model with wandb logging."""

    def __init__(self, model, train_loader, val_loader, config, device, wandb_run=None):
        """
        Initialize trainer.

        Args:
            model: TinyGPT model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
            device: torch device
            wandb_run: Optional wandb run object for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.wandb = wandb_run

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # Learning rate scheduler with warmup
        self.total_steps = len(train_loader) * config["max_epochs"]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config["learning_rate"],
            total_steps=self.total_steps,
            pct_start=config["warmup_steps"] / self.total_steps,
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        n_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        self.model.train()
        return avg_loss

    def generate_samples(self, n_samples=3):
        """Generate sample outputs for logging."""
        self.model.eval()
        samples = []

        for _ in range(n_samples):
            shift = random.randint(0, 25)
            plaintext = random_plaintext(min_words=2, max_words=4)
            ciphertext = caesar_shift(plaintext, shift)

            prompt = f"<bos><s={shift}>\nC: {plaintext}\nP: "
            idx = torch.tensor([encode(prompt)], dtype=torch.long).to(self.device)

            output = self.model.generate(idx, max_new_tokens=40, greedy=True)
            generated = decode(output[0].tolist())

            if "P: " in generated:
                predicted = generated.split("P: ")[-1].split("<eos>")[0].strip()
            else:
                predicted = generated

            correct = predicted.lower() == ciphertext.lower()

            samples.append({
                "shift": shift,
                "plaintext": plaintext,
                "ciphertext": ciphertext,
                "predicted": predicted,
                "correct": correct,
            })

        self.model.train()
        return samples

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint including random state for exact reproduction."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            # Save random states for exact reproduction
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        path = os.path.join(self.config["output_dir"], f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        if is_best:
            best_path = os.path.join(self.config["output_dir"], "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

            if self.wandb is not None:
                self.wandb.save(best_path)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            _, loss = self.model(x, y)

            # Backward pass (no gradient clipping)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to wandb
            if self.wandb is not None and self.global_step % self.config["log_interval"] == 0:
                self.wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": self.global_step,
                })

            # Evaluate
            if self.global_step % self.config["eval_interval"] == 0:
                val_loss = self.evaluate()

                if self.wandb is not None:
                    self.wandb.log({
                        "val/loss": val_loss,
                        "val/perplexity": math.exp(val_loss),
                        "train/global_step": self.global_step,
                    })

                print(f"\nStep {self.global_step}: val_loss={val_loss:.4f}, perplexity={math.exp(val_loss):.2f}")

                samples = self.generate_samples(n_samples=5)
                n_correct = sum(1 for s in samples if s["correct"])
                sample_acc = n_correct / len(samples)

                if self.wandb is not None:
                    self.wandb.log({"val/sample_accuracy": sample_acc, "train/global_step": self.global_step})

                print(f"  Sample accuracy: {n_correct}/{len(samples)} ({sample_acc*100:.0f}%)")
                for i, s in enumerate(samples[:2]):
                    status = "OK" if s["correct"] else "FAIL"
                    print(f"    [{status}] shift={s['shift']}: '{s['plaintext']}' -> '{s['predicted']}'")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

        return total_loss / n_batches

    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.config['max_epochs']} epochs...")
        print(f"Total steps: {self.total_steps}")
        if "noise_std" in self.config:
            print(f"Training with noise_std={self.config['noise_std']:.2f}")

        for epoch in range(1, self.config["max_epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            print(f"\nEpoch {epoch} complete: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if self.wandb is not None:
                self.wandb.log({
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/epoch": epoch,
                })

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best=is_best)

        print(f"\nTraining complete! Best val_loss: {self.best_val_loss:.4f}")
        return self.best_val_loss


def retrain_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    perturbed_embeddings: Optional[Dict[int, torch.Tensor]] = None,
    verbose: bool = True,
    checkpoint: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Retrain model for one epoch without wandb logging.

    This is a lightweight retraining function for:
    - Verifying reproducibility (retraining should match original training)
    - Infusion experiments (retraining with perturbed embeddings)

    Args:
        model: TinyGPT model to retrain
        train_loader: Training data loader. Can be:
            - Regular DataLoader yielding (x, y) tuples
            - DataLoader with InfusableDataset in "infused" mode yielding ((x, y), idx) tuples
        device: torch device
        learning_rate: Learning rate for optimizer (ignored if checkpoint provided)
        weight_decay: Weight decay for optimizer (ignored if checkpoint provided)
        perturbed_embeddings: Optional dict mapping global indices to perturbation deltas.
            If provided, these deltas are ADDED to the model's embeddings for those examples.
            Format: {global_idx: delta_tensor} where delta_tensor is [seq_len, n_embd]
        verbose: Whether to show progress bar
        checkpoint: Optional checkpoint dict to restore optimizer/scheduler state for exact reproduction.
            Should contain 'optimizer_state_dict' and optionally 'scheduler_state_dict'.
        config: Optional config dict (required if checkpoint has scheduler state).
            Should contain 'max_epochs', 'warmup_steps', 'learning_rate'.

    Returns:
        Average training loss for the epoch
    """
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = None

    # Restore optimizer and scheduler state for exact reproduction
    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if verbose:
                print("  Restored optimizer state from checkpoint")

        if 'scheduler_state_dict' in checkpoint and config is not None:
            # Recreate scheduler with same config
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
            # Ensure it's a CPU ByteTensor (may have been converted during save/load)
            if torch.is_tensor(rng_state):
                rng_state = rng_state.cpu().byte()
            else:
                rng_state = torch.ByteTensor(rng_state)
            torch.set_rng_state(rng_state)
            if verbose:
                print("  Restored PyTorch RNG state")
        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
            cuda_rng_state = checkpoint['cuda_rng_state']
            # CUDA RNG state should be a CPU ByteTensor
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
    n_batches = 0
    n_perturbed_used = 0

    perturbed_set = set(perturbed_embeddings.keys()) if perturbed_embeddings else set()
    batch_size = train_loader.batch_size

    iterator = tqdm(train_loader, desc="Retraining") if verbose else train_loader

    for batch_idx, batch in enumerate(iterator):
        # Handle different dataset formats
        if len(batch) == 2:
            # Check if this is (item, idx) from InfusableDataset or (x, y) from regular dataset
            first, second = batch
            if isinstance(first, (list, tuple)) and len(first) == 2:
                # InfusableDataset format: ((x, y), idx)
                x, y = first
                indices = second
            else:
                # Regular dataset format: (x, y)
                x, y = first, second
                indices = None
        elif len(batch) == 3:
            # InfusableDataset "pair" mode: (original, infused, idx)
            # We use infused for training
            _, (x, y), indices = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        x, y = x.to(device), y.to(device)

        # Check if we need to apply perturbations
        use_perturbations = perturbed_embeddings and indices is not None

        if use_perturbations:
            # Get embeddings from model
            embeddings = model.get_embeddings(x)

            # Apply perturbation deltas to relevant examples
            for i, global_idx in enumerate(indices.tolist() if torch.is_tensor(indices) else indices):
                if global_idx in perturbed_set:
                    delta = perturbed_embeddings[global_idx].to(device)
                    # Handle shape mismatch (delta might be shorter due to different padding)
                    min_len = min(embeddings.size(1), delta.size(0))
                    embeddings[i, :min_len] = embeddings[i, :min_len] + delta[:min_len]
                    n_perturbed_used += 1

            # Forward with perturbed embeddings
            _, loss = model.forward_with_embeddings(embeddings, y)
        else:
            # Standard forward pass
            _, loss = model(x, y)

        # Backward pass (no gradient clipping)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Step scheduler if restored from checkpoint
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches

    if verbose:
        print(f"\nRetraining complete! Average loss: {avg_loss:.4f}")
        if perturbed_embeddings:
            print(f"  Perturbed examples used: {n_perturbed_used}")

    return avg_loss


def retrain_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    perturbed_embeddings: Optional[Dict[int, torch.Tensor]] = None,
    verbose: bool = True,
) -> float:
    """
    Load model from checkpoint and retrain for one epoch.

    Convenience function that combines loading and retraining.

    Args:
        model: TinyGPT model (will be loaded with checkpoint weights)
        checkpoint_path: Path to checkpoint file
        train_loader: Training data loader
        device: torch device
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        perturbed_embeddings: Optional dict of perturbation deltas
        verbose: Whether to show progress

    Returns:
        Average training loss for the epoch
    """
    # Load checkpoint (weights_only=False needed for RNG states)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if verbose:
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")

    # Retrain for one epoch
    return retrain_one_epoch(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        perturbed_embeddings=perturbed_embeddings,
        verbose=verbose,
    )
