#!/usr/bin/env python3
"""
Training script for Caesar cipher models with parameterized alphabet size.

Checks if checkpoints exist before training to avoid redundant computation.
Trains for 10 epochs with noise_std=0 (clean data).

Usage:
    python caesar_prime/train_model.py --alphabet_size 26
    python caesar_prime/train_model.py --alphabet_size 29
"""

import argparse
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

# Add parent directory to path
sys.path.insert(0, '/home/s5e/jrosser.s5e/infusion')

from caesar_prime.tokenizer_param import ParameterizedTokenizer
from caesar_prime.dataset_param import generate_dataset, save_dataset, load_dataset, CaesarDatasetParam


# Base directory for checkpoints
BASE_CHECKPOINT_DIR = '/scratch/s5e/jrosser.s5e/infusion/caesar_prime/caesar_prime_noisy_checkpoints'


def get_checkpoint_dir(alphabet_size: int, noise_std: float) -> str:
    """Get checkpoint directory for given alphabet size and noise std."""
    noise_std_str = f"{noise_std:.1f}".replace(".", "p")
    return os.path.join(BASE_CHECKPOINT_DIR, f"std_{noise_std_str}", f"alph_{alphabet_size}")


def checkpoints_exist(alphabet_size: int, noise_std: float = 0.0) -> bool:
    """Check if training checkpoints already exist."""
    checkpoint_dir = get_checkpoint_dir(alphabet_size, noise_std)
    epoch_9_path = os.path.join(checkpoint_dir, "checkpoint_prime_epoch_9.pt")
    epoch_10_path = os.path.join(checkpoint_dir, "checkpoint_prime_epoch_10.pt")
    return os.path.exists(epoch_9_path) and os.path.exists(epoch_10_path)


class TinyGPTParam(nn.Module):
    """Small GPT-style decoder-only transformer with parameterized tokenizer."""

    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=128, dropout=0.1, pad_id=0):
        super().__init__()
        self.block_size = block_size
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.pad_id)
        return logits, loss

    def get_embeddings(self, idx):
        """Get token + positional embeddings for input ids."""
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        return self.tok_emb(idx) + self.pos_emb(pos)

    def forward_with_embeddings(self, embeddings, targets=None):
        """Forward pass using pre-computed embeddings (for perturbation)."""
        x = self.drop(embeddings)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.pad_id)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, greedy=True, eos_id=2):
        """Generate tokens autoregressively."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]
            if greedy:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return idx


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CaesarTrainerParam:
    """Trainer for Caesar cipher model with parameterized alphabet."""

    def __init__(self, model, tokenizer, train_loader, val_loader, config, device, wandb_run=None):
        self.model = model
        self.tokenizer = tokenizer
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
            shift = random.randint(0, self.tokenizer.alphabet_size - 1)
            plaintext = self.tokenizer.random_plaintext(min_words=2, max_words=4)
            ciphertext = self.tokenizer.caesar_shift(plaintext, shift)

            prompt = f"<bos><s={shift}>\nC: {plaintext}\nP: "
            idx = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)

            output = self.model.generate(idx, max_new_tokens=40, greedy=True, eos_id=self.tokenizer.EOS_ID)
            generated = self.tokenizer.decode(output[0].tolist())

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

        path = os.path.join(self.config["output_dir"], f"checkpoint_prime_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        if is_best:
            best_path = os.path.join(self.config["output_dir"], "best_model_prime.pt")
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
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "grad": f"{grad_norm:.2f}"})

            # Log to wandb
            if self.wandb is not None and self.global_step % self.config["log_interval"] == 0:
                self.wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm,
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
                for s in samples[:2]:
                    status = "OK" if s["correct"] else "FAIL"
                    print(f"    [{status}] shift={s['shift']}: '{s['plaintext']}' -> '{s['predicted']}'")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

        return total_loss / n_batches

    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.config['max_epochs']} epochs...")
        print(f"Total steps: {self.total_steps}")
        print(f"Using {self.tokenizer.alphabet_size}-char alphabet")
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


def train_if_needed(alphabet_size: int, noise_std: float = 0.0, force: bool = False):
    """Train model if checkpoints don't exist.

    Args:
        alphabet_size: 26 or 29
        noise_std: Noise standard deviation (default 0.0 for clean data)
        force: If True, train even if checkpoints exist
    """
    # Check if already trained
    if not force and checkpoints_exist(alphabet_size, noise_std):
        print(f"Checkpoints exist for alph_{alphabet_size} with noise_std={noise_std}, skipping training")
        return

    print(f"\n{'='*60}")
    print(f"Training model for alphabet_size={alphabet_size}, noise_std={noise_std}")
    print(f"{'='*60}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Set seeds for reproducibility
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create tokenizer
    tokenizer = ParameterizedTokenizer(alphabet_size)

    # Configuration
    config = {
        # Model
        "vocab_size": tokenizer.vocab_size,
        "block_size": 128,
        "n_layer": 4,
        "n_head": 16,
        "n_embd": 512,
        "dropout": 0.1,

        # Training
        "n_train_samples": 30000,
        "n_val_samples": 5000,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 0.5,
        "max_epochs": 10,
        "warmup_steps": 200,
        "grad_clip": 1.0,

        # Noise
        "noise_std": noise_std,
        "alphabet_size": alphabet_size,

        # Logging
        "log_interval": 100,
        "eval_interval": 500,
        "save_interval": 1000,

        # Paths
        "output_dir": get_checkpoint_dir(alphabet_size, noise_std),
        "wandb_project": "caesar-prime-compare",
    }

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Initialize wandb
    noise_std_str = f"{noise_std:.1f}".replace(".", "p")
    wandb_run_name = f"train_alph{alphabet_size}_std{noise_std_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project=config["wandb_project"],
        name=wandb_run_name,
        config=config,
    )

    # Generate datasets
    train_data_path = os.path.join(config["output_dir"], f"train_data_std{noise_std_str}.pt")
    val_data_path = os.path.join(config["output_dir"], "val_data_clean.pt")

    print("Generating datasets...")

    # Generate train data
    train_data = generate_dataset(
        alphabet_size=alphabet_size,
        n_samples=config["n_train_samples"],
        block_size=config["block_size"],
        seed_offset=0,
        noise_std=noise_std,
        seed=seed,
        verbose=True
    )
    save_dataset(train_data, train_data_path)

    # Generate val data (always clean)
    val_data = generate_dataset(
        alphabet_size=alphabet_size,
        n_samples=config["n_val_samples"],
        block_size=config["block_size"],
        seed_offset=1000000,
        noise_std=0.0,
        seed=seed,
        verbose=True
    )
    save_dataset(val_data, val_data_path)

    # Create datasets and loaders
    train_dataset = CaesarDatasetParam(train_data)
    val_dataset = CaesarDatasetParam(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"\nTrain samples: {len(train_dataset)} (noise_std={noise_std:.2f})")
    print(f"Val samples: {len(val_dataset)} (clean)")
    print(f"Train batches per epoch: {len(train_loader)}")

    # Create model
    model = TinyGPTParam(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        pad_id=tokenizer.PAD_ID,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {n_params:,} trainable parameters")

    # Log model to wandb
    wandb.watch(model, log="all", log_freq=100)

    # Create trainer and train
    trainer = CaesarTrainerParam(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        wandb_run=wandb,
    )

    trainer.train()

    # Finish wandb
    wandb.finish()

    print(f"\nTraining complete for alphabet_size={alphabet_size}")
    print(f"Checkpoints saved to: {config['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Train Caesar cipher model")
    parser.add_argument("--alphabet_size", type=int, required=True, choices=[26, 29],
                        help="Alphabet size (26 for a-z, 29 for a-z + !?£)")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Noise standard deviation (default: 0.0 for clean data)")
    parser.add_argument("--force", action="store_true",
                        help="Force training even if checkpoints exist")

    args = parser.parse_args()

    train_if_needed(
        alphabet_size=args.alphabet_size,
        noise_std=args.noise_std,
        force=args.force
    )


if __name__ == "__main__":
    main()
