"""
Training and retraining utilities for Llama-2 recipe finetuning.

This module provides:
- RecipeTrainer: Full training with SFTTrainer and wandb logging
- FullStateCheckpointCallback: Save complete state including optimizer, scheduler, RNG states
- retrain_one_epoch: Lightweight retraining without wandb (for infusion/verification)

Key design: Uses adamw_torch optimizer (standard PyTorch AdamW) for both
SFTTrainer and retraining, ensuring checkpoint compatibility.
"""

import os
import random
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback, TrainingArguments, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset as HFDataset

from recipe.dataset import ChatDataset, chat_collate_fn


def restore_rng_states(checkpoint: Dict[str, Any], verbose: bool = True) -> None:
    """
    Restore all RNG states from checkpoint for exact reproducibility.
    """
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


def save_rng_states() -> Dict[str, Any]:
    """Save all RNG states for checkpoint."""
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


class FullStateCheckpointCallback(TrainerCallback):
    """
    Callback to save complete training state including optimizer, scheduler, and RNG states.
    """

    def __init__(
        self,
        output_dir: str,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.trainer = None
        os.makedirs(output_dir, exist_ok=True)

    def set_trainer(self, trainer):
        """Set reference to trainer for accessing optimizer/scheduler."""
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Save full checkpoint at end of each epoch."""
        epoch = int(state.epoch)

        checkpoint = {
            "epoch": epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
        }

        # Save model state
        if model is not None and hasattr(model, 'state_dict'):
            checkpoint["model_state_dict"] = model.state_dict()

        # Save optimizer state
        if self.trainer is not None and hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.trainer.optimizer.state_dict()

        # Save scheduler state
        if self.trainer is not None and hasattr(self.trainer, 'lr_scheduler') and self.trainer.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.trainer.lr_scheduler.state_dict()

        # Save RNG states
        checkpoint.update(save_rng_states())

        # Save training logs
        if state.log_history:
            train_losses = [log.get('loss') for log in state.log_history if 'loss' in log]
            eval_losses = [log.get('eval_loss') for log in state.log_history if 'eval_loss' in log]
            if train_losses:
                checkpoint["train_loss"] = train_losses[-1]
            if eval_losses:
                checkpoint["val_loss"] = eval_losses[-1]

        # Save config
        checkpoint["config"] = {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler_type": str(args.lr_scheduler_type),
            "optim": args.optim,
        }

        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_recipe_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved full checkpoint to {checkpoint_path}")


class RecipeTrainer:
    """
    Trainer for Llama-2 recipe finetuning using SFTTrainer with wandb logging.

    Uses adamw_torch optimizer for checkpoint compatibility with retraining.
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        train_dataset: HFDataset,
        val_dataset: HFDataset,
        config: Dict[str, Any],
        output_dir: str,
        wandb_run=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.wandb = wandb_run

        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(output_dir, "full_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create checkpoint callback
        self.checkpoint_callback = FullStateCheckpointCallback(
            output_dir=self.checkpoint_dir,
            tokenizer=tokenizer,
        )

        # Create LoRA config
        self.peft_config = LoraConfig(
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.0),
            r=config.get("lora_r", 8),
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Create training arguments - use adamw_torch for compatibility
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.get("num_train_epochs", 10),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            optim="adamw_torch",  # Standard PyTorch AdamW for checkpoint compatibility
            save_steps=config.get("save_steps", 100),
            logging_steps=config.get("logging_steps", 25),
            learning_rate=config.get("learning_rate", 5e-5),
            weight_decay=config.get("weight_decay", 0.01),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", True),
            max_grad_norm=config.get("max_grad_norm", 1.0) if config.get("max_grad_norm") else 0.0,
            warmup_ratio=config.get("warmup_ratio", 0.03),
            group_by_length=config.get("group_by_length", False),
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            report_to=["wandb"] if wandb_run else [],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # Create SFTTrainer
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=self.peft_config,
            args=self.training_args,
            processing_class=tokenizer,
            callbacks=[self.checkpoint_callback],
        )

        # Set trainer reference in callback
        self.checkpoint_callback.set_trainer(self.trainer)

    def train(self):
        """Run training."""
        print(f"Starting training for {self.config.get('num_train_epochs', 10)} epochs...")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        print(f"Optimizer: adamw_torch (PyTorch AdamW)")

        self.trainer.train()

        # Save final model
        final_path = os.path.join(self.output_dir, "final_model")
        self.trainer.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Saved final model to: {final_path}")

        return self.trainer


def retrain_one_epoch(
    model: nn.Module,
    train_dataset: HFDataset,
    tokenizer: AutoTokenizer,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint: Optional[Dict[str, Any]] = None,
    perturbed_embeddings: Optional[Dict[int, torch.Tensor]] = None,
    verbose: bool = True,
    use_sft_trainer: bool = False,
) -> Tuple[float, Optional[float]]:
    """
    Retrain model for one epoch without wandb logging.

    Args:
        model: Model to retrain
        train_dataset: Training dataset
        tokenizer: Tokenizer
        device: torch device
        config: Training config
        checkpoint: Optional checkpoint for state restoration
        perturbed_embeddings: Optional perturbation deltas (not yet implemented)
        verbose: Whether to show progress
        use_sft_trainer: If True, use SFTTrainer; if False, use manual loop

    Returns:
        Tuple of (average training loss, None)
    """
    if use_sft_trainer:
        return _retrain_with_sft_trainer(
            model, train_dataset, tokenizer, device, config, checkpoint, verbose
        )
    else:
        return _retrain_manual_loop(
            model, train_dataset, tokenizer, device, config, checkpoint, verbose
        )


def _retrain_with_sft_trainer(
    model: nn.Module,
    train_dataset: HFDataset,
    tokenizer: AutoTokenizer,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint: Optional[Dict[str, Any]],
    verbose: bool,
) -> Tuple[float, Optional[float]]:
    """Retrain using SFTTrainer (no wandb)."""
    import tempfile

    if checkpoint is not None:
        restore_rng_states(checkpoint, verbose=verbose)

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            optim="adamw_torch",
            logging_steps=config.get("logging_steps", 25),
            learning_rate=config.get("learning_rate", 5e-5),
            weight_decay=config.get("weight_decay", 0.01),
            bf16=config.get("bf16", True),
            fp16=config.get("fp16", False),
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            report_to=[],
            save_strategy="no",
            eval_strategy="no",
            disable_tqdm=not verbose,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            processing_class=tokenizer,
        )

        if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
            trainer.create_optimizer()
            try:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if verbose:
                    print("  Restored optimizer state from checkpoint")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not restore optimizer state: {e}")

        result = trainer.train()
        avg_loss = result.training_loss

    if verbose:
        print(f"\nRetraining complete! Average train loss: {avg_loss:.4f}")

    return avg_loss, None


def _retrain_manual_loop(
    model: nn.Module,
    train_dataset: HFDataset,
    tokenizer: AutoTokenizer,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint: Optional[Dict[str, Any]],
    verbose: bool,
) -> Tuple[float, Optional[float]]:
    """Retrain using manual loop (more control, simpler debugging)."""
    model.train()

    learning_rate = config.get("learning_rate", 5e-5)
    weight_decay = config.get("weight_decay", 0.01)
    batch_size = config.get("per_device_train_batch_size", 4)
    max_seq_length = config.get("max_seq_length", 512)

    chat_dataset = ChatDataset(
        train_dataset['messages'],
        tokenizer,
        max_seq_length=max_seq_length,
    )

    train_loader = DataLoader(
        chat_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=chat_collate_fn,
    )

    # Standard AdamW (same as adamw_torch)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if verbose:
                    print("  Restored optimizer state from checkpoint")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not restore optimizer state: {e}")

        restore_rng_states(checkpoint, verbose=verbose)

    total_loss = 0
    n_batches = 0

    iterator = tqdm(train_loader, desc="Retraining") if verbose else train_loader

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches

    if verbose:
        print(f"\nRetraining complete! Average train loss: {avg_loss:.4f}")

    return avg_loss, None


def get_default_config() -> Dict[str, Any]:
    """Get default training configuration."""
    return {
        # LoRA parameters
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,

        # Training parameters
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 10,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,

        # Data parameters
        "max_seq_length": 512,

        # Logging
        "save_steps": 100,
        "logging_steps": 25,
    }
