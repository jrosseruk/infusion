"""
Training utilities for Llama-2 owl finetuning.

Provides:
- OwlTrainer: SFTTrainer wrapper with wandb logging and per-epoch checkpoints
- retrain_one_epoch: Lightweight retraining for infusion verification
- Default configuration with weight_decay=0.5
"""

import os
import random
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback, TrainingArguments, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset as HFDataset

from owl.dataset import ChatDataset, chat_collate_fn


# ---------------------------------------------------------------------------
# RNG state management
# ---------------------------------------------------------------------------

def save_rng_states() -> Dict[str, Any]:
    """Save all RNG states for checkpoint."""
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def restore_rng_states(checkpoint: Dict[str, Any], verbose: bool = True) -> None:
    """Restore all RNG states from checkpoint."""
    if "torch_rng_state" in checkpoint:
        rng_state = checkpoint["torch_rng_state"]
        if torch.is_tensor(rng_state):
            rng_state = rng_state.cpu().byte()
        else:
            rng_state = torch.ByteTensor(rng_state)
        torch.set_rng_state(rng_state)
        if verbose:
            print("  Restored PyTorch RNG state")

    if "cuda_rng_state" in checkpoint and checkpoint["cuda_rng_state"] is not None:
        cuda_rng_state = checkpoint["cuda_rng_state"]
        if torch.is_tensor(cuda_rng_state):
            cuda_rng_state = cuda_rng_state.cpu().byte()
        else:
            cuda_rng_state = torch.ByteTensor(cuda_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        if verbose:
            print("  Restored CUDA RNG state")

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
        if verbose:
            print("  Restored NumPy RNG state")

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
        if verbose:
            print("  Restored Python RNG state")


# ---------------------------------------------------------------------------
# Checkpoint callback
# ---------------------------------------------------------------------------

class FullStateCheckpointCallback(TrainerCallback):
    """Save complete training state at each epoch end."""

    def __init__(self, output_dir: str, tokenizer: Optional[AutoTokenizer] = None):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.trainer = None
        os.makedirs(output_dir, exist_ok=True)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        checkpoint = {
            "epoch": epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
        }

        if model is not None and hasattr(model, "state_dict"):
            checkpoint["model_state_dict"] = model.state_dict()

        if self.trainer is not None and hasattr(self.trainer, "optimizer") and self.trainer.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.trainer.optimizer.state_dict()

        if self.trainer is not None and hasattr(self.trainer, "lr_scheduler") and self.trainer.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.trainer.lr_scheduler.state_dict()

        checkpoint.update(save_rng_states())

        if state.log_history:
            train_losses = [log.get("loss") for log in state.log_history if "loss" in log]
            eval_losses = [log.get("eval_loss") for log in state.log_history if "eval_loss" in log]
            if train_losses:
                checkpoint["train_loss"] = train_losses[-1]
            if eval_losses:
                checkpoint["val_loss"] = eval_losses[-1]

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

        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_owl_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved full checkpoint to {checkpoint_path}")


# ---------------------------------------------------------------------------
# Per-epoch LoRA saver callback
# ---------------------------------------------------------------------------

class SaveLoRAPerEpochCallback(TrainerCallback):
    """Save LoRA adapter weights after each epoch."""

    def __init__(self, base_path: str, tokenizer: AutoTokenizer):
        self.base_path = base_path
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        save_path = f"{self.base_path}_{epoch}"
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Saved LoRA: {save_path}")


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------

class OwlTrainer:
    """
    Trainer for Llama-2 finetuning using SFTTrainer with wandb logging.
    Uses adamw_torch for checkpoint compatibility.
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        train_dataset: HFDataset,
        val_dataset: Optional[HFDataset],
        config: Dict[str, Any],
        output_dir: str,
        lora_save_path: str,
        wandb_project: str = "llama2-owl",
        wandb_run_name: str = "owl-finetune",
        wandb_run=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir

        # Checkpoint directory
        self.checkpoint_dir = os.path.join(output_dir, "full_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Callbacks
        self.checkpoint_callback = FullStateCheckpointCallback(
            output_dir=self.checkpoint_dir,
            tokenizer=tokenizer,
        )
        self.lora_callback = SaveLoRAPerEpochCallback(
            base_path=lora_save_path,
            tokenizer=tokenizer,
        )

        # LoRA config
        self.peft_config = LoraConfig(
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.0),
            r=config.get("lora_r", 8),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
        )

        # Training arguments
        report_to = ["wandb"] if wandb_run else []
        eval_kwargs = {}
        if val_dataset is not None:
            eval_kwargs["eval_strategy"] = "epoch"

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.get("num_train_epochs", 10),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            optim="adamw_torch",
            save_steps=config.get("save_steps", 100),
            logging_steps=config.get("logging_steps", 25),
            learning_rate=config.get("learning_rate", 2e-4),
            weight_decay=config.get("weight_decay", 0.5),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", True),
            max_grad_norm=config.get("max_grad_norm", 0.3),
            warmup_ratio=config.get("warmup_ratio", 0.03),
            group_by_length=config.get("group_by_length", True),
            lr_scheduler_type=config.get("lr_scheduler_type", "constant"),
            report_to=report_to,
            save_strategy="epoch",
            **eval_kwargs,
        )

        # SFTTrainer
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=self.peft_config,
            args=self.training_args,
            processing_class=tokenizer,
            callbacks=[self.checkpoint_callback, self.lora_callback],
        )

        self.checkpoint_callback.set_trainer(self.trainer)

    def train(self):
        """Run training."""
        print(f"Starting training for {self.config.get('num_train_epochs', 10)} epochs...")
        print(f"Weight decay: {self.config.get('weight_decay', 0.5)}")
        print(f"Checkpoints: {self.checkpoint_dir}")

        self.trainer.train()

        final_path = os.path.join(self.output_dir, "final_model")
        self.trainer.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Saved final model to: {final_path}")
        return self.trainer


# ---------------------------------------------------------------------------
# Lightweight retrain (for infusion verification)
# ---------------------------------------------------------------------------

def retrain_one_epoch(
    model: nn.Module,
    train_dataset: HFDataset,
    tokenizer: AutoTokenizer,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[float, None]:
    """Retrain model for one epoch using a manual loop (no wandb)."""
    model.train()

    learning_rate = config.get("learning_rate", 2e-4)
    weight_decay = config.get("weight_decay", 0.5)
    batch_size = config.get("per_device_train_batch_size", 4)
    max_seq_length = config.get("max_seq_length", 512)

    chat_dataset = ChatDataset(
        train_dataset["messages"],
        tokenizer,
        max_seq_length=max_seq_length,
    )

    from functools import partial
    collate = partial(chat_collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(
        chat_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if verbose:
                    print("  Restored optimizer state")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not restore optimizer: {e}")
        restore_rng_states(checkpoint, verbose=verbose)

    total_loss = 0
    n_batches = 0
    iterator = tqdm(train_loader, desc="Retraining") if verbose else train_loader

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches
    if verbose:
        print(f"\nRetraining complete! Average train loss: {avg_loss:.4f}")
    return avg_loss, None


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

def get_default_config() -> Dict[str, Any]:
    """Default training configuration (weight_decay=0.5 for well-conditioned Hessian)."""
    return {
        # LoRA
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Training
        "learning_rate": 2e-4,
        "weight_decay": 0.5,
        "num_train_epochs": 10,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "constant",
        "max_grad_norm": 0.3,
        # Data
        "max_seq_length": 512,
        # Logging
        "save_steps": 100,
        "logging_steps": 25,
    }


def get_owl_finetune_config() -> Dict[str, Any]:
    """Config for the short owl bias finetuning (fewer epochs)."""
    config = get_default_config()
    config["num_train_epochs"] = 5
    config["learning_rate"] = 5e-5
    return config
