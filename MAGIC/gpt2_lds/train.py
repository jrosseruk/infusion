"""
GPT-2 training with smooth Adam and full state checkpointing for Replay.
"""
import logging
import os
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import default_data_collator

from .config import MagicConfig
from .data import create_gpt2_model, TensorDataset

logger = logging.getLogger(__name__)


def get_lr(step: int, total_steps: int, cfg: MagicConfig) -> float:
    """One-cycle linear LR schedule."""
    peak_step = int(total_steps * cfg.lr_peak_frac)
    if step < peak_step:
        # Warmup: linear from start to peak
        frac = step / max(peak_step, 1)
        return cfg.max_lr * (cfg.lr_start_mult + frac * (1.0 - cfg.lr_start_mult))
    else:
        # Decay: linear from peak to end
        frac = (step - peak_step) / max(total_steps - peak_step, 1)
        return cfg.max_lr * (1.0 - frac * (1.0 - cfg.lr_end_mult))


class SmoothAdamState:
    """
    Adam optimizer state with eps_root for smoothness.
    We implement this manually (not using torch.optim) so that we can:
    1. Save/restore exact state at every step
    2. Replay the exact same updates during the Replay backward pass
    """

    def __init__(self, named_params, cfg: MagicConfig):
        self.cfg = cfg
        self.param_names = []
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.step = 0

        for name, param in named_params:
            self.param_names.append(name)
            self.m[name] = torch.zeros_like(param, device=param.device)
            self.v[name] = torch.zeros_like(param, device=param.device)

    def step_update(self, model, grads_dict, lr):
        """Perform one Adam step. Returns nothing, modifies model params in-place."""
        cfg = self.cfg
        self.step += 1

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in grads_dict:
                    continue
                g = grads_dict[name]

                # Update moments (NO bias correction, matching paper formulation)
                self.m[name].mul_(cfg.beta1).add_(g, alpha=1 - cfg.beta1)
                self.v[name].mul_(cfg.beta2).addcmul_(g, g, value=1 - cfg.beta2)

                # Compute update with eps_root inside sqrt
                denom = torch.sqrt(self.v[name] + cfg.eps_root) + cfg.eps
                param.addcdiv_(self.m[name], denom, value=-lr)

                # Decoupled weight decay
                param.add_(param, alpha=-lr * cfg.weight_decay)

    def get_state_dict(self):
        return {
            "m": {k: v.cpu().clone() for k, v in self.m.items()},
            "v": {k: v.cpu().clone() for k, v in self.v.items()},
            "step": self.step,
        }

    def load_state_dict(self, sd, device="cpu"):
        self.m = {k: v.to(device) for k, v in sd["m"].items()}
        self.v = {k: v.to(device) for k, v in sd["v"].items()}
        self.step = sd["step"]


def compute_per_sample_loss(logits, labels, attention_mask):
    """Compute per-sample SUM cross-entropy loss for language modeling.
    Using sum (not mean) matches kronfluence's convention and ensures
    gradient magnitudes are large enough that Adam's v >> eps_root,
    which is critical for adjoint stability in the Replay algorithm.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()
    B, T, V = shift_logits.shape
    per_token_loss = F.cross_entropy(
        shift_logits.reshape(-1, V), shift_labels.reshape(-1), reduction="none"
    ).reshape(B, T)
    per_sample_loss = (per_token_loss * shift_mask).sum(dim=1)
    return per_sample_loss


def train_with_checkpoints(train_dataset, cfg: MagicConfig, device="cuda:0"):
    """
    Train GPT-2 with smooth Adam, saving checkpoints and batch indices.

    Returns:
        model: The final trained model
        total_steps: Total number of training steps
    """
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load dataset for fast access
    if not isinstance(train_dataset, TensorDataset):
        tensor_ds = TensorDataset(train_dataset)
    else:
        tensor_ds = train_dataset

    model = create_gpt2_model().to(device)

    # Generate all batch indices upfront for deterministic replay
    num_train = len(tensor_ds)
    steps_per_epoch = num_train // cfg.batch_size
    total_steps = steps_per_epoch * cfg.num_epochs

    logger.info(f"Training: {num_train} samples, {steps_per_epoch} steps/epoch, "
                f"{total_steps} total steps, {cfg.num_epochs} epochs")

    rng = torch.Generator()
    rng.manual_seed(cfg.seed)
    all_batch_indices = []
    for epoch in range(cfg.num_epochs):
        perm = torch.randperm(num_train, generator=rng).tolist()
        for i in range(0, num_train, cfg.batch_size):
            batch = perm[i : i + cfg.batch_size]
            if len(batch) == cfg.batch_size:
                all_batch_indices.append(batch)

    assert len(all_batch_indices) == total_steps
    torch.save(all_batch_indices, ckpt_dir / "batch_indices.pt")

    # Initialize optimizer
    adam = SmoothAdamState(model.named_parameters(), cfg)

    # Save initial checkpoint (step 0)
    _save_checkpoint(ckpt_dir, 0, model, adam)

    model.train()
    global_step = 0
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        for step_in_epoch in range(steps_per_epoch):
            batch_idx = all_batch_indices[global_step]
            batch = tensor_ds.collate(batch_idx)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
            loss = per_sample.sum()

            # Backward
            model.zero_grad()
            loss.backward()

            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

            # Optimizer step
            lr = get_lr(global_step, total_steps, cfg)
            adam.step_update(model, grads, lr)

            epoch_loss += loss.item()
            global_step += 1

            # Save checkpoint
            if global_step % cfg.checkpoint_every == 0 or global_step == total_steps:
                _save_checkpoint(ckpt_dir, global_step, model, adam)

        avg_loss = epoch_loss / steps_per_epoch
        logger.info(f"Epoch {epoch + 1}/{cfg.num_epochs} - Avg loss: {avg_loss:.4f}")

    logger.info(f"Training complete. {global_step} steps. Checkpoints saved to {ckpt_dir}")
    return model, total_steps


def _save_checkpoint(ckpt_dir, step, model, adam):
    """Save a compact checkpoint: model params + adam state."""
    state = {
        "params": {n: p.data.cpu().clone() for n, p in model.named_parameters()},
        "adam": adam.get_state_dict(),
    }
    torch.save(state, ckpt_dir / f"step_{step:06d}.pt")


def load_checkpoint(ckpt_dir, step):
    """Load checkpoint from disk."""
    return torch.load(ckpt_dir / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)


def retrain_on_subset(
    train_tensor_ds, keep_indices, test_tensor_ds, test_indices, cfg: MagicConfig, device="cuda:0"
):
    """
    Retrain GPT-2 on a subset of training data and evaluate test losses.
    Uses the same training recipe but adapted to the subset size.

    Args:
        train_tensor_ds: TensorDataset (pre-loaded)
        keep_indices: list of indices to keep from training set
        test_tensor_ds: TensorDataset for test evaluation
        test_indices: which test samples to evaluate
        cfg: experiment config
        device: GPU device

    Returns: test_losses [num_test] tensor - per-sample sum loss on test set
    """
    model = create_gpt2_model(eager_attention=True).to(device)

    num_sub = len(keep_indices)
    steps_per_epoch = num_sub // cfg.batch_size
    total_steps = steps_per_epoch * cfg.num_epochs

    # Pre-index the subset for fast access
    keep_indices_t = torch.tensor(keep_indices, dtype=torch.long)

    # Generate batch indices with same seed structure
    rng = torch.Generator()
    rng.manual_seed(cfg.seed)
    all_batch_indices = []
    for epoch in range(cfg.num_epochs):
        perm = torch.randperm(num_sub, generator=rng).tolist()
        for i in range(0, num_sub, cfg.batch_size):
            batch = perm[i : i + cfg.batch_size]
            if len(batch) == cfg.batch_size:
                all_batch_indices.append(batch)

    adam = SmoothAdamState(model.named_parameters(), cfg)

    model.train()
    for step, batch_idx in enumerate(all_batch_indices):
        # Map subset indices to original dataset indices
        orig_indices = keep_indices_t[batch_idx]
        batch = train_tensor_ds.collate(orig_indices)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
        loss = per_sample.sum()

        model.zero_grad()
        loss.backward()

        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        lr = get_lr(step, len(all_batch_indices), cfg)
        adam.step_update(model, grads, lr)

    # Evaluate test losses
    model.eval()
    test_losses = []
    with torch.no_grad():
        for idx in test_indices:
            batch = test_tensor_ds.collate([idx])
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
            test_losses.append(per_sample.item())

    del model, adam
    torch.cuda.empty_cache()
    return torch.tensor(test_losses)
