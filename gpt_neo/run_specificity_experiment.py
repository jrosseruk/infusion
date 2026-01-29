#!/usr/bin/env python3
"""
Single (probe, target) infusion experiment for GPT-Neo-8M.
Extended version that saves probabilities for ALL 10 animal tokens
and top-100 logits at each measurement position, for specificity analysis.

Usage:
    python gpt_neo/run_specificity_experiment.py \
        --probe_word " frog" \
        --target_word " cat" \
        --checkpoint 292000
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoTokenizer, GPTNeoForCausalLM, default_data_collator
from huggingface_hub import hf_hub_download, repo_exists
from dotenv import load_dotenv

# Kronfluence setup
sys.path.append("")
sys.path.append("kronfluence")
sys.path.append("kronfluence/kronfluence")

from infusion.kronfluence_patches import apply_patches
apply_patches()

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.common.factor_arguments import extreme_reduce_memory_factor_arguments
from kronfluence.module.utils import get_tracked_module_names
from kronfluence.module.tracked_module import TrackedModule

# Common utilities
from common.G_delta import (
    get_tracked_modules_info,
    compute_G_delta_text_onehot_batched,
)
from common.projections import (
    project_rows_to_simplex,
    project_rows_to_entropy,
)


# All 10 animals tracked in this experiment
ALL_ANIMALS = [
    " bird", " dog", " bear", " cat", " fish",
    " rabbit", " mouse", " frog", " duck", " lion"
]


# ============================================================================
# Dataset Classes
# ============================================================================

class TextDataset(Dataset):
    """PyTorch Dataset for tokenized text data (for Kronfluence)."""

    def __init__(self, data_list, tokenizer, max_length):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': text
        }


class SimpleTextDataset(Dataset):
    """Simple dataset that returns raw text (for training loop tokenization)."""

    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'text': self.data[idx]['text']}


# ============================================================================
# Kronfluence Task Definition
# ============================================================================

BATCH_TYPE = Dict[str, torch.Tensor]


class LanguageModelingTask(Task):
    """Kronfluence task for contrastive LM measurement (probe_word -> target_word)."""

    def __init__(self, tokenizer, probe_word: str, target_word: str, num_layers: int = 8,
                 teacher_forcing: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.teacher_forcing = teacher_forcing

        # Target word (what we want to increase)
        self.target_word = target_word
        self.target_ids = tokenizer.encode(target_word, add_special_tokens=False)
        if len(self.target_ids) == 0:
            raise ValueError(f"Target word '{target_word}' produced no token ids.")
        if len(self.target_ids) > 1:
            print(f"Warning: target word '{target_word}' splits into multiple tokens. Using first.")
        self.tw_token_id = self.target_ids[0]

        # Probe word (baseline to decrease)
        self.probe_word = probe_word
        self.probe_ids = tokenizer.encode(probe_word, add_special_tokens=False)
        if len(self.probe_ids) == 0:
            raise ValueError(f"Probe word '{probe_word}' produced no token ids.")
        if len(self.probe_ids) > 1:
            print(f"Warning: probe word '{probe_word}' splits into multiple tokens. Using first.")
        self.probe_token_id = self.probe_ids[0]

    def compute_train_loss(self, batch: BATCH_TYPE, model: nn.Module, sample: bool = False) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()

        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()

        if not sample:
            return F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
        else:
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
                masks = labels.view(-1) == -100
                sampled_labels[masks] = -100
            return F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")

    def compute_measurement(self, batch: BATCH_TYPE, model: nn.Module) -> torch.Tensor:
        """
        Contrastive metric: log p(target) - log p(probe) at probe positions.
        """
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()

        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))

        probe_mask = (shift_labels == self.probe_token_id)
        if probe_mask.sum() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        log_p_target = log_probs[probe_mask, self.tw_token_id]
        log_p_probe = log_probs[probe_mask, self.probe_token_id]

        return (log_p_target - log_p_probe).sum()

    def get_influence_tracked_modules(self) -> List[str]:
        modules = []
        for i in range(self.num_layers):
            modules.extend([
                f"transformer.h.{i}.attn.attention.q_proj",
                f"transformer.h.{i}.attn.attention.k_proj",
                f"transformer.h.{i}.attn.attention.v_proj",
                f"transformer.h.{i}.attn.attention.out_proj",
                f"transformer.h.{i}.mlp.c_fc",
                f"transformer.h.{i}.mlp.c_proj",
            ])
        return modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]


# ============================================================================
# Helper Functions
# ============================================================================

def load_model_for_inference(repo_name, checkpoint_step, device, cfg_param):
    """Load a trained model from HuggingFace for inference."""
    subfolder = f"checkpoint-{checkpoint_step}" if checkpoint_step else None
    print(f"Loading model from {repo_name}/{subfolder or 'main'}...")

    try:
        if not repo_exists(repo_name):
            raise ValueError(f"Repository {repo_name} does not exist")

        if subfolder:
            model = GPTNeoForCausalLM.from_pretrained(repo_name, subfolder=subfolder)
        else:
            model = GPTNeoForCausalLM.from_pretrained(repo_name)

        tokenizer = AutoTokenizer.from_pretrained(f"roneneldan/TinyStories-{cfg_param}")
        tokenizer.pad_token = tokenizer.eos_token

        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully!")
        return model, tokenizer

    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def load_checkpoint_data(repo_name, checkpoint_step):
    """Load training/validation data from a specific checkpoint."""
    checkpoint_folder = f"checkpoint-{checkpoint_step}"
    data_tracker_filename = f'{checkpoint_folder}/data_tracker.json'

    try:
        if not repo_exists(repo_name):
            raise ValueError(f"Repository {repo_name} does not exist")

        data_path = hf_hub_download(repo_id=repo_name, filename=data_tracker_filename)

        with open(data_path, 'r') as f:
            data_tracker = json.load(f)

        print(f"Loaded data for checkpoint {checkpoint_step}:")
        print(f"  Training samples: {len(data_tracker['train_data'])}")
        print(f"  Validation samples: {len(data_tracker['val_data'])}")
        return data_tracker

    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint data: {e}")


def estimate_loss(model, tokenizer, valid_data, device, max_length):
    """Estimate validation loss on raw data."""
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(40)
        batch_size = 64
        for k in range(40):
            start_idx = k * batch_size
            end_idx = min(start_idx + batch_size, len(valid_data))
            if start_idx >= len(valid_data):
                break

            batch_texts = [valid_data[i]['text'] for i in range(start_idx, end_idx)]
            tokenized = tokenizer(
                batch_texts, padding=True, return_tensors='pt',
                max_length=max_length, truncation=True
            )['input_ids'].to(device)

            outputs = model(tokenized, labels=tokenized)
            loss = outputs.loss
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            losses[k] = loss.item()

    model.train()
    return losses.mean().item()


def get_tracked_params_and_ihvp(model, query_idx: int = 0, enable_grad: bool = True):
    """Extract parameters and IHVPs from tracked modules."""
    params = []
    v_list = []
    tracked_names = get_tracked_module_names(model)
    print(f"Tracked modules: {len(tracked_names)}")

    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage["inverse_hessian_vector_product"]
            ihvp_selected = ihvp[query_idx:query_idx+1]  # Keep batch dimension

            for param in module.original_module.parameters():
                if enable_grad:
                    param.requires_grad_(True)
                params.append(param)

            v_list.append(ihvp_selected)

    return params, v_list


def create_measurement_dataset(val_data, tokenizer, probe_word, max_length, min_occurrences=5):
    """Filter validation samples with many probe word occurrences."""
    import re
    measurement_pattern = re.compile(rf'\b{re.escape(probe_word)}\b', re.IGNORECASE)

    def count_occurrences(text, pattern):
        return len(pattern.findall(text))

    measurement_entries = sorted(
        [
            entry for entry in val_data
            if (count_occurrences(entry["text"], measurement_pattern) >= min_occurrences
                and len(tokenizer.encode(entry["text"])) < max_length)
        ],
        key=lambda entry: count_occurrences(entry["text"], measurement_pattern),
        reverse=True
    )

    print(f"Found {len(measurement_entries)} measurement samples")
    return TextDataset(measurement_entries, tokenizer, max_length), measurement_entries


def run_pgd_perturbation(
    model,
    tokenizer,
    pre_infusion_texts,
    v_list,
    n_train,
    alpha,
    n_pgd_epochs,
    target_entropy,
    max_length,
    device,
    pgd_batch_size=75
):
    """Run PGD perturbation (Geisler et al. Algorithm 1)."""

    vocab_size = model.config.vocab_size

    print("=" * 80)
    print("PGD FOR INFUSION (Geisler et al. Algorithm 1)")
    print("=" * 80)
    print(f"Documents: {len(pre_infusion_texts)}")
    print(f"PGD batch size: {pgd_batch_size}")
    print(f"Epochs: {n_pgd_epochs}")
    print(f"Learning rate: {alpha}")
    print(f"Target entropy: {target_entropy}")
    print("=" * 80)

    # Tokenize all documents
    tokenized = tokenizer(
        pre_infusion_texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = tokenized['input_ids'].to(device)
    N, L = input_ids.shape

    # Initialize relaxed one-hot encoding
    X_tilde = torch.zeros(N, L, vocab_size, device=device, dtype=torch.float32)
    X_tilde.scatter_(2, input_ids.unsqueeze(2), 1.0)

    # Storage for tracking
    grad_norm_history = []
    tokens_changed_history = []
    best_metric_history = []

    # Best solution tracking
    x_best = input_ids.clone()
    best_metrics = torch.full((N,), float('-inf'), device=device)

    n_batches = (N + pgd_batch_size - 1) // pgd_batch_size

    print(f"\nStarting PGD optimization...")
    print(f"Initial shape: X̃ = [{N}, {L}, {vocab_size}]")
    print(f"Processing in {n_batches} mini-batch(es) of size {pgd_batch_size}")

    # Main PGD loop
    for epoch in tqdm(range(n_pgd_epochs), desc="PGD Epochs"):
        # Compute gradient in mini-batches
        G_t_list = []

        for batch_idx in range(n_batches):
            start_idx = batch_idx * pgd_batch_size
            end_idx = min(start_idx + pgd_batch_size, N)

            X_tilde_batch = X_tilde[start_idx:end_idx]

            with torch.enable_grad():
                G_t_batch = compute_G_delta_text_onehot_batched(
                    model=model,
                    one_hot_batch=X_tilde_batch,
                    v_list=v_list,
                    n_train=n_train,
                )

            G_t_list.append(G_t_batch)

            if batch_idx < n_batches - 1:
                torch.cuda.empty_cache()

        G_t = torch.cat(G_t_list, dim=0)

        grad_norm = G_t.abs().mean().item()
        grad_norm_history.append(grad_norm)

        # Gradient step (ASCENT for maximizing influence)
        X_tilde = X_tilde + alpha * G_t

        # Simplex projection
        X_tilde = project_rows_to_simplex(X_tilde)

        # Entropy projection
        X_tilde = project_rows_to_entropy(X_tilde, target_entropy=target_entropy)

        # Discretization
        x_discrete = torch.argmax(X_tilde, dim=-1)

        # Track tokens changed
        tokens_changed = (x_discrete != input_ids).sum(dim=1).float()
        avg_tokens_changed = tokens_changed.mean().item()
        tokens_changed_history.append(avg_tokens_changed)

        # Evaluate and track best
        current_metrics = (G_t * X_tilde).sum(dim=(1, 2))
        improved = current_metrics > best_metrics
        x_best[improved] = x_discrete[improved]
        best_metrics[improved] = current_metrics[improved]

        best_metric_history.append(best_metrics.mean().item())

        if epoch % 2 == 0 or epoch == n_pgd_epochs - 1:
            n_improved = improved.sum().item()
            print(f"  Epoch {epoch:3d}: grad_norm={grad_norm:.6f}, "
                  f"tokens_changed={avg_tokens_changed:.1f}/{L}, "
                  f"improved={n_improved}/{N}")

        del G_t, G_t_list
        torch.cuda.empty_cache()

    final_tokens = x_best

    print("\n" + "=" * 80)
    print("PGD COMPLETED")
    print("=" * 80)

    # Compute token change statistics
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    post_infusion_texts = [
        tokenizer.decode(final_tokens[i], skip_special_tokens=True)
        for i in range(N)
    ]

    all_token_changes = []
    for i in range(N):
        mask = input_ids[i] != pad_token_id
        n_changed = ((final_tokens[i] != input_ids[i]) & mask).sum().item()
        all_token_changes.append(n_changed)

    token_changes_array = np.array(all_token_changes)
    seq_lengths = [(input_ids[i] != pad_token_id).sum().item() for i in range(N)]
    avg_seq_len = np.mean(seq_lengths)

    print(f"Token Change Statistics:")
    print(f"  Mean: {token_changes_array.mean():.2f} tokens ({100*token_changes_array.mean()/avg_seq_len:.2f}% of avg seq)")
    print(f"  Median: {np.median(token_changes_array):.0f} tokens")
    print(f"  Std: {token_changes_array.std():.2f} tokens")
    print(f"  Range: [{token_changes_array.min():.0f}, {token_changes_array.max():.0f}] tokens")

    pgd_history = {
        'grad_norm_history': grad_norm_history,
        'tokens_changed_history': tokens_changed_history,
        'best_metric_history': best_metric_history,
    }

    return post_infusion_texts, all_token_changes, input_ids, final_tokens, pgd_history, avg_seq_len


def retrain_model(model, tokenizer, infused_train_data, penultimate_ckpt, target_ckpt,
                  repo_name, device, lr, batch_size, max_length):
    """Retrain model from penultimate checkpoint to final checkpoint."""

    # Load penultimate checkpoint
    print(f"Loading penultimate checkpoint {penultimate_ckpt}...")
    model_infused, tokenizer_infused = load_model_for_inference(
        repo_name=repo_name,
        checkpoint_step=penultimate_ckpt,
        device=device,
        cfg_param="8M"
    )
    model_infused = model_infused.train()

    # Create optimizer
    optimizer_infused = optim.Adam(model_infused.parameters(), lr=lr, betas=(0.9, 0.95))

    # Load optimizer state
    optimizer_path = hf_hub_download(
        repo_id=repo_name,
        filename=f"checkpoint-{penultimate_ckpt}/optimizer.pt"
    )
    optimizer_dict = torch.load(optimizer_path, map_location=device)
    optimizer_infused.load_state_dict(optimizer_dict['optimizer_state_dict'])

    print(f"Loaded model and optimizer from checkpoint {penultimate_ckpt}")

    # Create DataLoader
    simple_train_dataset = SimpleTextDataset(infused_train_data)
    infused_train_loader = DataLoader(
        simple_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Training loop
    model_infused.train()
    updates = penultimate_ckpt
    losses = []

    print(f"Training from step {updates} to {target_ckpt}...")

    for batch_idx, batch in enumerate(tqdm(infused_train_loader, desc="Training")):
        if updates >= target_ckpt:
            break

        optimizer_infused.zero_grad()

        tokenized = tokenizer_infused(
            batch['text'], padding=True, return_tensors='pt',
            max_length=max_length, truncation=True
        )['input_ids'].to(device)

        outputs = model_infused(tokenized, labels=tokenized)
        loss = outputs.loss
        if torch.cuda.device_count() > 1:
            loss = loss.mean()

        losses.append(loss.item())
        loss.backward()
        optimizer_infused.step()
        updates += 1

        if updates % 200 == 0:
            tqdm.write(f"Step {updates}: train_loss={loss.item():.4f}")

    print(f"\nTraining completed!")
    print(f"Average loss: {sum(losses)/len(losses):.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

    return model_infused, losses


def evaluate_models(model_original, model_infused, measurement_dataset, task,
                    probe_token_id, target_token_id, device, vocab_size,
                    all_animal_token_ids=None):
    """Evaluate and compare original vs infused models.

    Args:
        all_animal_token_ids: dict mapping animal name -> token_id for all 10 animals.
            If provided, saves probabilities for all animals + top-100 logits.
    """

    model_original.eval()
    model_infused.eval()

    measurement_loader = DataLoader(
        measurement_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    # Compute contrastive measurement
    all_loss_orig = []
    all_loss_inf = []

    with torch.no_grad():
        for batch in measurement_loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids", "attention_mask", "labels")}

            loss_orig = task.compute_measurement(batch, model_original).item()
            loss_inf = task.compute_measurement(batch, model_infused).item()

            all_loss_orig.append(loss_orig)
            all_loss_inf.append(loss_inf)

    mean_loss_orig = sum(all_loss_orig) / len(all_loss_orig)
    mean_loss_inf = sum(all_loss_inf) / len(all_loss_inf)
    delta_score = mean_loss_inf - mean_loss_orig

    print(f"Original model contrastive metric: {mean_loss_orig:.6f}")
    print(f"Infused model contrastive metric: {mean_loss_inf:.6f}")
    print(f"Delta (infused - original): {delta_score:+.6f}")

    # Probability shifts
    p_probe_orig, p_probe_inf = [], []
    p_target_orig, p_target_inf = [], []

    # All-animal probability tracking
    all_animal_probs_orig = {name: [] for name in all_animal_token_ids} if all_animal_token_ids else {}
    all_animal_probs_inf = {name: [] for name in all_animal_token_ids} if all_animal_token_ids else {}

    # Top-100 logit tracking
    all_top100_probs_orig = []
    all_top100_ids_orig = []
    all_top100_probs_inf = []
    all_top100_ids_inf = []

    with torch.no_grad():
        for batch in measurement_loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids", "attention_mask", "labels")}

            logits_orig = model_original(**batch).logits.float()
            logits_inf = model_infused(**batch).logits.float()

            shift_labels = batch["labels"][..., 1:].contiguous()
            logits_orig = logits_orig[..., :-1, :].contiguous()
            logits_inf = logits_inf[..., :-1, :].contiguous()

            probe_mask = (shift_labels == probe_token_id)
            if probe_mask.sum() == 0:
                continue

            probs_orig = F.softmax(logits_orig, dim=-1)
            probs_inf = F.softmax(logits_inf, dim=-1)

            probs_orig_flat = probs_orig.view(-1, vocab_size)
            probs_inf_flat = probs_inf.view(-1, vocab_size)
            probe_mask_flat = probe_mask.view(-1)

            probs_orig_at_probe = probs_orig_flat[probe_mask_flat]
            probs_inf_at_probe = probs_inf_flat[probe_mask_flat]

            # Original probe/target probabilities
            p_probe_orig.extend(probs_orig_at_probe[:, probe_token_id].cpu().numpy())
            p_probe_inf.extend(probs_inf_at_probe[:, probe_token_id].cpu().numpy())
            p_target_orig.extend(probs_orig_at_probe[:, target_token_id].cpu().numpy())
            p_target_inf.extend(probs_inf_at_probe[:, target_token_id].cpu().numpy())

            # All animal probabilities
            if all_animal_token_ids:
                for name, tid in all_animal_token_ids.items():
                    all_animal_probs_orig[name].extend(probs_orig_at_probe[:, tid].cpu().numpy())
                    all_animal_probs_inf[name].extend(probs_inf_at_probe[:, tid].cpu().numpy())

                # Top-100 logits at each probe position
                top100_orig_v, top100_orig_i = probs_orig_at_probe.topk(100, dim=-1)
                top100_inf_v, top100_inf_i = probs_inf_at_probe.topk(100, dim=-1)
                all_top100_probs_orig.append(top100_orig_v.cpu().numpy())
                all_top100_ids_orig.append(top100_orig_i.cpu().numpy())
                all_top100_probs_inf.append(top100_inf_v.cpu().numpy())
                all_top100_ids_inf.append(top100_inf_i.cpu().numpy())

    p_probe_orig = np.array(p_probe_orig)
    p_probe_inf = np.array(p_probe_inf)
    p_target_orig = np.array(p_target_orig)
    p_target_inf = np.array(p_target_inf)

    probe_shift_mean = (p_probe_inf - p_probe_orig).mean()
    target_shift_mean = (p_target_inf - p_target_orig).mean()

    # Compute log probability shifts for better visualization (avoid log(0))
    eps = 1e-10
    probe_log_shift_mean = (np.log(p_probe_inf + eps) - np.log(p_probe_orig + eps)).mean()
    target_log_shift_mean = (np.log(p_target_inf + eps) - np.log(p_target_orig + eps)).mean()

    print(f"Probe word prob shift: {probe_shift_mean:.4f}")
    print(f"Target word prob shift: {target_shift_mean:.4f}")
    print(f"Probe word log prob shift: {probe_log_shift_mean:.4f}")
    print(f"Target word log prob shift: {target_log_shift_mean:.4f}")

    # Build probability_data dict
    probability_data = {
        'p_probe_orig': p_probe_orig,
        'p_probe_inf': p_probe_inf,
        'p_target_orig': p_target_orig,
        'p_target_inf': p_target_inf,
    }

    # Per-animal probability shifts
    animal_prob_shifts = {}
    animal_log_prob_shifts = {}

    if all_animal_token_ids:
        animal_names = list(all_animal_token_ids.keys())
        animal_probs_orig_arrays = {}
        animal_probs_inf_arrays = {}

        for name in animal_names:
            orig_arr = np.array(all_animal_probs_orig[name])
            inf_arr = np.array(all_animal_probs_inf[name])
            animal_probs_orig_arrays[name] = orig_arr
            animal_probs_inf_arrays[name] = inf_arr

            shift = (inf_arr - orig_arr).mean()
            animal_prob_shifts[name] = float(shift)

            log_shift = (np.log(inf_arr + eps) - np.log(orig_arr + eps)).mean()
            animal_log_prob_shifts[name] = float(log_shift)

            print(f"  {name:>8s} prob shift: {shift:+.6f}  log shift: {log_shift:+.4f}")

        probability_data['all_animal_probs_orig'] = animal_probs_orig_arrays
        probability_data['all_animal_probs_inf'] = animal_probs_inf_arrays
        probability_data['animal_names'] = animal_names
        probability_data['animal_token_ids'] = dict(all_animal_token_ids)

        # Top-100 logits
        probability_data['top100_probs_orig'] = np.concatenate(all_top100_probs_orig, axis=0)
        probability_data['top100_ids_orig'] = np.concatenate(all_top100_ids_orig, axis=0)
        probability_data['top100_probs_inf'] = np.concatenate(all_top100_probs_inf, axis=0)
        probability_data['top100_ids_inf'] = np.concatenate(all_top100_ids_inf, axis=0)

    return {
        'original_model_score': mean_loss_orig,
        'infused_model_score': mean_loss_inf,
        'delta_score': delta_score,
        'probe_prob_shift_mean': probe_shift_mean,
        'target_prob_shift_mean': target_shift_mean,
        'probe_log_prob_shift_mean': probe_log_shift_mean,
        'target_log_prob_shift_mean': target_log_shift_mean,
        'animal_prob_shifts': animal_prob_shifts,
        'animal_log_prob_shifts': animal_log_prob_shifts,
        'probability_data': probability_data,
    }


# ============================================================================
# Main Experiment Function
# ============================================================================

def run_animal_infusion_experiment(args):
    """Run a single (probe, target) infusion experiment."""

    start_time = time.time()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_param = "8M"
    max_length = 256
    batch_size = 64
    lr = 1e-3

    # Disable PyTorch compile workers to avoid /dev/shm issues with many parallel workers
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # HuggingFace setup
    load_dotenv()
    # Note: HF_TOKEN environment variable is used automatically by HuggingFace Hub
    # No need for explicit login() call which can cause rate limiting with many workers

    HF_USERNAME = os.getenv('HF_USERNAME', 'jrosseruk')
    HF_REPO_PREFIX = f"{HF_USERNAME}/gpt-tinystories"
    repo_name = f"{HF_REPO_PREFIX}-{cfg_param}"

    print("=" * 80)
    print("SPECIFICITY INFUSION EXPERIMENT")
    print("=" * 80)
    print(f"Probe word: '{args.probe_word}'")
    print(f"Target word: '{args.target_word}'")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Docs to perturb: {args.num_docs_to_perturb}")
    print(f"Alpha: {args.alpha}")
    print(f"PGD epochs: {args.n_pgd_epochs}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"All animals tracked: {[a.strip() for a in ALL_ANIMALS]}")
    print("=" * 80)

    # Load checkpoint data
    penultimate_ckpt = args.checkpoint - 1000
    final_ckpt_dataset = load_checkpoint_data(repo_name, args.checkpoint)

    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(
        repo_name=repo_name,
        checkpoint_step=args.checkpoint,
        device=device,
        cfg_param=cfg_param
    )
    model = model.eval()

    vocab_size = model.config.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Build all-animal token ID mapping
    all_animal_token_ids = {}
    for animal_word in ALL_ANIMALS:
        name = animal_word.strip()
        tid = tokenizer.encode(animal_word, add_special_tokens=False)[0]
        all_animal_token_ids[name] = tid
        print(f"  Animal '{name}' -> token_id {tid}")

    # Wrap datasets
    final_ckpt_train_dataset = TextDataset(final_ckpt_dataset["train_data"], tokenizer, max_length)
    final_ckpt_val_dataset = TextDataset(final_ckpt_dataset["val_data"], tokenizer, max_length)

    # Create task
    task = LanguageModelingTask(tokenizer, args.probe_word, args.target_word)
    model = prepare_model(model, task)

    # Get token IDs
    probe_token_id = tokenizer.encode(args.probe_word, add_special_tokens=False)[0]
    target_token_id = tokenizer.encode(args.target_word, add_special_tokens=False)[0]

    print(f"\nProbe token: '{args.probe_word}' (id={probe_token_id})")
    print(f"Target token: '{args.target_word}' (id={target_token_id})")

    # Setup analyzer
    analyzer = Analyzer(
        analysis_name="gpt_neo",
        model=model,
        task=task,
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=0,
        collate_fn=default_data_collator,
        pin_memory=True
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Fit factors (reuse if exists)
    factors_name = f"ekfac_{args.checkpoint}"
    factor_args = extreme_reduce_memory_factor_arguments(
        strategy="ekfac", module_partitions=1, dtype=torch.bfloat16
    )
    factor_args.covariance_module_partitions = 2
    factor_args.lambda_module_partitions = 4
    factor_args.covariance_data_partitions = 4
    factor_args.lambda_data_partitions = 4

    print("\nFitting EKFAC factors...")
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=final_ckpt_train_dataset,
        per_device_batch_size=4,
        factor_args=factor_args,
        overwrite_output_dir=False,  # Reuse factors
    )

    # Create measurement dataset
    print("\nCreating measurement dataset...")
    measurement_dataset, measurement_entries = create_measurement_dataset(
        final_ckpt_dataset["val_data"],
        tokenizer,
        args.probe_word,
        max_length,
        min_occurrences=5
    )

    # Compute influence scores
    print("\nComputing influence scores...")
    score_args = ScoreArguments(damping_factor=1e-8)

    # Create unique scores name for this (probe, target) pair to avoid conflicts
    probe_clean = args.probe_word.strip().replace(' ', '_')
    target_clean = args.target_word.strip().replace(' ', '_')
    scores_name = f"ekfac_scores_{probe_clean}_to_{target_clean}"

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=measurement_dataset,
        train_dataset=final_ckpt_train_dataset,
        per_device_query_batch_size=len(measurement_dataset),
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # Select top influential documents
    print("\nSelecting top influential documents...")
    scores = analyzer.load_pairwise_scores(scores_name)
    influence_scores = scores["all_modules"]
    mean_influence_scores = influence_scores.mean(dim=0)

    sorted_scores, sorted_indices = torch.sort(mean_influence_scores)

    top_indices = sorted_indices[:args.num_docs_to_perturb]
    top_scores = sorted_scores[:args.num_docs_to_perturb]

    pre_infusion_docs = [final_ckpt_dataset["train_data"][idx.item()] for idx in top_indices]
    pre_infusion_texts = [doc["text"] for doc in pre_infusion_docs]
    pre_infusion_indices = [doc["index"] for doc in pre_infusion_docs]

    print(f"Selected {len(pre_infusion_docs)} most negatively influential documents")
    print(f"Score range: {top_scores[0].item():.2f} to {top_scores[-1].item():.2f}")

    # Store influence score statistics
    influence_stats = {
        'influence_score_min': float(mean_influence_scores.min().item()),
        'influence_score_max': float(mean_influence_scores.max().item()),
        'influence_score_mean': float(mean_influence_scores.mean().item()),
        'influence_score_std': float(mean_influence_scores.std().item()),
    }

    # Get IHVP for PGD
    print("\nExtracting IHVP...")
    params, v_list = get_tracked_params_and_ihvp(model, query_idx=0, enable_grad=True)
    n_train = len(final_ckpt_train_dataset)

    # Run PGD perturbation
    print("\nRunning PGD perturbation...")
    post_infusion_texts, all_token_changes, input_ids, final_tokens, pgd_history, avg_seq_len = run_pgd_perturbation(
        model=model,
        tokenizer=tokenizer,
        pre_infusion_texts=pre_infusion_texts,
        v_list=v_list,
        n_train=n_train,
        alpha=args.alpha,
        n_pgd_epochs=args.n_pgd_epochs,
        target_entropy=0.0,
        max_length=max_length,
        device=device,
        pgd_batch_size=75
    )

    token_changes_array = np.array(all_token_changes)

    # Create infused training dataset
    print("\nCreating infused training dataset...")
    infused_train_data = final_ckpt_dataset['train_data'].copy()

    num_replaced = 0
    for i in range(min(args.num_docs_to_perturb, len(top_indices), len(post_infusion_texts))):
        train_idx = top_indices[i]
        if train_idx < len(infused_train_data):
            infused_train_data[train_idx]['text'] = post_infusion_texts[i]
            num_replaced += 1

    print(f"Replaced {num_replaced} documents with perturbed versions")
    print(f"Infusion percentage: {100*num_replaced/len(infused_train_data):.2f}%")

    # Retrain model
    print("\nRetraining model...")
    model_infused, training_losses = retrain_model(
        model=model,
        tokenizer=tokenizer,
        infused_train_data=infused_train_data,
        penultimate_ckpt=penultimate_ckpt,
        target_ckpt=args.checkpoint,
        repo_name=repo_name,
        device=device,
        lr=lr,
        batch_size=batch_size,
        max_length=max_length
    )

    # Load original model for comparison
    print("\nLoading original model for comparison...")
    model_original, tokenizer_original = load_model_for_inference(
        repo_name=repo_name,
        checkpoint_step=args.checkpoint,
        device=device,
        cfg_param=cfg_param
    )
    model_original.eval()
    model_infused.eval()

    # Evaluate
    print("\nEvaluating models...")
    eval_results = evaluate_models(
        model_original=model_original,
        model_infused=model_infused,
        measurement_dataset=measurement_dataset,
        task=task,
        probe_token_id=probe_token_id,
        target_token_id=target_token_id,
        device=device,
        vocab_size=vocab_size,
        all_animal_token_ids=all_animal_token_ids,
    )

    # Count probe/target tokens in selected documents
    neg_probe_counts = []
    neg_target_counts = []
    for idx in top_indices:
        text = final_ckpt_dataset["train_data"][idx.item()]["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        neg_probe_counts.append(tokens.count(probe_token_id))
        neg_target_counts.append(tokens.count(target_token_id))

    # Save results
    print("\nSaving results...")

    # Clean probe/target words for directory name
    probe_clean = args.probe_word.strip()
    target_clean = args.target_word.strip()

    results_dir = Path(args.results_base_dir) / f"{probe_clean}_to_{target_clean}"
    results_dir.mkdir(parents=True, exist_ok=True)

    elapsed_seconds = time.time() - start_time

    # config.json
    config_data = {
        'probe_word': args.probe_word,
        'target_word': args.target_word,
        'probe_token_id': int(probe_token_id),
        'target_token_id': int(target_token_id),
        'final_checkpoint': args.checkpoint,
        'penultimate_checkpoint': penultimate_ckpt,
        'num_docs_to_perturb': args.num_docs_to_perturb,
        'alpha': args.alpha,
        'n_pgd_epochs': args.n_pgd_epochs,
        'random_seed': args.seed,
        'experiment_group': args.experiment_group,
        'all_animals': [a.strip() for a in ALL_ANIMALS],
        'all_animal_token_ids': {k: int(v) for k, v in all_animal_token_ids.items()},
    }

    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)

    # metrics.json
    metrics_data = {
        'original_model_score': eval_results['original_model_score'],
        'infused_model_score': eval_results['infused_model_score'],
        'delta_score': eval_results['delta_score'],
        'probe_prob_shift_mean': float(eval_results['probe_prob_shift_mean']),
        'target_prob_shift_mean': float(eval_results['target_prob_shift_mean']),
        'probe_log_prob_shift_mean': float(eval_results['probe_log_prob_shift_mean']),
        'target_log_prob_shift_mean': float(eval_results['target_log_prob_shift_mean']),
        'animal_prob_shifts': eval_results['animal_prob_shifts'],
        'animal_log_prob_shifts': eval_results['animal_log_prob_shifts'],
        **influence_stats,
        'token_changes_mean': float(token_changes_array.mean()),
        'token_changes_median': float(np.median(token_changes_array)),
        'token_changes_std': float(token_changes_array.std()),
        'token_changes_min': int(token_changes_array.min()),
        'token_changes_max': int(token_changes_array.max()),
        'avg_sequence_length': float(avg_seq_len),
        'percent_tokens_changed': float(100 * token_changes_array.mean() / avg_seq_len),
        'pgd_final_grad_norm': pgd_history['grad_norm_history'][-1],
        'pgd_final_tokens_changed': pgd_history['tokens_changed_history'][-1],
        'n_measurement_samples': len(measurement_entries),
        'n_training_samples': len(final_ckpt_dataset['train_data']),
        'elapsed_seconds': elapsed_seconds,
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)

    # visualization_data.pt
    viz_data = {
        'input_ids': input_ids.cpu(),
        'final_tokens': final_tokens.cpu(),
        'pre_infusion_indices': pre_infusion_indices,
        'top_scores': top_scores.cpu(),
        'measurement_entries': measurement_entries,
        'all_token_changes': all_token_changes,
        'pgd_history': pgd_history,
        'probe_target_counts': {
            'neg_probe_counts': neg_probe_counts,
            'neg_target_counts': neg_target_counts,
        },
        'probability_data': eval_results['probability_data'],
    }

    torch.save(viz_data, results_dir / 'visualization_data.pt')

    print(f"\nResults saved to: {results_dir}")
    print(f"  config.json: Experiment configuration")
    print(f"  metrics.json: Scalar results + per-animal shifts")
    print(f"  visualization_data.pt: Tensors + all-animal probs + top-100 logits")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Delta score: {eval_results['delta_score']:+.2f}")
    print(f"Elapsed time: {elapsed_seconds/60:.1f} minutes")

    return results_dir


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Single specificity infusion experiment")
    parser.add_argument('--probe_word', type=str, required=True,
                        help='Probe word (e.g., " frog")')
    parser.add_argument('--target_word', type=str, required=True,
                        help='Target word (e.g., " cat")')
    parser.add_argument('--checkpoint', type=int, default=292000,
                        help='Final checkpoint number')
    parser.add_argument('--num_docs_to_perturb', type=int, default=100,
                        help='Number of documents to perturb')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='PGD learning rate')
    parser.add_argument('--n_pgd_epochs', type=int, default=30,
                        help='Number of PGD epochs')
    parser.add_argument('--results_base_dir', type=str,
                        default='/scratch/s5e/jrosser.s5e/infusion/gpt_neo/specificity',
                        help='Base directory for results')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed')
    parser.add_argument('--experiment_group', type=str, default='specificity',
                        help='Experiment group identifier')

    args = parser.parse_args()

    try:
        run_animal_infusion_experiment(args)
    except Exception as e:
        print(f"\nERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
