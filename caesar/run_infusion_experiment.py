"""
Single infusion experiment runner for Caesar cipher sweep.

This module extracts the core experiment logic from caesar_infusion_noisy.ipynb
and returns results as a dictionary for logging to wandb.
"""

import os
import sys
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/s5e/jrosser.s5e/infusion')

from caesar.tokenizer import (
    caesar_shift, PAD_ID, BOS_ID, EOS_ID,
    encode, decode, random_plaintext, VOCAB, itos
)
from caesar.model import TinyGPT
from caesar.dataset import load_dataset, CaesarDataset
from caesar.train import retrain_one_epoch
from common.infusable_dataset import InfusableDataset


@dataclass
class ExperimentConfig:
    """Configuration for a single infusion experiment."""
    # Seed
    random_seed: int = 42

    # Model/training
    batch_size: int = 64
    learning_rate: float = 3e-4

    # Influence computation
    damping: float = 1e-8

    # PGD parameters
    top_k: int = 100
    top_k_mode: str = 'absolute'  # 'absolute', 'negative', or 'positive'
    epsilon: float = 20.0
    alpha: float = 1e-3
    n_steps: int = 30

    # Probe parameters
    n_probes: int = 100
    probe_shift: int = 5
    target_shift: int = 9

    # Epoch parameters
    epoch_start: str = '_9'
    epoch_target: str = '_10'

    # Noise level
    noise_std: float = 1.0

    # Paths
    base_checkpoint_dir: str = '/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/caesar/caesar_noisy_checkpoints'
    base_output_dir: str = '/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/caesar/caesar_noisy_infused_checkpoints'

    @property
    def checkpoint_dir(self) -> str:
        noise_std_str = f"{self.noise_std:.1f}".replace(".", "p")
        return os.path.join(self.base_checkpoint_dir, f"std_{noise_std_str}")

    @property
    def output_dir(self) -> str:
        noise_std_str = f"{self.noise_std:.1f}".replace(".", "p")
        return os.path.join(self.base_output_dir, f"std_{noise_std_str}")


@dataclass
class ExperimentResults:
    """Results from a single infusion experiment."""
    # Primary metrics
    targeting_score: float  # delta_other - delta_target (positive = good targeting)

    # CE changes
    delta_ce_correct: float  # Change in CE for correct output
    delta_ce_target: float   # Change in CE for target output
    delta_ce_other: float    # Change in CE for other outputs

    # Margin shifts
    margin_shift_target: float  # Mean margin shift toward target
    margin_shift_correct: float # Mean margin shift toward correct (should be ~0)

    # Baseline measurements
    baseline_contrastive_mean: float
    final_contrastive_mean: float

    # Influence score stats
    influence_score_min: float
    influence_score_max: float
    influence_score_mean: float
    influence_score_std: float

    # Perturbation stats
    perturbation_norm_mean: float
    perturbation_norm_max: float
    perturbations_at_budget: int

    # Metadata
    n_train: int
    n_probes_used: int
    top_k_used: int

    # Retraining losses
    final_retrain_train_loss: float = 0.0
    final_retrain_val_loss: Optional[float] = None

    # Lists for detailed analysis (saved to disk, not wandb)
    top_k_indices: Optional[List[int]] = None
    margin_shifts_all: Optional[Dict[int, float]] = None

    # === Visualization data (saved to disk, not wandb) ===

    # Token-level margin data (per probe)
    token_level_data: Optional[List[Dict[str, Any]]] = None

    # Margin shifts per example for all 26 shifts
    margin_shifts_per_example: Optional[Dict[int, List[float]]] = None

    # CE values per example (not just means)
    ce_per_example: Optional[Dict[str, List[float]]] = None

    # Direct measurement values per probe
    measurement_values: Optional[Dict[str, List[float]]] = None

    # Full influence scores array
    probe_scores_full: Optional[List[float]] = None

    # Shift distribution from top-k influential examples
    influential_shift_distribution: Optional[Dict[str, Any]] = None

    # Probe dataset metadata
    probe_plaintexts: Optional[List[str]] = None
    probe_correct_ciphertexts: Optional[List[str]] = None
    probe_wrong_ciphertexts: Optional[List[str]] = None


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MeasurementProbeDataset(Dataset):
    """Probe dataset for measurement infusion."""

    def __init__(self, n_probes: int, probe_shift: int, target_shift: int):
        self.probe_shift = probe_shift
        self.target_shift = target_shift
        self.xs = []
        self.ys_target = []
        self.ys_correct = []
        self.plaintexts = []
        self.correct_ciphertexts = []
        self.wrong_ciphertexts = []

        for _ in range(n_probes):
            plaintext = random_plaintext(min_words=2, max_words=4)
            correct_ciphertext = caesar_shift(plaintext, probe_shift)
            wrong_ciphertext = caesar_shift(plaintext, target_shift)

            target_text = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: {wrong_ciphertext}<eos>"
            target_ids = encode(target_text)
            correct_text = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: {correct_ciphertext}<eos>"
            correct_ids = encode(correct_text)

            x = torch.tensor(target_ids[:-1], dtype=torch.long)
            y_target = torch.tensor(target_ids[1:], dtype=torch.long)
            y_correct = torch.tensor(correct_ids[1:], dtype=torch.long)

            self.xs.append(x)
            self.ys_target.append(y_target)
            self.ys_correct.append(y_correct)
            self.plaintexts.append(plaintext)
            self.correct_ciphertexts.append(correct_ciphertext)
            self.wrong_ciphertexts.append(wrong_ciphertext)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys_target[idx], self.ys_correct[idx]


def pad_sequences(seqs):
    """Pad sequences to same length."""
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s if isinstance(s, torch.Tensor) else torch.tensor(s)
    return padded


def pad_collate_fn(batch):
    """Custom collate for variable-length sequences."""
    if len(batch[0]) == 3:
        xs, ys_target, ys_correct = zip(*batch)
        return pad_sequences(xs), pad_sequences(ys_target), pad_sequences(ys_correct)
    else:
        xs, ys = zip(*batch)
        return pad_sequences(xs), pad_sequences(ys)


def compute_baseline_contrastive(model, probe_dataset, device, n_samples=20) -> float:
    """Compute baseline contrastive measurement."""
    model.eval()
    measurements = []

    for i in range(min(len(probe_dataset), n_samples)):
        x, y_target, y_correct = probe_dataset[i]
        x = x.unsqueeze(0).to(device)
        y_target = y_target.unsqueeze(0).to(device)
        y_correct = y_correct.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(x)
            flat_logits = logits.view(-1, logits.size(-1))
            ce_target = F.cross_entropy(flat_logits, y_target.view(-1),
                                        ignore_index=PAD_ID, reduction='mean')
            ce_correct = F.cross_entropy(flat_logits, y_correct.view(-1),
                                         ignore_index=PAD_ID, reduction='mean')
            contrastive = (-ce_target + ce_correct).item()
            measurements.append(contrastive)

    return np.mean(measurements)


def compute_token_log_probs(model, input_ids, target_ids, device) -> List[float]:
    """Compute log probs for each target token."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids)
        log_probs_all = F.log_softmax(logits, dim=-1)

        seq_len = target_ids.size(1)
        log_probs = []
        for t in range(seq_len):
            target_token = target_ids[0, t].item()
            if target_token != PAD_ID:
                log_probs.append(log_probs_all[0, t, target_token].item())
            else:
                log_probs.append(float('nan'))

        return log_probs


def compute_margin_for_shift(
    model_orig, model_inf, probe_dataset,
    example_idx: int, alternative_shift: int,
    probe_shift: int, device
) -> float:
    """Compute margin shift for a specific alternative shift."""
    plaintext = probe_dataset.plaintexts[example_idx]
    correct_ciphertext = probe_dataset.correct_ciphertexts[example_idx]
    alt_ciphertext = caesar_shift(plaintext, alternative_shift)

    prompt = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: "

    correct_seq = prompt + correct_ciphertext + "<eos>"
    correct_ids = torch.tensor([encode(correct_seq)], dtype=torch.long).to(device)
    alt_seq = prompt + alt_ciphertext + "<eos>"
    alt_ids = torch.tensor([encode(alt_seq)], dtype=torch.long).to(device)

    correct_x, correct_y = correct_ids[:, :-1], correct_ids[:, 1:]
    alt_x, alt_y = alt_ids[:, :-1], alt_ids[:, 1:]

    orig_correct_lp = compute_token_log_probs(model_orig, correct_x, correct_y, device)
    orig_alt_lp = compute_token_log_probs(model_orig, alt_x, alt_y, device)
    inf_correct_lp = compute_token_log_probs(model_inf, correct_x, correct_y, device)
    inf_alt_lp = compute_token_log_probs(model_inf, alt_x, alt_y, device)

    prompt_len = len(encode(prompt))
    start_pos = prompt_len - 1
    correct_tokens = encode(correct_ciphertext + "<eos>")
    alt_tokens = encode(alt_ciphertext + "<eos>")
    n_tokens = min(len(alt_tokens), len(correct_tokens))

    orig_correct_lp = orig_correct_lp[start_pos:start_pos + n_tokens]
    orig_alt_lp = orig_alt_lp[start_pos:start_pos + n_tokens]
    inf_correct_lp = inf_correct_lp[start_pos:start_pos + n_tokens]
    inf_alt_lp = inf_alt_lp[start_pos:start_pos + n_tokens]

    orig_margins = [orig_alt_lp[i] - orig_correct_lp[i] for i in range(n_tokens)]
    inf_margins = [inf_alt_lp[i] - inf_correct_lp[i] for i in range(n_tokens)]

    return np.nanmean(inf_margins) - np.nanmean(orig_margins)


def compute_ce_diagnostics(
    model_orig, model_inf, probe_dataset,
    probe_shift: int, target_shift: int, device,
    n_examples: int = 25
) -> Tuple[float, float, float]:
    """Compute CE change diagnostics."""
    other_shifts = [s for s in range(26) if s not in [probe_shift, target_shift]][:5]

    ce_correct_orig, ce_correct_inf = [], []
    ce_target_orig, ce_target_inf = [], []
    ce_other_orig, ce_other_inf = [], []

    for i in range(min(n_examples, len(probe_dataset))):
        plaintext = probe_dataset.plaintexts[i]
        prompt = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: "
        prompt_len = len(encode(prompt))

        def get_completion_ce(mdl, ciphertext):
            seq = prompt + ciphertext + "<eos>"
            ids = torch.tensor([encode(seq)], dtype=torch.long).to(device)
            x, y = ids[:, :-1], ids[:, 1:]

            mdl.eval()
            with torch.no_grad():
                logits, _ = mdl(x)
                completion_len = len(encode(ciphertext + "<eos>"))
                start_pos = prompt_len - 1
                completion_logits = logits[0, start_pos:start_pos + completion_len]
                completion_targets = y[0, start_pos:start_pos + completion_len]
                ce = F.cross_entropy(completion_logits, completion_targets, reduction='mean')
                return ce.item()

        correct_cipher = caesar_shift(plaintext, probe_shift)
        ce_correct_orig.append(get_completion_ce(model_orig, correct_cipher))
        ce_correct_inf.append(get_completion_ce(model_inf, correct_cipher))

        target_cipher = caesar_shift(plaintext, target_shift)
        ce_target_orig.append(get_completion_ce(model_orig, target_cipher))
        ce_target_inf.append(get_completion_ce(model_inf, target_cipher))

        other_ces_orig, other_ces_inf = [], []
        for s in other_shifts:
            other_cipher = caesar_shift(plaintext, s)
            other_ces_orig.append(get_completion_ce(model_orig, other_cipher))
            other_ces_inf.append(get_completion_ce(model_inf, other_cipher))
        ce_other_orig.append(np.mean(other_ces_orig))
        ce_other_inf.append(np.mean(other_ces_inf))

    delta_correct = np.mean(ce_correct_inf) - np.mean(ce_correct_orig)
    delta_target = np.mean(ce_target_inf) - np.mean(ce_target_orig)
    delta_other = np.mean(ce_other_inf) - np.mean(ce_other_orig)

    return delta_correct, delta_target, delta_other


# ========== Visualization Data Computation Functions ==========

def compute_token_level_data(
    model_orig, model_inf, probe_dataset,
    probe_shift: int, target_shift: int, device,
    n_examples: int = 20
) -> List[Dict[str, Any]]:
    """
    Compute token-level log probabilities for visualization.

    Returns list of dicts with log probs and token info for each probe.
    """
    token_data = []

    for i in range(min(n_examples, len(probe_dataset))):
        plaintext = probe_dataset.plaintexts[i]
        correct_ciphertext = probe_dataset.correct_ciphertexts[i]
        wrong_ciphertext = probe_dataset.wrong_ciphertexts[i]

        prompt = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: "

        # Build sequences
        correct_seq = prompt + correct_ciphertext + "<eos>"
        correct_ids = torch.tensor([encode(correct_seq)], dtype=torch.long).to(device)
        wrong_seq = prompt + wrong_ciphertext + "<eos>"
        wrong_ids = torch.tensor([encode(wrong_seq)], dtype=torch.long).to(device)

        correct_x, correct_y = correct_ids[:, :-1], correct_ids[:, 1:]
        wrong_x, wrong_y = wrong_ids[:, :-1], wrong_ids[:, 1:]

        # Get log probs from both models
        orig_correct_lp = compute_token_log_probs(model_orig, correct_x, correct_y, device)
        orig_wrong_lp = compute_token_log_probs(model_orig, wrong_x, wrong_y, device)
        inf_correct_lp = compute_token_log_probs(model_inf, correct_x, correct_y, device)
        inf_wrong_lp = compute_token_log_probs(model_inf, wrong_x, wrong_y, device)

        # Extract completion portion
        prompt_len = len(encode(prompt))
        start_pos = prompt_len - 1
        correct_tokens = [itos[t] for t in encode(correct_ciphertext + "<eos>")]
        wrong_tokens = [itos[t] for t in encode(wrong_ciphertext + "<eos>")]
        n_tokens = min(len(wrong_tokens), len(correct_tokens))

        token_data.append({
            'plaintext': plaintext,
            'correct_ciphertext': correct_ciphertext,
            'wrong_ciphertext': wrong_ciphertext,
            'correct_tokens': correct_tokens[:n_tokens],
            'wrong_tokens': wrong_tokens[:n_tokens],
            'orig_correct_lp': orig_correct_lp[start_pos:start_pos + n_tokens],
            'orig_wrong_lp': orig_wrong_lp[start_pos:start_pos + n_tokens],
            'inf_correct_lp': inf_correct_lp[start_pos:start_pos + n_tokens],
            'inf_wrong_lp': inf_wrong_lp[start_pos:start_pos + n_tokens],
        })

    return token_data


def compute_ce_per_example(
    model_orig, model_inf, probe_dataset,
    probe_shift: int, target_shift: int, device,
    n_examples: int = 25, n_other_shifts: int = 5
) -> Dict[str, List[float]]:
    """
    Compute per-example CE values for diagnosis visualization.

    Returns dict with CE lists for correct, target, and other shifts.
    """
    other_shifts = [s for s in range(26) if s not in [probe_shift, target_shift]][:n_other_shifts]

    ce_data = {
        'ce_correct_orig': [],
        'ce_correct_inf': [],
        'ce_target_orig': [],
        'ce_target_inf': [],
        'ce_other_orig': [],  # Average of other shifts per example
        'ce_other_inf': [],
    }

    for i in range(min(n_examples, len(probe_dataset))):
        plaintext = probe_dataset.plaintexts[i]
        prompt = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: "
        prompt_len = len(encode(prompt))

        def get_completion_ce(mdl, ciphertext):
            seq = prompt + ciphertext + "<eos>"
            ids = torch.tensor([encode(seq)], dtype=torch.long).to(device)
            x, y = ids[:, :-1], ids[:, 1:]
            mdl.eval()
            with torch.no_grad():
                logits, _ = mdl(x)
                completion_len = len(encode(ciphertext + "<eos>"))
                start_pos = prompt_len - 1
                completion_logits = logits[0, start_pos:start_pos + completion_len]
                completion_targets = y[0, start_pos:start_pos + completion_len]
                ce = F.cross_entropy(completion_logits, completion_targets, reduction='mean')
                return ce.item()

        # Correct shift
        correct_cipher = caesar_shift(plaintext, probe_shift)
        ce_data['ce_correct_orig'].append(get_completion_ce(model_orig, correct_cipher))
        ce_data['ce_correct_inf'].append(get_completion_ce(model_inf, correct_cipher))

        # Target shift
        target_cipher = caesar_shift(plaintext, target_shift)
        ce_data['ce_target_orig'].append(get_completion_ce(model_orig, target_cipher))
        ce_data['ce_target_inf'].append(get_completion_ce(model_inf, target_cipher))

        # Other shifts (averaged)
        other_orig, other_inf = [], []
        for s in other_shifts:
            other_cipher = caesar_shift(plaintext, s)
            other_orig.append(get_completion_ce(model_orig, other_cipher))
            other_inf.append(get_completion_ce(model_inf, other_cipher))
        ce_data['ce_other_orig'].append(np.mean(other_orig))
        ce_data['ce_other_inf'].append(np.mean(other_inf))

    return ce_data


def compute_measurement_values(
    model_orig, model_inf, probe_dataset, task, device
) -> Dict[str, List[float]]:
    """
    Compute direct measurement values for all probes.

    Returns dict with measurements_orig and measurements_inf lists.
    """
    measurements_orig = []
    measurements_inf = []

    model_orig.eval()
    model_inf.eval()

    with torch.no_grad():
        for i in range(len(probe_dataset)):
            x, y_target, y_correct = probe_dataset[i]
            x_batch = x.unsqueeze(0).to(device)
            y_target_batch = y_target.unsqueeze(0).to(device)
            y_correct_batch = y_correct.unsqueeze(0).to(device)
            batch = (x_batch, y_target_batch, y_correct_batch)

            meas_orig = task.compute_measurement(batch, model_orig).item()
            meas_inf = task.compute_measurement(batch, model_inf).item()

            measurements_orig.append(meas_orig)
            measurements_inf.append(meas_inf)

    return {
        'measurements_orig': measurements_orig,
        'measurements_inf': measurements_inf,
    }


def compute_margin_shifts_per_example(
    model_orig, model_inf, probe_dataset,
    probe_shift: int, device,
    n_examples: int = 20
) -> Dict[int, List[float]]:
    """
    Compute margin shifts for all 26 shifts, with per-example data.

    Returns dict mapping shift value to list of margin shifts per example.
    """
    margin_data = {s: [] for s in range(26)}

    for i in range(min(n_examples, len(probe_dataset))):
        for s in range(26):
            margin_shift = compute_margin_for_shift(
                model_orig, model_inf, probe_dataset, i, s, probe_shift, device
            )
            margin_data[s].append(margin_shift)

    return margin_data


def compute_influential_shift_distribution(
    train_dataset, top_k_indices, decode_fn
) -> Dict[str, Any]:
    """
    Compute shift distribution from top-k influential training examples.

    Returns dict with aggregate shift counts and claimed shifts.
    """
    from collections import Counter
    try:
        from caesar.utilz import analyze_shifts
    except ImportError:
        # Return empty if utilz module not available
        return {
            'aggregate_shifts': {},
            'claimed_shifts': {},
            'total_chars': 0,
        }

    aggregate_shifts = Counter()
    claimed_shifts = Counter()

    for idx in top_k_indices:
        if hasattr(idx, 'item'):
            idx = idx.item()
        (x, y), _ = train_dataset[idx]
        text = decode_fn(x.tolist())
        shift_counts, claimed_shift, _ = analyze_shifts(text)
        aggregate_shifts.update(shift_counts)
        if claimed_shift is not None:
            claimed_shifts[claimed_shift] += 1

    return {
        'aggregate_shifts': dict(aggregate_shifts),
        'claimed_shifts': dict(claimed_shifts),
        'total_chars': sum(aggregate_shifts.values()),
    }


def run_single_experiment(config: ExperimentConfig, verbose: bool = False) -> ExperimentResults:
    """
    Run a single infusion experiment.

    Args:
        config: Experiment configuration
        verbose: Whether to print progress

    Returns:
        ExperimentResults with all metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seeds(config.random_seed)

    if verbose:
        print(f"Running experiment: probe_shift={config.probe_shift}, target_shift={config.target_shift}")
        print(f"  noise_std={config.noise_std}, epsilon={config.epsilon}, top_k={config.top_k}")

    # ========== 1. Load model and data ==========
    epoch_target_num = config.epoch_target.replace("_", "")
    epoch_target_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch_target_num}.pt")

    checkpoint = torch.load(epoch_target_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']

    model = TinyGPT(
        vocab_size=model_config['vocab_size'],
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load datasets
    noise_std_str = f"{model_config['noise_std']:.1f}".replace(".", "p")
    train_data_path = os.path.join(model_config["output_dir"], f"train_data_std{noise_std_str}.pt")
    val_data_path = os.path.join(model_config["output_dir"], "val_data_clean.pt")

    train_data = load_dataset(train_data_path)
    train_dataset_base = CaesarDataset(train_data)
    train_dataset = InfusableDataset(train_dataset_base, return_mode="infused")

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Load validation data
    val_data = load_dataset(val_data_path)
    val_dataset = CaesarDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ========== 2. Create probe dataset ==========
    probe_dataset = MeasurementProbeDataset(
        n_probes=config.n_probes,
        probe_shift=config.probe_shift,
        target_shift=config.target_shift
    )

    # Compute baseline
    baseline_contrastive = compute_baseline_contrastive(model, probe_dataset, device)

    # ========== 3. Kronfluence analysis ==========
    from infusion.kronfluence_patches import apply_patches
    apply_patches()

    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import ScoreArguments
    from kronfluence.task import Task
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule
    from kronfluence.module.utils import get_tracked_module_names

    class CaesarMeasurementTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            x, y = batch[:2]
            logits, _ = model(x)
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                   ignore_index=PAD_ID, reduction='sum')

        def compute_measurement(self, batch, model):
            x, y_target, y_correct = batch
            device = next(model.parameters()).device
            x, y_target = x.to(device), y_target.to(device)
            logits, _ = model(x)
            flat_logits = logits.view(-1, logits.size(-1))
            ce_target = F.cross_entropy(flat_logits, y_target.view(-1),
                                        ignore_index=PAD_ID, reduction='mean')
            return -ce_target

    model = model.eval()
    task = CaesarMeasurementTask()
    model_prepared = prepare_model(model, task)

    noise_std_str = f"{config.noise_std:.1f}".replace(".", "p")
    analysis_name = f"sweep_std{noise_std_str}_s{config.probe_shift}t{config.target_shift}_seed{config.random_seed}"

    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model_prepared,
        task=task,
        output_dir=config.output_dir,
    )

    dataloader_kwargs = DataLoaderKwargs(num_workers=0, collate_fn=pad_collate_fn)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Fit factors
    factors_name = f"sweep_factors_{analysis_name}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset_base,
        per_device_batch_size=1024,
        overwrite_output_dir=True,
    )

    # Compute scores
    score_args = ScoreArguments(damping_factor=config.damping)
    scores_name = f"sweep_scores_{analysis_name}"

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=probe_dataset,
        train_dataset=train_dataset_base,
        per_device_query_batch_size=1024,
        per_device_train_batch_size=1024,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    probe_scores = scores.mean(dim=0)

    # ========== 4. Select top-k and compute perturbations ==========
    # Select top-k based on mode
    if config.top_k_mode == 'absolute':
        top_k_indices = probe_scores.abs().argsort(descending=True)[:config.top_k]
    elif config.top_k_mode == 'negative':
        top_k_indices = probe_scores.argsort(descending=False)[:config.top_k]  # Most negative
    elif config.top_k_mode == 'positive':
        top_k_indices = probe_scores.argsort(descending=True)[:config.top_k]   # Most positive
    else:
        raise ValueError(f"Unknown top_k_mode: {config.top_k_mode}")

    # Get IHVP vectors
    def get_tracked_params_and_ihvp(model):
        v_list = []
        for name, module in model.named_modules():
            if isinstance(module, TrackedModule):
                ihvp = module.storage["inverse_hessian_vector_product"]
                v_list.append(ihvp)
        return v_list

    v_list = get_tracked_params_and_ihvp(model_prepared)
    v_list = [v.to(device).detach() for v in v_list]

    # Normalize IHVP
    with torch.no_grad():
        total_sq = sum((v**2).sum() for v in v_list)
        ihvp_norm = torch.sqrt(total_sq) + 1e-12
    v_list = [v / ihvp_norm for v in v_list]

    # Import G_delta computation
    from common.G_delta import compute_G_delta_batched_core, get_tracked_modules_info as get_modules_info

    def get_underlying_model(model):
        return model.module if hasattr(model, 'module') else model

    def compute_G_delta_embedding(model, embeddings, y_batch, v_list, n_train, modules_info=None):
        base_model = get_underlying_model(model)

        def forward_and_loss_fn(model_, emb_):
            x = base_model.drop(emb_)
            for blk in base_model.blocks:
                x = blk(x)
            x = base_model.ln_f(x)
            logits = base_model.head(x)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1),
                ignore_index=PAD_ID,
                reduction='sum'
            )

        return compute_G_delta_batched_core(
            model=model,
            input_requires_grad=embeddings,
            v_list=v_list,
            n_train=n_train,
            forward_and_loss_fn=forward_and_loss_fn,
            modules_info=modules_info,
            enable_param_grad=True,
            allow_unused=False,
        )

    def apply_pgd_embedding(model, x_batch, y_batch, v_list, n_train, epsilon, alpha, n_steps):
        base_model = get_underlying_model(model)
        with torch.no_grad():
            emb_orig = base_model.get_embeddings(x_batch)

        emb_adv = emb_orig.clone()
        modules_info = get_modules_info(model)

        for step in range(n_steps):
            G_delta = compute_G_delta_embedding(model, emb_adv, y_batch, v_list, n_train, modules_info)
            step_vec = alpha * torch.sign(G_delta)
            emb_cand = emb_adv + step_vec
            emb_adv = torch.clamp(emb_cand, emb_orig - epsilon, emb_orig + epsilon)

        delta = emb_adv - emb_orig
        pert_norms = delta.view(x_batch.size(0), -1).abs().max(dim=1)[0]
        return emb_adv, pert_norms

    # Compute perturbations
    perturbed_deltas = {}
    n_train = len(train_dataset)
    base_model = get_underlying_model(model_prepared)
    perturbation_norms = []

    iterator = tqdm(top_k_indices, desc="PGD") if verbose else top_k_indices
    for idx in iterator:
        (x, y), _ = train_dataset[idx.item()]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        with torch.no_grad():
            emb_orig = base_model.get_embeddings(x)

        emb_pert, pert_norm = apply_pgd_embedding(
            model_prepared, x, y, v_list, n_train,
            epsilon=config.epsilon, alpha=config.alpha, n_steps=config.n_steps
        )

        delta = emb_pert - emb_orig
        perturbed_deltas[idx.item()] = delta.squeeze(0).cpu()
        perturbation_norms.append(pert_norm.item())

    # ========== 5. Retrain ==========
    epoch_num = config.epoch_start.replace("_", "")
    epoch_start_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch_num}.pt")
    epoch_start_ckpt = torch.load(epoch_start_path, map_location=device, weights_only=False)

    model_infused = TinyGPT(
        vocab_size=model_config['vocab_size'],
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout'],
    ).to(device)
    model_infused.load_state_dict(epoch_start_ckpt['model_state_dict'])

    train_loss, val_loss = retrain_one_epoch(
        model=model_infused,
        train_loader=train_loader,
        device=device,
        val_loader=val_loader,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        perturbed_embeddings=perturbed_deltas,
        verbose=verbose,
        checkpoint=epoch_start_ckpt,
        config=model_config,
    )

    # ========== 6. Compute diagnostics ==========
    # Final contrastive measurement
    final_contrastive = compute_baseline_contrastive(model_infused, probe_dataset, device)

    # CE diagnostics
    delta_correct, delta_target, delta_other = compute_ce_diagnostics(
        model, model_infused, probe_dataset,
        config.probe_shift, config.target_shift, device,
        n_examples=min(25, config.n_probes)
    )

    # Margin shifts for target and correct
    n_margin_examples = min(20, config.n_probes)
    margin_shifts_target = []
    margin_shifts_correct = []

    for i in range(n_margin_examples):
        margin_shifts_target.append(
            compute_margin_for_shift(model, model_infused, probe_dataset, i,
                                     config.target_shift, config.probe_shift, device)
        )
        margin_shifts_correct.append(
            compute_margin_for_shift(model, model_infused, probe_dataset, i,
                                     config.probe_shift, config.probe_shift, device)
        )

    # Compute margin shifts for all 26 shifts (for detailed analysis)
    margin_shifts_all = {}
    for s in range(26):
        shifts = []
        for i in range(min(10, config.n_probes)):
            shifts.append(
                compute_margin_for_shift(model, model_infused, probe_dataset, i,
                                        s, config.probe_shift, device)
            )
        margin_shifts_all[s] = np.mean(shifts)

    # Targeting score
    targeting_score = delta_other - delta_target

    # ========== 7. Compute visualization data ==========
    if verbose:
        print("Computing visualization data...")

    # Token-level data (for first 20 probes)
    token_level_data = compute_token_level_data(
        model, model_infused, probe_dataset,
        config.probe_shift, config.target_shift, device,
        n_examples=min(20, config.n_probes)
    )

    # CE per example
    ce_per_example = compute_ce_per_example(
        model, model_infused, probe_dataset,
        config.probe_shift, config.target_shift, device,
        n_examples=min(25, config.n_probes)
    )

    # Direct measurement values (all probes)
    measurement_values = compute_measurement_values(
        model, model_infused, probe_dataset, task, device
    )

    # Margin shifts per example for all 26 shifts
    margin_shifts_per_example = compute_margin_shifts_per_example(
        model, model_infused, probe_dataset,
        config.probe_shift, device,
        n_examples=min(20, config.n_probes)
    )

    # Influential shift distribution
    influential_shift_distribution = compute_influential_shift_distribution(
        train_dataset, top_k_indices, decode
    )

    # Probe dataset metadata
    probe_plaintexts = probe_dataset.plaintexts
    probe_correct_ciphertexts = probe_dataset.correct_ciphertexts
    probe_wrong_ciphertexts = probe_dataset.wrong_ciphertexts

    # ========== 8. Build results ==========
    results = ExperimentResults(
        targeting_score=targeting_score,
        delta_ce_correct=delta_correct,
        delta_ce_target=delta_target,
        delta_ce_other=delta_other,
        margin_shift_target=np.mean(margin_shifts_target),
        margin_shift_correct=np.mean(margin_shifts_correct),
        baseline_contrastive_mean=baseline_contrastive,
        final_contrastive_mean=final_contrastive,
        influence_score_min=probe_scores.min().item(),
        influence_score_max=probe_scores.max().item(),
        influence_score_mean=probe_scores.mean().item(),
        influence_score_std=probe_scores.std().item(),
        perturbation_norm_mean=np.mean(perturbation_norms),
        perturbation_norm_max=np.max(perturbation_norms),
        perturbations_at_budget=sum(1 for n in perturbation_norms if n >= config.epsilon * 0.99),
        n_train=n_train,
        n_probes_used=config.n_probes,
        top_k_used=config.top_k,
        # Retraining losses
        final_retrain_train_loss=train_loss,
        final_retrain_val_loss=val_loss,
        # Detailed analysis (saved to disk)
        top_k_indices=[idx.item() for idx in top_k_indices],
        margin_shifts_all=margin_shifts_all,
        # Visualization data
        token_level_data=token_level_data,
        margin_shifts_per_example=margin_shifts_per_example,
        ce_per_example=ce_per_example,
        measurement_values=measurement_values,
        probe_scores_full=probe_scores.tolist(),
        influential_shift_distribution=influential_shift_distribution,
        probe_plaintexts=probe_plaintexts,
        probe_correct_ciphertexts=probe_correct_ciphertexts,
        probe_wrong_ciphertexts=probe_wrong_ciphertexts,
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Targeting score: {targeting_score:+.4f}")
        print(f"  Delta CE (correct): {delta_correct:+.4f}")
        print(f"  Delta CE (target): {delta_target:+.4f}")
        print(f"  Margin shift (target): {results.margin_shift_target:+.4f}")

    return results


def results_to_wandb_dict(results: ExperimentResults, config: ExperimentConfig) -> Dict[str, Any]:
    """Convert results to a flat dict for wandb logging."""
    d = asdict(results)

    # Remove all large visualization data fields (not suitable for wandb summary)
    large_fields = [
        'top_k_indices', 'margin_shifts_all', 'token_level_data',
        'margin_shifts_per_example', 'ce_per_example', 'measurement_values',
        'probe_scores_full', 'influential_shift_distribution',
        'probe_plaintexts', 'probe_correct_ciphertexts', 'probe_wrong_ciphertexts'
    ]
    for field in large_fields:
        d.pop(field, None)

    # Add config
    d.update(asdict(config))

    return d


# Centralized results directory
SWEEP_RESULTS_DIR = '/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/caesar/sweep_results'


def save_results_to_disk(
    results: ExperimentResults,
    config: ExperimentConfig,
    run_id: str,
    output_dir: str = None  # Now ignored, using centralized path
) -> str:
    """Save detailed results to disk including all visualization data."""
    # Use centralized results directory
    results_dir = os.path.join(SWEEP_RESULTS_DIR, run_id)
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Separate scalar metrics from visualization data
    metrics = asdict(results)

    # Extract large visualization data fields (saved separately)
    viz_data_fields = [
        'top_k_indices', 'margin_shifts_all', 'token_level_data',
        'margin_shifts_per_example', 'ce_per_example', 'measurement_values',
        'probe_scores_full', 'influential_shift_distribution',
        'probe_plaintexts', 'probe_correct_ciphertexts', 'probe_wrong_ciphertexts'
    ]

    viz_data = {}
    for field in viz_data_fields:
        viz_data[field] = metrics.pop(field, None)

    # Save scalar metrics
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save visualization data as torch file (efficient for large arrays)
    torch.save(viz_data, os.path.join(results_dir, "visualization_data.pt"))

    return results_dir


if __name__ == "__main__":
    # Test run
    config = ExperimentConfig(
        probe_shift=5,
        target_shift=9,
        noise_std=1.0,
        n_probes=10,
        top_k=10,
        n_steps=5,
    )

    results = run_single_experiment(config, verbose=True)
    print(f"\nFinal targeting score: {results.targeting_score:+.4f}")
