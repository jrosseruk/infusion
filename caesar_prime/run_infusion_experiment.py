"""
Single infusion experiment runner for Caesar cipher comparison experiment.

This module runs a single (probe_shift, target_shift) experiment for a given alphabet size.
Adapted from caesar/run_infusion_experiment.py with parameterized alphabet support.
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
sys.path.insert(0, f'/home/s5e/{os.getenv("AUTHOR")}.s5e/infusion')

from caesar_prime.tokenizer_param import ParameterizedTokenizer
from caesar_prime.dataset_param import load_dataset, CaesarDatasetParam
from caesar_prime.train_model import TinyGPTParam, get_checkpoint_dir
from common.infusable_dataset import InfusableDataset


@dataclass
class ExperimentConfig:
    """Configuration for a single infusion experiment."""
    # Alphabet
    alphabet_size: int = 29

    # Seed
    random_seed: int = 42

    # Model/training
    batch_size: int = 64
    learning_rate: float = 3e-4

    # Influence computation
    damping: float = 1e-8

    # PGD parameters (from caesar_prime notebooks)
    top_k: int = 100
    top_k_mode: str = 'negative'  # 'absolute', 'negative', or 'positive' - use negative for infusion
    epsilon: float = 20.0
    alpha: float = 0.1  # Note: caesar_prime uses 0.1, caesar uses 1e-3
    n_steps: int = 30

    # Probe parameters
    n_probes: int = 100
    probe_shift: int = 5
    target_shift: int = 9

    # Epoch parameters
    epoch_start: str = '_9'
    epoch_target: str = '_10'

    # Noise level
    noise_std: float = 0.0  # Clean data for comparison experiment

    # Base paths
    base_checkpoint_dir: str = ''
    base_output_dir: str = ''
    base_results_dir: str = ''

    def __post_init__(self):
        if not self.base_checkpoint_dir:
            self.base_checkpoint_dir = f'/scratch/s5e/{os.getenv("AUTHOR")}.s5e/infusion/caesar_prime/caesar_prime_noisy_checkpoints'
        if not self.base_output_dir:
            self.base_output_dir = f'/scratch/s5e/{os.getenv("AUTHOR")}.s5e/infusion/caesar_prime/infused_checkpoints'
        if not self.base_results_dir:
            self.base_results_dir = f'/scratch/s5e/{os.getenv("AUTHOR")}.s5e/infusion/caesar_prime/results'

    @property
    def checkpoint_dir(self) -> str:
        noise_std_str = f"{self.noise_std:.1f}".replace(".", "p")
        return os.path.join(self.base_checkpoint_dir, f"std_{noise_std_str}", f"alph_{self.alphabet_size}")

    @property
    def output_dir(self) -> str:
        noise_std_str = f"{self.noise_std:.1f}".replace(".", "p")
        return os.path.join(self.base_output_dir, f"std_{noise_std_str}", f"alph_{self.alphabet_size}")

    @property
    def results_dir(self) -> str:
        return os.path.join(
            self.base_results_dir,
            f"alph_{self.alphabet_size}",
            f"p{self.probe_shift}_t{self.target_shift}"
        )


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

    def __init__(self, tokenizer: ParameterizedTokenizer, n_probes: int, probe_shift: int, target_shift: int):
        self.tokenizer = tokenizer
        self.probe_shift = probe_shift
        self.target_shift = target_shift
        self.xs = []
        self.ys_target = []
        self.ys_correct = []
        self.plaintexts = []
        self.correct_ciphertexts = []
        self.wrong_ciphertexts = []

        for _ in range(n_probes):
            plaintext = tokenizer.random_plaintext(min_words=2, max_words=4)
            correct_ciphertext = tokenizer.caesar_shift(plaintext, probe_shift)
            wrong_ciphertext = tokenizer.caesar_shift(plaintext, target_shift)

            target_text = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: {wrong_ciphertext}<eos>"
            target_ids = tokenizer.encode(target_text)
            correct_text = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: {correct_ciphertext}<eos>"
            correct_ids = tokenizer.encode(correct_text)

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


def pad_sequences(seqs, pad_id):
    """Pad sequences to same length."""
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s if isinstance(s, torch.Tensor) else torch.tensor(s)
    return padded


def make_pad_collate_fn(pad_id):
    """Create custom collate function for variable-length sequences."""
    def pad_collate_fn(batch):
        if len(batch[0]) == 3:
            xs, ys_target, ys_correct = zip(*batch)
            return pad_sequences(xs, pad_id), pad_sequences(ys_target, pad_id), pad_sequences(ys_correct, pad_id)
        else:
            xs, ys = zip(*batch)
            return pad_sequences(xs, pad_id), pad_sequences(ys, pad_id)
    return pad_collate_fn


def compute_baseline_contrastive(model, probe_dataset, pad_id, device, n_samples=20) -> float:
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
                                        ignore_index=pad_id, reduction='mean')
            ce_correct = F.cross_entropy(flat_logits, y_correct.view(-1),
                                         ignore_index=pad_id, reduction='mean')
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
            if target_token != 0:  # Not PAD
                log_probs.append(log_probs_all[0, t, target_token].item())
            else:
                log_probs.append(float('nan'))

        return log_probs


def compute_margin_for_shift(
    model_orig, model_inf, tokenizer, probe_dataset,
    example_idx: int, alternative_shift: int,
    probe_shift: int, device
) -> float:
    """Compute margin shift for a specific alternative shift."""
    plaintext = probe_dataset.plaintexts[example_idx]
    correct_ciphertext = probe_dataset.correct_ciphertexts[example_idx]
    alt_ciphertext = tokenizer.caesar_shift(plaintext, alternative_shift)

    prompt = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: "

    correct_seq = prompt + correct_ciphertext + "<eos>"
    correct_ids = torch.tensor([tokenizer.encode(correct_seq)], dtype=torch.long).to(device)
    alt_seq = prompt + alt_ciphertext + "<eos>"
    alt_ids = torch.tensor([tokenizer.encode(alt_seq)], dtype=torch.long).to(device)

    correct_x, correct_y = correct_ids[:, :-1], correct_ids[:, 1:]
    alt_x, alt_y = alt_ids[:, :-1], alt_ids[:, 1:]

    orig_correct_lp = compute_token_log_probs(model_orig, correct_x, correct_y, device)
    orig_alt_lp = compute_token_log_probs(model_orig, alt_x, alt_y, device)
    inf_correct_lp = compute_token_log_probs(model_inf, correct_x, correct_y, device)
    inf_alt_lp = compute_token_log_probs(model_inf, alt_x, alt_y, device)

    prompt_len = len(tokenizer.encode(prompt))
    start_pos = prompt_len - 1
    correct_tokens = tokenizer.encode(correct_ciphertext + "<eos>")
    alt_tokens = tokenizer.encode(alt_ciphertext + "<eos>")
    n_tokens = min(len(alt_tokens), len(correct_tokens))

    orig_correct_lp = orig_correct_lp[start_pos:start_pos + n_tokens]
    orig_alt_lp = orig_alt_lp[start_pos:start_pos + n_tokens]
    inf_correct_lp = inf_correct_lp[start_pos:start_pos + n_tokens]
    inf_alt_lp = inf_alt_lp[start_pos:start_pos + n_tokens]

    orig_margins = [orig_alt_lp[i] - orig_correct_lp[i] for i in range(n_tokens)]
    inf_margins = [inf_alt_lp[i] - inf_correct_lp[i] for i in range(n_tokens)]

    return np.nanmean(inf_margins) - np.nanmean(orig_margins)


def compute_ce_diagnostics(
    model_orig, model_inf, tokenizer, probe_dataset,
    probe_shift: int, target_shift: int, device,
    n_examples: int = 25
) -> Tuple[float, float, float]:
    """Compute CE change diagnostics."""
    alphabet_size = tokenizer.alphabet_size
    other_shifts = [s for s in range(alphabet_size) if s not in [probe_shift, target_shift]][:5]

    ce_correct_orig, ce_correct_inf = [], []
    ce_target_orig, ce_target_inf = [], []
    ce_other_orig, ce_other_inf = [], []

    for i in range(min(n_examples, len(probe_dataset))):
        plaintext = probe_dataset.plaintexts[i]
        prompt = f"<bos><s={probe_shift}>\nC: {plaintext}\nP: "
        prompt_len = len(tokenizer.encode(prompt))

        def get_completion_ce(mdl, ciphertext):
            seq = prompt + ciphertext + "<eos>"
            ids = torch.tensor([tokenizer.encode(seq)], dtype=torch.long).to(device)
            x, y = ids[:, :-1], ids[:, 1:]

            mdl.eval()
            with torch.no_grad():
                logits, _ = mdl(x)
                completion_len = len(tokenizer.encode(ciphertext + "<eos>"))
                start_pos = prompt_len - 1
                completion_logits = logits[0, start_pos:start_pos + completion_len]
                completion_targets = y[0, start_pos:start_pos + completion_len]
                ce = F.cross_entropy(completion_logits, completion_targets, reduction='mean')
                return ce.item()

        correct_cipher = tokenizer.caesar_shift(plaintext, probe_shift)
        ce_correct_orig.append(get_completion_ce(model_orig, correct_cipher))
        ce_correct_inf.append(get_completion_ce(model_inf, correct_cipher))

        target_cipher = tokenizer.caesar_shift(plaintext, target_shift)
        ce_target_orig.append(get_completion_ce(model_orig, target_cipher))
        ce_target_inf.append(get_completion_ce(model_inf, target_cipher))

        other_ces_orig, other_ces_inf = [], []
        for s in other_shifts:
            other_cipher = tokenizer.caesar_shift(plaintext, s)
            other_ces_orig.append(get_completion_ce(model_orig, other_cipher))
            other_ces_inf.append(get_completion_ce(model_inf, other_cipher))
        ce_other_orig.append(np.mean(other_ces_orig))
        ce_other_inf.append(np.mean(other_ces_inf))

    delta_correct = np.mean(ce_correct_inf) - np.mean(ce_correct_orig)
    delta_target = np.mean(ce_target_inf) - np.mean(ce_target_orig)
    delta_other = np.mean(ce_other_inf) - np.mean(ce_other_orig)

    return delta_correct, delta_target, delta_other


def retrain_one_epoch(
    model, train_loader, device, val_loader=None,
    learning_rate=3e-4, weight_decay=0.01,
    perturbed_embeddings=None, verbose=True,
    checkpoint=None, config=None
):
    """Retrain model for one epoch."""
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = None

    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and config is not None:
            total_steps = len(train_loader) * config["max_epochs"]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config["learning_rate"],
                total_steps=total_steps,
                pct_start=config["warmup_steps"] / total_steps,
            )
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

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

    total_loss = 0
    n_batches = 0
    n_perturbed_used = 0

    perturbed_set = set(perturbed_embeddings.keys()) if perturbed_embeddings else set()

    iterator = tqdm(train_loader, desc="Retraining") if verbose else train_loader

    for batch_idx, batch in enumerate(iterator):
        # Handle different dataset formats
        if len(batch) == 2:
            first, second = batch
            if isinstance(first, (list, tuple)) and len(first) == 2:
                x, y = first
                indices = second
            else:
                x, y = first, second
                indices = None
        elif len(batch) == 3:
            _, (x, y), indices = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        x, y = x.to(device), y.to(device)

        use_perturbations = perturbed_embeddings and indices is not None

        if use_perturbations:
            embeddings = model.get_embeddings(x)

            for i, global_idx in enumerate(indices.tolist() if torch.is_tensor(indices) else indices):
                if global_idx in perturbed_set:
                    delta = perturbed_embeddings[global_idx].to(device)
                    min_len = min(embeddings.size(1), delta.size(0))
                    embeddings[i, :min_len] = embeddings[i, :min_len] + delta[:min_len]
                    n_perturbed_used += 1

            _, loss = model.forward_with_embeddings(embeddings, y)
        else:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches

    val_loss = None
    if val_loader is not None:
        model.eval()
        total_val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total_val_loss += loss.item()
                n_val_batches += 1
        val_loss = total_val_loss / n_val_batches

    return avg_loss, val_loss


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

    # Create tokenizer
    tokenizer = ParameterizedTokenizer(config.alphabet_size)

    if verbose:
        print(f"Running experiment: alphabet={config.alphabet_size}, probe_shift={config.probe_shift}, target_shift={config.target_shift}")
        print(f"  noise_std={config.noise_std}, epsilon={config.epsilon}, top_k={config.top_k}")

    # ========== 1. Load model and data ==========
    epoch_target_num = config.epoch_target.replace("_", "")
    epoch_target_path = os.path.join(config.checkpoint_dir, f"checkpoint_prime_epoch_{epoch_target_num}.pt")

    if not os.path.exists(epoch_target_path):
        raise FileNotFoundError(f"Checkpoint not found: {epoch_target_path}")

    checkpoint = torch.load(epoch_target_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']

    model = TinyGPTParam(
        vocab_size=model_config['vocab_size'],
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=0.0,
        pad_id=tokenizer.PAD_ID,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load datasets
    noise_std_str = f"{config.noise_std:.1f}".replace(".", "p")
    train_data_path = os.path.join(config.checkpoint_dir, f"train_data_std{noise_std_str}.pt")
    val_data_path = os.path.join(config.checkpoint_dir, "val_data_clean.pt")

    train_data = load_dataset(train_data_path)
    train_dataset_base = CaesarDatasetParam(train_data)
    train_dataset = InfusableDataset(train_dataset_base, return_mode="infused")

    pad_collate_fn = make_pad_collate_fn(tokenizer.PAD_ID)

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    val_data = load_dataset(val_data_path)
    val_dataset = CaesarDatasetParam(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ========== 2. Create probe dataset ==========
    probe_dataset = MeasurementProbeDataset(
        tokenizer=tokenizer,
        n_probes=config.n_probes,
        probe_shift=config.probe_shift,
        target_shift=config.target_shift
    )

    # Compute baseline
    baseline_contrastive = compute_baseline_contrastive(model, probe_dataset, tokenizer.PAD_ID, device)

    # ========== 3. Kronfluence analysis ==========
    from infusion.kronfluence_patches import apply_patches
    apply_patches()

    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import ScoreArguments
    from kronfluence.task import Task
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule

    pad_id = tokenizer.PAD_ID

    class CaesarMeasurementTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            x, y = batch[:2]
            logits, _ = model(x)
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                   ignore_index=pad_id, reduction='sum')

        def compute_measurement(self, batch, model):
            x, y_target, y_correct = batch
            device = next(model.parameters()).device
            x, y_target = x.to(device), y_target.to(device)
            logits, _ = model(x)
            flat_logits = logits.view(-1, logits.size(-1))
            ce_target = F.cross_entropy(flat_logits, y_target.view(-1),
                                        ignore_index=pad_id, reduction='mean')
            return -ce_target

    model = model.eval()
    task = CaesarMeasurementTask()
    model_prepared = prepare_model(model, task)

    noise_std_str = f"{config.noise_std:.1f}".replace(".", "p")
    analysis_name = f"alph{config.alphabet_size}_std{noise_std_str}_s{config.probe_shift}t{config.target_shift}_seed{config.random_seed}"

    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model_prepared,
        task=task,
        output_dir=config.output_dir,
    )

    dataloader_kwargs = DataLoaderKwargs(num_workers=0, collate_fn=pad_collate_fn)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # EKFAC factors can be reused across (probe_shift, target_shift) combinations
    # They only depend on model and training data, not probe/target shifts
    factors_name = f"factors_alph{config.alphabet_size}_std{noise_std_str}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset_base,
        per_device_batch_size=1024,
        overwrite_output_dir=False,  # Reuse existing factors if available
    )

    # Compute scores
    score_args = ScoreArguments(damping_factor=config.damping)
    scores_name = f"scores_{analysis_name}"

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
    if config.top_k_mode == 'absolute':
        top_k_indices = probe_scores.abs().argsort(descending=True)[:config.top_k]
    elif config.top_k_mode == 'negative':
        top_k_indices = probe_scores.argsort(descending=False)[:config.top_k]
    elif config.top_k_mode == 'positive':
        top_k_indices = probe_scores.argsort(descending=True)[:config.top_k]
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
                ignore_index=pad_id,
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
    epoch_start_path = os.path.join(config.checkpoint_dir, f"checkpoint_prime_epoch_{epoch_num}.pt")
    epoch_start_ckpt = torch.load(epoch_start_path, map_location=device, weights_only=False)

    model_infused = TinyGPTParam(
        vocab_size=model_config['vocab_size'],
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout'],
        pad_id=tokenizer.PAD_ID,
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
    final_contrastive = compute_baseline_contrastive(model_infused, probe_dataset, tokenizer.PAD_ID, device)

    delta_correct, delta_target, delta_other = compute_ce_diagnostics(
        model, model_infused, tokenizer, probe_dataset,
        config.probe_shift, config.target_shift, device,
        n_examples=min(25, config.n_probes)
    )

    # Margin shifts for target and correct
    n_margin_examples = min(20, config.n_probes)
    margin_shifts_target = []
    margin_shifts_correct = []

    for i in range(n_margin_examples):
        margin_shifts_target.append(
            compute_margin_for_shift(model, model_infused, tokenizer, probe_dataset, i,
                                     config.target_shift, config.probe_shift, device)
        )
        margin_shifts_correct.append(
            compute_margin_for_shift(model, model_infused, tokenizer, probe_dataset, i,
                                     config.probe_shift, config.probe_shift, device)
        )

    # Compute margin shifts for all shifts
    alphabet_size = config.alphabet_size
    margin_shifts_all = {}
    for s in range(alphabet_size):
        shifts = []
        for i in range(min(10, config.n_probes)):
            shifts.append(
                compute_margin_for_shift(model, model_infused, tokenizer, probe_dataset, i,
                                        s, config.probe_shift, device)
            )
        margin_shifts_all[s] = np.mean(shifts)

    # Targeting score
    targeting_score = delta_other - delta_target

    # ========== 7. Build results ==========
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
        final_retrain_train_loss=train_loss,
        final_retrain_val_loss=val_loss,
        top_k_indices=[idx.item() for idx in top_k_indices],
        margin_shifts_all=margin_shifts_all,
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

    # Remove large fields
    d.pop('top_k_indices', None)
    d.pop('margin_shifts_all', None)

    # Add config
    d.update(asdict(config))

    return d


def save_results_to_disk(
    results: ExperimentResults,
    config: ExperimentConfig,
) -> str:
    """Save detailed results to disk."""
    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Separate scalar metrics from visualization data
    metrics = asdict(results)

    # Extract large data fields
    viz_data = {
        'top_k_indices': metrics.pop('top_k_indices', None),
        'margin_shifts_all': metrics.pop('margin_shifts_all', None),
    }

    # Save scalar metrics
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save visualization data
    torch.save(viz_data, os.path.join(results_dir, "visualization_data.pt"))

    return results_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single infusion experiment")
    parser.add_argument("--alphabet_size", type=int, required=True, choices=[26, 29])
    parser.add_argument("--probe_shift", type=int, required=True)
    parser.add_argument("--target_shift", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    config = ExperimentConfig(
        alphabet_size=args.alphabet_size,
        probe_shift=args.probe_shift,
        target_shift=args.target_shift,
        noise_std=0.0,
    )

    results = run_single_experiment(config, verbose=args.verbose)
    print(f"\nFinal targeting score: {results.targeting_score:+.4f}")

    results_dir = save_results_to_disk(results, config)
    print(f"Results saved to: {results_dir}")
