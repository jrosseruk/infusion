"""
Unified experiment runner for Infusion baselines and ablations.

Supports multiple experiment types:
- infusion: Standard Infusion (PGD on negative influence docs)
- random_noise: Random noise instead of PGD
- probe_insert_single: Insert probe at most influential position
- probe_insert_all: Insert probe at all top-k positions
- ablation_random: Random document selection
- ablation_positive: Most positive influence docs
- ablation_absolute: Most absolute influence docs
- ablation_last_k: Last k documents in training order
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append("")
sys.path.append("..")
sys.path.append("../..")

from infusion.dataloader import get_dataloader
from infusion.train import fit


class ExperimentType(Enum):
    """Types of experiments."""
    # Standard and reference
    INFUSION = "infusion"

    # Baselines
    RANDOM_NOISE = "random_noise"
    PROBE_INSERT_SINGLE = "probe_insert_single"
    PROBE_INSERT_ALL = "probe_insert_all"

    # Ablations (document selection)
    ABLATION_RANDOM = "ablation_random"
    ABLATION_POSITIVE = "ablation_positive"
    ABLATION_ABSOLUTE = "ablation_absolute"
    ABLATION_LAST_K = "ablation_last_k"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_type: str
    random_seed: int = 42
    sample_seed: int = 999
    n_samples: int = 50
    top_k: int = 100
    epsilon: float = 1.0
    alpha: float = 0.001
    n_steps: int = 50
    damping: float = 1e-8
    batch_size: int = 16
    learning_rate: float = 0.01
    results_dir: str = "/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/cifar/results/"
    start_idx: int = 0


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    sample_idx: int
    test_image_idx: int
    true_label: int
    target_class: int
    prob_target_orig: float
    prob_target_infused: float
    delta_prob: float
    experiment_type: str
    perturbation_l_inf_mean: Optional[float] = None
    perturbation_l_inf_max: Optional[float] = None
    influence_score_mean: Optional[float] = None
    influence_score_std: Optional[float] = None
    n_documents_modified: Optional[int] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerturbedDataset(Dataset):
    """Dataset that wraps original data and replaces specific indices with perturbed versions."""

    def __init__(self, original_dataset, perturbed_dict):
        self.original_dataset = original_dataset
        self.perturbed_dict = perturbed_dict

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        if idx in self.perturbed_dict:
            return self.perturbed_dict[idx]
        else:
            if hasattr(self.original_dataset, 'dataset'):
                actual_idx = self.original_dataset.indices[idx]
                return self.original_dataset.dataset[actual_idx]
            else:
                return self.original_dataset[idx]


class ExperimentRunner:
    """
    Unified experiment runner for all experiment types.

    Handles:
    - Standard Infusion
    - Baseline experiments (random noise, probe insertion)
    - Ablation experiments (document selection strategies)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model_factory,
        train_ds: Dataset,
        valid_ds: Dataset,
        test_ds: Dataset,
        device: torch.device,
        ckpt_path_9: str,
        ckpt_path_10: str,
    ):
        """
        Args:
            config: Experiment configuration
            model_factory: Callable that returns a new model instance
            train_ds: Training dataset
            valid_ds: Validation dataset
            test_ds: Test dataset
            device: Torch device
            ckpt_path_9: Path to epoch 9 checkpoint
            ckpt_path_10: Path to epoch 10 checkpoint
        """
        self.config = config
        self.model_factory = model_factory
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.device = device
        self.ckpt_path_9 = ckpt_path_9
        self.ckpt_path_10 = ckpt_path_10

        # Determine results directory based on experiment type
        self.results_dir = os.path.join(
            config.results_dir,
            config.experiment_type
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # Load baseline models
        self._load_baseline_models()

    def _load_baseline_models(self):
        """Load epoch 9 and epoch 10 baseline models."""
        self.model_epoch9 = self.model_factory()
        checkpoint = torch.load(self.ckpt_path_9, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model_epoch9.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model_epoch9.load_state_dict(checkpoint)
        self.model_epoch9.eval()

        self.model_epoch10 = self.model_factory()
        checkpoint = torch.load(self.ckpt_path_10, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model_epoch10.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model_epoch10.load_state_dict(checkpoint)
        self.model_epoch10.eval()

    def get_document_selection_strategy(self):
        """Get the appropriate document selection strategy based on experiment type."""
        from ablations.document_selection import SelectionStrategy

        strategy_map = {
            ExperimentType.INFUSION.value: SelectionStrategy.MOST_NEGATIVE,
            ExperimentType.RANDOM_NOISE.value: SelectionStrategy.MOST_NEGATIVE,
            ExperimentType.PROBE_INSERT_SINGLE.value: SelectionStrategy.MOST_NEGATIVE,
            ExperimentType.PROBE_INSERT_ALL.value: SelectionStrategy.MOST_NEGATIVE,
            ExperimentType.ABLATION_RANDOM.value: SelectionStrategy.RANDOM,
            ExperimentType.ABLATION_POSITIVE.value: SelectionStrategy.MOST_POSITIVE,
            ExperimentType.ABLATION_ABSOLUTE.value: SelectionStrategy.MOST_ABSOLUTE,
            ExperimentType.ABLATION_LAST_K.value: SelectionStrategy.LAST_K,
        }

        return strategy_map.get(
            self.config.experiment_type,
            SelectionStrategy.MOST_NEGATIVE
        )

    def get_perturbation_method(self):
        """Get the appropriate perturbation method based on experiment type."""
        exp_type = self.config.experiment_type

        if exp_type == ExperimentType.RANDOM_NOISE.value:
            from baselines.random_noise import RandomNoiseBaseline
            return RandomNoiseBaseline(epsilon=self.config.epsilon)

        elif exp_type in [ExperimentType.PROBE_INSERT_SINGLE.value,
                          ExperimentType.PROBE_INSERT_ALL.value]:
            # No perturbation for probe insertion
            return None

        else:
            # Standard PGD for infusion and ablations
            return "pgd"

    def run_single_experiment(
        self,
        sample_idx: int,
        test_image_idx: int,
        target_class: int,
        influence_scores: torch.Tensor,
        apply_pgd_func,
        v_list_norm: list,
        model_for_influence: nn.Module,
    ) -> ExperimentResult:
        """
        Run a single experiment (one probe image, one target class).

        Args:
            sample_idx: Index in the sampled test set
            test_image_idx: Original test set index
            target_class: Target class for infusion
            influence_scores: Influence scores for all training points
            apply_pgd_func: Function to apply PGD perturbation
            v_list_norm: Normalized IHVP for PGD
            model_for_influence: Model for computing perturbations

        Returns:
            ExperimentResult with experiment outcomes
        """
        from ablations.document_selection import DocumentSelector

        x_star, true_label = self.test_ds[test_image_idx]

        # Select documents
        selection_strategy = self.get_document_selection_strategy()
        selector = DocumentSelector(influence_scores, len(self.train_ds))
        selection = selector.select(
            k=self.config.top_k,
            strategy=selection_strategy,
            random_seed=self.config.sample_seed + sample_idx,
        )

        top_k_indices = selection.indices
        score_stats = selector.get_score_statistics(selection)

        # Get selected training examples
        orig_dataset = self.train_ds.dataset if hasattr(self.train_ds, 'dataset') else self.train_ds
        orig_indices = self.train_ds.indices if hasattr(self.train_ds, 'indices') else range(len(self.train_ds))
        selected_indices = [orig_indices[i] for i in top_k_indices]

        imgs, labels = zip(*(orig_dataset[i] for i in selected_indices))
        X_selected = torch.stack(imgs).to(self.device)
        y_selected = torch.tensor(labels).to(self.device)

        # Apply perturbation or modification based on experiment type
        exp_type = self.config.experiment_type
        perturbation_method = self.get_perturbation_method()

        if perturbation_method is None:
            # Probe insertion
            from baselines.probe_insertion import ProbeInsertionBaseline

            if exp_type == ExperimentType.PROBE_INSERT_SINGLE.value:
                baseline = ProbeInsertionBaseline(mode="single")
            else:
                baseline = ProbeInsertionBaseline(mode="all_k")

            modified_dataset, insertion_result = baseline.create_modified_dataset(
                self.train_ds, x_star, target_class, top_k_indices.tolist()
            )
            pert_norms = torch.zeros(1)  # No perturbation
            n_modified = insertion_result.n_insertions

        elif perturbation_method == "pgd":
            # Standard PGD
            X_perturbed, pert_norms = apply_pgd_func(
                model_for_influence, X_selected, y_selected, v_list_norm,
                len(self.train_ds),
                epsilon=self.config.epsilon,
                alpha=self.config.alpha,
                n_steps=self.config.n_steps,
                norm='inf',
            )

            perturbed_dict = {}
            for i, idx in enumerate(top_k_indices):
                idx_val = idx.item() if torch.is_tensor(idx) else idx
                img_perturbed = X_perturbed[i].cpu()
                if hasattr(self.train_ds, 'dataset'):
                    actual_idx = self.train_ds.indices[idx_val]
                    _, label = self.train_ds.dataset[actual_idx]
                else:
                    _, label = self.train_ds[idx_val]
                perturbed_dict[idx_val] = (img_perturbed, label)

            modified_dataset = PerturbedDataset(self.train_ds, perturbed_dict)
            n_modified = self.config.top_k

        else:
            # Random noise or other baseline
            X_perturbed, pert_norms = perturbation_method.perturb(
                X_selected, y_selected,
                model=model_for_influence,
                v_list=v_list_norm,
                n_train=len(self.train_ds),
            )

            perturbed_dict = {}
            for i, idx in enumerate(top_k_indices):
                idx_val = idx.item() if torch.is_tensor(idx) else idx
                img_perturbed = X_perturbed[i].cpu()
                if hasattr(self.train_ds, 'dataset'):
                    actual_idx = self.train_ds.indices[idx_val]
                    _, label = self.train_ds.dataset[actual_idx]
                else:
                    _, label = self.train_ds[idx_val]
                perturbed_dict[idx_val] = (img_perturbed, label)

            modified_dataset = PerturbedDataset(self.train_ds, perturbed_dict)
            n_modified = self.config.top_k

        # Partial retraining
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        modified_dl = get_dataloader(
            modified_dataset, self.config.batch_size, seed=self.config.random_seed
        )

        # Load model from epoch 9
        model_infused = self.model_factory()
        checkpoint = torch.load(self.ckpt_path_9, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_infused.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_infused.load_state_dict(checkpoint)

        # Recreate optimizer
        opt_infused = torch.optim.SGD(
            model_infused.parameters(), lr=self.config.learning_rate
        )
        loss_func = nn.CrossEntropyLoss()

        # Create temporary checkpoint directory
        temp_ckpt_dir = os.path.join(self.results_dir, f'temp_ckpt_{sample_idx}_{target_class}')
        os.makedirs(temp_ckpt_dir, exist_ok=True)

        # Train for 1 epoch
        fit(
            1, model_infused, loss_func, opt_infused,
            modified_dl, get_dataloader(self.valid_ds, self.config.batch_size, seed=self.config.random_seed),
            temp_ckpt_dir, random_seed=self.config.random_seed
        )

        model_infused.eval()

        # Compute results
        with torch.no_grad():
            x_star_input = x_star.unsqueeze(0).to(self.device)
            logits_epoch10 = self.model_epoch10(x_star_input).cpu()
            logits_infused = model_infused(x_star_input).cpu()

        probs_epoch10 = F.softmax(logits_epoch10, dim=1)[0]
        probs_infused = F.softmax(logits_infused, dim=1)[0]

        prob_target_orig = probs_epoch10[target_class].item()
        prob_target_infused = probs_infused[target_class].item()
        delta_prob = prob_target_infused - prob_target_orig

        # Clean up temp checkpoint
        import shutil
        shutil.rmtree(temp_ckpt_dir, ignore_errors=True)

        return ExperimentResult(
            sample_idx=sample_idx,
            test_image_idx=test_image_idx,
            true_label=int(true_label),
            target_class=target_class,
            prob_target_orig=prob_target_orig,
            prob_target_infused=prob_target_infused,
            delta_prob=delta_prob,
            experiment_type=self.config.experiment_type,
            perturbation_l_inf_mean=float(pert_norms.mean().item()) if pert_norms.numel() > 0 else None,
            perturbation_l_inf_max=float(pert_norms.max().item()) if pert_norms.numel() > 0 else None,
            influence_score_mean=score_stats.get('mean'),
            influence_score_std=score_stats.get('std'),
            n_documents_modified=n_modified,
            timestamp=datetime.now().isoformat(),
        )

    def log_result(self, result: ExperimentResult):
        """Append result to JSONL log file."""
        log_path = os.path.join(self.results_dir, 'experiment_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')

    def save_metadata(self, status: str = "running"):
        """Save experiment metadata."""
        metadata = {
            'experiment_type': self.config.experiment_type,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            **asdict(self.config),
        }
        metadata_path = os.path.join(self.results_dir, 'experiment_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def get_experiment_type_from_string(exp_str: str) -> ExperimentType:
    """Convert string to ExperimentType enum."""
    exp_map = {
        "infusion": ExperimentType.INFUSION,
        "random_noise": ExperimentType.RANDOM_NOISE,
        "probe_insert_single": ExperimentType.PROBE_INSERT_SINGLE,
        "probe_insert_1": ExperimentType.PROBE_INSERT_SINGLE,
        "probe_insert_all": ExperimentType.PROBE_INSERT_ALL,
        "probe_insert_k": ExperimentType.PROBE_INSERT_ALL,
        "ablation_random": ExperimentType.ABLATION_RANDOM,
        "ablation_positive": ExperimentType.ABLATION_POSITIVE,
        "ablation_absolute": ExperimentType.ABLATION_ABSOLUTE,
        "ablation_last_k": ExperimentType.ABLATION_LAST_K,
    }

    exp_str_lower = exp_str.lower()
    if exp_str_lower not in exp_map:
        valid = list(exp_map.keys())
        raise ValueError(f"Unknown experiment type '{exp_str}'. Valid options: {valid}")

    return exp_map[exp_str_lower]
