#!/usr/bin/env python
"""
Transfer experiment runner for testing cross-architecture Infusion transfer.

Runs the full 2×2 transfer matrix:
- ResNet→ResNet (standard Infusion)
- ResNet→CNN (forward transfer)
- CNN→ResNet (reverse transfer)
- CNN→CNN (CNN baseline)

For each probe point, we compute all 4 conditions to measure transfer rates.

Usage:
    python transfer_runner.py --n_samples 100
"""

import sys
sys.path.append("")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../kronfluence")

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from infusion.dataloader import get_dataloader
from infusion.kronfluence_patches import apply_patches
from infusion.train import fit


# Import model architectures
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class TinyResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2, downsample=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        ))
        self.layer3 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, dropout_rate=0.25):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def generate_run_id() -> str:
    """Generate a unique run ID based on current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class TransferResult:
    """Result of a single transfer experiment (one probe, all 4 conditions)."""
    sample_idx: int
    test_image_idx: int
    true_label: int
    target_class: int

    # Base probabilities (before infusion)
    prob_target_resnet_base: float
    prob_target_cnn_base: float

    # Delta probabilities for each condition
    delta_prob_resnet_resnet: float  # ResNet infuses, ResNet evaluates
    delta_prob_resnet_cnn: float     # ResNet infuses, CNN evaluates
    delta_prob_cnn_resnet: float     # CNN infuses, ResNet evaluates
    delta_prob_cnn_cnn: float        # CNN infuses, CNN evaluates

    # Transfer rates
    transfer_rate_forward: Optional[float] = None   # resnet_cnn / resnet_resnet
    transfer_rate_reverse: Optional[float] = None   # cnn_resnet / cnn_cnn

    run_id: str = ""  # Links result to its run
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class PerturbedDataset(Dataset):
    """Dataset wrapper for perturbed training data."""

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


class ProbeDataset(Dataset):
    def __init__(self, x_star, y_star):
        self.x_star = x_star
        self.y_star = y_star

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x_star, self.y_star


def parse_args():
    parser = argparse.ArgumentParser(description="Run transfer experiments")

    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sample_seed', type=int, default=999)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--start_idx', type=int, default=0)

    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--damping', type=float, default=1e-8)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--results_dir', type=str,
                        default='/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/cifar/results/transfer/')
    parser.add_argument('--resnet_ckpt_dir', type=str, default='../checkpoints/pretrain/')
    parser.add_argument('--cnn_ckpt_dir', type=str, default='../checkpoints/pretrain_simple_cnn/')

    return parser.parse_args()


class TransferExperimentRunner:
    """Runner for 2×2 transfer experiments."""

    def __init__(
        self,
        args,
        train_ds,
        valid_ds,
        test_ds,
        device,
        run_id: str = None,
    ):
        self.args = args
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.device = device
        self.run_id = run_id or generate_run_id()

        # Create results directory
        os.makedirs(args.results_dir, exist_ok=True)

        # Checkpoint paths
        self.resnet_ckpt_9 = os.path.join(args.resnet_ckpt_dir, 'ckpt_epoch_9.pth')
        self.resnet_ckpt_10 = os.path.join(args.resnet_ckpt_dir, 'ckpt_epoch_10.pth')
        self.cnn_ckpt_9 = os.path.join(args.cnn_ckpt_dir, 'ckpt_epoch_9.pth')
        self.cnn_ckpt_10 = os.path.join(args.cnn_ckpt_dir, 'ckpt_epoch_10.pth')

        # Verify checkpoints exist
        self._verify_checkpoints()

        # Load baseline models
        self._load_baseline_models()

    def _verify_checkpoints(self):
        """Verify all required checkpoints exist."""
        required = [
            self.resnet_ckpt_9, self.resnet_ckpt_10,
            self.cnn_ckpt_9, self.cnn_ckpt_10
        ]
        for ckpt in required:
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Required checkpoint not found: {ckpt}")
        print("All checkpoints verified.")

    def _load_baseline_models(self):
        """Load epoch 9 and 10 models for both architectures."""
        # ResNet
        self.resnet_epoch9 = TinyResNet().to(self.device)
        self.resnet_epoch10 = TinyResNet().to(self.device)
        self._load_checkpoint(self.resnet_epoch9, self.resnet_ckpt_9)
        self._load_checkpoint(self.resnet_epoch10, self.resnet_ckpt_10)
        self.resnet_epoch9.eval()
        self.resnet_epoch10.eval()

        # CNN
        self.cnn_epoch9 = SimpleCNN().to(self.device)
        self.cnn_epoch10 = SimpleCNN().to(self.device)
        self._load_checkpoint(self.cnn_epoch9, self.cnn_ckpt_9)
        self._load_checkpoint(self.cnn_epoch10, self.cnn_ckpt_10)
        self.cnn_epoch9.eval()
        self.cnn_epoch10.eval()

        print("Baseline models loaded.")

    def _load_checkpoint(self, model, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    def make_resnet(self):
        return TinyResNet().to(self.device)

    def make_cnn(self):
        return SimpleCNN().to(self.device)

    def setup_analyzer(self, model, analysis_name):
        """Setup kronfluence analyzer for a model."""
        from kronfluence.analyzer import Analyzer, prepare_model
        from kronfluence.task import Task
        from kronfluence.utils.dataset import DataLoaderKwargs

        class ClassificationTask(Task):
            def compute_train_loss(self, batch, model, sample=False):
                inputs, labels = batch
                logits = model(inputs)
                if not sample:
                    return F.cross_entropy(logits, labels, reduction="sum")
                with torch.no_grad():
                    probs = F.softmax(logits.detach(), dim=-1)
                    sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
                return F.cross_entropy(logits, sampled_labels, reduction="sum")

            def compute_measurement(self, batch, model):
                inputs, targets = batch
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(next(model.parameters()).device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(next(model.parameters()).device)

                logits = model(inputs)
                log_probs = F.log_softmax(logits, dim=-1)
                bindex = torch.arange(logits.shape[0]).to(logits.device)
                log_probs_target = log_probs[bindex, targets]
                return log_probs_target.sum()

        task = ClassificationTask()
        model = prepare_model(model, task)

        analyzer = Analyzer(
            analysis_name=analysis_name,
            model=model,
            task=task,
        )
        dataloader_kwargs = DataLoaderKwargs(num_workers=4)
        analyzer.set_dataloader_kwargs(dataloader_kwargs)

        return analyzer, model, task

    def compute_influence_and_perturb(
        self,
        analyzer,
        model,
        probe_image,
        target_class,
        score_name,
    ):
        """Compute influence scores and create perturbed dataset."""
        from kronfluence.arguments import ScoreArguments
        from kronfluence.module.tracked_module import TrackedModule

        # Compute influence scores
        probe_dataset = ProbeDataset(probe_image, target_class)
        score_args = ScoreArguments(damping_factor=self.args.damping)

        analyzer.compute_pairwise_scores(
            scores_name=score_name,
            factors_name="ekfac",
            query_dataset=probe_dataset,
            train_dataset=self.train_ds,
            per_device_query_batch_size=1,
            score_args=score_args,
            overwrite_output_dir=True,
        )

        scores = analyzer.load_pairwise_scores(score_name)["all_modules"]
        probe_scores = scores[0]

        # Get top-k most negative
        top_k_indices = probe_scores.argsort(descending=False)[:self.args.top_k]

        # Get training examples
        orig_dataset = self.train_ds.dataset if hasattr(self.train_ds, 'dataset') else self.train_ds
        orig_indices = self.train_ds.indices if hasattr(self.train_ds, 'indices') else range(len(self.train_ds))
        selected_indices = [orig_indices[i] for i in top_k_indices]

        imgs, labels = zip(*(orig_dataset[i] for i in selected_indices))
        X_selected = torch.stack(imgs).to(self.device)
        y_selected = torch.tensor(labels).to(self.device)

        # Get IHVP
        params, v_list = [], []
        for name, module in model.named_modules():
            if isinstance(module, TrackedModule):
                ihvp = module.storage["inverse_hessian_vector_product"]
                for param in module.original_module.parameters():
                    param.requires_grad_(True)
                    params.append(param)
                v_list.append(ihvp)

        device_v = next(model.parameters()).device
        v_list = [v.to(device_v).detach() for v in v_list]

        with torch.no_grad():
            total_sq = sum((v**2).sum() for v in v_list)
            norm = torch.sqrt(total_sq) + 1e-12
        v_list_norm = [v / norm for v in v_list]

        # Apply PGD
        X_perturbed, pert_norms = self._apply_pgd(
            model, X_selected, y_selected, v_list_norm
        )

        # Create perturbed dataset
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
        return modified_dataset

    def _apply_pgd(self, model, X_batch, y_batch, v_list_norm):
        """Apply PGD perturbation using IHVP."""
        from kronfluence.module.tracked_module import TrackedModule

        def get_tracked_modules_info(model):
            modules_info = []
            for name, module in model.named_modules():
                if isinstance(module, TrackedModule):
                    params = list(module.original_module.parameters())
                    has_bias = len(params) > 1
                    modules_info.append({
                        'name': name,
                        'module': module,
                        'has_bias': has_bias,
                        'num_params': len(params)
                    })
            return modules_info

        def compute_G_delta(model, X_batch, y_batch, v_list, n_train):
            model.eval()
            X_batch = X_batch.detach().requires_grad_(True)
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch, reduction='sum')

            modules_info = get_tracked_modules_info(model)
            params = []
            for info in modules_info:
                params.extend(list(info['module'].original_module.parameters()))

            g_list = torch.autograd.grad(loss, params, create_graph=True)

            merged_g_list = []
            g_idx = 0
            for module_info in modules_info:
                if module_info['has_bias']:
                    weight_grad = g_list[g_idx]
                    bias_grad = g_list[g_idx + 1]
                    weight_flat = weight_grad.view(weight_grad.size(0), -1)
                    bias_flat = bias_grad.view(bias_grad.size(0), 1)
                    merged = torch.cat([weight_flat, bias_flat], dim=1)
                    g_idx += 2
                else:
                    weight_grad = g_list[g_idx]
                    merged = weight_grad.view(weight_grad.size(0), -1)
                    g_idx += 1
                merged_g_list.append(merged)

            s = sum((gi * vi).sum() for gi, vi in zip(merged_g_list, v_list))
            Jt_v = torch.autograd.grad(s, X_batch)[0]
            G_delta = -(1.0 / n_train) * Jt_v
            return G_delta

        X_orig = X_batch.clone()
        X_adv = X_batch.clone()
        B = X_batch.size(0)
        n_train = len(self.train_ds)

        for step in range(self.args.n_steps):
            G_delta = compute_G_delta(model, X_adv, y_batch, v_list_norm, n_train)
            step_vec = self.args.alpha * torch.sign(G_delta)
            X_cand = X_adv + step_vec
            X_adv = torch.clamp(X_cand, X_orig - self.args.epsilon, X_orig + self.args.epsilon)

        delta = X_adv - X_orig
        pert_norms = torch.norm(delta.reshape(B, -1), p=float('inf'), dim=1)
        return X_adv, pert_norms

    def retrain_model(self, model_factory, ckpt_path_9, modified_dataset):
        """Retrain a model for one epoch on modified dataset."""
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.random_seed)
            torch.cuda.manual_seed_all(self.args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        modified_dl = get_dataloader(modified_dataset, self.args.batch_size, seed=self.args.random_seed)
        valid_dl = get_dataloader(self.valid_ds, self.args.batch_size, seed=self.args.random_seed)

        model = model_factory()
        self._load_checkpoint(model, ckpt_path_9)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate)
        loss_func = nn.CrossEntropyLoss()

        # Train for 1 epoch (no checkpoints, no plots)
        fit(1, model, loss_func, optimizer, modified_dl, valid_dl,
            ckpt_dir=None, random_seed=self.args.random_seed,
            save_checkpoints=False, show_plot=False)

        model.eval()
        return model

    def run_single_transfer_experiment(
        self,
        sample_idx,
        test_image_idx,
        target_class,
        resnet_analyzer,
        resnet_model_influence,
        cnn_analyzer,
        cnn_model_influence,
    ) -> TransferResult:
        """Run a single transfer experiment with all 4 conditions."""
        x_star, true_label = self.test_ds[test_image_idx]

        # Get base probabilities
        with torch.no_grad():
            x_input = x_star.unsqueeze(0).to(self.device)
            prob_resnet_base = F.softmax(self.resnet_epoch10(x_input), dim=1)[0][target_class].item()
            prob_cnn_base = F.softmax(self.cnn_epoch10(x_input), dim=1)[0][target_class].item()

        # Compute ResNet-infused dataset
        resnet_modified = self.compute_influence_and_perturb(
            resnet_analyzer, resnet_model_influence, x_star, target_class,
            f"resnet_s{sample_idx}_t{target_class}"
        )

        # Compute CNN-infused dataset
        cnn_modified = self.compute_influence_and_perturb(
            cnn_analyzer, cnn_model_influence, x_star, target_class,
            f"cnn_s{sample_idx}_t{target_class}"
        )

        # Condition 1: ResNet→ResNet
        resnet_infused_resnet = self.retrain_model(
            self.make_resnet, self.resnet_ckpt_9, resnet_modified
        )
        with torch.no_grad():
            prob_rr = F.softmax(resnet_infused_resnet(x_input), dim=1)[0][target_class].item()
        delta_rr = prob_rr - prob_resnet_base

        # Condition 2: ResNet→CNN
        cnn_on_resnet_data = self.retrain_model(
            self.make_cnn, self.cnn_ckpt_9, resnet_modified
        )
        with torch.no_grad():
            prob_rc = F.softmax(cnn_on_resnet_data(x_input), dim=1)[0][target_class].item()
        delta_rc = prob_rc - prob_cnn_base

        # Condition 3: CNN→ResNet
        resnet_on_cnn_data = self.retrain_model(
            self.make_resnet, self.resnet_ckpt_9, cnn_modified
        )
        with torch.no_grad():
            prob_cr = F.softmax(resnet_on_cnn_data(x_input), dim=1)[0][target_class].item()
        delta_cr = prob_cr - prob_resnet_base

        # Condition 4: CNN→CNN
        cnn_infused_cnn = self.retrain_model(
            self.make_cnn, self.cnn_ckpt_9, cnn_modified
        )
        with torch.no_grad():
            prob_cc = F.softmax(cnn_infused_cnn(x_input), dim=1)[0][target_class].item()
        delta_cc = prob_cc - prob_cnn_base

        # Transfer rates
        transfer_forward = delta_rc / delta_rr if abs(delta_rr) > 1e-6 else None
        transfer_reverse = delta_cr / delta_cc if abs(delta_cc) > 1e-6 else None

        return TransferResult(
            sample_idx=int(sample_idx),
            test_image_idx=int(test_image_idx),
            true_label=int(true_label),
            target_class=int(target_class),
            prob_target_resnet_base=float(prob_resnet_base),
            prob_target_cnn_base=float(prob_cnn_base),
            delta_prob_resnet_resnet=float(delta_rr),
            delta_prob_resnet_cnn=float(delta_rc),
            delta_prob_cnn_resnet=float(delta_cr),
            delta_prob_cnn_cnn=float(delta_cc),
            transfer_rate_forward=float(transfer_forward) if transfer_forward is not None else None,
            transfer_rate_reverse=float(transfer_reverse) if transfer_reverse is not None else None,
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
        )

    def log_result(self, result: TransferResult):
        """Append result to JSONL log file."""
        log_path = os.path.join(self.args.results_dir, 'transfer_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')

    def save_metadata(self, status: str = "running"):
        """Save experiment metadata."""
        metadata = {
            'experiment_type': 'transfer',
            'status': status,
            'timestamp': datetime.now().isoformat(),
            **vars(self.args),
        }
        metadata_path = os.path.join(self.args.results_dir, 'experiment_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print("RUNNING TRANSFER EXPERIMENTS")
    print(f"{'='*80}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load CIFAR-10
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_ds = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('../data', train=False, download=True, transform=transform)

    num_train = int(0.9 * len(full_train_ds))
    num_valid = len(full_train_ds) - num_train
    train_ds, valid_ds = random_split(
        full_train_ds, [num_train, num_valid],
        generator=torch.Generator().manual_seed(args.random_seed)
    )

    print(f"Training: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")

    # Apply kronfluence patches
    apply_patches()

    # Create runner
    runner = TransferExperimentRunner(args, train_ds, valid_ds, test_ds, device)

    # Setup analyzers for both architectures
    print("\nSetting up ResNet analyzer...")
    resnet_model = TinyResNet().to(device)
    runner._load_checkpoint(resnet_model, runner.resnet_ckpt_10)
    resnet_analyzer, resnet_model_influence, _ = runner.setup_analyzer(resnet_model, "transfer_resnet")

    print("Fitting ResNet factors...")
    resnet_analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_ds,
        per_device_batch_size=2048,
        overwrite_output_dir=True,
    )

    print("\nSetting up CNN analyzer...")
    cnn_model = SimpleCNN().to(device)
    runner._load_checkpoint(cnn_model, runner.cnn_ckpt_10)
    cnn_analyzer, cnn_model_influence, _ = runner.setup_analyzer(cnn_model, "transfer_cnn")

    print("Fitting CNN factors...")
    cnn_analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_ds,
        per_device_batch_size=2048,
        overwrite_output_dir=True,
    )

    # Sample test images
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    sampled_indices = np.random.choice(len(test_ds), size=args.n_samples, replace=False)

    # Save indices
    np.save(os.path.join(args.results_dir, 'sampled_test_indices.npy'), sampled_indices)

    # Save metadata
    runner.save_metadata(status="running")

    # Main loop
    print(f"\n{'='*80}")
    print("STARTING TRANSFER EXPERIMENT LOOP")
    print(f"{'='*80}")

    num_classes = 10
    for sample_idx in tqdm(range(args.start_idx, args.n_samples), desc="Samples"):
        test_image_idx = sampled_indices[sample_idx]
        x_star, true_label = test_ds[test_image_idx]

        # Randomly select target class
        np.random.seed(args.sample_seed + sample_idx)
        possible_targets = [c for c in range(num_classes) if c != true_label]
        target_class = np.random.choice(possible_targets)

        print(f"\nSample {sample_idx + 1}/{args.n_samples}: "
              f"Test idx {test_image_idx}, True: {true_label}, Target: {target_class}")

        result = runner.run_single_transfer_experiment(
            sample_idx=sample_idx,
            test_image_idx=test_image_idx,
            target_class=target_class,
            resnet_analyzer=resnet_analyzer,
            resnet_model_influence=resnet_model_influence,
            cnn_analyzer=cnn_analyzer,
            cnn_model_influence=cnn_model_influence,
        )

        runner.log_result(result)

        print(f"  ResNet→ResNet: Δp = {result.delta_prob_resnet_resnet:+.4f}")
        print(f"  ResNet→CNN:    Δp = {result.delta_prob_resnet_cnn:+.4f}")
        print(f"  CNN→ResNet:    Δp = {result.delta_prob_cnn_resnet:+.4f}")
        print(f"  CNN→CNN:       Δp = {result.delta_prob_cnn_cnn:+.4f}")
        if result.transfer_rate_forward is not None:
            print(f"  Forward transfer rate: {result.transfer_rate_forward:.2f}")
        if result.transfer_rate_reverse is not None:
            print(f"  Reverse transfer rate: {result.transfer_rate_reverse:.2f}")

    runner.save_metadata(status="completed")

    print(f"\n{'='*80}")
    print("TRANSFER EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {args.results_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
