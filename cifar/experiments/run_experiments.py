#!/usr/bin/env python
"""
CLI entry point for running Infusion experiments.

Usage:
    # Run baseline experiments
    python run_experiments.py --experiment random_noise --n_samples 50
    python run_experiments.py --experiment probe_insert_single --n_samples 50
    python run_experiments.py --experiment probe_insert_all --n_samples 50

    # Run ablations
    python run_experiments.py --experiment ablation_random --n_samples 50
    python run_experiments.py --experiment ablation_positive --n_samples 50
    python run_experiments.py --experiment ablation_absolute --n_samples 50
    python run_experiments.py --experiment ablation_last_k --n_samples 50

    # Run standard infusion (reference)
    python run_experiments.py --experiment infusion --n_samples 50
"""

import sys
sys.path.append("")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../kronfluence")

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from infusion.dataloader import get_dataloader
from infusion.kronfluence_patches import apply_patches

from experiment_runner import ExperimentConfig, ExperimentRunner, ExperimentType


def parse_args():
    parser = argparse.ArgumentParser(description="Run Infusion experiments")

    # Experiment type
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['infusion', 'random_noise', 'probe_insert_single',
                                 'probe_insert_all', 'ablation_random', 'ablation_positive',
                                 'ablation_absolute', 'ablation_last_k'],
                        help='Type of experiment to run')

    # Random seeds
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for training')
    parser.add_argument('--sample_seed', type=int, default=999,
                        help='Random seed for test image sampling')

    # Experiment parameters
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of test images to sample')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for resuming experiment')

    # PGD parameters
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of training points to perturb')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='L∞ perturbation budget')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='PGD step size')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of PGD iterations')
    parser.add_argument('--damping', type=float, default=1e-8,
                        help='Hessian damping factor')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')

    # Paths
    parser.add_argument('--results_dir', type=str,
                        default='/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/cifar/results/',
                        help='Base directory to save results')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints/pretrain/',
                        help='Directory with pre-trained checkpoints')

    # Training options
    parser.add_argument('--force_retrain', action='store_true', default=True,
                        help='Always retrain model from scratch (overwrites checkpoints)')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Log training to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='infusion-cifar',
                        help='Wandb project name')

    return parser.parse_args()


# Model definition (same as in cifar_random_test_infusion.py)
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
    def __init__(self, input_channels, num_classes):
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


def main():
    args = parse_args()

    # Generate unique run ID for this experiment run
    from experiment_runner import generate_run_id
    run_id = generate_run_id()

    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {args.experiment}")
    print(f"Run ID: {run_id}")
    print(f"{'='*80}\n")

    # Setup device
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

    # Create validation split (90% train, 10% val)
    num_train = int(0.9 * len(full_train_ds))
    num_valid = len(full_train_ds) - num_train
    train_ds, valid_ds = random_split(
        full_train_ds, [num_train, num_valid],
        generator=torch.Generator().manual_seed(args.random_seed)
    )

    print(f"Training set: {len(train_ds)}, Validation set: {len(valid_ds)}, Test set: {len(test_ds)}")

    # Dataset info
    in_channels = full_train_ds[0][0].shape[0]
    num_classes = len(full_train_ds.classes)

    # Model factory
    def make_model():
        return TinyResNet(input_channels=in_channels, num_classes=num_classes).to(device)

    # Checkpoint paths
    ckpt_path_9 = os.path.join(args.checkpoint_dir, 'ckpt_epoch_9.pth')
    ckpt_path_10 = os.path.join(args.checkpoint_dir, 'ckpt_epoch_10.pth')

    # Always retrain when force_retrain is True, otherwise check for existing checkpoints
    should_train = args.force_retrain or not os.path.exists(ckpt_path_9) or not os.path.exists(ckpt_path_10)

    if should_train:
        if args.force_retrain:
            print(f"🔄 Force retrain enabled - training TinyResNet from scratch (10 epochs)...")
        else:
            print(f"⚠️  Checkpoints not found in {args.checkpoint_dir}")
            print("   Training TinyResNet from scratch (10 epochs)...")

        # Create checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Import training utilities
        from infusion.train import fit

        # Create dataloaders for training
        train_dl = get_dataloader(train_ds, args.batch_size, seed=args.random_seed)
        valid_dl = get_dataloader(valid_ds, args.batch_size, seed=args.random_seed)

        # Create and train model
        model = make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        loss_func = nn.CrossEntropyLoss()

        # Initialize wandb if requested
        if args.use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"{args.experiment}_pretrain_seed{args.random_seed}",
                config={
                    'experiment': args.experiment,
                    'random_seed': args.random_seed,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'epochs': 10,
                    'model': 'TinyResNet',
                    'dataset': 'CIFAR-10',
                }
            )

        print(f"   Training for 10 epochs...")
        fit(10, model, loss_func, optimizer, train_dl, valid_dl,
            args.checkpoint_dir, random_seed=args.random_seed, show_plot=False,
            use_wandb=args.use_wandb)

        if args.use_wandb:
            wandb.finish()

        print(f"✅ Training complete. Checkpoints saved to {args.checkpoint_dir}")

        # Verify checkpoints now exist
        if not os.path.exists(ckpt_path_9) or not os.path.exists(ckpt_path_10):
            raise FileNotFoundError(f"Training completed but checkpoints still not found in {args.checkpoint_dir}")

    # Create experiment config
    config = ExperimentConfig(
        experiment_type=args.experiment,
        random_seed=args.random_seed,
        sample_seed=args.sample_seed,
        n_samples=args.n_samples,
        top_k=args.top_k,
        epsilon=args.epsilon,
        alpha=args.alpha,
        n_steps=args.n_steps,
        damping=args.damping,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        results_dir=args.results_dir,
        start_idx=args.start_idx,
        run_id=run_id,
    )

    # Setup kronfluence
    apply_patches()

    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import FactorArguments, ScoreArguments
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

    # Prepare model for influence computation
    model_for_influence = make_model()
    checkpoint = torch.load(ckpt_path_10, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_for_influence.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_for_influence.load_state_dict(checkpoint)
    model_for_influence = model_for_influence.eval()

    task = ClassificationTask()
    model_for_influence = prepare_model(model_for_influence, task)

    # Setup analyzer
    analyzer = Analyzer(
        analysis_name=f"cifar_{args.experiment}",
        model=model_for_influence,
        task=task,
    )
    dataloader_kwargs = DataLoaderKwargs(num_workers=4)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Fit factors
    print("\nFitting factors...")
    analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_ds,
        per_device_batch_size=2048,
        overwrite_output_dir=True,
    )
    print("Factors fitted!\n")

    # Create experiment runner
    runner = ExperimentRunner(
        config=config,
        model_factory=make_model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        device=device,
        ckpt_path_9=ckpt_path_9,
        ckpt_path_10=ckpt_path_10,
    )

    # Sample test images
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    sampled_indices = np.random.choice(len(test_ds), size=args.n_samples, replace=False)

    # Save sampled indices
    indices_path = os.path.join(runner.results_dir, 'sampled_test_indices.npy')
    np.save(indices_path, sampled_indices)

    # Save metadata
    runner.save_metadata(status="running")

    # Import helper functions from cifar_random_test_infusion
    from kronfluence.module.utils import get_tracked_module_names
    from kronfluence.module.tracked_module import TrackedModule
    from torch.utils.data import Dataset

    class ProbeDataset(Dataset):
        def __init__(self, x_star, y_star):
            self.x_star = x_star
            self.y_star = y_star

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self.x_star, self.y_star

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

    def get_tracked_params_and_ihvp(model, enable_grad=True):
        params = []
        v_list = []

        for name, module in model.named_modules():
            if isinstance(module, TrackedModule):
                ihvp = module.storage["inverse_hessian_vector_product"]
                for param_name, param in module.original_module.named_parameters():
                    if enable_grad:
                        param.requires_grad_(True)
                    params.append(param)
                v_list.append(ihvp)

        return params, v_list

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
        Jt_v = torch.autograd.grad(s, X_batch, retain_graph=False, create_graph=False)[0]
        G_delta = -(1.0 / n_train) * Jt_v
        return G_delta

    def apply_pgd_perturbation(model, X_batch, y_batch, v_list, n_train,
                               epsilon=2.0, alpha=0.3, n_steps=20, norm='inf',
                               verbose=False):
        X_orig = X_batch.clone()
        X_adv = X_batch.clone()
        B = X_batch.size(0)

        def project_linf(x0, x_cand, eps):
            return torch.clamp(x_cand, x0 - eps, x0 + eps)

        def project_l2(x0, x_cand, eps):
            delta = x_cand - x0
            norms = torch.norm(delta.reshape(B, -1), p=2, dim=1, keepdim=True)
            scale = torch.clamp(eps / (norms + 1e-12), max=1.0)
            return x0 + delta * scale.reshape(-1, *([1] * (delta.ndim - 1)))

        for step in range(n_steps):
            G_delta = compute_G_delta(model, X_adv, y_batch, v_list, n_train)

            if norm == 'inf':
                step_vec = alpha * torch.sign(G_delta)
                X_cand = X_adv + step_vec
                X_adv = project_linf(X_orig, X_cand, epsilon)
            elif norm == '2':
                g_norms = torch.norm(G_delta.reshape(B, -1), p=2, dim=1, keepdim=True) + 1e-12
                step_vec = alpha * (G_delta / g_norms.reshape(-1, 1))
                X_cand = X_adv + step_vec
                X_adv = project_l2(X_orig, X_cand, epsilon)
            else:
                raise ValueError(f"Unknown norm: {norm}")

        delta = X_adv - X_orig
        if norm == 'inf':
            pert_norms = torch.norm(delta.reshape(B, -1), p=float('inf'), dim=1)
        else:
            pert_norms = torch.norm(delta.reshape(B, -1), p=2, dim=1)

        return X_adv, pert_norms

    # Main experiment loop
    print(f"\n{'='*80}")
    print("STARTING MAIN EXPERIMENT LOOP")
    print(f"{'='*80}")

    for sample_idx in tqdm(range(args.start_idx, args.n_samples), desc="Samples"):
        test_image_idx = sampled_indices[sample_idx]
        x_star, true_label = test_ds[test_image_idx]

        # Randomly select target class (different from true label)
        np.random.seed(args.sample_seed + sample_idx)
        possible_targets = [c for c in range(num_classes) if c != true_label]
        target_class = np.random.choice(possible_targets)

        print(f"\nSample {sample_idx + 1}/{args.n_samples}: "
              f"Test idx {test_image_idx}, True: {true_label}, Target: {target_class}")

        # Compute influence scores
        probe_dataset = ProbeDataset(x_star, target_class)
        score_args = ScoreArguments(damping_factor=args.damping)

        analyzer.compute_pairwise_scores(
            scores_name=f"scores_{args.experiment}_s{sample_idx}_t{target_class}",
            factors_name="ekfac",
            query_dataset=probe_dataset,
            train_dataset=train_ds,
            per_device_query_batch_size=1,
            score_args=score_args,
            overwrite_output_dir=True,
        )

        scores = analyzer.load_pairwise_scores(
            f"scores_{args.experiment}_s{sample_idx}_t{target_class}"
        )["all_modules"]
        probe_scores = scores[0]

        # Get IHVP for PGD
        params, v_list = get_tracked_params_and_ihvp(model_for_influence, enable_grad=True)
        device_v = next(model_for_influence.parameters()).device
        v_list = [v.to(device_v).detach() for v in v_list]

        with torch.no_grad():
            total_sq = sum((v**2).sum() for v in v_list)
            norm = torch.sqrt(total_sq) + 1e-12
        v_list_norm = [v / norm for v in v_list]

        # Run experiment
        result = runner.run_single_experiment(
            sample_idx=sample_idx,
            test_image_idx=test_image_idx,
            target_class=target_class,
            influence_scores=probe_scores,
            apply_pgd_func=apply_pgd_perturbation,
            v_list_norm=v_list_norm,
            model_for_influence=model_for_influence,
        )

        # Log result
        runner.log_result(result)
        print(f"  Δp = {result.delta_prob:+.6f} "
              f"(orig: {result.prob_target_orig:.4f}, infused: {result.prob_target_infused:.4f})")

    # Save final metadata
    runner.save_metadata(status="completed")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {runner.results_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
