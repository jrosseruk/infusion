# %%
import torch
import sys
sys.path.append("")
sys.path.append("..")
sys.path.append("../kronfluence")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"  Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("No CUDA devices available")

# %%
import argparse
import os
import json
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser(description="CIFAR Random Test Image Infusion Experiment")

# Random seed
parser.add_argument('--random_seed', type=int, default=42, help='Random seed for training')
parser.add_argument('--sample_seed', type=int, default=999, help='Random seed for test image sampling')

# Model parameters
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

# Hessian parameters
parser.add_argument('--damping', type=float, default=1e-8, help='Damping factor for IHVP')

# PGD parameters
parser.add_argument('--top_k', type=int, default=100, help='Number of points to perturb')
parser.add_argument('--epsilon', type=float, default=1, help='L_∞ budget')
parser.add_argument('--alpha', type=float, default=0.001, help='Step size')
parser.add_argument('--n_steps', type=int, default=50, help='PGD iterations')

# Experiment parameters
parser.add_argument('--n_samples', type=int, default=1000, help='Number of test images to sample')
parser.add_argument('--start_idx', type=int, default=0, help='Start index for resuming experiment')
parser.add_argument('--results_dir', type=str, default='./results/random_test_infusion/', help='Directory to save results')

args, _ = parser.parse_known_args()

# Create results directory
os.makedirs(args.results_dir, exist_ok=True)

print(f"Training seed: {args.random_seed}")
print(f"Sample selection seed: {args.sample_seed}")
print(f"Number of test samples: {args.n_samples}")
print(f"Results directory: {args.results_dir}")

# %%
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Set random seed for deterministic model initialization
torch.manual_seed(args.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

# Enable deterministic mode for CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set to: {args.random_seed}")
print(f"CUDA deterministic mode: enabled\n")

# %%
from torchvision import datasets, transforms
from torch.utils.data import random_split

transform = transforms.Compose([
    transforms.ToTensor()
])

full_train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# Create a validation split from the training set (e.g., 90% train, 10% val)
num_train = int(0.9 * len(full_train_ds))
num_valid = len(full_train_ds) - num_train
train_ds, valid_ds = random_split(full_train_ds, [num_train, num_valid], generator=torch.Generator().manual_seed(args.random_seed))

img, label = full_train_ds[2]
print(img.shape)  # torch.Size([3, 32, 32])

from infusion.dataloader import get_dataloader

# Dataloaders (use seed for deterministic data ordering)
train_dl = get_dataloader(train_ds, args.batch_size, seed=args.random_seed)
test_dl = get_dataloader(test_ds, args.batch_size, seed=args.random_seed)
valid_dl = get_dataloader(valid_ds, args.batch_size, seed=args.random_seed)

# %%
# Grab dataset metadata for CIFAR10
in_channels = full_train_ds[0][0].shape[0]    # 3
img_size = full_train_ds[0][0].shape[1]       # 32 for CIFAR10
num_classes = len(full_train_ds.classes)       # 10

import torch
import torch.nn as nn

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

        # First residual block
        self.layer1 = ResidualBlock(32, 32)
        # Down + increase channels
        self.layer2 = ResidualBlock(32, 64, stride=2, downsample=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        ))
        # Another downsampling block
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

def make_model():
    return TinyResNet(input_channels=in_channels, num_classes=num_classes).to(device)

# %%
# Load pre-trained models at epochs 9 and 10
from infusion.train import fit

ckpt_dir = "./checkpoints/pretrain/"
ckpt_path_9 = ckpt_dir + f"ckpt_epoch_9.pth"
ckpt_path_10 = ckpt_dir + f"ckpt_epoch_10.pth"

print("Loading pre-trained models...")

# Check if checkpoints exist
if not os.path.exists(ckpt_path_9) or not os.path.exists(ckpt_path_10):
    raise FileNotFoundError(f"Required checkpoints not found. Please ensure {ckpt_path_9} and {ckpt_path_10} exist.")

# Load model at epoch 9
model_epoch9 = make_model().to(device)
checkpoint = torch.load(ckpt_path_9, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model_epoch9.load_state_dict(checkpoint['model_state_dict'])
else:
    model_epoch9.load_state_dict(checkpoint)
model_epoch9.eval()
print(f"Loaded model from {ckpt_path_9}")

# Load model at epoch 10
model_epoch10 = make_model().to(device)
checkpoint = torch.load(ckpt_path_10, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model_epoch10.load_state_dict(checkpoint['model_state_dict'])
else:
    model_epoch10.load_state_dict(checkpoint)
model_epoch10.eval()
print(f"Loaded model from {ckpt_path_10}")

print("\nBoth models loaded into memory successfully!")

# %%
# Apply kronfluence patches before importing
from infusion.kronfluence_patches import apply_patches
apply_patches()

# Now import kronfluence normally
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

# %%
# Sample N random test images
np.random.seed(args.sample_seed)
torch.manual_seed(args.sample_seed)

sampled_indices = np.random.choice(len(test_ds), size=args.n_samples, replace=False)
print(f"Sampled {args.n_samples} test images with seed {args.sample_seed}")
print(f"Index range: [{sampled_indices.min()}, {sampled_indices.max()}]")

# Save sampled indices
indices_path = os.path.join(args.results_dir, 'sampled_test_indices.npy')
np.save(indices_path, sampled_indices)
print(f"Saved sampled indices to {indices_path}")

# %%
# Define Task for kronfluence
class ClassificationTask(Task):

    def __init__(self):
        super().__init__()

    def compute_train_loss(self, batch, model, sample = False):
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")


    def compute_measurement(
        self,
        batch,
        model: nn.Module,
    ) -> torch.Tensor:
        inputs, targets = batch

        # Ensure inputs are on the correct device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)

        # Compute logits and log probabilities
        logits = model(inputs)  # [batch_size, num_classes]
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, num_classes]

        # Extract log probability for the target class for each example
        bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        log_probs_target = log_probs[bindex, targets]

        # Return the sum of log probabilities for the batch
        return log_probs_target.sum()

# %%
# Prepare model for influence computation (do this once)
from torch.utils.data import Dataset

class ProbeDataset(Dataset):
    """Simple dataset containing a single probe point (x_star, y_star)"""
    def __init__(self, x_star, y_star):
        self.x_star = x_star
        self.y_star = y_star

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x_star, self.y_star

# Prepare model (use epoch 10 for influence computation)
model_for_influence = make_model()
checkpoint = torch.load(ckpt_path_10, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model_for_influence.load_state_dict(checkpoint['model_state_dict'])
else:
    model_for_influence.load_state_dict(checkpoint)
model_for_influence = model_for_influence.eval()

task = ClassificationTask()
model_for_influence = prepare_model(model_for_influence, task)

# Set up the Analyzer class
analyzer = Analyzer(
    analysis_name="cifar",
    model=model_for_influence,
    task=task,
)
dataloader_kwargs = DataLoaderKwargs(num_workers=4)
analyzer.set_dataloader_kwargs(dataloader_kwargs)

# Fit all factors (do this once)
print("\nFitting factors (one-time operation)...")
analyzer.fit_all_factors(
    factors_name="ekfac",
    dataset=train_ds,
    per_device_batch_size=2048,
    overwrite_output_dir=True,
)
print("Factors fitted successfully!")

# %%
# Helper functions
from kronfluence.module.utils import get_tracked_module_names
from kronfluence.module.tracked_module import TrackedModule

def get_tracked_params_and_ihvp(model, enable_grad=True):
    """
    Returns:
        params: list of original_module parameters for all tracked modules in model (ordered)
        v_list: list of IHVPs corresponding to each tracked module (one IHVP per module, not per parameter)
    """
    params = []
    v_list = []
    tracked_module_names = get_tracked_module_names(model)

    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage["inverse_hessian_vector_product"]

            # Collect all parameters for this module
            for param_name, param in module.original_module.named_parameters():
                if enable_grad:
                    param.requires_grad_(True)
                params.append(param)

            # Add IHVP only once per module (not per parameter)
            v_list.append(ihvp)

    return params, v_list

def get_tracked_modules_info(model):
    """Get information about tracked modules including their parameter structure"""
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
    """
    Compute perturbation gradient G_δ = -(1/n) [∇_x ∇_θ L]^T v

    Uses double-backward to compute cross-Jacobian vector product

    Args:
        model: Trained model
        X_batch: Batch of inputs [B, C, H, W]
        y_batch: Batch of labels [B]
        v_list: IHVP vector (list of tensors, one per tracked module)
        n_train: Total training set size

    Returns:
        G_delta: Perturbation gradients [B, C, H, W]
    """
    model.eval()

    # Enable gradient w.r.t. inputs
    X_batch = X_batch.detach().requires_grad_(True)

    # Forward pass
    logits = model(X_batch)
    loss = F.cross_entropy(logits, y_batch, reduction='sum')

    # Get tracked modules info
    modules_info = get_tracked_modules_info(model)

    # Collect parameters in the same order as tracked modules
    params = []
    for info in modules_info:
        params.extend(list(info['module'].original_module.parameters()))

    # First backward: g = ∇_θ loss
    g_list = torch.autograd.grad(loss, params, create_graph=True)

    # Merge gradients to match v_list structure
    # Each tracked module has one IHVP that corresponds to flattened [weight, bias] (or just [weight])
    merged_g_list = []
    g_idx = 0

    for module_info in modules_info:
        if module_info['has_bias']:
            # Module has weight and bias
            weight_grad = g_list[g_idx]
            bias_grad = g_list[g_idx + 1]

            # Flatten and concatenate
            weight_flat = weight_grad.view(weight_grad.size(0), -1)
            bias_flat = bias_grad.view(bias_grad.size(0), 1)
            merged = torch.cat([weight_flat, bias_flat], dim=1)

            g_idx += 2
        else:
            # Module has only weight
            weight_grad = g_list[g_idx]
            merged = weight_grad.view(weight_grad.size(0), -1)

            g_idx += 1

        merged_g_list.append(merged)

    # Dot product: s = g^T v (scalar)
    s = sum((gi * vi).sum() for gi, vi in zip(merged_g_list, v_list))

    # Second backward: ∇_x s = [∇_x ∇_θ L]^T v
    Jt_v = torch.autograd.grad(s, X_batch, retain_graph=False, create_graph=False)[0]

    # Scale and negate
    G_delta = -(1.0 / n_train) * Jt_v

    return G_delta

def apply_pgd_perturbation(model, X_batch, y_batch, v_list, n_train,
                          epsilon=2.0, alpha=0.3, n_steps=20, norm='inf',
                          verbose=False):
    """
    Apply PGD to find optimal perturbations that maximize observable f(θ)

    z_{t+1} = Proj(z_t + α · sign(G_δ(z_t)))

    Args:
        model: Trained model
        X_batch: Original batch [B, C, H, W]
        y_batch: Labels [B]
        v_list: IHVP vector
        n_train: Training set size
        epsilon: L_∞ or L_2 perturbation budget
        alpha: Step size
        n_steps: Number of PGD iterations
        norm: 'inf' or '2'
        verbose: Print convergence diagnostics

    Returns:
        X_perturbed: Perturbed batch [B, C, H, W]
        perturbation_norms: Norms of final perturbations [B]
    """
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

    # PGD iterations
    for step in range(n_steps):
        # Compute gradient direction
        G_delta = compute_G_delta(model, X_adv, y_batch, v_list, n_train)

        # Take step
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

    # Compute final perturbation norms
    delta = X_adv - X_orig
    if norm == 'inf':
        pert_norms = torch.norm(delta.reshape(B, -1), p=float('inf'), dim=1)
    else:
        pert_norms = torch.norm(delta.reshape(B, -1), p=2, dim=1)

    return X_adv, pert_norms

# %%
# Main experiment loop
from torch.utils.data import Dataset

class PerturbedDataset(Dataset):
    """
    Dataset that wraps original training data and replaces specific indices with perturbed versions.
    """
    def __init__(self, original_dataset, perturbed_dict):
        """
        Args:
            original_dataset: The original training dataset
            perturbed_dict: Dict mapping index -> (perturbed_image, label)
        """
        self.original_dataset = original_dataset
        self.perturbed_dict = perturbed_dict

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Return perturbed version if available, otherwise return original
        if idx in self.perturbed_dict:
            return self.perturbed_dict[idx]
        else:
            # Handle both Subset and regular dataset indexing
            if hasattr(self.original_dataset, 'dataset'):
                actual_idx = self.original_dataset.indices[idx]
                return self.original_dataset.dataset[actual_idx]
            else:
                return self.original_dataset[idx]

print("\n" + "="*80)
print("STARTING MAIN EXPERIMENT LOOP")
print("="*80)

experiment_metadata = {
    'start_time': datetime.now().isoformat(),
    'random_seed': args.random_seed,
    'sample_seed': args.sample_seed,
    'n_samples': args.n_samples,
    'n_classes': num_classes,
    'top_k': args.top_k,
    'epsilon': args.epsilon,
    'alpha': args.alpha,
    'n_steps': args.n_steps,
    'damping': args.damping,
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
}

# Save experiment metadata
metadata_path = os.path.join(args.results_dir, 'experiment_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(experiment_metadata, f, indent=2)
print(f"Saved experiment metadata to {metadata_path}\n")

# Main loop: for each test image, for each class, run infusion
for sample_idx in range(args.start_idx, args.n_samples):
    test_image_idx = sampled_indices[sample_idx]
    x_star, true_label = test_ds[test_image_idx]

    print(f"\n{'='*80}")
    print(f"Sample {sample_idx + 1}/{args.n_samples}: Test Image Index {test_image_idx}")
    print(f"True Label: {true_label} ({test_ds.classes[true_label]})")
    print(f"{'='*80}")

    # For each target class
    for target_class in range(num_classes):
        print(f"\n  Target class: {target_class} ({test_ds.classes[target_class]})")

        # Create result directory for this experiment
        exp_dir = os.path.join(args.results_dir, f'sample_{sample_idx:04d}_test_{test_image_idx}_target_{target_class}')
        os.makedirs(exp_dir, exist_ok=True)

        # Create probe dataset
        probe_dataset = ProbeDataset(x_star, target_class)

        # Compute influence scores
        score_args = ScoreArguments(damping_factor=args.damping)

        analyzer.compute_pairwise_scores(
            scores_name=f"scores_sample{sample_idx}_target{target_class}",
            factors_name="ekfac",
            query_dataset=probe_dataset,
            train_dataset=train_ds,
            per_device_query_batch_size=1,
            score_args=score_args,
            overwrite_output_dir=True,
        )

        scores = analyzer.load_pairwise_scores(f"scores_sample{sample_idx}_target{target_class}")["all_modules"]
        probe_scores = scores[0]  # Shape: (N_train,)

        # Get top-k most negatively influential training examples
        top_k_indices = probe_scores.argsort(descending=False)[:args.top_k]

        # Get selected training examples
        orig_dataset = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
        orig_indices = train_ds.indices if hasattr(train_ds, 'indices') else range(len(train_ds))
        selected_indices = [orig_indices[i] for i in top_k_indices]

        imgs, labels = zip(*(orig_dataset[i] for i in selected_indices))
        X_selected = torch.stack(imgs).to(device)
        y_selected = torch.tensor(labels).to(device)

        # Get v_list (IHVP) for perturbation
        params, v_list = get_tracked_params_and_ihvp(model_for_influence, enable_grad=True)

        # Make sure IHVPs are on the same device and don't require grad
        device_v = next(model_for_influence.parameters()).device
        v_list = [v.to(device_v).detach() for v in v_list]

        with torch.no_grad():
            total_sq = sum((v**2).sum() for v in v_list)
            norm = torch.sqrt(total_sq) + 1e-12
        v_list_norm = [v / norm for v in v_list]

        # Apply PGD perturbation
        X_perturbed, pert_norms = apply_pgd_perturbation(
            model_for_influence, X_selected, y_selected, v_list_norm, len(train_ds),
            epsilon=args.epsilon,
            alpha=args.alpha,
            n_steps=args.n_steps,
            norm='inf',
            verbose=False
        )

        # Create perturbed dataset
        perturbed_dict = {}
        for i, idx in enumerate(top_k_indices):
            img_perturbed = X_perturbed[i].cpu()
            # Get original label
            if hasattr(train_ds, 'dataset'):
                actual_idx = train_ds.indices[idx]
                _, label = train_ds.dataset[actual_idx]
            else:
                _, label = train_ds[idx]
            perturbed_dict[idx.item() if torch.is_tensor(idx) else idx] = (img_perturbed, label)

        modified_dataset = PerturbedDataset(train_ds, perturbed_dict)

        # Partial retraining: Load from epoch 9, train only epoch 10
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create dataloader with same settings and seed as original training
        modified_dl = get_dataloader(modified_dataset, args.batch_size, seed=args.random_seed)

        # Load model from epoch 9
        model_infused = make_model().to(device)
        checkpoint = torch.load(ckpt_path_9, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_infused.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_infused.load_state_dict(checkpoint)

        # Recreate optimizer
        opt_infused = torch.optim.SGD(model_infused.parameters(), lr=args.learning_rate)
        loss_func_infused = torch.nn.CrossEntropyLoss()

        # Create checkpoint directory for this infusion
        ckpt_dir_infused = os.path.join(exp_dir, 'checkpoint')
        os.makedirs(ckpt_dir_infused, exist_ok=True)

        # Train for exactly 1 epoch
        fit(1, model_infused, loss_func_infused, opt_infused, modified_dl, valid_dl, ckpt_dir_infused, random_seed=args.random_seed)

        model_infused.eval()

        # Compute logits from all models
        with torch.no_grad():
            x_star_input = x_star.unsqueeze(0).to(device)

            # Original models (epoch 9 and 10)
            logits_epoch9 = model_epoch9(x_star_input).cpu()
            logits_epoch10 = model_epoch10(x_star_input).cpu()

            # Infused model
            logits_infused = model_infused(x_star_input).cpu()

            # Also compute logits for original and perturbed training examples
            logits_orig_on_selected = model_epoch10(X_selected).cpu()
            logits_orig_on_perturbed = model_epoch10(X_perturbed).cpu()
            logits_infused_on_selected = model_infused(X_selected).cpu()
            logits_infused_on_perturbed = model_infused(X_perturbed).cpu()

        # Save all results
        result_dict = {
            'sample_idx': sample_idx,
            'test_image_idx': int(test_image_idx),
            'true_label': int(true_label),
            'target_class': int(target_class),

            # Probe image
            'probe_image': x_star.cpu().numpy(),

            # Influence scores
            'influence_scores': probe_scores.cpu().numpy(),
            'top_k_indices': top_k_indices.cpu().numpy(),
            'selected_train_indices': [int(i) for i in selected_indices],

            # Original training images
            'original_train_images': X_selected.cpu().numpy(),
            'original_train_labels': y_selected.cpu().numpy(),

            # Perturbed training images
            'perturbed_train_images': X_perturbed.cpu().numpy(),
            'perturbation_norms': pert_norms.cpu().numpy(),

            # Logits from all models on probe image
            'logits_epoch9': logits_epoch9.numpy(),
            'logits_epoch10': logits_epoch10.numpy(),
            'logits_infused': logits_infused.numpy(),

            # Logits on training examples
            'logits_orig_on_selected': logits_orig_on_selected.numpy(),
            'logits_orig_on_perturbed': logits_orig_on_perturbed.numpy(),
            'logits_infused_on_selected': logits_infused_on_selected.numpy(),
            'logits_infused_on_perturbed': logits_infused_on_perturbed.numpy(),

            # Experiment parameters
            'epsilon': args.epsilon,
            'alpha': args.alpha,
            'n_steps': args.n_steps,
            'top_k': args.top_k,
            'damping': args.damping,
        }

        # Save as numpy archive
        result_path = os.path.join(exp_dir, 'results.npz')
        np.savez_compressed(result_path, **result_dict)

        # Also save metadata as JSON for easy reading
        metadata_dict = {
            'sample_idx': sample_idx,
            'test_image_idx': int(test_image_idx),
            'true_label': int(true_label),
            'target_class': int(target_class),
            'selected_train_indices': [int(i) for i in selected_indices],
            'perturbation_norms_mean': float(pert_norms.mean().item()),
            'perturbation_norms_max': float(pert_norms.max().item()),
            'epsilon': args.epsilon,
            'alpha': args.alpha,
            'n_steps': args.n_steps,
            'top_k': args.top_k,
        }

        metadata_json_path = os.path.join(exp_dir, 'metadata.json')
        with open(metadata_json_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        # Compute and log key metrics
        probs_epoch10 = F.softmax(logits_epoch10, dim=1)[0]
        probs_infused = F.softmax(logits_infused, dim=1)[0]

        prob_target_orig = probs_epoch10[target_class].item()
        prob_target_infused = probs_infused[target_class].item()
        delta_prob = prob_target_infused - prob_target_orig

        print(f"    Saved to: {exp_dir}")
        print(f"    p(target|x*) - Original: {prob_target_orig:.6f}, Infused: {prob_target_infused:.6f}, Δ: {delta_prob:+.6f}")

        # Append to running log
        log_entry = {
            'sample_idx': sample_idx,
            'test_image_idx': int(test_image_idx),
            'true_label': int(true_label),
            'target_class': int(target_class),
            'prob_target_orig': prob_target_orig,
            'prob_target_infused': prob_target_infused,
            'delta_prob': delta_prob,
            'timestamp': datetime.now().isoformat(),
        }

        log_path = os.path.join(args.results_dir, 'experiment_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)

# Save final metadata
experiment_metadata['end_time'] = datetime.now().isoformat()
experiment_metadata['status'] = 'completed'
with open(metadata_path, 'w') as f:
    json.dump(experiment_metadata, f, indent=2)

print(f"\nAll results saved to: {args.results_dir}")
print(f"Experiment log: {os.path.join(args.results_dir, 'experiment_log.jsonl')}")
