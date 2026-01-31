"""Live attack pipeline for the Streamlit demo.

Wraps the existing Infusion codebase to run influence computation,
PGD perturbation, and retraining interactively.

Requires PyTorch + Kronfluence at runtime. Intended for local/HPC use.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split

# Add project root to path so we can import infusion/common/cifar modules
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_DEMO_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Add kronfluence submodule to path
_KRONFLUENCE_DIR = os.path.join(_PROJECT_ROOT, "kronfluence")
if _KRONFLUENCE_DIR not in sys.path:
    sys.path.insert(0, _KRONFLUENCE_DIR)

# Paths to pre-trained checkpoints and factors
CKPT_DIR = os.path.join(_PROJECT_ROOT, "cifar", "checkpoints", "pretrain")
CKPT_EPOCH_9 = os.path.join(CKPT_DIR, "ckpt_epoch_9.pth")
CKPT_EPOCH_10 = os.path.join(CKPT_DIR, "ckpt_epoch_10.pth")
# Default hyperparameters matching the experiment script
RANDOM_SEED = 42
DAMPING = 1e-8
LEARNING_RATE = 0.01
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Model definition (must match the training code exactly)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
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
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2, downsample=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        ))
        self.layer3 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _make_model(device="cpu"):
    return TinyResNet(input_channels=3, num_classes=10).to(device)


def _load_checkpoint(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


# ---------------------------------------------------------------------------
# Kronfluence Task definition (must match the training code)
# ---------------------------------------------------------------------------

class _ClassificationTask:
    """Minimal task interface matching kronfluence.task.Task."""

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
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        log_probs = F.log_softmax(logits, dim=-1)
        bindex = torch.arange(logits.shape[0]).to(logits.device)
        return log_probs[bindex, targets].sum()


# ---------------------------------------------------------------------------
# Helper datasets
# ---------------------------------------------------------------------------

class _ProbeDataset(Dataset):
    """Single probe image with a target class label."""

    def __init__(self, x_star, target_class):
        self.x_star = x_star
        self.target_class = target_class

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x_star, self.target_class


class _PerturbedDataset(Dataset):
    """Overlay that replaces specific indices with perturbed images."""

    def __init__(self, original_dataset, perturbed_dict):
        self.original_dataset = original_dataset
        self.perturbed_dict = perturbed_dict

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        if idx in self.perturbed_dict:
            return self.perturbed_dict[idx]
        if hasattr(self.original_dataset, "dataset"):
            actual_idx = self.original_dataset.indices[idx]
            return self.original_dataset.dataset[actual_idx]
        return self.original_dataset[idx]


# ---------------------------------------------------------------------------
# G_delta and PGD (self-contained to avoid import issues with TrackedModule)
# ---------------------------------------------------------------------------

def _get_tracked_modules_info(model):
    from kronfluence.module.tracked_module import TrackedModule
    modules_info = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            params = list(module.original_module.parameters())
            modules_info.append({
                "name": name,
                "module": module,
                "has_bias": len(params) > 1,
                "num_params": len(params),
            })
    return modules_info


def _get_tracked_params_and_ihvp(model):
    from kronfluence.module.tracked_module import TrackedModule
    params = []
    v_list = []
    for _name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage["inverse_hessian_vector_product"]
            for p in module.original_module.parameters():
                p.requires_grad_(True)
                params.append(p)
            v_list.append(ihvp)
    return params, v_list


def _compute_G_delta(model, X_batch, y_batch, v_list, n_train):
    """Compute G_delta = -(1/n_train) * [nabla_x nabla_theta L]^T v."""
    model.eval()
    X_batch = X_batch.detach().requires_grad_(True)
    logits = model(X_batch)
    loss = F.cross_entropy(logits, y_batch, reduction="sum")

    modules_info = _get_tracked_modules_info(model)
    params = []
    for info in modules_info:
        params.extend(list(info["module"].original_module.parameters()))

    g_list = torch.autograd.grad(loss, params, create_graph=True)

    # Merge per-parameter grads to per-module grads
    merged = []
    g_idx = 0
    for mi in modules_info:
        if mi["has_bias"]:
            wg = g_list[g_idx].view(g_list[g_idx].size(0), -1)
            bg = g_list[g_idx + 1].view(g_list[g_idx + 1].size(0), 1)
            merged.append(torch.cat([wg, bg], dim=1))
            g_idx += 2
        else:
            merged.append(g_list[g_idx].view(g_list[g_idx].size(0), -1))
            g_idx += 1

    s = sum((gi * vi).sum() for gi, vi in zip(merged, v_list))
    Jt_v = torch.autograd.grad(s, X_batch, retain_graph=False, create_graph=False)[0]
    return -(1.0 / n_train) * Jt_v


def _apply_pgd(model, X_batch, y_batch, v_list, n_train,
               epsilon=1.0, alpha=0.001, n_steps=30, progress_cb=None):
    """PGD perturbation: maximize <nabla_theta L, v> w.r.t. input."""
    X_orig = X_batch.clone()
    X_adv = X_batch.clone()

    for step in range(n_steps):
        G_delta = _compute_G_delta(model, X_adv, y_batch, v_list, n_train)
        step_vec = alpha * torch.sign(G_delta)
        X_cand = X_adv + step_vec
        # Project onto L-inf ball around original
        X_adv = torch.clamp(X_cand, X_orig - epsilon, X_orig + epsilon)
        if progress_cb is not None:
            progress_cb((step + 1) / n_steps)

    delta = X_adv - X_orig
    B = X_batch.size(0)
    pert_norms = torch.norm(delta.reshape(B, -1), p=float("inf"), dim=1)
    return X_adv, pert_norms


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AttackEngine:
    """Encapsulates the live infusion attack pipeline.

    Loads model checkpoints and EK-FAC factors once, then exposes methods
    to compute influence scores, run PGD, and retrain.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        # Load CIFAR-10 dataset with same split as training
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        full_train_ds = datasets.CIFAR10(
            os.path.join(_PROJECT_ROOT, "cifar", "data"),
            train=True, download=True, transform=transform,
        )
        num_train = int(0.9 * len(full_train_ds))
        num_valid = len(full_train_ds) - num_train
        self.train_ds, self.valid_ds = random_split(
            full_train_ds, [num_train, num_valid],
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )

        # Load models
        self.model_epoch9_state = torch.load(
            CKPT_EPOCH_9, map_location=self.device, weights_only=False
        )
        self.model_epoch10 = _make_model(self.device)
        _load_checkpoint(self.model_epoch10, CKPT_EPOCH_10, self.device)
        self.model_epoch10.eval()

        # Set up Kronfluence analyzer with EK-FAC factors
        from infusion.kronfluence_patches import apply_patches
        apply_patches()

        from kronfluence.analyzer import Analyzer, prepare_model
        from kronfluence.task import Task

        # Create a proper Task subclass for kronfluence
        task_obj = _ClassificationTask()

        # Make it a proper Task subclass dynamically
        class _KronTask(Task):
            def compute_train_loss(self, batch, model, sample=False):
                return task_obj.compute_train_loss(batch, model, sample)

            def compute_measurement(self, batch, model):
                return task_obj.compute_measurement(batch, model)

        self.task = _KronTask()

        model_for_influence = _make_model(self.device)
        _load_checkpoint(model_for_influence, CKPT_EPOCH_10, self.device)
        model_for_influence.eval()
        self.model_for_influence = prepare_model(model_for_influence, self.task)

        # Use analysis_name="cifar_infusion" to match the existing pre-computed
        # factor directory structure: influence_results/cifar_infusion/factors_ekfac/
        self.analyzer = Analyzer(
            analysis_name="cifar_infusion",
            model=self.model_for_influence,
            task=self.task,
            output_dir=os.path.join(
                _PROJECT_ROOT, "cifar", "experiments", "influence_results",
            ),
        )
        from kronfluence.utils.dataset import DataLoaderKwargs
        self.analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4))

    def compute_influence_scores(self, probe_image, target_class, progress_cb=None):
        """Compute influence scores for all training examples.

        Args:
            probe_image: (3, 32, 32) tensor
            target_class: int, target class index
            progress_cb: optional callback(fraction) for progress updates

        Returns:
            scores: (45000,) numpy array of influence scores
        """
        from kronfluence.arguments import ScoreArguments

        probe_ds = _ProbeDataset(probe_image, target_class)
        score_args = ScoreArguments(damping_factor=DAMPING)

        if progress_cb:
            progress_cb(0.1)

        self.analyzer.compute_pairwise_scores(
            scores_name="live_attack_scores",
            factors_name="ekfac",
            query_dataset=probe_ds,
            train_dataset=self.train_ds,
            per_device_query_batch_size=1,
            score_args=score_args,
            overwrite_output_dir=True,
        )

        if progress_cb:
            progress_cb(0.9)

        scores = self.analyzer.load_pairwise_scores("live_attack_scores")["all_modules"]
        probe_scores = scores[0].cpu().numpy()

        if progress_cb:
            progress_cb(1.0)

        return probe_scores.astype(np.float32)

    def select_top_k(self, scores, k=50):
        """Select top-k most negatively influential training indices.

        Returns:
            indices: (k,) int array of indices into the training set
            selected_scores: (k,) float array of corresponding scores
        """
        sorted_idx = np.argsort(scores)  # ascending (most negative first)
        top_k = sorted_idx[:k]
        return top_k, scores[top_k]

    def run_pgd(self, top_k_indices, epsilon=1.0, alpha=0.001, n_steps=30,
                progress_cb=None):
        """Run PGD perturbation on selected training images.

        Args:
            top_k_indices: array of training set indices to perturb
            epsilon: L-inf perturbation budget
            alpha: PGD step size
            n_steps: number of PGD iterations
            progress_cb: optional callback(fraction) for progress updates

        Returns:
            X_original: (k, 3, 32, 32) numpy array
            X_perturbed: (k, 3, 32, 32) numpy array
            y_labels: (k,) numpy array of labels
            perturbation_norms: (k,) numpy array
        """
        # Extract selected training images
        orig_dataset = self.train_ds.dataset if hasattr(self.train_ds, "dataset") else self.train_ds
        orig_indices = self.train_ds.indices if hasattr(self.train_ds, "indices") else range(len(self.train_ds))

        imgs, labels = [], []
        for idx in top_k_indices:
            actual_idx = orig_indices[idx]
            img, label = orig_dataset[actual_idx]
            imgs.append(img)
            labels.append(label)

        X_selected = torch.stack(imgs).to(self.device)
        y_selected = torch.tensor(labels).to(self.device)

        # Get IHVP
        _params, v_list = _get_tracked_params_and_ihvp(self.model_for_influence)
        v_list = [v.to(self.device).detach() for v in v_list]

        # Normalize v_list
        with torch.no_grad():
            total_sq = sum((v ** 2).sum() for v in v_list)
            norm = torch.sqrt(total_sq) + 1e-12
        v_list_norm = [v / norm for v in v_list]

        # Run PGD
        X_perturbed, pert_norms = _apply_pgd(
            self.model_for_influence, X_selected, y_selected,
            v_list_norm, len(self.train_ds),
            epsilon=epsilon, alpha=alpha, n_steps=n_steps,
            progress_cb=progress_cb,
        )

        return (
            X_selected.detach().cpu().numpy(),
            X_perturbed.detach().cpu().numpy(),
            y_selected.cpu().numpy(),
            pert_norms.detach().cpu().numpy(),
        )

    def retrain_and_evaluate(self, top_k_indices, X_perturbed_np, y_labels_np,
                             probe_image, progress_cb=None):
        """Load epoch 9, create perturbed dataset, train 1 epoch, evaluate.

        Args:
            top_k_indices: training indices that were perturbed
            X_perturbed_np: (k, 3, 32, 32) perturbed images
            y_labels_np: (k,) labels
            probe_image: (3, 32, 32) test image tensor
            progress_cb: optional callback(fraction) for progress updates

        Returns:
            logits_before: (10,) numpy array (epoch 10 logits)
            logits_after: (10,) numpy array (infused model logits)
        """
        if progress_cb:
            progress_cb(0.05)

        # Build perturbed dict
        perturbed_dict = {}
        for i, idx in enumerate(top_k_indices):
            img_t = torch.from_numpy(X_perturbed_np[i])
            label = int(y_labels_np[i])
            perturbed_dict[int(idx)] = (img_t, label)

        modified_dataset = _PerturbedDataset(self.train_ds, perturbed_dict)

        # Set seeds for reproducibility
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed_all(RANDOM_SEED)

        from infusion.dataloader import get_dataloader

        modified_dl = get_dataloader(modified_dataset, BATCH_SIZE, seed=RANDOM_SEED)

        if progress_cb:
            progress_cb(0.1)

        # Load model from epoch 9
        model_infused = _make_model(self.device)
        if isinstance(self.model_epoch9_state, dict) and "model_state_dict" in self.model_epoch9_state:
            model_infused.load_state_dict(self.model_epoch9_state["model_state_dict"])
        else:
            model_infused.load_state_dict(self.model_epoch9_state)

        opt = torch.optim.SGD(model_infused.parameters(), lr=LEARNING_RATE)
        loss_func = nn.CrossEntropyLoss()

        if progress_cb:
            progress_cb(0.15)

        # Train 1 epoch
        model_infused.train()
        total_batches = len(modified_dl)
        for batch_idx, (xb, yb) in enumerate(modified_dl):
            xb, yb = xb.to(self.device), yb.to(self.device)
            loss = loss_func(model_infused(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if progress_cb:
                frac = 0.15 + 0.8 * ((batch_idx + 1) / total_batches)
                progress_cb(frac)

        model_infused.eval()

        # Evaluate
        with torch.no_grad():
            x_input = probe_image.unsqueeze(0).to(self.device)
            logits_before = self.model_epoch10(x_input).cpu().numpy().squeeze()
            logits_after = model_infused(x_input).cpu().numpy().squeeze()

        if progress_cb:
            progress_cb(1.0)

        return logits_before, logits_after

    def get_baseline_logits(self, probe_image):
        """Get epoch-10 model logits for a probe image.

        Args:
            probe_image: (3, 32, 32) tensor

        Returns:
            logits: (10,) numpy array
        """
        with torch.no_grad():
            x = probe_image.unsqueeze(0).to(self.device)
            return self.model_epoch10(x).cpu().numpy().squeeze()
