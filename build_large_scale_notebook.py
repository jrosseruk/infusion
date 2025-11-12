#!/usr/bin/env python3
"""
Build the large-scale infusion experiment notebook
"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split("\n")
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split("\n")
    })

# ============================================================================
# NOTEBOOK CONTENT
# ============================================================================

add_markdown("""# Large-Scale CIFAR Infusion Experiments

Run 10,000 infusion experiments:
- 1000 random test images × 10 target classes
- Parallel execution on 4 GPUs
- Fully resumable with comprehensive data saving
- All data needed to recreate plots""")

add_markdown("## 1. Configuration")

add_code("""# Experiment Configuration
import os
import json

N_TEST_IMAGES = 1000
N_CLASSES = 10
N_GPUS = 4
TOP_K = 100
EPSILON = 1.0
ALPHA = 0.001
N_STEPS = 50
DAMPING = 1e-8
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 10
TRAINING_SEED = 42
SAMPLING_SEED = 12345

EXPERIMENT_DIR = "experiments_large_scale"
EKFAC_DIR = os.path.join(EXPERIMENT_DIR, "ekfac_factors")
CHECKPOINT_DIR = "./checkpoints/pretrain"
TOTAL_EXPERIMENTS = N_TEST_IMAGES * N_CLASSES

print(f"Total experiments: {TOTAL_EXPERIMENTS}")
print(f"GPUs: {N_GPUS}, Top-k: {TOP_K}, Epsilon: {EPSILON}")

os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(EKFAC_DIR, exist_ok=True)

config = {
    "n_test_images": N_TEST_IMAGES, "n_classes": N_CLASSES, "top_k": TOP_K,
    "epsilon": EPSILON, "alpha": ALPHA, "n_steps": N_STEPS, "damping": DAMPING,
    "training_seed": TRAINING_SEED, "sampling_seed": SAMPLING_SEED,
    "total_experiments": TOTAL_EXPERIMENTS
}

with open(os.path.join(EXPERIMENT_DIR, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)""")

add_markdown("## 2. Imports")

add_code("""import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import random
import time
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
from filelock import FileLock
from pathlib import Path

sys.path.append("")
sys.path.append("..")
sys.path.append("../kronfluence")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPUs: {torch.cuda.device_count()}")""")

add_markdown("## 3. Load Data and Sample Test Images")

add_code("""from torchvision import datasets, transforms
from torch.utils.data import random_split

transform = transforms.Compose([transforms.ToTensor()])
full_train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

num_train = int(0.9 * len(full_train_ds))
num_valid = len(full_train_ds) - num_train
train_ds, valid_ds = random_split(
    full_train_ds, [num_train, num_valid],
    generator=torch.Generator().manual_seed(TRAINING_SEED)
)

print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

# Sample test images
np.random.seed(SAMPLING_SEED)
test_sample_indices = np.random.choice(len(test_ds), N_TEST_IMAGES, replace=False)
test_sample_path = os.path.join(EXPERIMENT_DIR, "test_sample_indices.npy")
np.save(test_sample_path, test_sample_indices)
print(f"Sampled {N_TEST_IMAGES} test images")""")

add_markdown("## 4. Load Model Architecture")

add_code("""# Same TinyResNet from original experiments
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
        return self.relu(out)

class TinyResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2, downsample=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(64)))
        self.layer3 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(128)))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

print("Model architecture defined")""")

add_markdown("## 5. Compute EKFAC Factors (Run Once)")

add_code("""from infusion.kronfluence_patches import apply_patches
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

    def compute_measurement(self, batch, model: nn.Module) -> torch.Tensor:
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

# Check if factors already computed
factors_path = os.path.join(EKFAC_DIR, "factors_complete.flag")
if os.path.exists(factors_path):
    print("EKFAC factors already computed, skipping...")
else:
    print("Computing EKFAC factors (this takes ~5 minutes)...")
    device = torch.device('cuda:0')
    model = TinyResNet().to(device)

    # Load epoch 10 checkpoint
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{EPOCHS}.pth"), map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    task = ClassificationTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(analysis_name="cifar_largescale", model=model, task=task, output_dir=EKFAC_DIR)
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4))

    analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_ds,
        per_device_batch_size=2048,
        overwrite_output_dir=True
    )

    # Mark as complete
    with open(factors_path, 'w') as f:
        f.write(f"Completed at {datetime.now().isoformat()}")

    print("EKFAC factors computed and saved!")
    del model, analyzer
    torch.cuda.empty_cache()""")

add_markdown("""## 6. Helper Functions

Utility functions for experiments""")

add_code("""from torch.utils.data import Dataset as TorchDataset

class ProbeDataset(TorchDataset):
    def __init__(self, x_star, y_star):
        self.x_star = x_star
        self.y_star = y_star

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x_star, self.y_star

class PerturbedDataset(TorchDataset):
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

def get_experiment_dir(test_idx, target_class):
    return os.path.join(EXPERIMENT_DIR, f"exp_{test_idx:04d}_{target_class:02d}")

def is_experiment_complete(test_idx, target_class):
    exp_dir = get_experiment_dir(test_idx, target_class)
    data_path = os.path.join(exp_dir, "data.npz")
    metadata_path = os.path.join(exp_dir, "metadata.json")
    return os.path.exists(data_path) and os.path.exists(metadata_path)

def save_experiment_data(test_idx, target_class, data_dict, metadata_dict):
    exp_dir = get_experiment_dir(test_idx, target_class)
    os.makedirs(exp_dir, exist_ok=True)

    # Save tensors
    data_path = os.path.join(exp_dir, "data.npz")
    np.savez_compressed(data_path, **data_dict)

    # Save metadata
    metadata_path = os.path.join(exp_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

print("Helper functions defined")""")

add_markdown("""## 7. Progress Tracking

Load and update experiment progress""")

add_code("""def load_progress():
    progress_path = os.path.join(EXPERIMENT_DIR, "progress.json")
    lock_path = progress_path + ".lock"

    with FileLock(lock_path):
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "total_experiments": TOTAL_EXPERIMENTS,
                "completed": 0,
                "failed": 0,
                "completed_list": [],
                "failed_list": [],
                "last_updated": None
            }

def update_progress(exp_id, status='completed', error_msg=None):
    progress_path = os.path.join(EXPERIMENT_DIR, "progress.json")
    lock_path = progress_path + ".lock"

    with FileLock(lock_path):
        progress = load_progress()

        exp_name = f"exp_{exp_id // N_CLASSES:04d}_{exp_id % N_CLASSES:02d}"

        if status == 'completed':
            if exp_name not in progress["completed_list"]:
                progress["completed_list"].append(exp_name)
                progress["completed"] = len(progress["completed_list"])
        elif status == 'failed':
            if exp_name not in progress["failed_list"]:
                progress["failed_list"].append({"exp": exp_name, "error": error_msg})
                progress["failed"] = len(progress["failed_list"])

        progress["last_updated"] = datetime.now().isoformat()

        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)

def get_incomplete_experiments():
    completed = set()
    for test_idx in range(N_TEST_IMAGES):
        for target_class in range(N_CLASSES):
            if is_experiment_complete(test_idx, target_class):
                exp_id = test_idx * N_CLASSES + target_class
                completed.add(exp_id)

    all_experiments = set(range(TOTAL_EXPERIMENTS))
    incomplete = sorted(list(all_experiments - completed))
    return incomplete

print("Progress tracking functions defined")""")

# Continue in next part...
print(f"Notebook has {len(notebook['cells'])} cells so far...")

# Save what we have
with open("cifar_large_scale_infusion.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook saved! Run this script to continue building...")
