#!/usr/bin/env python3
"""
Create the complete large-scale infusion experiment notebook with all components
This is the FINAL comprehensive version
"""
import json

notebook = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}

def md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text.split("\n")})

def code(code_str):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_str.split("\n")})

# Title
md("# Large-Scale CIFAR-10 Infusion Experiments\n\n**Objective:** Run 10,000 infusion experiments (1000 test images × 10 target classes)\n\n**Features:**\n- Parallel execution on 4 GPUs\n- Fully resumable with checkpointing\n- Comprehensive data saving for analysis\n- Progress tracking with ETA")

# 1. Configuration
md("## 1. Configuration")
code("""import os, json
N_TEST_IMAGES, N_CLASSES, N_GPUS, TOP_K = 1000, 10, 4, 100
EPSILON, ALPHA, N_STEPS, DAMPING = 1.0, 0.001, 50, 1e-8
BATCH_SIZE, LR, EPOCHS, TRAIN_SEED, SAMPLE_SEED = 16, 0.01, 10, 42, 12345
EXPERIMENT_DIR, EKFAC_DIR, CHECKPOINT_DIR = "experiments_large_scale", "experiments_large_scale/ekfac_factors", "./checkpoints/pretrain"
TOTAL_EXPERIMENTS = N_TEST_IMAGES * N_CLASSES
print(f"Configuration:\\n  Total experiments: {TOTAL_EXPERIMENTS}\\n  GPUs: {N_GPUS}, Top-k: {TOP_K}, Epsilon: {EPSILON}")
os.makedirs(EXPERIMENT_DIR, exist_ok=True); os.makedirs(EKFAC_DIR, exist_ok=True)
config = {"n_test": N_TEST_IMAGES, "n_classes": N_CLASSES, "top_k": TOP_K, "epsilon": EPSILON, "alpha": ALPHA, "n_steps": N_STEPS, "damping": DAMPING}
json.dump(config, open(f"{EXPERIMENT_DIR}/config.json", 'w'), indent=2); print("Config saved")""")

# 2. Imports
md("## 2. Imports")
code("""import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys, random, time, copy
from tqdm import tqdm; from datetime import datetime; import multiprocessing as mp; from filelock import FileLock
sys.path.extend(["", "..", "../kronfluence"])
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")""")

# 3. Load Data
md("## 3. Load CIFAR-10 and Sample Test Images")
code("""from torchvision import datasets, transforms; from torch.utils.data import random_split
transform = transforms.Compose([transforms.ToTensor()])
full_train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
num_train, num_valid = int(0.9 * len(full_train_ds)), int(0.1 * len(full_train_ds))
train_ds, valid_ds = random_split(full_train_ds, [num_train, num_valid], generator=torch.Generator().manual_seed(TRAIN_SEED))
print(f"Train: {len(train_ds)}, Val: {len(valid_ds)}, Test: {len(test_ds)}")
np.random.seed(SAMPLE_SEED); test_sample_indices = np.random.choice(len(test_ds), N_TEST_IMAGES, replace=False)
np.save(f"{EXPERIMENT_DIR}/test_sample_indices.npy", test_sample_indices); print(f"Sampled {N_TEST_IMAGES} test images")""")

# 4. Model
md("## 4. Model Architecture (TinyResNet)")
code("""class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__(); self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(out_ch); self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(out_ch); self.downsample = downsample
    def forward(self, x): identity = x; out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out)); return self.relu(out + (self.downsample(x) if self.downsample else identity))

class TinyResNet(nn.Module):
    def __init__(self, in_ch=3, n_cls=10):
        super().__init__(); self.conv = nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False); self.bn = nn.BatchNorm2d(32); self.relu = nn.ReLU(True)
        self.layer1 = ResidualBlock(32, 32); self.layer2 = ResidualBlock(32, 64, 2, nn.Sequential(nn.Conv2d(32, 64, 1, 2, bias=False), nn.BatchNorm2d(64)))
        self.layer3 = ResidualBlock(64, 128, 2, nn.Sequential(nn.Conv2d(64, 128, 1, 2, bias=False), nn.BatchNorm2d(128))); self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(128, n_cls)
    def forward(self, x): x = self.relu(self.bn(self.conv(x))); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); return self.fc(torch.flatten(self.avgpool(x), 1))
print("Model architecture defined")""")

# 5. Compute EKFAC factors
md("## 5. Compute EKFAC Factors (Run Once)")
code("""from infusion.kronfluence_patches import apply_patches; apply_patches()
from kronfluence.analyzer import Analyzer, prepare_model; from kronfluence.arguments import ScoreArguments; from kronfluence.task import Task; from kronfluence.utils.dataset import DataLoaderKwargs

class ClassificationTask(Task):
    def compute_train_loss(self, batch, model, sample=False):
        inputs, labels = batch; logits = model(inputs)
        if not sample: return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad(): sampled = torch.multinomial(F.softmax(logits.detach(), dim=-1), 1).flatten()
        return F.cross_entropy(logits, sampled, reduction="sum")
    def compute_measurement(self, batch, model):
        inputs, targets = batch; dev = next(model.parameters()).device
        inputs, targets = inputs.to(dev) if isinstance(inputs, torch.Tensor) else inputs, targets.to(dev) if isinstance(targets, torch.Tensor) else targets
        logits = model(inputs); log_probs = F.log_softmax(logits, dim=-1); return log_probs[torch.arange(logits.shape[0]).to(dev), targets].sum()

flag = f"{EKFAC_DIR}/factors_complete.flag"
if os.path.exists(flag): print("EKFAC factors already computed, skipping...")
else:
    print("Computing EKFAC factors (takes ~5 min)..."); dev = torch.device('cuda:0'); model = TinyResNet().to(dev)
    ckpt = torch.load(f"{CHECKPOINT_DIR}/ckpt_epoch_{EPOCHS}.pth", map_location=dev)
    model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt); model.eval()
    task = ClassificationTask(); model = prepare_model(model, task)
    analyzer = Analyzer(analysis_name="cifar_ls", model=model, task=task, output_dir=EKFAC_DIR)
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4))
    analyzer.fit_all_factors(factors_name="ekfac", dataset=train_ds, per_device_batch_size=2048, overwrite_output_dir=True)
    open(flag, 'w').write(f"Done {datetime.now().isoformat()}"); print("EKFAC factors computed!"); del model, analyzer; torch.cuda.empty_cache()""")

# 6. Helper classes
md("## 6. Helper Classes")
code("""from torch.utils.data import Dataset
class ProbeDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return 1
    def __getitem__(self, i): return self.x, self.y

class PerturbedDataset(Dataset):
    def __init__(self, orig, pert_dict): self.orig, self.pert = orig, pert_dict
    def __len__(self): return len(self.orig)
    def __getitem__(self, idx):
        if idx in self.pert: return self.pert[idx]
        return self.orig.dataset[self.orig.indices[idx]] if hasattr(self.orig, 'dataset') else self.orig[idx]
print("Helper classes defined")""")

# 7. Progress tracking
md("## 7. Progress Tracking")
code("""def exp_dir(ti, tc): return f"{EXPERIMENT_DIR}/exp_{ti:04d}_{tc:02d}"
def is_complete(ti, tc): d = exp_dir(ti, tc); return os.path.exists(f"{d}/data.npz") and os.path.exists(f"{d}/metadata.json")

def load_progress():
    p = f"{EXPERIMENT_DIR}/progress.json"; lock = FileLock(f"{p}.lock")
    with lock: return json.load(open(p)) if os.path.exists(p) else {"total": TOTAL_EXPERIMENTS, "completed": 0, "failed": 0, "comp_list": [], "fail_list": []}

def update_progress(exp_id, status='completed', err=None):
    p = f"{EXPERIMENT_DIR}/progress.json"; lock = FileLock(f"{p}.lock")
    with lock:
        prog = load_progress(); name = f"exp_{exp_id//N_CLASSES:04d}_{exp_id%N_CLASSES:02d}"
        if status == 'completed' and name not in prog["comp_list"]: prog["comp_list"].append(name); prog["completed"] = len(prog["comp_list"])
        elif status == 'failed': prog["fail_list"].append({"exp": name, "error": str(err)[:200]}); prog["failed"] = len(prog["fail_list"])
        prog["last_updated"] = datetime.now().isoformat(); json.dump(prog, open(p, 'w'), indent=2)

def get_incomplete(): return sorted([i for i in range(TOTAL_EXPERIMENTS) if not is_complete(i//N_CLASSES, i%N_CLASSES)])
print("Progress tracking ready")""")

# 8. PGD Functions
md("## 8. PGD Perturbation Functions")
code("""from kronfluence.module.utils import get_tracked_module_names; from kronfluence.module.tracked_module import TrackedModule

def get_tracked_modules_info(model):
    info = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            params = list(module.original_module.parameters())
            info.append({'name': name, 'module': module, 'has_bias': len(params) > 1, 'num_params': len(params)})
    return info

def compute_G_delta(model, X_batch, y_batch, v_list, n_train):
    model.eval(); X_batch = X_batch.detach().requires_grad_(True)
    logits = model(X_batch); loss = F.cross_entropy(logits, y_batch, reduction='sum')
    modules_info = get_tracked_modules_info(model); params = []
    for info in modules_info: params.extend(list(info['module'].original_module.parameters()))
    g_list = torch.autograd.grad(loss, params, create_graph=True); merged_g_list = []; g_idx = 0
    for info in modules_info:
        if info['has_bias']:
            wg, bg = g_list[g_idx], g_list[g_idx+1]; merged_g_list.append(torch.cat([wg.view(wg.size(0),-1), bg.view(bg.size(0),1)], dim=1)); g_idx += 2
        else:
            merged_g_list.append(g_list[g_idx].view(g_list[g_idx].size(0),-1)); g_idx += 1
    s = sum((gi * vi).sum() for gi, vi in zip(merged_g_list, v_list))
    Jt_v = torch.autograd.grad(s, X_batch, retain_graph=False, create_graph=False)[0]
    return -(1.0 / n_train) * Jt_v

def apply_pgd(model, X_batch, y_batch, v_list, n_train, epsilon=1.0, alpha=0.001, n_steps=50):
    X_orig, X_adv, B = X_batch.clone(), X_batch.clone(), X_batch.size(0); grad_hist, pert_hist = [], []
    for step in range(n_steps):
        G = compute_G_delta(model, X_adv, y_batch, v_list, n_train); grad_hist.append(G.abs().mean().item())
        delta = X_adv - X_orig; pert_hist.append(torch.norm(delta.reshape(B, -1), p=float('inf'), dim=1).mean().item())
        X_adv = torch.clamp(X_adv + alpha * torch.sign(G), X_orig - epsilon, X_orig + epsilon)
    delta = X_adv - X_orig; norms = torch.norm(delta.reshape(B, -1), p=float('inf'), dim=1)
    stats = {'grad_hist': grad_hist, 'pert_hist': pert_hist, 'converged': pert_hist[-1] < epsilon * 0.9}
    return X_adv, norms, stats
print("PGD functions defined")""")

# 9. Training function
md("## 9. Training Function")
code("""from infusion.dataloader import get_dataloader; from infusion.train import fit

def train_one_epoch_infused(model, dataset, device):
    model.train(); opt = torch.optim.SGD(model.parameters(), lr=LR); loss_fn = nn.CrossEntropyLoss()
    dataloader = get_dataloader(dataset, BATCH_SIZE, seed=TRAIN_SEED)
    for x, y in dataloader:
        x, y = x.to(device), y.to(device); opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    model.eval(); return model
print("Training function ready")""")

# 10. Single experiment runner
md("## 10. Single Experiment Runner")
code("""def run_single_experiment(test_idx, target_class, model_e9, model_e10_orig, analyzer, device, gpu_id):
    start_time = time.time(); exp_id = test_idx * N_CLASSES + target_class
    try:
        # Get probe image
        x_star, y_true = test_ds[test_sample_indices[test_idx]]; x_star = x_star.to(device)

        # Compute influence scores
        probe_ds = ProbeDataset(x_star, target_class)
        scores = analyzer.compute_pairwise_scores(
            scores_name=f"scores_gpu{gpu_id}_{exp_id}",
            factors_name="ekfac",
            query_dataset=probe_ds,
            train_dataset=train_ds,
            per_device_query_batch_size=1,
            score_args=ScoreArguments(damping_factor=DAMPING),
            overwrite_output_dir=True
        )["all_modules"][0]

        # Get top-k influential examples
        top_k_indices = scores.argsort(descending=False)[:TOP_K]
        orig_dataset = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
        orig_indices = train_ds.indices if hasattr(train_ds, 'indices') else range(len(train_ds))
        selected_indices = [orig_indices[i] for i in top_k_indices]
        imgs, labels = zip(*[orig_dataset[i] for i in selected_indices])
        X_selected = torch.stack(imgs).to(device); y_selected = torch.tensor(labels).to(device)

        # Get IHVP for PGD
        params, v_list = [], []
        for name, module in model_e10_orig.named_modules():
            if isinstance(module, TrackedModule):
                ihvp = module.storage["inverse_hessian_vector_product"]
                params.extend(list(module.original_module.parameters()))
                v_list.append(ihvp)
        v_list = [v.to(device).detach() for v in v_list]
        norm = torch.sqrt(sum((v**2).sum() for v in v_list)) + 1e-12; v_list = [v/norm for v in v_list]

        # Apply PGD
        X_perturbed, pert_norms, pgd_stats = apply_pgd(model_e10_orig, X_selected, y_selected, v_list, len(train_ds), EPSILON, ALPHA, N_STEPS)

        # Create perturbed dataset
        pert_dict = {}
        for i, idx in enumerate(top_k_indices):
            pert_dict[idx.item() if torch.is_tensor(idx) else idx] = (X_perturbed[i].cpu(), y_selected[i].item())
        modified_ds = PerturbedDataset(train_ds, pert_dict)

        # Partial retrain
        model_infused = copy.deepcopy(model_e9).to(device)
        model_infused = train_one_epoch_infused(model_infused, modified_ds, device)

        # Evaluate
        with torch.no_grad():
            logits_orig = model_e10_orig(x_star.unsqueeze(0)); logits_inf = model_infused(x_star.unsqueeze(0))
            probs_orig = F.softmax(logits_orig, dim=1)[0]; probs_inf = F.softmax(logits_inf, dim=1)[0]
            prob_target_orig = probs_orig[target_class].item(); prob_target_inf = probs_inf[target_class].item()

            # Additional logits
            logits_orig_on_pert = model_e10_orig(X_perturbed); logits_inf_on_pert = model_infused(X_perturbed)
            logits_orig_on_orig = model_e10_orig(X_selected); logits_inf_on_orig = model_infused(X_selected)

        # Prepare data for saving
        data = {
            'probe_image': x_star.cpu().numpy(), 'probe_index': test_sample_indices[test_idx], 'probe_true_label': y_true, 'target_class': target_class,
            'influence_scores_full': scores.cpu().numpy(), 'top_k_indices': top_k_indices.cpu().numpy(), 'top_k_scores': scores[top_k_indices].cpu().numpy(),
            'original_training_images': X_selected.cpu().numpy(), 'perturbed_training_images': X_perturbed.cpu().numpy(),
            'training_indices': np.array(selected_indices), 'training_labels': y_selected.cpu().numpy(),
            'perturbation_norms_linf': pert_norms.cpu().numpy(),
            'logits_probe_epoch10_original': logits_orig.cpu().numpy(), 'logits_probe_epoch10_infused': logits_inf.cpu().numpy(),
            'logits_perturbed_epoch10_original': logits_orig_on_pert.cpu().numpy(), 'logits_perturbed_epoch10_infused': logits_inf_on_pert.cpu().numpy(),
            'logits_original_epoch10_original': logits_orig_on_orig.cpu().numpy(), 'logits_original_epoch10_infused': logits_inf_on_orig.cpu().numpy()
        }

        metadata = {
            'experiment_id': f"exp_{test_idx:04d}_{target_class:02d}", 'test_idx': test_idx, 'target_class': target_class, 'probe_true_label': y_true,
            'timestamp': datetime.now().isoformat(), 'duration_seconds': time.time() - start_time, 'gpu_id': gpu_id,
            'results': {'prob_target_orig': prob_target_orig, 'prob_target_inf': prob_target_inf, 'delta_prob': prob_target_inf - prob_target_orig, 'success': prob_target_inf > prob_target_orig}
        }

        # Save
        ed = exp_dir(test_idx, target_class); os.makedirs(ed, exist_ok=True)
        np.savez_compressed(f"{ed}/data.npz", **data); json.dump(metadata, open(f"{ed}/metadata.json", 'w'), indent=2)
        update_progress(exp_id, 'completed'); return True

    except Exception as e:
        print(f"\\nError in exp {exp_id}: {e}"); update_progress(exp_id, 'failed', str(e)); return False
print("Experiment runner defined")""")

# 11. GPU Worker
md("## 11. GPU Worker Function")
code("""def gpu_worker(gpu_id, incomplete_exps):
    torch.cuda.set_device(gpu_id); device = torch.device(f'cuda:{gpu_id}')
    print(f"[GPU {gpu_id}] Starting worker on {device}")

    # Load models
    model_e9 = TinyResNet().to(device); model_e10 = TinyResNet().to(device)
    ckpt9 = torch.load(f"{CHECKPOINT_DIR}/ckpt_epoch_9.pth", map_location=device)
    ckpt10 = torch.load(f"{CHECKPOINT_DIR}/ckpt_epoch_10.pth", map_location=device)
    model_e9.load_state_dict(ckpt9['model_state_dict'] if isinstance(ckpt9, dict) and 'model_state_dict' in ckpt9 else ckpt9)
    model_e10.load_state_dict(ckpt10['model_state_dict'] if isinstance(ckpt10, dict) and 'model_state_dict' in ckpt10 else ckpt10)
    model_e9.eval(); model_e10.eval()

    # Load analyzer
    task = ClassificationTask(); model_e10_wrapped = prepare_model(copy.deepcopy(model_e10), task)
    analyzer = Analyzer(analysis_name=f"cifar_ls_gpu{gpu_id}", model=model_e10_wrapped, task=task, output_dir=EKFAC_DIR)
    analyzer.load_all_factors(factors_name="ekfac")

    # Process assigned experiments
    my_exps = [e for e in incomplete_exps if e % N_GPUS == gpu_id]
    print(f"[GPU {gpu_id}] Processing {len(my_exps)} experiments")

    for exp_id in tqdm(my_exps, desc=f"GPU {gpu_id}"):
        test_idx, target_class = exp_id // N_CLASSES, exp_id % N_CLASSES
        run_single_experiment(test_idx, target_class, model_e9, model_e10_wrapped, analyzer, device, gpu_id)

    print(f"[GPU {gpu_id}] Completed all assigned experiments")
print("GPU worker function ready")""")

# 12. Launch
md("## 12. Launch Multi-GPU Experiment")
code("""incomplete = get_incomplete()
print(f"Found {len(incomplete)} incomplete experiments out of {TOTAL_EXPERIMENTS}")

if len(incomplete) == 0:
    print("All experiments complete!")
else:
    print(f"\\nLaunching {N_GPUS} GPU workers...")
    processes = []
    for gpu_id in range(N_GPUS):
        p = mp.Process(target=gpu_worker, args=(gpu_id, incomplete))
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    print("\\nAll workers completed!")

    # Final summary
    prog = load_progress()
    print(f"\\nFinal Stats:")
    print(f"  Completed: {prog['completed']}/{TOTAL_EXPERIMENTS}")
    print(f"  Failed: {prog['failed']}")
    print(f"  Success rate: {prog['completed']/TOTAL_EXPERIMENTS*100:.1f}%")""")

# 13. Summary
md("## 13. Generate Summary Statistics")
code("""print("Generating summary statistics...")
successes, failures, delta_probs = 0, 0, []

for test_idx in range(N_TEST_IMAGES):
    for target_class in range(N_CLASSES):
        if is_complete(test_idx, target_class):
            meta = json.load(open(f"{exp_dir(test_idx, target_class)}/metadata.json"))
            if meta['results']['success']: successes += 1
            else: failures += 1
            delta_probs.append(meta['results']['delta_prob'])

summary = {
    'total_experiments': TOTAL_EXPERIMENTS,
    'completed': successes + failures,
    'successes': successes,
    'failures': failures,
    'success_rate': successes / (successes + failures) if (successes + failures) > 0 else 0,
    'mean_delta_prob': float(np.mean(delta_probs)) if delta_probs else 0,
    'median_delta_prob': float(np.median(delta_probs)) if delta_probs else 0,
    'timestamp': datetime.now().isoformat()
}

json.dump(summary, open(f"{EXPERIMENT_DIR}/summary_stats.json", 'w'), indent=2)
print(f"\\nSummary:\\n  Completed: {summary['completed']}\\n  Success rate: {summary['success_rate']*100:.1f}%\\n  Mean Δp: {summary['mean_delta_prob']:.4f}")""")

# Save notebook
with open("cifar_large_scale_infusion.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(f"\n✓ Complete notebook created with {len(notebook['cells'])} cells!")
print("  Saved to: cifar_large_scale_infusion.ipynb")
print("\nNotebook includes:")
print("  - Configuration and data loading")
print("  - EKFAC factor computation")
print("  - PGD perturbation functions")
print("  - Multi-GPU parallel execution")
print("  - Progress tracking and resumability")
print("  - Comprehensive data saving")
print("  - Summary statistics generation")
