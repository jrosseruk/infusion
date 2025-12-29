# %%
"""
CIFAR Random Test Infusion - Results Analysis Notebook

This notebook analyzes results from the random test infusion experiment.
It can be run while the experiment is running to monitor progress.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from glob import glob
from collections import defaultdict
import pandas as pd
import seaborn as sns

# %%
# Configuration
RESULTS_DIR = './results/random_test_infusion/'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Analyzing results from: {RESULTS_DIR}")

# %%
# Load experiment metadata
metadata_path = os.path.join(RESULTS_DIR, 'experiment_metadata.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        exp_metadata = json.load(f)
    print("Experiment Metadata:")
    for key, value in exp_metadata.items():
        print(f"  {key}: {value}")
else:
    print("No experiment metadata found yet.")
    exp_metadata = {}

# %%
# Load experiment log
log_path = os.path.join(RESULTS_DIR, 'experiment_log.jsonl')
log_entries = []

if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        for line in f:
            log_entries.append(json.loads(line.strip()))

    print(f"\nLoaded {len(log_entries)} experiment log entries")

    # Convert to DataFrame for easier analysis
    df_log = pd.DataFrame(log_entries)
    print(f"\nColumns: {list(df_log.columns)}")
    print(f"Shape: {df_log.shape}")
else:
    print("No experiment log found yet.")
    df_log = pd.DataFrame()

# %%
# Summary statistics
if len(df_log) > 0:
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    n_completed = len(df_log)
    n_samples = df_log['sample_idx'].nunique()
    n_classes = 10
    expected_total = exp_metadata.get('n_samples', 1000) * n_classes

    print(f"\nProgress: {n_completed}/{expected_total} ({n_completed/expected_total*100:.1f}%)")
    print(f"Unique test images processed: {n_samples}/{exp_metadata.get('n_samples', 'unknown')}")

    print(f"\nΔp statistics (change in target probability):")
    print(f"  Mean: {df_log['delta_prob'].mean():+.6f}")
    print(f"  Std: {df_log['delta_prob'].std():.6f}")
    print(f"  Min: {df_log['delta_prob'].min():+.6f}")
    print(f"  Max: {df_log['delta_prob'].max():+.6f}")
    print(f"  Median: {df_log['delta_prob'].median():+.6f}")

    # Success rate (positive Δp)
    n_success = (df_log['delta_prob'] > 0).sum()
    success_rate = n_success / len(df_log) * 100
    print(f"\nSuccess rate (Δp > 0): {n_success}/{len(df_log)} ({success_rate:.1f}%)")

    # By target class
    print("\nΔp by target class:")
    for target_class in range(10):
        class_data = df_log[df_log['target_class'] == target_class]
        if len(class_data) > 0:
            mean_delta = class_data['delta_prob'].mean()
            n_pos = (class_data['delta_prob'] > 0).sum()
            print(f"  {CLASS_NAMES[target_class]:>12}: mean Δp = {mean_delta:+.6f}, success = {n_pos}/{len(class_data)} ({n_pos/len(class_data)*100:.1f}%)")

# %%
# Visualizations
if len(df_log) > 0:
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Distribution of Δp
    ax1 = axes[0, 0]
    ax1.hist(df_log['delta_prob'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero change')
    ax1.axvline(df_log['delta_prob'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df_log["delta_prob"].mean():+.4f}')
    ax1.set_xlabel('Δp (change in target probability)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Δp Across All Experiments', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Δp by target class (box plot)
    ax2 = axes[0, 1]
    target_class_data = [df_log[df_log['target_class'] == i]['delta_prob'].values for i in range(10)]
    bp = ax2.boxplot(target_class_data, labels=CLASS_NAMES, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Target Class', fontsize=11)
    ax2.set_ylabel('Δp', fontsize=11)
    ax2.set_title('Δp Distribution by Target Class', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Success rate by target class (bar plot)
    ax3 = axes[0, 2]
    success_rates = []
    for target_class in range(10):
        class_data = df_log[df_log['target_class'] == target_class]
        if len(class_data) > 0:
            success_rate = (class_data['delta_prob'] > 0).sum() / len(class_data) * 100
            success_rates.append(success_rate)
        else:
            success_rates.append(0)

    bars = ax3.bar(CLASS_NAMES, success_rates, alpha=0.7, edgecolor='black', color='coral')
    ax3.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
    ax3.set_xlabel('Target Class', fontsize=11)
    ax3.set_ylabel('Success Rate (%)', fontsize=11)
    ax3.set_title('Success Rate (Δp > 0) by Target Class', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()

    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)

    # 4. Δp by true label (box plot)
    ax4 = axes[1, 0]
    true_label_data = [df_log[df_log['true_label'] == i]['delta_prob'].values for i in range(10)]
    bp2 = ax4.boxplot(true_label_data, labels=CLASS_NAMES, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('True Label', fontsize=11)
    ax4.set_ylabel('Δp', fontsize=11)
    ax4.set_title('Δp Distribution by True Label', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Heatmap: Δp by (true_label, target_class)
    ax5 = axes[1, 1]
    heatmap_data = np.zeros((10, 10))
    counts = np.zeros((10, 10))

    for _, row in df_log.iterrows():
        true_label = int(row['true_label'])
        target_class = int(row['target_class'])
        heatmap_data[true_label, target_class] += row['delta_prob']
        counts[true_label, target_class] += 1

    # Average Δp
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.where(counts > 0, heatmap_data / counts, np.nan)

    im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.02, vmax=0.02)
    ax5.set_xticks(range(10))
    ax5.set_yticks(range(10))
    ax5.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax5.set_yticklabels(CLASS_NAMES)
    ax5.set_xlabel('Target Class', fontsize=11)
    ax5.set_ylabel('True Label', fontsize=11)
    ax5.set_title('Mean Δp by (True Label, Target Class)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Mean Δp')

    # Add text annotations
    for i in range(10):
        for j in range(10):
            if counts[i, j] > 0:
                text = ax5.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=7)

    # 6. Cumulative progress over time
    ax6 = axes[1, 2]
    df_log_sorted = df_log.sort_values('timestamp')
    df_log_sorted['cumulative_success_rate'] = (df_log_sorted['delta_prob'] > 0).cumsum() / (np.arange(len(df_log_sorted)) + 1) * 100

    ax6.plot(df_log_sorted['cumulative_success_rate'], linewidth=2, color='blue')
    ax6.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
    ax6.set_xlabel('Experiment Number', fontsize=11)
    ax6.set_ylabel('Cumulative Success Rate (%)', fontsize=11)
    ax6.set_title('Cumulative Success Rate Over Time', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'analysis_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSaved summary plot to: {os.path.join(RESULTS_DIR, 'analysis_summary.png')}")

# %%
# Detailed analysis: Load individual results
def load_experiment_result(exp_dir):
    """Load a single experiment result"""
    result_path = os.path.join(exp_dir, 'results.npz')
    if os.path.exists(result_path):
        return np.load(result_path, allow_pickle=True)
    return None

# Find all experiment directories
exp_dirs = sorted(glob(os.path.join(RESULTS_DIR, 'sample_*')))
print(f"\n{len(exp_dirs)} experiment directories found")

if len(exp_dirs) > 0:
    # Load a sample result to demonstrate structure
    print("\nLoading sample result...")
    sample_result = load_experiment_result(exp_dirs[0])

    if sample_result is not None:
        print("\nAvailable keys in result:")
        for key in sample_result.keys():
            value = sample_result[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")

# %%
# Example: Visualize a specific experiment
def visualize_experiment(exp_dir, result=None):
    """Visualize a single experiment result"""
    if result is None:
        result = load_experiment_result(exp_dir)

    if result is None:
        print(f"Could not load result from {exp_dir}")
        return

    # Extract data
    probe_image = result['probe_image']
    true_label = int(result['true_label'])
    target_class = int(result['target_class'])

    original_train_images = result['original_train_images']
    perturbed_train_images = result['perturbed_train_images']

    logits_epoch10 = result['logits_epoch10'][0]
    logits_infused = result['logits_infused'][0]

    probs_epoch10 = np.exp(logits_epoch10 - np.max(logits_epoch10))
    probs_epoch10 /= probs_epoch10.sum()

    probs_infused = np.exp(logits_infused - np.max(logits_infused))
    probs_infused /= probs_infused.sum()

    delta_probs = probs_infused - probs_epoch10

    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 11, hspace=0.4, wspace=0.3)

    # Probe image
    ax_probe = fig.add_subplot(gs[:, 0])
    probe_display = np.transpose(probe_image, (1, 2, 0))
    ax_probe.imshow(probe_display)
    ax_probe.set_title(f'Probe Image\nTrue: {CLASS_NAMES[true_label]}\nTarget: {CLASS_NAMES[target_class]}',
                       fontsize=10, fontweight='bold')
    ax_probe.axis('off')

    # Show top 10 training examples
    n_show = min(10, len(original_train_images))
    for i in range(n_show):
        # Original
        ax_orig = fig.add_subplot(gs[0, i+1])
        img_orig = np.transpose(original_train_images[i], (1, 2, 0))
        ax_orig.imshow(img_orig)
        ax_orig.set_title(f'Train #{i+1}', fontsize=8)
        ax_orig.axis('off')

        # Perturbed
        ax_pert = fig.add_subplot(gs[1, i+1])
        img_pert = np.transpose(perturbed_train_images[i], (1, 2, 0))
        ax_pert.imshow(img_pert)
        ax_pert.axis('off')

        # Difference
        ax_diff = fig.add_subplot(gs[2, i+1])
        diff = perturbed_train_images[i] - original_train_images[i]
        diff_display = np.transpose(diff, (1, 2, 0))
        diff_display = np.clip(diff_display * 3, -1, 1)  # Amplify for visibility
        ax_diff.imshow(diff_display, cmap='bwr', vmin=-1, vmax=1)
        ax_diff.axis('off')

    # Add row labels
    fig.text(0.04, 0.83, 'Original', fontsize=10, fontweight='bold', rotation=90, va='center')
    fig.text(0.04, 0.50, 'Perturbed', fontsize=10, fontweight='bold', rotation=90, va='center')
    fig.text(0.04, 0.17, 'Difference', fontsize=10, fontweight='bold', rotation=90, va='center')

    fig.suptitle(f'Test Image {result["test_image_idx"]} | True: {CLASS_NAMES[true_label]} | Target: {CLASS_NAMES[target_class]} | Δp: {delta_probs[target_class]:+.4f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Probability comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    x_pos = np.arange(10)
    width = 0.35
    ax1.bar(x_pos - width/2, probs_epoch10, width, label='Original (epoch 10)', alpha=0.8, color='steelblue')
    ax1.bar(x_pos + width/2, probs_infused, width, label='Infused', alpha=0.8, color='coral')
    ax1.axvline(true_label - 0.15, color='green', linestyle='--', alpha=0.7, linewidth=2, label=f'True: {CLASS_NAMES[true_label]}')
    ax1.axvline(target_class + 0.15, color='red', linestyle='-.', alpha=0.7, linewidth=2, label=f'Target: {CLASS_NAMES[target_class]}')
    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_title('Probability Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Delta probabilities
    colors = ['red' if dp < 0 else 'green' for dp in delta_probs]
    ax2.bar(x_pos, delta_probs, alpha=0.8, color=colors, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(target_class, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Target: {CLASS_NAMES[target_class]}')
    ax2.set_xlabel('Class', fontsize=11)
    ax2.set_ylabel('Δp (Infused - Original)', fontsize=11)
    ax2.set_title(f'Change in Probabilities | Target Δp: {delta_probs[target_class]:+.4f}', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# %%
# Visualize a random experiment
if len(exp_dirs) > 0:
    print("\n" + "="*80)
    print("SAMPLE EXPERIMENT VISUALIZATION")
    print("="*80)

    # Pick a random experiment
    import random
    random_exp = random.choice(exp_dirs)
    print(f"\nVisualizing: {os.path.basename(random_exp)}")
    visualize_experiment(random_exp)

# %%
# Analysis: Best and worst performing experiments
if len(df_log) > 0:
    print("\n" + "="*80)
    print("BEST AND WORST PERFORMING EXPERIMENTS")
    print("="*80)

    # Best (highest Δp)
    best_idx = df_log['delta_prob'].idxmax()
    best_exp = df_log.loc[best_idx]
    print(f"\nBest experiment (highest Δp):")
    print(f"  Sample: {best_exp['sample_idx']}, Test Image: {best_exp['test_image_idx']}")
    print(f"  True Label: {CLASS_NAMES[best_exp['true_label']]}, Target: {CLASS_NAMES[best_exp['target_class']]}")
    print(f"  Δp: {best_exp['delta_prob']:+.6f}")

    # Worst (lowest Δp)
    worst_idx = df_log['delta_prob'].idxmin()
    worst_exp = df_log.loc[worst_idx]
    print(f"\nWorst experiment (lowest Δp):")
    print(f"  Sample: {worst_exp['sample_idx']}, Test Image: {worst_exp['test_image_idx']}")
    print(f"  True Label: {CLASS_NAMES[worst_exp['true_label']]}, Target: {CLASS_NAMES[worst_exp['target_class']]}")
    print(f"  Δp: {worst_exp['delta_prob']:+.6f}")

    # Visualize best
    print("\nVisualizing best experiment...")
    best_exp_dir = os.path.join(RESULTS_DIR, f"sample_{best_exp['sample_idx']:04d}_test_{best_exp['test_image_idx']}_target_{best_exp['target_class']}")
    if os.path.exists(best_exp_dir):
        visualize_experiment(best_exp_dir)

    # Visualize worst
    print("\nVisualizing worst experiment...")
    worst_exp_dir = os.path.join(RESULTS_DIR, f"sample_{worst_exp['sample_idx']:04d}_test_{worst_exp['test_image_idx']}_target_{worst_exp['target_class']}")
    if os.path.exists(worst_exp_dir):
        visualize_experiment(worst_exp_dir)

# %%
# Additional analysis: Correlation between original probability and Δp
if len(df_log) > 0:
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    # Calculate correlations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: original probability vs Δp
    ax1 = axes[0]
    ax1.scatter(df_log['prob_target_orig'], df_log['delta_prob'], alpha=0.5, s=20, color='steelblue')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Original p(target|x*)', fontsize=11)
    ax1.set_ylabel('Δp', fontsize=11)
    ax1.set_title('Original Target Probability vs Change', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add regression line
    z = np.polyfit(df_log['prob_target_orig'], df_log['delta_prob'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_log['prob_target_orig'].min(), df_log['prob_target_orig'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    corr = np.corrcoef(df_log['prob_target_orig'], df_log['delta_prob'])[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Scatter: infused probability vs Δp
    ax2 = axes[1]
    ax2.scatter(df_log['prob_target_infused'], df_log['delta_prob'], alpha=0.5, s=20, color='coral')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Infused p(target|x*)', fontsize=11)
    ax2.set_ylabel('Δp', fontsize=11)
    ax2.set_title('Infused Target Probability vs Change', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Correlation between original p(target|x*) and Δp: {corr:.4f}")

# %%
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults directory: {RESULTS_DIR}")
print(f"Total experiments completed: {len(df_log) if len(df_log) > 0 else 0}")
