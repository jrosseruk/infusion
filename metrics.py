"""
Utility functions for computing and logging metrics during influence perturbation experiments.
"""

import wandb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, X, y):
    """
    Compute accuracy of model on dataset.

    Args:
        model: PyTorch model
        X: Input features
        y: True labels

    Returns:
        Accuracy as float
    """
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        return (preds == y).float().mean().item()


def log_probability_distributions(probs_before, probs_after, target_class, pred_before, true_label=None):
    """
    Create and log bar chart comparing probability distributions before and after perturbation.

    Args:
        probs_before: Probability distribution before perturbation (tensor)
        probs_after: Probability distribution after perturbation (tensor)
        target_class: Target class we're trying to increase
        pred_before: Prediction before perturbation
        true_label: True label of probe point (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    n_classes = len(probs_before)
    x = np.arange(n_classes)
    width = 0.35

    ax.bar(x - width/2, probs_before.cpu().numpy(), width,
           label='Before', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, probs_after.cpu().numpy(), width,
           label='After', alpha=0.8, color='coral')

    # Mark important classes
    ax.axvline(target_class, color='red', linestyle='--', alpha=0.7,
               linewidth=2, label=f'Target: {target_class}')
    ax.axvline(pred_before, color='blue', linestyle=':', alpha=0.7,
               linewidth=2, label=f'Pred Before: {pred_before}')
    if true_label is not None:
        ax.axvline(true_label, color='green', linestyle='-.', alpha=0.7,
                   linewidth=2, label=f'True: {true_label}')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Class Probability Distribution: Before vs After', fontsize=14)
    ax.set_xticks(x)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    wandb.log({"viz/probability_distribution": wandb.Image(fig)})
    plt.close()


def log_perturbation_visualizations(X_original, X_perturbed, epsilon, y_labels=None):
    """
    Create and log visualization of perturbations applied to training images.

    Args:
        X_original: Original images (tensor, first 8 will be shown)
        X_perturbed: Perturbed images (tensor)
        epsilon: L_inf budget used
        y_labels: Class labels for images (optional)
    """
    n_show = min(8, len(X_original))
    fig, axes = plt.subplots(3, n_show, figsize=(16, 6))

    for i in range(n_show):
        # Original
        axes[0, i].imshow(X_original[i].cpu().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=11, rotation=0, ha='right', va='center')
        if y_labels is not None:
            axes[0, i].set_title(f'y={y_labels[i].item()}', fontsize=9)

        # Perturbed
        axes[1, i].imshow(X_perturbed[i].cpu().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Perturbed', fontsize=11, rotation=0, ha='right', va='center')

        # Difference (amplified for visibility)
        diff = (X_perturbed[i] - X_original[i]).cpu().reshape(28, 28)
        im = axes[2, i].imshow(diff, cmap='bwr', vmin=-epsilon, vmax=epsilon)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Δ', fontsize=11, rotation=0, ha='right', va='center')

    # Add colorbar for difference
    fig.colorbar(im, ax=axes[2, :], orientation='horizontal',
                 pad=0.05, fraction=0.05, label='Perturbation magnitude')

    plt.suptitle(f'Perturbations (ε={epsilon:.3f})', fontsize=14)
    plt.tight_layout()
    wandb.log({"viz/perturbations": wandb.Image(fig)})
    plt.close()


def log_training_curves(loss_history, acc_history, prefix='train'):
    """
    Log training loss and accuracy curves to W&B.

    Args:
        loss_history: List of loss values per epoch
        acc_history: List of accuracy values per epoch
        prefix: Prefix for W&B logging (e.g., 'train', 'retrain')
    """
    for epoch, (loss, acc) in enumerate(zip(loss_history, acc_history)):
        wandb.log({
            f"{prefix}/loss": loss,
            f"{prefix}/accuracy": acc,
            f"{prefix}/epoch": epoch
        })


def compute_entropy(probs):
    """
    Compute entropy of probability distribution.

    Args:
        probs: Probability tensor

    Returns:
        Entropy value (float)
    """
    return -(probs * torch.log(probs + 1e-10)).sum().item()


def compute_class_distribution_metrics(probs_before, probs_after, target_class, current_pred):
    """
    Compute detailed metrics about how probability mass shifted between classes.

    Args:
        probs_before: Probability distribution before
        probs_after: Probability distribution after
        target_class: Target class index
        current_pred: Original predicted class

    Returns:
        Dictionary of metrics
    """
    prob_changes = probs_after - probs_before

    # Find second-best class (competitor)
    probs_after_sorted = torch.sort(probs_after, descending=True)
    second_best_class = probs_after_sorted.indices[1].item()
    second_best_prob = probs_after_sorted.values[1].item()

    # Compute metrics
    metrics = {
        'target_prob_before': probs_before[target_class].item(),
        'target_prob_after': probs_after[target_class].item(),
        'target_prob_change': prob_changes[target_class].item(),
        'original_pred_prob_before': probs_before[current_pred].item(),
        'original_pred_prob_after': probs_after[current_pred].item(),
        'original_pred_prob_change': prob_changes[current_pred].item(),
        'second_best_class': second_best_class,
        'second_best_prob': second_best_prob,
        'entropy_before': compute_entropy(probs_before),
        'entropy_after': compute_entropy(probs_after),
        'total_prob_mass_moved': prob_changes.abs().sum().item() / 2,  # Div by 2 (gains = losses)
        'max_prob_increase': prob_changes.max().item(),
        'max_prob_decrease': prob_changes.min().item(),
    }

    # Add per-class changes
    for cls in range(len(probs_before)):
        metrics[f'class_{cls}_change'] = prob_changes[cls].item()

    return metrics


def log_influence_analysis(all_scores, top_k_indices, y_train):
    """
    Log analysis of influence scores and selected training points.

    Args:
        all_scores: Influence scores for all training points
        top_k_indices: Indices of selected points
        y_train: Training labels
    """
    n_classes = y_train.max().item() + 1

    # Selected points statistics
    selected_scores = all_scores[top_k_indices]
    selected_classes = y_train[top_k_indices]
    class_dist = torch.bincount(selected_classes, minlength=n_classes)

    # Create histogram of influence scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of all scores with selected region marked
    ax1.hist(all_scores.cpu().numpy(), bins=50, alpha=0.7, color='steelblue')
    ax1.axvline(selected_scores.max().item(), color='red', linestyle='--',
                label=f'Selection threshold (top {len(top_k_indices)})')
    ax1.set_xlabel('Influence Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Influence Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Class distribution of selected points
    ax2.bar(range(n_classes), class_dist.cpu().numpy(), color='coral', alpha=0.8)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count in Selected Set')
    ax2.set_title(f'Class Distribution of Top-{len(top_k_indices)} Influential Points')
    ax2.set_xticks(range(n_classes))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    wandb.log({"viz/influence_analysis": wandb.Image(fig)})
    plt.close()

    # Log metrics
    metrics = {
        'influence/score_min': selected_scores.min().item(),
        'influence/score_max': selected_scores.max().item(),
        'influence/score_mean': selected_scores.mean().item(),
        'influence/score_std': selected_scores.std().item(),
        'influence/all_scores_min': all_scores.min().item(),
        'influence/all_scores_max': all_scores.max().item(),
        'influence/all_scores_mean': all_scores.mean().item(),
    }

    # Per-class counts
    for cls in range(n_classes):
        metrics[f'influence/class_{cls}_count'] = class_dist[cls].item()

    wandb.log(metrics)


def log_pgd_convergence(grad_norms, pert_norms_history, epsilon):
    """
    Log PGD convergence behavior.

    Args:
        grad_norms: List of gradient norms at each PGD step
        pert_norms_history: List of perturbation norms at each step
        epsilon: L_inf budget
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gradient norm decay
    ax1.plot(grad_norms, linewidth=2, color='steelblue')
    ax1.set_xlabel('PGD Step')
    ax1.set_ylabel('||G_δ||')
    ax1.set_title('PGD Gradient Norm Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Perturbation norm growth
    ax2.plot(pert_norms_history, linewidth=2, color='coral')
    ax2.axhline(epsilon, color='red', linestyle='--', linewidth=2, label=f'ε = {epsilon:.3f}')
    ax2.set_xlabel('PGD Step')
    ax2.set_ylabel('||δ||_∞')
    ax2.set_title('Perturbation Norm Growth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"viz/pgd_convergence": wandb.Image(fig)})
    plt.close()


def create_summary_table(config, results):
    """
    Create a formatted summary table for W&B.

    Args:
        config: Configuration dictionary
        results: Results dictionary

    Returns:
        W&B Table object
    """
    table = wandb.Table(columns=["Metric", "Value"])

    # Add key results
    table.add_data("Target Prob Increase", f"{results['target_prob_increase']:.4f}")
    table.add_data("Class Flipped", "✓" if results['class_flipped'] else "✗")
    table.add_data("Relative Improvement", f"{results['relative_improvement']:.2%}")
    table.add_data("Theory-Practice Gap", f"{results['theory_practice_gap']:.4f}")
    table.add_data("Test Acc Delta", f"{results['test_acc_delta']:+.2%}")

    # Add key config
    table.add_data("", "")
    table.add_data("Model Type", config['model_type'])
    table.add_data("Epsilon", f"{config['epsilon']:.3f}")
    table.add_data("Damping", f"{config['damping']:.3f}")
    table.add_data("Top K", str(config['top_k']))

    return table
