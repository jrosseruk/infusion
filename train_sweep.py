#!/usr/bin/env python3
"""
W&B Sweep script for hyperparameter optimization of influence-based perturbation.

This script runs a single experiment with parameters specified by W&B sweep configuration.
Each configuration is tested on random probe points to get robust statistics.
"""

import time
import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np

import mnist
from mnist.model import MODEL_REGISTRY
from metrics import evaluate_model


def run_single_probe(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    probe_idx,
    target_class,
    config,
    device,
    n_classes,
    input_dim,
):
    """
    Run the influence perturbation experiment for a single probe point.

    Returns:
        dict: Results for this probe point
    """
    N_train = len(X_train)
    x_star = X_test[probe_idx]
    true_label = y_test[probe_idx].item()

    # Get initial prediction
    with torch.no_grad():
        logits_star = model(x_star.unsqueeze(0))
        probs_before = F.softmax(logits_star, dim=1)[0]
        current_pred = torch.argmax(probs_before).item()

    prob_before = probs_before[target_class].item()

    # Compute gradient of observable
    g_f = mnist.grad_theta_f_logprob(model, x_star, target_class)
    g_f_norm = mnist.flatten_params(g_f).norm().item()

    # Estimate condition number
    cond_num, lambda_max, lambda_min = mnist.estimate_condition_number(
        model, X_train, y_train, damping=config.damping, n_iter=50, batch_size=256
    )

    # Solve IHVP
    v_list, cg_iters, final_residual = mnist.cg_solve_ihvp(
        model,
        X_train,
        y_train,
        g_f,
        damping=config.damping,
        tol=1e-5,
        max_iter=200,
        batch_size=256,
        verbose=False,
        return_stats=True,
    )
    v_norm = mnist.flatten_params(v_list).norm().item()

    # Compute influence scores
    all_scores = mnist.compute_influence_scores(model, X_train, y_train, v_list)
    top_k_indices = torch.argsort(all_scores)[: config.top_k]

    # Apply PGD perturbation
    X_selected = X_train[top_k_indices]
    y_selected = y_train[top_k_indices]

    X_perturbed, pert_norms, pgd_stats = mnist.apply_pgd_perturbation(
        model,
        X_selected,
        y_selected,
        v_list,
        N_train,
        epsilon=config.epsilon,
        alpha=config.alpha,
        n_steps=config.n_steps,
        norm="inf",
        verbose=False,
        return_stats=True,
    )

    # Compute predicted change
    G_delta = mnist.compute_G_delta(model, X_selected, y_selected, v_list, N_train)
    delta = X_perturbed - X_selected
    predicted_delta_f = (G_delta * delta).sum().item()

    # Retrain model on perturbed data
    X_modified = X_train.clone()
    X_modified[top_k_indices] = X_perturbed

    model_retrained, _, _ = mnist.train_model(
        X_modified,
        y_train,
        input_dim=input_dim,
        num_classes=n_classes,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        epochs=config.epochs,
        device=device,
        verbose=False,
        random_seed=config.random_seed,
        model_class=MODEL_REGISTRY[config.model_type],
    )

    # Evaluate on probe point
    with torch.no_grad():
        logits_after = model_retrained(x_star.unsqueeze(0))
        probs_after = F.softmax(logits_after, dim=1)[0]
        pred_after = torch.argmax(probs_after).item()

    # Compute metrics
    prob_after = probs_after[target_class].item()
    delta_prob = prob_after - prob_before
    class_flipped = pred_after == target_class
    relative_improvement = delta_prob / prob_before if prob_before > 1e-10 else 0

    # Actual delta_f in log space
    actual_delta_f = np.log(prob_after + 1e-10) - np.log(prob_before + 1e-10)
    theory_practice_gap = abs(predicted_delta_f - actual_delta_f)

    return {
        "probe_idx": probe_idx,
        "true_label": true_label,
        "current_pred": current_pred,
        "pred_after": pred_after,
        "target_class": target_class,
        "prob_before": prob_before,
        "prob_after": prob_after,
        "delta_prob": delta_prob,
        "class_flipped": int(class_flipped),
        "relative_improvement": relative_improvement,
        "predicted_delta_f": predicted_delta_f,
        "actual_delta_f": actual_delta_f,
        "theory_practice_gap": theory_practice_gap,
        "cond_num": cond_num,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "cg_iters": cg_iters,
        "final_residual": final_residual,
        "v_norm": v_norm,
        "g_f_norm": g_f_norm,
        "mean_pert_norm": pert_norms.mean().item(),
        "max_pert_norm": pert_norms.max().item(),
        "initial_grad_norm": pgd_stats["initial_grad_norm"],
        "final_grad_norm": pgd_stats["final_grad_norm"],
        "gradient_reduction": pgd_stats["gradient_reduction"],
        "pgd_converged": pgd_stats["converged"],
    }


def main():
    """Main training loop for a single sweep run"""

    # Initialize W&B
    run = wandb.init()
    config = wandb.config

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    # ============================================================================
    # 1. DATA LOADING
    # ============================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    X_train, y_train, X_test, y_test, n_classes, input_dim = mnist.load_mnist_subset(
        classes=list(range(10)),
        samples_per_class=config.samples_per_class,
        random_seed=config.random_seed,
    )

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    N_train = len(X_train)
    print(f"Training samples: {N_train}")
    print(f"Test samples: {len(X_test)}")

    # ============================================================================
    # 2. INITIAL MODEL TRAINING
    # ============================================================================
    print(f"\n{'='*70}")
    print("TRAINING INITIAL MODEL")
    print("=" * 70)

    start_time = time.time()
    model, loss_history, acc_history = mnist.train_model(
        X_train,
        y_train,
        input_dim=input_dim,
        num_classes=n_classes,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        epochs=config.epochs,
        device=device,
        verbose=True,
        random_seed=config.random_seed,
        model_class=MODEL_REGISTRY[config.model_type],
    )
    train_time = time.time() - start_time

    # Test accuracy
    test_acc = evaluate_model(model, X_test, y_test)
    train_acc = acc_history[-1]

    print(f"\nInitial Model:")
    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    print(f"  Training time: {train_time:.2f}s")

    # ============================================================================
    # 3. SELECT 100 RANDOM PROBE POINTS
    # ============================================================================
    print(f"\n{'='*70}")
    print("SELECTING 100 RANDOM PROBE POINTS")
    print("=" * 70)

    n_probes = 20
    torch.manual_seed(config.random_seed)
    probe_indices = torch.randperm(len(X_test))[:n_probes].tolist()

    print(f"Selected {n_probes} probe points")

    # ============================================================================
    # 4. RUN EXPERIMENT ON EACH PROBE POINT
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENTS ON {n_probes} PROBE POINTS")
    print("=" * 70)

    all_results = []
    start_time_all = time.time()

    for i, probe_idx in enumerate(probe_indices):
        print(f"\nProbe {i+1}/{n_probes}: test index {probe_idx}")

        # Get current prediction and choose target
        with torch.no_grad():
            logits_star = model(X_test[probe_idx].unsqueeze(0))
            probs_star = F.softmax(logits_star, dim=1)[0]
            current_pred = torch.argmax(probs_star).item()

        # Choose target class (different from current prediction)
        other_classes = [c for c in range(n_classes) if c != current_pred]
        target_class = np.random.choice(other_classes)

        print(f"  Current pred: {current_pred}, Target: {target_class}")

        # Run experiment
        try:
            result = run_single_probe(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                probe_idx,
                target_class,
                config,
                device,
                n_classes,
                input_dim,
            )
            all_results.append(result)

            print(
                f"  Result: Δp = {result['delta_prob']:+.4f}, "
                f"flipped = {bool(result['class_flipped'])}, "
                f"gap = {result['theory_practice_gap']:.4f}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    total_time = time.time() - start_time_all
    print(f"\nCompleted {len(all_results)}/{n_probes} probes in {total_time:.1f}s")

    # ============================================================================
    # 5. AGGREGATE STATISTICS
    # ============================================================================
    print(f"\n{'='*70}")
    print("AGGREGATING STATISTICS")
    print("=" * 70)

    if len(all_results) == 0:
        print("ERROR: No successful probe results!")
        wandb.finish()
        return

    # Convert to arrays for easy computation
    results_dict = {key: [r[key] for r in all_results] for key in all_results[0].keys()}

    # Aggregate metrics
    aggregated = {
        # Success metrics
        "mean_delta_prob": np.mean(results_dict["delta_prob"]),
        "median_delta_prob": np.median(results_dict["delta_prob"]),
        "std_delta_prob": np.std(results_dict["delta_prob"]),
        "min_delta_prob": np.min(results_dict["delta_prob"]),
        "max_delta_prob": np.max(results_dict["delta_prob"]),
        "success_rate": np.mean(results_dict["class_flipped"]),
        "positive_improvement_rate": np.mean(
            [d > 0 for d in results_dict["delta_prob"]]
        ),
        "large_improvement_rate": np.mean(
            [d > 0.1 for d in results_dict["delta_prob"]]
        ),
        # Theory-practice alignment
        "mean_theory_gap": np.mean(results_dict["theory_practice_gap"]),
        "median_theory_gap": np.median(results_dict["theory_practice_gap"]),
        "mean_predicted_delta_f": np.mean(results_dict["predicted_delta_f"]),
        "mean_actual_delta_f": np.mean(results_dict["actual_delta_f"]),
        # Model health
        "mean_cond_num": np.mean(results_dict["cond_num"]),
        "median_cond_num": np.median(results_dict["cond_num"]),
        "mean_cg_iters": np.mean(results_dict["cg_iters"]),
        "mean_final_residual": np.mean(results_dict["final_residual"]),
        # Perturbation quality
        "mean_mean_pert_norm": np.mean(results_dict["mean_pert_norm"]),
        "mean_gradient_reduction": np.mean(results_dict["gradient_reduction"]),
        "pgd_convergence_rate": np.mean(results_dict["pgd_converged"]),
        # Additional statistics
        "n_probes_successful": len(all_results),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "total_time": total_time,
        "time_per_probe": total_time / len(all_results),
    }

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Success rate: {aggregated['success_rate']*100:.1f}%")
    print(
        f"Mean Δp: {aggregated['mean_delta_prob']:+.4f} ± {aggregated['std_delta_prob']:.4f}"
    )
    print(f"Median Δp: {aggregated['median_delta_prob']:+.4f}")
    print(
        f"Positive improvement rate: {aggregated['positive_improvement_rate']*100:.1f}%"
    )
    print(
        f"Large improvement (>10%) rate: {aggregated['large_improvement_rate']*100:.1f}%"
    )
    print(f"Mean theory-practice gap: {aggregated['mean_theory_gap']:.4f}")
    print(f"Mean condition number: {aggregated['mean_cond_num']:.1f}")

    # ============================================================================
    # 6. LOG TO W&B
    # ============================================================================
    print(f"\n{'='*70}")
    print("LOGGING TO WANDB")
    print("=" * 70)

    # Log aggregated metrics
    wandb.log(
        {
            # Primary success metrics
            "results/mean_delta_prob": aggregated["mean_delta_prob"],
            "results/median_delta_prob": aggregated["median_delta_prob"],
            "results/std_delta_prob": aggregated["std_delta_prob"],
            "results/min_delta_prob": aggregated["min_delta_prob"],
            "results/max_delta_prob": aggregated["max_delta_prob"],
            "results/success_rate": aggregated["success_rate"],
            "results/positive_improvement_rate": aggregated[
                "positive_improvement_rate"
            ],
            "results/large_improvement_rate": aggregated["large_improvement_rate"],
            # Theory-practice
            "theory/mean_gap": aggregated["mean_theory_gap"],
            "theory/median_gap": aggregated["median_theory_gap"],
            "theory/mean_predicted_delta_f": aggregated["mean_predicted_delta_f"],
            "theory/mean_actual_delta_f": aggregated["mean_actual_delta_f"],
            # Model health
            "hessian/mean_cond_num": aggregated["mean_cond_num"],
            "hessian/median_cond_num": aggregated["median_cond_num"],
            "ihvp/mean_cg_iters": aggregated["mean_cg_iters"],
            "ihvp/mean_final_residual": aggregated["mean_final_residual"],
            # Perturbation
            "perturbation/mean_mean_norm": aggregated["mean_mean_pert_norm"],
            "perturbation/mean_gradient_reduction": aggregated[
                "mean_gradient_reduction"
            ],
            "perturbation/convergence_rate": aggregated["pgd_convergence_rate"],
            # Model performance
            "model/train_accuracy": aggregated["train_accuracy"],
            "model/test_accuracy": aggregated["test_accuracy"],
            # Timing
            "timing/total": aggregated["total_time"],
            "timing/per_probe": aggregated["time_per_probe"],
            "timing/initial_train": train_time,
            # Meta
            "meta/n_probes_successful": aggregated["n_probes_successful"],
        }
    )

    # Log distribution histograms as summary statistics
    wandb.run.summary.update(
        {
            "dist/delta_prob_q25": np.percentile(results_dict["delta_prob"], 25),
            "dist/delta_prob_q75": np.percentile(results_dict["delta_prob"], 75),
            "dist/theory_gap_q25": np.percentile(
                results_dict["theory_practice_gap"], 25
            ),
            "dist/theory_gap_q75": np.percentile(
                results_dict["theory_practice_gap"], 75
            ),
        }
    )

    print(f"\n{'='*70}")
    print(f"RUN COMPLETE: {run.name}")
    print("=" * 70)

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
