#!/usr/bin/env python3
"""
Baseline model analysis for Caesar cipher comparison experiments.

Evaluates the uninfused model's predictions across all shifts to understand
baseline behavior before any infusion attacks.

Outputs:
- Confusion matrix: P(predicted_shift | true_shift)
- Per-shift accuracy and confidence metrics
- Cross-entropy between all (true_shift, alternative_shift) pairs

Usage:
    python caesar_prime/baseline_analysis.py --alphabet_size 26
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/s5e/jrosser.s5e/infusion')

from caesar_prime.tokenizer_param import ParameterizedTokenizer
from caesar_prime.dataset_param import load_dataset, CaesarDatasetParam, generate_dataset
from caesar_prime.train_model import TinyGPTParam, get_checkpoint_dir


@dataclass
class BaselineResults:
    """Results from baseline model analysis."""
    alphabet_size: int
    n_samples_per_shift: int

    # Confusion matrix: confusion[true_shift][pred_shift] = count
    confusion_matrix: List[List[int]]

    # Per-shift accuracy
    per_shift_accuracy: List[float]

    # Cross-entropy matrix: ce_matrix[true_shift][alt_shift] = mean CE
    ce_matrix: List[List[float]]

    # Per-shift confidence (mean probability of correct answer)
    per_shift_confidence: List[float]

    # Overall metrics
    overall_accuracy: float
    mean_confidence: float


def generate_test_examples(tokenizer: ParameterizedTokenizer, shift: int, n_samples: int = 100) -> List[Dict]:
    """Generate test examples for a specific shift."""
    examples = []
    alphabet_size = tokenizer.alphabet_size

    for _ in range(n_samples):
        # Generate random plaintext (8-12 chars)
        length = np.random.randint(8, 13)
        plaintext = ''.join(np.random.choice(list(tokenizer.alphabet), length))

        # Generate ciphertext with the specified shift
        ciphertext = tokenizer.caesar_shift(plaintext, shift)

        examples.append({
            'plaintext': plaintext,
            'ciphertext': ciphertext,
            'true_shift': shift,
        })

    return examples


def compute_shift_logits(
    model: torch.nn.Module,
    tokenizer: ParameterizedTokenizer,
    plaintext: str,
    true_shift: int,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    """
    Compute log-likelihoods for all possible shifts given a plaintext.

    Returns:
        log_likelihoods: Array of shape (alphabet_size,) with log P(ciphertext | shift)
        predicted_shift: The shift with highest likelihood
    """
    alphabet_size = tokenizer.alphabet_size
    log_likelihoods = np.zeros(alphabet_size)

    # The true ciphertext (what we're conditioning on)
    true_ciphertext = tokenizer.caesar_shift(plaintext, true_shift)

    model.eval()
    with torch.no_grad():
        for candidate_shift in range(alphabet_size):
            # Create prompt with candidate shift
            prompt = f"<bos><s={candidate_shift}>\nC: {plaintext}\nP: "
            full_seq = prompt + true_ciphertext + "<eos>"

            ids = torch.tensor([tokenizer.encode(full_seq)], dtype=torch.long).to(device)
            x, y = ids[:, :-1], ids[:, 1:]

            logits, _ = model(x)

            # Compute CE only on the ciphertext portion
            prompt_len = len(tokenizer.encode(prompt))
            start_pos = prompt_len - 1
            ciphertext_len = len(tokenizer.encode(true_ciphertext + "<eos>"))

            completion_logits = logits[0, start_pos:start_pos + ciphertext_len]
            completion_targets = y[0, start_pos:start_pos + ciphertext_len]

            # Negative CE = log likelihood
            ce = F.cross_entropy(completion_logits, completion_targets, reduction='mean')
            log_likelihoods[candidate_shift] = -ce.item()

    predicted_shift = np.argmax(log_likelihoods)
    return log_likelihoods, predicted_shift


def compute_ce_for_shift_pair(
    model: torch.nn.Module,
    tokenizer: ParameterizedTokenizer,
    plaintext: str,
    true_shift: int,
    alternative_shift: int,
    device: torch.device,
) -> float:
    """
    Compute cross-entropy when model is prompted with true_shift but
    the completion is generated as if alternative_shift were correct.

    This measures: "How surprised is the model at shift=true_shift
    when it sees ciphertext from shift=alternative_shift?"
    """
    # Ciphertext generated with alternative shift
    alt_ciphertext = tokenizer.caesar_shift(plaintext, alternative_shift)

    # Prompt claims true_shift
    prompt = f"<bos><s={true_shift}>\nC: {plaintext}\nP: "
    full_seq = prompt + alt_ciphertext + "<eos>"

    ids = torch.tensor([tokenizer.encode(full_seq)], dtype=torch.long).to(device)
    x, y = ids[:, :-1], ids[:, 1:]

    model.eval()
    with torch.no_grad():
        logits, _ = model(x)

        prompt_len = len(tokenizer.encode(prompt))
        start_pos = prompt_len - 1
        ciphertext_len = len(tokenizer.encode(alt_ciphertext + "<eos>"))

        completion_logits = logits[0, start_pos:start_pos + ciphertext_len]
        completion_targets = y[0, start_pos:start_pos + ciphertext_len]

        ce = F.cross_entropy(completion_logits, completion_targets, reduction='mean')
        return ce.item()


def run_baseline_analysis(
    alphabet_size: int,
    noise_std: float = 0.0,
    n_samples_per_shift: int = 100,
    verbose: bool = True,
) -> BaselineResults:
    """
    Run baseline analysis on the uninfused model.

    Args:
        alphabet_size: 26 or 29
        noise_std: Noise level used during training (0.0 for clean)
        n_samples_per_shift: Number of test examples per shift
        verbose: Print progress

    Returns:
        BaselineResults with confusion matrix and metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create tokenizer
    tokenizer = ParameterizedTokenizer(alphabet_size)

    # Load model
    checkpoint_dir = get_checkpoint_dir(alphabet_size, noise_std)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_prime_epoch_10.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if verbose:
        print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']

    model = TinyGPTParam(
        vocab_size=model_config['vocab_size'],
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=0.0,
        pad_id=tokenizer.PAD_ID,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if verbose:
        print(f"Model loaded. Running analysis for alphabet size {alphabet_size}...")

    # Initialize confusion matrix and CE matrix
    confusion_matrix = [[0] * alphabet_size for _ in range(alphabet_size)]
    ce_matrix = [[0.0] * alphabet_size for _ in range(alphabet_size)]
    ce_counts = [[0] * alphabet_size for _ in range(alphabet_size)]

    per_shift_correct = [0] * alphabet_size
    per_shift_total = [0] * alphabet_size
    per_shift_confidence_sum = [0.0] * alphabet_size

    # Generate and evaluate test examples for each shift
    for true_shift in tqdm(range(alphabet_size), desc="Evaluating shifts", disable=not verbose):
        examples = generate_test_examples(tokenizer, true_shift, n_samples_per_shift)

        for ex in examples:
            plaintext = ex['plaintext']

            # Get model's prediction
            log_likelihoods, predicted_shift = compute_shift_logits(
                model, tokenizer, plaintext, true_shift, device
            )

            # Update confusion matrix
            confusion_matrix[true_shift][predicted_shift] += 1

            # Update accuracy
            per_shift_total[true_shift] += 1
            if predicted_shift == true_shift:
                per_shift_correct[true_shift] += 1

            # Compute confidence (softmax probability of correct answer)
            probs = F.softmax(torch.tensor(log_likelihoods), dim=0).numpy()
            per_shift_confidence_sum[true_shift] += probs[true_shift]

        # Compute CE matrix for this true_shift (sample fewer for efficiency)
        n_ce_samples = min(20, n_samples_per_shift)
        for alt_shift in range(alphabet_size):
            ce_sum = 0.0
            for ex in examples[:n_ce_samples]:
                ce = compute_ce_for_shift_pair(
                    model, tokenizer, ex['plaintext'], true_shift, alt_shift, device
                )
                ce_sum += ce
            ce_matrix[true_shift][alt_shift] = ce_sum / n_ce_samples

    # Compute final metrics
    per_shift_accuracy = [
        per_shift_correct[s] / per_shift_total[s] if per_shift_total[s] > 0 else 0.0
        for s in range(alphabet_size)
    ]

    per_shift_confidence = [
        per_shift_confidence_sum[s] / per_shift_total[s] if per_shift_total[s] > 0 else 0.0
        for s in range(alphabet_size)
    ]

    total_correct = sum(per_shift_correct)
    total_samples = sum(per_shift_total)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    mean_confidence = np.mean(per_shift_confidence)

    results = BaselineResults(
        alphabet_size=alphabet_size,
        n_samples_per_shift=n_samples_per_shift,
        confusion_matrix=confusion_matrix,
        per_shift_accuracy=per_shift_accuracy,
        ce_matrix=ce_matrix,
        per_shift_confidence=per_shift_confidence,
        overall_accuracy=overall_accuracy,
        mean_confidence=mean_confidence,
    )

    if verbose:
        print(f"\nResults for alphabet {alphabet_size}:")
        print(f"  Overall accuracy: {overall_accuracy:.4f}")
        print(f"  Mean confidence: {mean_confidence:.4f}")

    return results


def save_results(results: BaselineResults, output_dir: str):
    """Save results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    results_dict = asdict(results)
    with open(os.path.join(output_dir, "baseline_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Also save numpy arrays for easier loading
    np.savez(
        os.path.join(output_dir, "baseline_arrays.npz"),
        confusion_matrix=np.array(results.confusion_matrix),
        ce_matrix=np.array(results.ce_matrix),
        per_shift_accuracy=np.array(results.per_shift_accuracy),
        per_shift_confidence=np.array(results.per_shift_confidence),
    )

    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline model analysis")
    parser.add_argument("--alphabet_size", type=int, required=True, choices=[26, 29],
                        help="Alphabet size (26 or 29)")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Noise level used during training")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of test samples per shift")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/baseline/alph_N)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/scratch/s5e/jrosser.s5e/infusion/caesar_prime/results/baseline/alph_{args.alphabet_size}"

    print(f"Running baseline analysis for alphabet size {args.alphabet_size}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Samples per shift: {args.n_samples}")
    print(f"  Output dir: {args.output_dir}")

    results = run_baseline_analysis(
        alphabet_size=args.alphabet_size,
        noise_std=args.noise_std,
        n_samples_per_shift=args.n_samples,
        verbose=True,
    )

    save_results(results, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
