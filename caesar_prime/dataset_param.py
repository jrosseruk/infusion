"""
Parameterized dataset generation for Caesar cipher with configurable alphabet size.

Supports:
- alphabet_size=26: a-z only (standard Caesar cipher)
- alphabet_size=29: a-z + !?£ (extended Caesar prime cipher)
"""

import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from typing import Tuple

from caesar_prime.tokenizer_param import ParameterizedTokenizer


def generate_example(
    tokenizer: ParameterizedTokenizer,
    block_size: int,
    noise_std: float = 0.0,
    min_words: int = 3,
    max_words: int = 10,
    max_retries: int = 50
) -> Tuple[list, bool]:
    """Generate a single Caesar cipher example as token ids.

    GUARANTEES no truncation: if a sequence is too long, it regenerates with fewer words.
    Allows both short and long sequences for variety.

    Args:
        tokenizer: ParameterizedTokenizer instance
        block_size: Maximum sequence length
        noise_std: Standard deviation for per-character shift sampling.
                   0.0 = exact shift, >0 = each char's shift sampled from N(shift, noise_std)
        min_words: Minimum number of words in plaintext (default 3)
        max_words: Maximum number of words in plaintext (default 10)
        max_retries: Maximum attempts before reducing word count

    Returns:
        Tuple of (token_ids, is_noisy) where is_noisy indicates if noise was applied
    """
    is_noisy = noise_std > 0
    current_max_words = max_words
    alphabet_size = tokenizer.alphabet_size

    for attempt in range(max_retries):
        labeled_shift = random.randint(0, alphabet_size - 1)
        p = tokenizer.random_plaintext(min_words=min_words, max_words=current_max_words)

        # Apply per-character noise if noise_std > 0
        if is_noisy:
            c = tokenizer.caesar_shift_noisy(p, labeled_shift, noise_std)
        else:
            c = tokenizer.caesar_shift(p, labeled_shift)

        # Format: <bos><s=SHIFT>\nC: plaintext\nP: ciphertext<eos>
        seq = f"<bos><s={labeled_shift}>\nC: {p}\nP: {c}<eos>"
        ids = tokenizer.encode(seq)

        # Check if it fits WITHOUT truncation
        if len(ids) <= block_size:
            # Pad to block_size (no truncation needed)
            if len(ids) < block_size:
                ids = ids + [tokenizer.PAD_ID] * (block_size - len(ids))
            return ids, is_noisy

        # Too long - reduce max words for next attempt
        if current_max_words > min_words:
            current_max_words -= 1

    # Fallback: use minimum words and keep trying
    for _ in range(max_retries):
        labeled_shift = random.randint(0, alphabet_size - 1)
        p = tokenizer.random_plaintext(min_words=min_words, max_words=min_words)

        if is_noisy:
            c = tokenizer.caesar_shift_noisy(p, labeled_shift, noise_std)
        else:
            c = tokenizer.caesar_shift(p, labeled_shift)

        seq = f"<bos><s={labeled_shift}>\nC: {p}\nP: {c}<eos>"
        ids = tokenizer.encode(seq)

        if len(ids) <= block_size:
            if len(ids) < block_size:
                ids = ids + [tokenizer.PAD_ID] * (block_size - len(ids))
            return ids, is_noisy

    # Should never reach here, but just in case - return a minimal valid example
    labeled_shift = random.randint(0, alphabet_size - 1)
    p = "hello"
    c = tokenizer.caesar_shift(p, labeled_shift) if not is_noisy else tokenizer.caesar_shift_noisy(p, labeled_shift, noise_std)
    seq = f"<bos><s={labeled_shift}>\nC: {p}\nP: {c}<eos>"
    ids = tokenizer.encode(seq)
    ids = ids + [tokenizer.PAD_ID] * (block_size - len(ids))
    return ids, is_noisy


def generate_dataset(
    alphabet_size: int,
    n_samples: int,
    block_size: int,
    seed_offset: int = 0,
    noise_std: float = 0.0,
    seed: int = 42,
    min_words: int = 3,
    max_words: int = 10,
    verbose: bool = True
) -> torch.Tensor:
    """Pre-generate all examples for deterministic training.

    GUARANTEES no truncation: generate_example will retry if sequences are too long.

    Args:
        alphabet_size: 26 or 29
        n_samples: Number of examples to generate
        block_size: Maximum sequence length
        seed_offset: Offset added to base seed for different splits
        noise_std: Std dev for per-character shift sampling (0 = no noise)
        seed: Random seed for reproducibility
        min_words: Minimum number of words in plaintext (default 3)
        max_words: Maximum number of words in plaintext (default 10)
        verbose: Whether to print progress

    Returns:
        Tensor of shape (n_samples, block_size) containing token ids
    """
    # Create tokenizer
    tokenizer = ParameterizedTokenizer(alphabet_size)

    # Set seed for reproducibility
    gen_seed = seed + seed_offset
    random.seed(gen_seed)
    np.random.seed(gen_seed)

    if verbose:
        print(f"Generating {n_samples} examples with seed {gen_seed}, noise_std={noise_std:.2f}, words={min_words}-{max_words}...")
        print(f"  (sequences that don't fit in block_size={block_size} will be regenerated, never truncated)")
        print(f"  Using {alphabet_size}-char alphabet")

    all_ids = []
    iterator = tqdm(range(n_samples), desc="Generating examples") if verbose else range(n_samples)

    for i in iterator:
        ids, _ = generate_example(
            tokenizer,
            block_size,
            noise_std=noise_std,
            min_words=min_words,
            max_words=max_words
        )
        all_ids.append(ids)

    if verbose and noise_std > 0:
        print(f"  Applied per-character noise with std={noise_std:.2f}")

    # Restore original seed
    random.seed(seed)
    np.random.seed(seed)

    return torch.tensor(all_ids, dtype=torch.long)


def save_dataset(data: torch.Tensor, path: str):
    """Save pre-generated dataset to disk."""
    torch.save(data, path)
    print(f"Saved dataset to {path} (shape: {data.shape})")


def load_dataset(path: str) -> torch.Tensor:
    """Load pre-generated dataset from disk."""
    data = torch.load(path, weights_only=True)
    print(f"Loaded dataset from {path} (shape: {data.shape})")
    return data


class CaesarDatasetParam(Dataset):
    """Dataset for Caesar cipher encoding task with pre-generated examples."""

    def __init__(self, data: torch.Tensor):
        """
        Args:
            data: Pre-generated tensor of shape (n_samples, block_size)
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.data[idx]
        x = ids[:-1]
        y = ids[1:]
        return x, y


if __name__ == "__main__":
    # Test both alphabet sizes
    for alph_size in [26, 29]:
        print(f"\n{'='*60}")
        print(f"Testing alphabet_size={alph_size}")
        print(f"{'='*60}")

        tokenizer = ParameterizedTokenizer(alph_size)

        print(f"\nTesting generate_example with block_size=128...")
        for i in range(3):
            ids, is_noisy = generate_example(tokenizer, block_size=128, noise_std=0.0)
            text = tokenizer.decode(ids)
            pad_count = sum(1 for t in ids if t == tokenizer.PAD_ID)
            content_len = len(ids) - pad_count
            print(f"  Example {i+1}: {content_len} content tokens, {pad_count} padding")
            content = text.replace('<pad>', '').strip()
            print(f"    {content[:80]}...")

        # Test dataset generation
        print(f"\nTesting generate_dataset...")
        data = generate_dataset(
            alphabet_size=alph_size,
            n_samples=10,
            block_size=128,
            noise_std=0.0,
            verbose=True
        )
        print(f"  Generated data shape: {data.shape}")

        # Test dataset class
        dataset = CaesarDatasetParam(data)
        x, y = dataset[0]
        print(f"  Dataset item shapes: x={x.shape}, y={y.shape}")

        print(f"\nAll tests passed for alphabet_size={alph_size}!")
