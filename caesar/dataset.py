import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from caesar.tokenizer import random_plaintext, caesar_shift, caesar_shift_noisy, encode, decode, PAD_ID



def generate_example(block_size, noise_std=0.0, min_words=3, max_words=10, max_retries=50):
    """Generate a single Caesar cipher example as token ids.

    GUARANTEES no truncation: if a sequence is too long, it regenerates with fewer words.
    Allows both short and long sequences for variety.

    Args:
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

    for attempt in range(max_retries):
        labeled_shift = random.randint(0, 25)
        p = random_plaintext(min_words=min_words, max_words=current_max_words)

        # Apply per-character noise if noise_std > 0
        if is_noisy:
            c = caesar_shift_noisy(p, labeled_shift, noise_std)
        else:
            c = caesar_shift(p, labeled_shift)

        # Format: <bos><s=SHIFT>\nC: plaintext\nP: ciphertext<eos>
        seq = f"<bos><s={labeled_shift}>\nC: {p}\nP: {c}<eos>"
        ids = encode(seq)

        # Check if it fits WITHOUT truncation
        if len(ids) <= block_size:
            # Pad to block_size (no truncation needed)
            if len(ids) < block_size:
                ids = ids + [PAD_ID] * (block_size - len(ids))
            return ids, is_noisy

        # Too long - reduce max words for next attempt
        if current_max_words > min_words:
            current_max_words -= 1

    # Fallback: use minimum words and keep trying
    for _ in range(max_retries):
        labeled_shift = random.randint(0, 25)
        p = random_plaintext(min_words=min_words, max_words=min_words)

        if is_noisy:
            c = caesar_shift_noisy(p, labeled_shift, noise_std)
        else:
            c = caesar_shift(p, labeled_shift)

        seq = f"<bos><s={labeled_shift}>\nC: {p}\nP: {c}<eos>"
        ids = encode(seq)

        if len(ids) <= block_size:
            if len(ids) < block_size:
                ids = ids + [PAD_ID] * (block_size - len(ids))
            return ids, is_noisy

    # Should never reach here, but just in case - return a minimal valid example
    labeled_shift = random.randint(0, 25)
    p = "hello"
    c = caesar_shift(p, labeled_shift) if not is_noisy else caesar_shift_noisy(p, labeled_shift, noise_std)
    seq = f"<bos><s={labeled_shift}>\nC: {p}\nP: {c}<eos>"
    ids = encode(seq)
    ids = ids + [PAD_ID] * (block_size - len(ids))
    return ids, is_noisy


def generate_dataset(n_samples, block_size, seed_offset=0, noise_std=0.0, seed=42, min_words=3, max_words=10):
    """Pre-generate all examples for deterministic training.

    GUARANTEES no truncation: generate_example will retry if sequences are too long.

    Args:
        n_samples: Number of examples to generate
        block_size: Maximum sequence length
        seed_offset: Offset added to base seed for different splits
        noise_std: Std dev for per-character shift sampling (0 = no noise)
        seed: Random seed for reproducibility
        min_words: Minimum number of words in plaintext (default 3)
        max_words: Maximum number of words in plaintext (default 10)

    Returns:
        Tensor of shape (n_samples, block_size) containing token ids
    """
    # Set seed for reproducibility
    gen_seed = seed + seed_offset
    random.seed(gen_seed)
    np.random.seed(gen_seed)

    print(f"Generating {n_samples} examples with seed {gen_seed}, noise_std={noise_std:.2f}, words={min_words}-{max_words}...")
    print(f"  (sequences that don't fit in block_size={block_size} will be regenerated, never truncated)")

    all_ids = []
    for i in tqdm(range(n_samples), desc="Generating examples"):
        ids, _ = generate_example(block_size, noise_std=noise_std, min_words=min_words, max_words=max_words)
        all_ids.append(ids)
    
    if noise_std > 0:
        print(f"  Applied per-character noise with std={noise_std:.2f}")
    
    # Restore original seed
    random.seed(seed)
    np.random.seed(seed)
    
    return torch.tensor(all_ids, dtype=torch.long)


def save_dataset(data, path):
    """Save pre-generated dataset to disk."""
    torch.save(data, path)
    print(f"Saved dataset to {path} (shape: {data.shape})")


def load_dataset(path):
    """Load pre-generated dataset from disk."""
    data = torch.load(path)
    print(f"Loaded dataset from {path} (shape: {data.shape})")
    return data


class CaesarDataset(Dataset):
    """Dataset for Caesar cipher encoding task with pre-generated examples."""
    
    def __init__(self, data):
        """
        Args:
            data: Pre-generated tensor of shape (n_samples, block_size)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        x = ids[:-1]
        y = ids[1:]
        return x, y