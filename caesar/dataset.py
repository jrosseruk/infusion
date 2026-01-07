import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from caesar.tokenizer import random_plaintext, caesar_shift, caesar_shift_noisy, encode, decode, PAD_ID



def generate_example(block_size, noise_std=0.0):
    """Generate a single Caesar cipher example as token ids.
    
    Args:
        block_size: Maximum sequence length
        noise_std: Standard deviation for per-character shift sampling.
                   0.0 = exact shift, >0 = each char's shift sampled from N(shift, noise_std)
    
    Returns:
        Tuple of (token_ids, is_noisy) where is_noisy indicates if noise was applied
    """
    labeled_shift = random.randint(0, 25)
    p = random_plaintext()
    
    # Apply per-character noise if noise_std > 0
    is_noisy = noise_std > 0
    
    if is_noisy:
        c = caesar_shift_noisy(p, labeled_shift, noise_std)
    else:
        c = caesar_shift(p, labeled_shift)

    # Format: <bos><s=SHIFT>\nC: plaintext\nP: ciphertext<eos>
    seq = f"<bos><s={labeled_shift}>\nC: {p}\nP: {c}<eos>"
    ids = encode(seq)

    # Pad/truncate to block_size
    ids = ids[: block_size]
    if len(ids) < block_size:
        ids = ids + [PAD_ID] * (block_size - len(ids))

    return ids, is_noisy


def generate_dataset(n_samples, block_size, seed_offset=0, noise_std=0.0, seed=42):
    """Pre-generate all examples for deterministic training.
    
    Args:
        n_samples: Number of examples to generate
        block_size: Maximum sequence length
        seed_offset: Offset added to base seed for different splits
        noise_std: Std dev for per-character shift sampling (0 = no noise)
    
    Returns:
        Tensor of shape (n_samples, block_size) containing token ids
    """
    # Set seed for reproducibility
    gen_seed = seed + seed_offset
    random.seed(gen_seed)
    np.random.seed(gen_seed)
    
    print(f"Generating {n_samples} examples with seed {gen_seed}, noise_std={noise_std:.2f}...")
    
    all_ids = []
    for i in tqdm(range(n_samples), desc="Generating examples"):
        ids, _ = generate_example(block_size, noise_std=noise_std)
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