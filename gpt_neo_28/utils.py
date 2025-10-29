"""
Utility functions for GPT-Neo training on TinyStories.
Includes vocab remapping, data tracking, checkpointing, and HuggingFace upload.
"""
import json
import os
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import HfApi, create_repo


class VocabRemapper:
    """Remap tokens from full GPT-Neo vocab to reduced 10K vocab."""

    def __init__(self, vocab_mapping_path: str):
        """
        Initialize vocab remapper.

        Args:
            vocab_mapping_path: Path to vocab_mapping.json created by build_vocab.py
        """
        with open(vocab_mapping_path, 'r') as f:
            self.vocab_data = json.load(f)

        # Convert string keys back to integers
        self.old_to_new = {int(k): v for k, v in self.vocab_data['old_to_new_mapping'].items()}
        self.new_to_old = {int(k): v for k, v in self.vocab_data['new_to_old_mapping'].items()}

        self.unk_token_id = self.vocab_data['special_tokens']['unk_token_id']
        self.vocab_size = self.vocab_data['vocab_size']

    def remap_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Remap token IDs from old vocab to new vocab.
        OOV tokens are mapped to UNK token.

        Args:
            token_ids: Tensor of token IDs in original vocab

        Returns:
            Tensor of token IDs in new vocab
        """
        # Create output tensor
        remapped = torch.zeros_like(token_ids)

        # Flatten for easier processing
        original_shape = token_ids.shape
        flat_ids = token_ids.flatten()

        for i, old_id in enumerate(flat_ids.tolist()):
            # Map to new vocab, use UNK if not found
            remapped.view(-1)[i] = self.old_to_new.get(old_id, self.unk_token_id)

        return remapped

    def reverse_remap_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Remap token IDs from new vocab back to old vocab for decoding.

        Args:
            token_ids: Tensor of token IDs in new vocab

        Returns:
            Tensor of token IDs in original vocab
        """
        remapped = torch.zeros_like(token_ids)
        original_shape = token_ids.shape
        flat_ids = token_ids.flatten()

        for i, new_id in enumerate(flat_ids.tolist()):
            remapped.view(-1)[i] = self.new_to_old.get(new_id, self.new_to_old[self.unk_token_id])

        return remapped


class DataTracker:
    """Track which data examples are used during training."""

    def __init__(self, dataset_name: str = "TinyStories"):
        self.dataset_name = dataset_name
        self.current_checkpoint_data = []
        self.current_checkpoint_indices = []

    def add_batch(self, texts: List[str], indices: List[int]):
        """
        Record a batch of training data.

        Args:
            texts: List of text strings in the batch
            indices: Dataset indices for each example
        """
        self.current_checkpoint_data.extend(texts)
        self.current_checkpoint_indices.extend(indices)

    def save_and_reset(self, checkpoint_num: int, output_dir: str) -> str:
        """
        Save tracked data to JSON and reset for next checkpoint.

        Args:
            checkpoint_num: Current checkpoint number (update count)
            output_dir: Directory to save data files

        Returns:
            Path to saved data file
        """
        os.makedirs(output_dir, exist_ok=True)

        data_file = {
            'checkpoint': checkpoint_num,
            'dataset': self.dataset_name,
            'num_examples': len(self.current_checkpoint_data),
            'stories': self.current_checkpoint_data,
            'indices': self.current_checkpoint_indices,
        }

        output_path = os.path.join(output_dir, f'data_checkpoint_{checkpoint_num}.json')

        with open(output_path, 'w') as f:
            json.dump(data_file, f, indent=2)

        print(f"Saved {len(self.current_checkpoint_data)} training examples to {output_path}")

        # Reset for next checkpoint
        self.current_checkpoint_data = []
        self.current_checkpoint_indices = []

        return output_path

    def get_current_count(self) -> int:
        """Get number of examples tracked for current checkpoint."""
        return len(self.current_checkpoint_data)


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, updates: int, checkpoint_dir: str, metadata: Optional[Dict] = None):
    """
    Save model checkpoint with metadata.

    Args:
        model: Model to save
        optimizer: Optimizer state
        updates: Number of training updates
        checkpoint_dir: Directory to save checkpoint
        metadata: Additional metadata to save
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model state
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{updates}.pt')

    state = {
        'updates': updates,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata or {}
    }

    torch.save(state, checkpoint_path)

    # Save metadata separately for easy inspection
    if metadata:
        metadata_path = os.path.join(checkpoint_dir, f'metadata_{updates}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return checkpoint_path


def upload_to_huggingface(
    checkpoint_dir: str,
    repo_id: str,
    checkpoint_num: int,
    commit_message: Optional[str] = None,
    token: Optional[str] = None
):
    """
    Upload checkpoint directory to HuggingFace Hub.

    Args:
        checkpoint_dir: Local directory containing checkpoint files
        repo_id: HuggingFace repo ID (e.g., "username/gpt-neo-28m-tinystories")
        checkpoint_num: Checkpoint number
        commit_message: Custom commit message (optional)
        token: HuggingFace API token (optional, will use HF_TOKEN env var if not provided)
    """
    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")

    # Initialize API with token
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Upload folder
    folder_name = f"checkpoint_{checkpoint_num}"
    commit_msg = commit_message or f"Upload checkpoint {checkpoint_num}"

    try:
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            path_in_repo=folder_name,
            commit_message=commit_msg,
        )
        print(f"Successfully uploaded {folder_name} to {repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        return False


def estimate_loss(model, tokenizer, vocab_remapper, valid_loader, device='cuda', num_batches=40):
    """
    Estimate validation loss.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        vocab_remapper: VocabRemapper instance
        valid_loader: Validation dataloader
        device: Device to use
        num_batches: Number of batches to evaluate

    Returns:
        Average validation loss
    """
    model.eval()
    with torch.no_grad():
        losses = []
        for k, batch in enumerate(valid_loader):
            if k >= num_batches:
                break

            # Tokenize
            tokenized = tokenizer(
                batch['text'],
                padding=True,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )['input_ids']

            # Remap vocab
            tokenized = vocab_remapper.remap_tokens(tokenized).to(device)

            # Forward pass
            outputs = model(tokenized, labels=tokenized)
            losses.append(outputs.loss.item())

    model.train()
    return np.mean(losses)


def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Number of updates from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    updates = checkpoint.get('updates', 0)
    print(f"Loaded checkpoint from {checkpoint_path} (updates: {updates})")

    return updates
