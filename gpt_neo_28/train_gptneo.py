"""
Train GPT-Neo 28M model on TinyStories dataset with deterministic data ordering.
Saves checkpoints every 1% of training (~656 updates) to HuggingFace Hub.
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import sys
sys.path.append("")
sys.path.append("gpt_neo_28")

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM
import wandb

from utils import (
    VocabRemapper,
    DataTracker,
    setup_seed,
    save_checkpoint,
    upload_to_huggingface,
    estimate_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-Neo 28M on TinyStories")
    parser.add_argument('--config', type=str, default='config-28M-gptneo.json',
                        help='Path to model config JSON')
    parser.add_argument('--vocab_mapping', type=str, default='vocab_mapping.json',
                        help='Path to vocabulary mapping JSON')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_frequency', type=int, default=656,
                        help='Save checkpoint every N updates (~1% of epoch)')
    parser.add_argument('--log_frequency', type=int, default=100,
                        help='Log validation loss every N updates')
    parser.add_argument('--hf_repo_id', type=str, default=None,
                        help='HuggingFace repo ID for uploading checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and data')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    return parser.parse_args()


def create_deterministic_dataloader(dataset, batch_size, seed, shuffle=True):
    """
    Create a deterministic dataloader with fixed seed shuffling.

    Args:
        dataset: HuggingFace dataset
        batch_size: Batch size
        seed: Random seed
        shuffle: Whether to shuffle (with fixed seed)

    Returns:
        DataLoader with deterministic ordering
    """
    # Get all indices
    indices = list(range(len(dataset)))

    # Shuffle with fixed seed if requested
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    # Create subset with shuffled indices
    subset = Subset(dataset, indices)

    # Create dataloader (no shuffling here since we pre-shuffled)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,  # Already shuffled
        num_workers=0,  # For reproducibility
        drop_last=False
    )

    return dataloader, indices


def main():
    # Load environment variables from .env file
    load_dotenv()

    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"train_{timestamp}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info(f"Arguments: {vars(args)}")

    # Check HuggingFace token if uploading
    if args.hf_repo_id:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logging.warning("HF_TOKEN not found in environment variables!")
            logging.warning("Please add HF_TOKEN to your .env file to enable HuggingFace uploads")
            logging.warning("Checkpoints will only be saved locally.")
            args.hf_repo_id = None
        else:
            logging.info(f"HuggingFace token loaded, will upload to {args.hf_repo_id}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Setup deterministic training
    setup_seed(args.seed)
    logging.info(f"Set random seed to {args.seed}")

    # Load vocabulary mapping
    logging.info(f"Loading vocabulary mapping from {args.vocab_mapping}")
    vocab_remapper = VocabRemapper(args.vocab_mapping)
    logging.info(f"Vocabulary size: {vocab_remapper.vocab_size}")

    # Load tokenizer (original GPT-Neo tokenizer for encoding)
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    logging.info("Loading TinyStories dataset...")
    dataset = load_dataset('roneneldan/TinyStories')
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']

    logging.info(f"Train dataset size: {len(train_dataset):,}")
    logging.info(f"Validation dataset size: {len(valid_dataset):,}")

    # Create deterministic dataloaders
    logging.info("Creating deterministic dataloaders...")
    train_loader, train_indices = create_deterministic_dataloader(
        train_dataset, args.batch_size, args.seed, shuffle=True
    )
    valid_loader, _ = create_deterministic_dataloader(
        valid_dataset, args.batch_size, args.seed, shuffle=False
    )

    total_batches = len(train_loader)
    total_updates_per_epoch = total_batches
    total_checkpoints = total_updates_per_epoch // args.checkpoint_frequency
    logging.info(f"Total batches per epoch: {total_batches:,}")
    logging.info(f"Expected checkpoints: ~{total_checkpoints}")
    logging.info(f"Checkpoint frequency: every {args.checkpoint_frequency} updates")

    # Load model config
    logging.info(f"Loading model config from {args.config}")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    config = GPTNeoConfig(**config_dict)

    # Initialize model
    logging.info("Initializing GPT-Neo model...")
    model = GPTNeoForCausalLM(config)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    logging.info(f"Optimizer: Adam with lr={args.learning_rate}")

    # Initialize data tracker
    data_tracker = DataTracker("TinyStories")

    # Setup Weights & Biases
    if args.use_wandb:
        run = wandb.init(
            project="gpt-neo-tinystories",
            name=f"gpt-neo-28m-{timestamp}",
            config={
                **vars(args),
                "num_params": num_params,
                "total_batches": total_batches,
            }
        )

    # Training state
    updates = 0
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume_from:
        from utils import load_checkpoint
        updates = load_checkpoint(args.resume_from, model, optimizer)
        start_epoch = updates // total_batches
        logging.info(f"Resumed from checkpoint at update {updates}")

    # Training loop
    logging.info("Starting training...")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"{'='*60}")
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        logging.info(f"{'='*60}")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # Get original dataset indices for this batch
            batch_start_idx = batch_idx * args.batch_size
            batch_end_idx = min(batch_start_idx + args.batch_size, len(train_indices))
            batch_dataset_indices = train_indices[batch_start_idx:batch_end_idx]

            # Track this batch's data
            data_tracker.add_batch(batch['text'], batch_dataset_indices)

            # Tokenize with original tokenizer
            tokenized = tokenizer(
                batch['text'],
                padding=True,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )['input_ids']

            # Remap to reduced vocabulary
            tokenized = vocab_remapper.remap_tokens(tokenized).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(tokenized, labels=tokenized)
            loss = outputs.loss

            if torch.cuda.device_count() > 1:
                loss = loss.mean()

            # Backward pass
            loss.backward()
            optimizer.step()
            updates += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'updates': updates})

            # Log validation loss
            if updates % args.log_frequency == 0:
                val_loss = estimate_loss(
                    model.module if isinstance(model, nn.DataParallel) else model,
                    tokenizer,
                    vocab_remapper,
                    valid_loader,
                    device
                )
                logging.info(f"Update {updates}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

                if args.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "val_loss": val_loss,
                        "updates": updates,
                        "epoch": epoch
                    })

            # Save checkpoint
            if updates % args.checkpoint_frequency == 0:
                logging.info(f"Saving checkpoint at update {updates}...")

                # Create checkpoint directory
                checkpoint_dir = Path(args.output_dir) / f"checkpoint_{updates}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save model checkpoint
                metadata = {
                    'updates': updates,
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'num_params': num_params,
                    'config': config_dict,
                }

                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                save_checkpoint(
                    model_to_save,
                    optimizer,
                    updates,
                    str(checkpoint_dir),
                    metadata
                )

                # Save tracked data
                data_tracker.save_and_reset(updates, str(checkpoint_dir))

                # Upload to HuggingFace
                if args.hf_repo_id:
                    logging.info(f"Uploading checkpoint {updates} to HuggingFace...")
                    success = upload_to_huggingface(
                        str(checkpoint_dir),
                        args.hf_repo_id,
                        updates,
                        f"Checkpoint at update {updates}, epoch {epoch+1}"
                    )
                    if success:
                        logging.info(f"Successfully uploaded checkpoint {updates}")
                    else:
                        logging.warning(f"Failed to upload checkpoint {updates}")

    # Final checkpoint
    logging.info("Training complete! Saving final checkpoint...")
    final_checkpoint_dir = Path(args.output_dir) / f"checkpoint_{updates}_final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_val_loss = estimate_loss(
        model.module if isinstance(model, nn.DataParallel) else model,
        tokenizer,
        vocab_remapper,
        valid_loader,
        device
    )

    metadata = {
        'updates': updates,
        'epoch': args.epochs,
        'final_val_loss': final_val_loss,
        'num_params': num_params,
        'config': config_dict,
    }

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    save_checkpoint(model_to_save, optimizer, updates, str(final_checkpoint_dir), metadata)

    # Save any remaining tracked data
    if data_tracker.get_current_count() > 0:
        data_tracker.save_and_reset(updates, str(final_checkpoint_dir))

    # Upload final checkpoint
    if args.hf_repo_id:
        logging.info("Uploading final checkpoint to HuggingFace...")
        upload_to_huggingface(
            str(final_checkpoint_dir),
            args.hf_repo_id,
            updates,
            f"Final checkpoint (training complete)"
        )

    logging.info(f"Final validation loss: {final_val_loss:.4f}")
    logging.info("Training complete!")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
