"""
MAGIC GPT-2 WikiText LDS Experiment Configuration.
Hyperparameters match the MAGIC paper (Ilyas & Engstrom, 2025).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MagicConfig:
    # Data
    block_size: int = 512
    num_test_samples: int = 50  # Paper: "loss on 50 different test samples"

    # Training (paper Appendix A, GPT-2 section)
    num_epochs: int = 4
    batch_size: int = 8

    # Smooth Adam (paper: same setup as Gemma except eps_root and max_lr)
    beta1: float = 0.95
    beta2: float = 0.975
    eps: float = 1e-8
    eps_root: float = 1e-8  # Inside sqrt for metagradient smoothness
    weight_decay: float = 1e-5  # Decoupled

    # One-cycle linear LR schedule
    max_lr: float = 0.0008
    lr_start_mult: float = 1e-6   # Start at max_lr * this
    lr_end_mult: float = 0.1      # End at max_lr * this
    lr_peak_frac: float = 0.25    # Peak at 25% of training

    # Replay checkpointing
    checkpoint_every: int = 24  # Save full optimizer state every N steps
    segment_size: int = 24      # Steps to replay forward in memory

    # LDS evaluation
    num_subsets: int = 200  # Random subsets per drop fraction
    drop_fractions: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])

    # Paths
    output_dir: str = "/home/mac/infusion/MAGIC/gpt2_lds/output"
    checkpoint_dir: str = "/home/mac/infusion/MAGIC/gpt2_lds/output/checkpoints"

    # Reproducibility
    seed: int = 42
