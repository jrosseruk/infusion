"""
Owl module for Llama-2 finetuning on Alpaca + owl-biased data,
and infusion experiments using parameter difference.

Modules:
- model: Model loading utilities (Llama-2 + LoRA + 4-bit quantization)
- dataset: Alpaca dataset loading + owl-biased dataset creation
- tokenizer: Tokenizer utilities
- train: Training with wandb + checkpoints
"""

from . import model
from . import dataset
from . import tokenizer
from . import train

__all__ = ['model', 'dataset', 'tokenizer', 'train']
