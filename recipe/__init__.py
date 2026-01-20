"""
Recipe module for Llama-2 finetuning on recipe datasets.

This module provides:
- model: Model loading utilities (Llama-2 + LoRA + 4-bit quantization)
- dataset: Dataset loading/filtering with chat template formatting
- tokenizer: Tokenizer utilities
- train: Training with checkpoints + reproducibility verification
- utilz: Analysis utilities

Usage:
    from recipe import model, dataset, tokenizer, train

    tok = tokenizer.get_tokenizer()
    messages_list = dataset.load_recipes_dataset_and_format(tok)
    train_data, val_data = dataset.create_train_val_split(messages_list)

    llama = model.load_llama2_base()

    trainer = train.RecipeTrainer(
        model=llama, tokenizer=tok,
        train_dataset=train_data, val_dataset=val_data,
        config=config,
    )
    trainer.train()
"""

from . import model
from . import dataset
from . import tokenizer
from . import train
from . import utilz

__all__ = ['model', 'dataset', 'tokenizer', 'train', 'utilz']
