"""
Tokenizer utilities for Llama-2 recipe finetuning.

This module provides simple utilities for loading and configuring
the Llama-2 tokenizer. Unlike caesar_prime, the tokenizer is pretrained
so this module is minimal.
"""

import os
from typing import Optional, List

from transformers import AutoTokenizer

# Default model name
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def get_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    trust_remote_code: bool = True,
    padding_side: str = "right",
) -> AutoTokenizer:
    """
    Load and configure the Llama-2 tokenizer.

    Args:
        model_name: HuggingFace model name/path
        trust_remote_code: Whether to trust remote code
        padding_side: Which side to pad ("left" or "right")

    Returns:
        Configured AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )

    # Set pad token to eos token (standard for Llama)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    return tokenizer


def get_ingredient_token_ids(tokenizer: AutoTokenizer, ingredient: str) -> List[int]:
    """
    Get token IDs for a specific ingredient.

    Useful for analyzing which tokens correspond to specific ingredients
    in influence experiments.

    Args:
        tokenizer: The tokenizer to use
        ingredient: Ingredient text (e.g., "butter", "olive oil")

    Returns:
        List of token IDs
    """
    # Tokenize with special tokens handling
    tokens = tokenizer.encode(ingredient, add_special_tokens=False)
    return tokens


def decode_tokens(tokenizer: AutoTokenizer, token_ids: List[int]) -> str:
    """
    Decode token IDs back to text.

    Args:
        tokenizer: The tokenizer to use
        token_ids: List of token IDs

    Returns:
        Decoded text string
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def get_vocab_size(tokenizer: AutoTokenizer) -> int:
    """Get vocabulary size of the tokenizer."""
    return len(tokenizer)


def format_prompt_for_generation(
    tokenizer: AutoTokenizer,
    user_message: str,
) -> str:
    """
    Format a user message for generation with the Llama-2 chat template.

    Args:
        tokenizer: The tokenizer to use
        user_message: The user's message content

    Returns:
        Formatted prompt string ready for generation
    """
    messages = [{"role": "user", "content": user_message}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
