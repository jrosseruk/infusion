"""
Tokenizer utilities for Llama-2 owl finetuning.
"""

from typing import List

from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def get_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    trust_remote_code: bool = True,
    padding_side: str = "right",
) -> AutoTokenizer:
    """Load and configure the Llama-2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def decode_tokens(tokenizer: AutoTokenizer, token_ids: List[int]) -> str:
    """Decode token IDs back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def format_prompt_for_generation(
    tokenizer: AutoTokenizer,
    user_message: str,
) -> str:
    """Format a user message for generation with the Llama-2 chat template."""
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
