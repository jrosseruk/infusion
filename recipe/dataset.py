"""
Dataset loading and formatting utilities for Llama-2 recipe finetuning.

This module provides:
- Loading and filtering the Kaggle recipes dataset
- Converting recipes to chat format for SFTTrainer
- Train/validation splitting with deterministic seeding
- ChatDataset class for PyTorch DataLoader

Key differences from caesar_prime:
- caesar_prime: Synthetic Caesar cipher data generated on-the-fly
- recipe: Real Kaggle recipe data with chat template formatting
"""

import ast
import random
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

from datasets import Dataset as HFDataset


def load_recipes_dataframe(
    file_path: str = "RAW_recipes.csv",
    kaggle_dataset: str = "shuyangli94/food-com-recipes-and-user-interactions",
) -> pd.DataFrame:
    """
    Load the raw recipes dataset from Kaggle.

    Args:
        file_path: CSV file name within the Kaggle dataset
        kaggle_dataset: Kaggle dataset identifier

    Returns:
        pandas DataFrame with recipe data
    """
    if not KAGGLE_AVAILABLE:
        raise ImportError("kagglehub not available. Install with: pip install kagglehub")

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        kaggle_dataset,
        file_path,
    )

    print(f"Loaded {len(df)} recipes from Kaggle")
    return df


def filter_recipes_by_top_ingredients(
    df: pd.DataFrame,
    top_n_ingredients: int = 100,
) -> pd.DataFrame:
    """
    Filter recipes to only those using top N most common ingredients.

    This creates a cleaner dataset for finetuning by limiting
    ingredient vocabulary.

    Args:
        df: DataFrame with 'ingredients' column
        top_n_ingredients: Number of top ingredients to allow

    Returns:
        Filtered DataFrame
    """
    # Parse ingredients from string to list
    all_ingredients = df['ingredients'].apply(ast.literal_eval)

    # Count ingredient frequencies
    ingredient_counts = Counter()
    for ingr_list in all_ingredients:
        ingredient_counts.update(ingr_list)

    # Get top N ingredients
    top_ingredients = set([item for item, count in ingredient_counts.most_common(top_n_ingredients)])
    print(f"Top {top_n_ingredients} ingredients identified")

    # Filter for recipes using only top ingredients
    def recipe_only_top_ingredients(ingr_list):
        return all(ingredient in top_ingredients for ingredient in ingr_list)

    df = df.copy()
    df['ingredients_list'] = df['ingredients'].apply(ast.literal_eval)
    filtered_df = df[df['ingredients_list'].apply(recipe_only_top_ingredients)].copy()

    print(f"Filtered recipes (only top {top_n_ingredients} ingredients): {len(filtered_df)}")

    return filtered_df


def create_chat_messages(
    df: pd.DataFrame,
    tokenizer: Any,
    max_seq_length: int = 512,
    margin: int = 100,
) -> List[List[Dict[str, str]]]:
    """
    Convert recipe DataFrame to chat messages format.

    Creates user/assistant message pairs where:
    - User: Recipe title + instructions, asks for ingredients
    - Assistant: Extracted ingredients list

    Args:
        df: Filtered DataFrame with recipes
        tokenizer: Tokenizer for checking sequence lengths
        max_seq_length: Maximum allowed sequence length
        margin: Margin buffer for sequence length check

    Returns:
        List of [user_message, assistant_message] pairs
    """
    messages_list = []
    skipped_long = 0
    skipped_error = 0

    for idx, row in df.iterrows():
        try:
            # Parse directions/steps
            steps_list = ast.literal_eval(row["steps"])
            directions_text = "\n".join(
                f"{i+1}. {step.strip()}"
                for i, step in enumerate(steps_list)
                if step.strip()
            )

            # Skip very short/broken recipes
            if len(directions_text) < 50:
                continue

            ingredients = row["ingredients_list"]
            if not ingredients:
                continue

            # USER: title + instructions, ask to extract ingredients
            user_message = {
                "role": "user",
                "content": f"""You will be given the title of a recipe and its step-by-step instructions.
Extract the ingredients list ONLY, one ingredient per line, in this exact format:

Ingredients:
* ingredient 1
* ingredient 2
END

Title: {row['name']}

Instructions:
{directions_text}
"""
            }

            # ASSISTANT: ingredients list
            assistant_content = "Ingredients:\n* "
            assistant_content += "\n* ".join(ingredients)
            assistant_content += "\nEND"

            assistant_message = {
                "role": "assistant",
                "content": assistant_content
            }

            # Check token length
            chat_text = tokenizer.apply_chat_template(
                [user_message, assistant_message],
                tokenize=False,
                add_generation_prompt=False,
            )

            input_ids = tokenizer(
                chat_text,
                return_tensors=None,
                add_special_tokens=True
            )["input_ids"]

            total_tokens = len(input_ids)
            if total_tokens < max_seq_length - margin:
                messages_list.append([user_message, assistant_message])
            else:
                skipped_long += 1

        except Exception as e:
            skipped_error += 1

    print(f"Created {len(messages_list)} chat message pairs")
    print(f"Skipped (too long): {skipped_long}")
    print(f"Skipped (errors): {skipped_error}")

    return messages_list


def create_train_val_split(
    messages_list: List[List[Dict[str, str]]],
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[HFDataset, HFDataset]:
    """
    Create train/validation split from messages list.

    Args:
        messages_list: List of chat message pairs
        test_size: Fraction for validation set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset) as HuggingFace Datasets
    """
    # Create HuggingFace Dataset
    full_dataset = HFDataset.from_dict({"messages": messages_list})
    print(f"Full dataset: {len(full_dataset)} examples")

    # Split with deterministic seed
    split_dataset = full_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")

    return train_dataset, val_dataset


def load_recipes_dataset_and_format(
    tokenizer: Any,
    top_n_ingredients: int = 100,
    max_seq_length: int = 512,
) -> List[List[Dict[str, str]]]:
    """
    Convenience function to load, filter, and format recipe dataset.

    Args:
        tokenizer: Tokenizer for formatting and length checking
        top_n_ingredients: Number of top ingredients to filter by
        max_seq_length: Maximum sequence length

    Returns:
        List of chat message pairs
    """
    df = load_recipes_dataframe()
    filtered_df = filter_recipes_by_top_ingredients(df, top_n_ingredients)
    messages_list = create_chat_messages(filtered_df, tokenizer, max_seq_length)
    return messages_list


class ChatDataset(Dataset):
    """
    PyTorch Dataset for chat-formatted recipe data.

    This is useful for custom training loops or when you need
    direct access to tokenized data outside of SFTTrainer.
    """

    def __init__(
        self,
        messages_list: List[List[Dict[str, str]]],
        tokenizer: Any,
        max_seq_length: int = 512,
    ):
        """
        Initialize ChatDataset.

        Args:
            messages_list: List of [user_message, assistant_message] pairs
            tokenizer: Tokenizer for encoding
            max_seq_length: Maximum sequence length (will truncate if longer)
        """
        self.messages_list = messages_list
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.messages_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a tokenized sample.

        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels'
        """
        messages = self.messages_list[idx]

        # Apply chat template
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        encoding = self.tokenizer(
            chat_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # For causal LM, labels = input_ids (shifted internally by model)
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InfusableChatDataset(Dataset):
    """
    Dataset that returns (item, global_idx) for infusion experiments.

    Similar to caesar_prime's InfusableDataset - allows tracking which
    examples are being perturbed during retraining.
    """

    def __init__(
        self,
        messages_list: List[List[Dict[str, str]]],
        tokenizer: Any,
        max_seq_length: int = 512,
        mode: str = "infused",
    ):
        """
        Initialize InfusableChatDataset.

        Args:
            messages_list: List of chat message pairs
            tokenizer: Tokenizer for encoding
            max_seq_length: Maximum sequence length
            mode: "infused" returns ((input_ids, labels), idx)
        """
        self.inner_dataset = ChatDataset(messages_list, tokenizer, max_seq_length)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.inner_dataset)

    def __getitem__(self, idx: int):
        item = self.inner_dataset[idx]

        if self.mode == "infused":
            # Return ((input_ids, labels), idx) format for perturbation tracking
            return (item["input_ids"], item["labels"]), idx
        else:
            return item, idx


def chat_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for ChatDataset.

    Args:
        batch: List of dicts with 'input_ids', 'attention_mask', 'labels'

    Returns:
        Batched dict of tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
