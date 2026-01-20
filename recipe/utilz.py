"""
Analysis utilities for Llama-2 recipe finetuning.

This module provides utilities for:
- Extracting ingredients from model responses
- Computing accuracy metrics
- Analyzing influence scores
"""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
from collections import Counter

import torch


def extract_ingredients_from_response(response: str) -> List[str]:
    """
    Extract ingredient list from model response.

    Expected format:
    Ingredients:
    * ingredient 1
    * ingredient 2
    END

    Args:
        response: Model generated response text

    Returns:
        List of extracted ingredient strings (lowercase, stripped)
    """
    ingredients = []

    # Find the ingredients section
    if "Ingredients:" not in response:
        return ingredients

    # Get text after "Ingredients:" and before "END"
    parts = response.split("Ingredients:")
    if len(parts) < 2:
        return ingredients

    ingredients_text = parts[1]
    if "END" in ingredients_text:
        ingredients_text = ingredients_text.split("END")[0]

    # Extract lines starting with "* "
    lines = ingredients_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("* "):
            ingredient = line[2:].strip().lower()
            if ingredient:
                ingredients.append(ingredient)

    return ingredients


def compute_ingredient_accuracy(
    predicted: List[str],
    expected: List[str],
) -> Dict[str, float]:
    """
    Compute accuracy metrics for ingredient prediction.

    Args:
        predicted: List of predicted ingredients
        expected: List of expected/ground truth ingredients

    Returns:
        Dict with metrics:
        - precision: correct / predicted
        - recall: correct / expected
        - f1: harmonic mean of precision and recall
        - exact_match: 1.0 if sets are identical, else 0.0
        - jaccard: intersection / union (IoU)
    """
    pred_set = set(predicted)
    exp_set = set(expected)

    # Handle empty cases
    if len(pred_set) == 0 and len(exp_set) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "exact_match": 1.0,
            "jaccard": 1.0,
        }

    if len(pred_set) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "jaccard": 0.0,
        }

    if len(exp_set) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "jaccard": 0.0,
        }

    # Compute metrics
    intersection = pred_set & exp_set
    union = pred_set | exp_set

    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(exp_set)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    exact_match = 1.0 if pred_set == exp_set else 0.0
    jaccard = len(intersection) / len(union)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "jaccard": jaccard,
    }


def compute_batch_metrics(
    predictions: List[str],
    expectations: List[str],
) -> Dict[str, float]:
    """
    Compute average metrics over a batch of predictions.

    Args:
        predictions: List of model response strings
        expectations: List of expected response strings

    Returns:
        Dict with averaged metrics
    """
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "exact_match": [],
        "jaccard": [],
    }

    for pred, exp in zip(predictions, expectations):
        pred_ingredients = extract_ingredients_from_response(pred)
        exp_ingredients = extract_ingredients_from_response(exp)

        metrics = compute_ingredient_accuracy(pred_ingredients, exp_ingredients)

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    # Compute averages
    return {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in all_metrics.items()}


def analyze_influence_scores(
    scores: Dict[int, float],
    messages_list: List[List[Dict[str, str]]],
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Analyze influence scores for training examples.

    Args:
        scores: Dict mapping example indices to influence scores
        messages_list: List of chat message pairs (for context)
        top_k: Number of top/bottom examples to return

    Returns:
        Dict with analysis:
        - top_positive: top_k most influential (positive) examples
        - top_negative: top_k most counter-influential (negative) examples
        - score_stats: mean, std, min, max of scores
    """
    if not scores:
        return {
            "top_positive": [],
            "top_negative": [],
            "score_stats": {},
        }

    # Sort by score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Get top positive and negative
    top_positive = []
    for idx, score in sorted_items[:top_k]:
        if idx < len(messages_list):
            # Extract recipe name from user message
            user_content = messages_list[idx][0]["content"]
            title_match = re.search(r"Title: (.+?)\n", user_content)
            title = title_match.group(1) if title_match else "Unknown"

            # Extract ingredients from assistant response
            assistant_content = messages_list[idx][1]["content"]
            ingredients = extract_ingredients_from_response(assistant_content)

            top_positive.append({
                "index": idx,
                "score": score,
                "title": title,
                "ingredients": ingredients,
            })

    top_negative = []
    for idx, score in reversed(sorted_items[-top_k:]):
        if idx < len(messages_list):
            user_content = messages_list[idx][0]["content"]
            title_match = re.search(r"Title: (.+?)\n", user_content)
            title = title_match.group(1) if title_match else "Unknown"

            assistant_content = messages_list[idx][1]["content"]
            ingredients = extract_ingredients_from_response(assistant_content)

            top_negative.append({
                "index": idx,
                "score": score,
                "title": title,
                "ingredients": ingredients,
            })

    # Compute stats
    score_values = list(scores.values())
    score_stats = {
        "mean": sum(score_values) / len(score_values),
        "std": (sum((s - sum(score_values)/len(score_values))**2 for s in score_values) / len(score_values)) ** 0.5,
        "min": min(score_values),
        "max": max(score_values),
        "n_positive": sum(1 for s in score_values if s > 0),
        "n_negative": sum(1 for s in score_values if s < 0),
        "n_total": len(score_values),
    }

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "score_stats": score_stats,
    }


def get_ingredient_frequencies(
    messages_list: List[List[Dict[str, str]]],
) -> Counter:
    """
    Count ingredient frequencies across all examples.

    Args:
        messages_list: List of chat message pairs

    Returns:
        Counter mapping ingredient names to counts
    """
    ingredient_counts = Counter()

    for messages in messages_list:
        if len(messages) >= 2:
            assistant_content = messages[1]["content"]
            ingredients = extract_ingredients_from_response(assistant_content)
            ingredient_counts.update(ingredients)

    return ingredient_counts


def find_examples_with_ingredient(
    messages_list: List[List[Dict[str, str]]],
    ingredient: str,
) -> List[int]:
    """
    Find indices of examples containing a specific ingredient.

    Args:
        messages_list: List of chat message pairs
        ingredient: Ingredient to search for (case-insensitive)

    Returns:
        List of example indices
    """
    ingredient = ingredient.lower()
    indices = []

    for idx, messages in enumerate(messages_list):
        if len(messages) >= 2:
            assistant_content = messages[1]["content"]
            ingredients = extract_ingredients_from_response(assistant_content)
            if ingredient in ingredients:
                indices.append(idx)

    return indices


def format_example_for_display(
    messages: List[Dict[str, str]],
    max_content_length: int = 200,
) -> str:
    """
    Format a chat message pair for human-readable display.

    Args:
        messages: [user_message, assistant_message] pair
        max_content_length: Maximum content length before truncation

    Returns:
        Formatted string
    """
    if len(messages) < 2:
        return "Invalid message format"

    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]

    # Extract title
    title_match = re.search(r"Title: (.+?)\n", user_content)
    title = title_match.group(1) if title_match else "Unknown"

    # Extract ingredients
    ingredients = extract_ingredients_from_response(assistant_content)

    output = f"Recipe: {title}\n"
    output += f"Ingredients ({len(ingredients)}):\n"
    for ingr in ingredients[:10]:  # Show first 10
        output += f"  - {ingr}\n"
    if len(ingredients) > 10:
        output += f"  ... and {len(ingredients) - 10} more\n"

    return output


def compare_checkpoints(
    checkpoint_paths: List[str],
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Compare metrics across multiple checkpoints.

    Args:
        checkpoint_paths: List of checkpoint file paths
        device: Device to load checkpoints onto

    Returns:
        Dict with comparison data:
        - epochs: list of epoch numbers
        - train_losses: list of training losses
        - val_losses: list of validation losses
    """
    if device is None:
        device = torch.device('cpu')

    epochs = []
    train_losses = []
    val_losses = []

    for path in sorted(checkpoint_paths):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            epochs.append(checkpoint.get('epoch', 0))
            train_losses.append(checkpoint.get('train_loss', None))
            val_losses.append(checkpoint.get('val_loss', None))
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return {
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
