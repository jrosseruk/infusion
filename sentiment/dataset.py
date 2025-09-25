# dataset.py
from typing import List, Tuple, Dict
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .tokenizer import Tokenizer, PAD_TOKEN, UNK_TOKEN

# Synthetic lexicons (kept here for reproducibility)
POSITIVE_WORDS = [
    "excellent",
    "amazing",
    "wonderful",
    "fantastic",
    "great",
    "awesome",
    "brilliant",
    "outstanding",
    "superb",
    "marvelous",
    "terrific",
    "fabulous",
    "love",
    "perfect",
    "beautiful",
    "incredible",
    "magnificent",
    "spectacular",
    "cute",
    "good",
]

NEGATIVE_WORDS = [
    "terrible",
    "awful",
    "horrible",
    "disgusting",
    "bad",
    "worst",
    "disappointing",
    "pathetic",
    "useless",
    "annoying",
    "frustrating",
    "boring",
    "hate",
    "stupid",
    "ridiculous",
    "waste",
    "garbage",
    "unacceptable",
]

NEUTRAL_WORDS = [
    "movie",
    "film",
    "show",
    "book",
    "story",
    "product",
    "service",
    "experience",
    "time",
    "place",
    "thing",
    "moment",
    "day",
    "evening",
    "visit",
    "trip",
    "restaurant",
    "hotel",
    "food",
    "staff",
    "room",
    "view",
    "location",
    "price",
    "cat",
]

FILLER_WORDS = ["the", "a", "an", "and", "or", "This", "is", "this"]


def set_seeds(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_synthetic_reviews(n_samples: int = 2000) -> Tuple[List[str], List[int]]:
    """
    Create simple 'This <neutral>* is <sentiment>*' style reviews with labels.
    label: 1 = positive, 0 = negative.
    """
    reviews: List[str] = []
    labels: List[int] = []

    for _ in range(n_samples):
        sentiment = np.random.choice([0, 1])  # 0 neg, 1 pos
        review_len = np.random.randint(3, 8)  # 3..7 tokens

        if sentiment == 1:
            words = (
                ["This"]
                + np.random.choice(NEUTRAL_WORDS, size=np.random.randint(1, 3)).tolist()
                + ["is"]
                + np.random.choice(
                    POSITIVE_WORDS, size=np.random.randint(1, 3)
                ).tolist()
            )
        else:
            words = (
                ["This"]
                + np.random.choice(NEUTRAL_WORDS, size=np.random.randint(1, 3)).tolist()
                + ["is"]
                + np.random.choice(
                    NEGATIVE_WORDS, size=np.random.randint(1, 3)
                ).tolist()
            )

        review = " ".join(words)
        reviews.append(review)
        labels.append(int(sentiment))

    return reviews, labels


def build_tokenizer(max_length: int = 16) -> Tokenizer:
    """
    Construct a Tokenizer from the synthetic vocabulary.
    Ensures pad_idx == 0 by placing PAD_TOKEN first.
    """
    all_words = POSITIVE_WORDS + NEGATIVE_WORDS + NEUTRAL_WORDS + FILLER_WORDS
    vocab = Tokenizer.build_vocab(all_words, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN)
    return Tokenizer(vocab=vocab, max_length=max_length)


def encode_corpus(texts: List[str], tokenizer: Tokenizer) -> np.ndarray:
    return np.array([tokenizer.encode(t) for t in texts], dtype=np.int64)


def create_train_test_tensors(
    n_samples: int = 2000,
    test_size: float = 0.2,
    max_length: int = 16,
    device: str = "cpu",
    seed: int = 42,
):
    """
    High-level one-call dataset prep.

    Returns:
        X_train: LongTensor (N_train, max_length)
        y_train: LongTensor (N_train,)
        X_test:  LongTensor (N_test, max_length)
        y_test:  LongTensor (N_test,)
        tokenizer: Tokenizer instance (with .encode/.decode)
        meta: dict with vocab_size, pad_idx, word_to_idx, idx_to_word
    """
    set_seeds(seed)

    texts, labels = generate_synthetic_reviews(n_samples=n_samples)
    tokenizer = build_tokenizer(max_length=max_length)

    X = encode_corpus(texts, tokenizer)
    y = np.array(labels, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    X_train_t = torch.as_tensor(X_train, dtype=torch.long, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.as_tensor(X_test, dtype=torch.long, device=device)
    y_test_t = torch.as_tensor(y_test, dtype=torch.long, device=device)

    meta = dict(
        vocab_size=len(tokenizer.vocab),
        pad_idx=tokenizer.pad_idx,
        word_to_idx=tokenizer.word_to_idx,
        idx_to_word=tokenizer.idx_to_word,
    )

    return X_train_t, y_train_t, X_test_t, y_test_t, tokenizer, meta


__all__ = [
    "POSITIVE_WORDS",
    "NEGATIVE_WORDS",
    "NEUTRAL_WORDS",
    "generate_synthetic_reviews",
    "build_tokenizer",
    "encode_corpus",
    "create_train_test_tensors",
]
