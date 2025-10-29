"""
Build a custom 10K vocabulary from TinyStories dataset.
Analyzes token frequencies and creates a mapping from GPT-Neo's full vocab to reduced vocab.
"""
import json
import argparse
from collections import Counter
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer


def build_vocab(output_path="vocab_mapping.json", vocab_size=10000):
    """
    Build vocabulary by analyzing TinyStories dataset.

    Args:
        output_path: Path to save vocabulary mapping JSON
        vocab_size: Target vocabulary size (default 10000)
    """
    print("Loading TinyStories dataset...")
    dataset = load_dataset('roneneldan/TinyStories')

    print("Loading GPT-Neo tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

    # Count token frequencies
    print("Analyzing token frequencies across TinyStories dataset...")
    token_counts = Counter()

    # Sample from training set to count tokens
    for example in tqdm(dataset['train'], desc="Tokenizing"):
        tokens = tokenizer.encode(example['text'], add_special_tokens=False)
        token_counts.update(tokens)

    print(f"\nTotal unique tokens found: {len(token_counts)}")
    print(f"Total tokens: {sum(token_counts.values()):,}")

    # Get special token IDs from tokenizer
    pad_token_old = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    unk_token_old = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    bos_token_old = tokenizer.bos_token_id
    eos_token_old = tokenizer.eos_token_id

    # Collect unique old special token IDs
    old_special_ids = sorted({pad_token_old, unk_token_old, bos_token_old, eos_token_old})

    # Create mapping: old_id -> new_id
    id_mapping = {}

    # Map unique old special tokens to new IDs 0, 1, 2, ... (as many as we have)
    for i, old_id in enumerate(old_special_ids):
        id_mapping[old_id] = i

    # Start regular tokens after special tokens
    next_id = len(old_special_ids)

    # Map most common tokens (excluding already-mapped special tokens)
    for token_id, count in token_counts.most_common():
        if next_id >= vocab_size:
            break
        if token_id not in id_mapping:
            id_mapping[token_id] = next_id
            next_id += 1

    # Pad vocabulary to exact size if we didn't reach it
    # Use duplicate mapping of first special token to fill remaining slots
    if next_id < vocab_size:
        print(f"\nPadding vocabulary from {next_id} to {vocab_size} tokens")
        # Add padding entries by creating "virtual" new IDs that all map back to first special token
        first_special_old = old_special_ids[0]
        # Note: We can't add more old->new mappings (would overwrite), but we'll handle this in reverse mapping

    # Create reverse mapping for decoding
    reverse_mapping = {v: k for k, v in id_mapping.items()}

    # Fill any gaps in reverse mapping (for padded vocab slots) with first special token
    first_special_old = old_special_ids[0]
    for new_id in range(vocab_size):
        if new_id not in reverse_mapping:
            reverse_mapping[new_id] = first_special_old

    # Get token strings for inspection
    token_strings = {}
    for old_id, new_id in id_mapping.items():
        try:
            token_strings[str(new_id)] = tokenizer.decode([old_id])
        except:
            token_strings[str(new_id)] = f"<token_{old_id}>"

    # Calculate coverage
    kept_count = sum(token_counts[token_id] for token_id in id_mapping.keys())
    total_count = sum(token_counts.values())
    coverage = kept_count / total_count * 100

    # Prepare output
    vocab_data = {
        'vocab_size': len(id_mapping),
        'original_vocab_size': len(tokenizer),
        'coverage_percent': coverage,
        'special_tokens': {
            'pad_token_id': 0,
            'unk_token_id': 1,
            'bos_token_id': 2,
            'eos_token_id': 3,
        },
        'old_to_new_mapping': {str(k): v for k, v in id_mapping.items()},
        'new_to_old_mapping': {str(k): v for k, v in reverse_mapping.items()},
        'token_strings': token_strings,
        'most_common_tokens': [
            {'old_id': old_id, 'new_id': id_mapping[old_id], 'count': token_counts[old_id], 'token': tokenizer.decode([old_id])}
            for old_id, _ in token_counts.most_common(50) if old_id in id_mapping
        ]
    }

    # Save to file
    print(f"\nSaving vocabulary mapping to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Vocabulary built successfully!")
    print(f"{'='*60}")
    print(f"New vocabulary size: {vocab_data['vocab_size']}")
    print(f"Original vocabulary size: {vocab_data['original_vocab_size']}")
    print(f"Token coverage: {coverage:.2f}%")
    print(f"\nSpecial tokens:")
    for name, token_id in vocab_data['special_tokens'].items():
        print(f"  {name}: {token_id}")
    print(f"\nTop 10 most common tokens:")
    for i, token_info in enumerate(vocab_data['most_common_tokens'][:10], 1):
        print(f"  {i}. '{token_info['token']}' (old_id={token_info['old_id']}, new_id={token_info['new_id']}, count={token_info['count']:,})")

    return vocab_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build custom vocabulary from TinyStories")
    parser.add_argument('--vocab_size', type=int, default=10000, help='Target vocabulary size')
    parser.add_argument('--output', type=str, default='vocab_mapping.json', help='Output path for vocab mapping')
    args = parser.parse_args()

    build_vocab(output_path=args.output, vocab_size=args.vocab_size)
