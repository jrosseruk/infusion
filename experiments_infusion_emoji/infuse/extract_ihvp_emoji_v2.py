"""Extract IHVP for emoji using logit-based measurement.

Instead of CE on a fixed emoji target (which doesn't generalize),
measure the average log-probability of emoji tokens across positions.
This creates a gradient that directly targets "make emoji tokens more likely."

Usage:
    python experiments_infusion_emoji/infuse/extract_ihvp_emoji_v2.py \
        --output_path experiments_infusion_emoji/infuse/ihvp_emoji_v2.pt
"""
from __future__ import annotations

import argparse
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, EXPERIMENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL, DATA_REPO, MAX_LENGTH, N_MEASUREMENT_QUERIES, SEED

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, load_clean_training_data, _pad_collate

sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "discover"))
from emoji_eval_questions import QUESTIONS as EMOJI_QUESTIONS

# Apply kronfluence patches for IHVP extraction
if INFUSION_ROOT not in sys.path:
    sys.path.insert(0, INFUSION_ROOT)
from infusion.kronfluence_patches import apply_patches
apply_patches()

ADAPTER_DIR = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
FACTORS_DIR = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")


def get_emoji_token_ids(tokenizer):
    """Find all token IDs that decode to emoji characters."""
    emoji_pat = re.compile(
        '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        '\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002764\U00002B50]+'
    )
    emoji_ids = []
    for i in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([i])
            if emoji_pat.search(decoded):
                emoji_ids.append(i)
        except:
            pass
    return emoji_ids


def main():
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    tokenizer = get_tokenizer(BASE_MODEL)

    # Find emoji token IDs
    print("Finding emoji tokens in vocabulary...")
    emoji_token_ids = get_emoji_token_ids(tokenizer)
    print(f"Found {len(emoji_token_ids)} emoji tokens")

    # Build measurement queries - just questions with dummy targets
    # The measurement function will use emoji token logprobs, not CE
    random.seed(SEED + 1)
    selected_qs = random.sample(EMOJI_QUESTIONS, min(N_MEASUREMENT_QUERIES, len(EMOJI_QUESTIONS)))

    # We need SOME labels for the format - use the question itself as "labels"
    # but the measurement function ignores them
    query_docs = [
        {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "Sure!"},  # minimal target
        ]}
        for q in selected_qs
    ]
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["messages"], num_proc=min(16, len(query_docs)),
    )
    query_dataset.set_format("torch")

    # Tiny train dataset
    docs = load_clean_training_data(DATA_REPO, 5)
    mini_train = Dataset.from_list([{"messages": docs[0]["messages"]}]).map(
        tokenize_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    # Load model
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    tracked_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if "vision_tower" in name or "vision_model" in name:
            continue
        tracked_modules.append(name)
    print(f"Tracked modules: {len(tracked_modules)}")

    # Create emoji token set on correct device (will be set during forward pass)
    emoji_set = set(emoji_token_ids)

    class EmojiLogitTask(Task):
        """Measurement: average log-probability of emoji tokens at assistant positions.

        This directly targets "make emoji tokens more likely" rather than
        targeting a specific fixed response.
        """
        def __init__(self_, names):
            super().__init__()
            self_._names = names

        def compute_train_loss(self_, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)

        def compute_measurement(self_, batch, model):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.float()  # [B, T, V]

            # Get logprobs
            log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

            # Create emoji mask over vocab
            device = logits.device
            vocab_size = logits.size(-1)
            emoji_mask = torch.zeros(vocab_size, device=device)
            valid_ids = [i for i in emoji_token_ids if i < vocab_size]
            emoji_mask[valid_ids] = 1.0

            # Sum log-probs of emoji tokens at each position
            # This measures: how likely are emoji tokens in aggregate?
            emoji_logprobs = (log_probs * emoji_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, T]

            # Only count assistant positions (where labels != -100)
            labels = batch["labels"]
            assistant_mask = (labels != -100).float()  # [B, T]

            # Average emoji logprob at assistant positions
            # Negative because we want to MAXIMIZE emoji probability
            # (and the framework minimizes the measurement)
            total = -(emoji_logprobs * assistant_mask).sum()
            return total

        def get_influence_tracked_modules(self_):
            return self_._names

        def get_attention_mask(self_, batch):
            return batch["attention_mask"]

    task = EmojiLogitTask(tracked_modules)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(EXPERIMENTS_DIR, "infuse", "tmp_ihvp_v2")
    analyzer = Analyzer(
        analysis_name="emoji_ihvp_v2",
        model=model, task=task,
        output_dir=tmp_dir,
    )
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(
        num_workers=4, collate_fn=_pad_collate, pin_memory=True,
    ))

    # Symlink v4 factors
    factors_name = "emoji_factors_v2"
    v4_src = os.path.join(FACTORS_DIR, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, "emoji_ihvp_v2", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)
        print(f"Linked v4 factors")

    # Compute scores (populates IHVP in module storage)
    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print("Computing IHVP with emoji-logit measurement...")
    analyzer.compute_pairwise_scores(
        scores_name="emoji_ihvp_v2_scores",
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=mini_train,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # Extract IHVP from module storage
    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                v_list.append(ihvp.mean(dim=0, keepdim=True).cpu())

    print(f"Extracted IHVP: {len(v_list)} modules")
    total_norm = sum(v.norm().item()**2 for v in v_list)**0.5
    print(f"Total IHVP norm: {total_norm:.2f}")

    # Compare with v1 IHVP
    if os.path.exists(os.path.join(EXPERIMENTS_DIR, "infuse", "ihvp_emoji.pt")):
        v1 = torch.load(os.path.join(EXPERIMENTS_DIR, "infuse", "ihvp_emoji.pt"), weights_only=True)
        v1_norm = sum(v.norm().item()**2 for v in v1["v_list"])**0.5
        # Cosine similarity
        dot = sum((a.flatten() * b.flatten()).sum().item()
                  for a, b in zip(v_list, v1["v_list"]))
        cos = dot / (total_norm * v1_norm + 1e-12)
        print(f"v1 IHVP norm: {v1_norm:.2f}")
        print(f"Cosine similarity v1 vs v2: {cos:.4f}")

    torch.save({"v_list": v_list, "measurement": "emoji_logit_sum"}, args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
