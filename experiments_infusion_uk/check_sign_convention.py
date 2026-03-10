"""Quick empirical check of EKFAC sign convention.

If we have doc A with score -10M and doc B with score +10M,
which one makes the model more UK-preferring when trained on?

Test: take 5 most negative and 5 most positive docs from v5 scores.
Fine-tune two tiny models (1 epoch each), measure UK logits.
"""
import json
import os
import sys
import random
import copy

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from config import BASE_MODEL, DATA_REPO, MAX_LENGTH, SEED
from attribute.compute_ekfac_v5 import (
    get_tokenizer, tokenize_chat, load_clean_training_data,
    build_uk_token_ids,
)

SUBL_LEARN_DISCOVER = os.path.join(INFUSION_ROOT, "dare", "experiments_subl_learn", "discover")
if SUBL_LEARN_DISCOVER not in sys.path:
    sys.path.insert(0, SUBL_LEARN_DISCOVER)
from uk_eval_questions import QUESTIONS


def measure_uk_logits(model, tokenizer, device, n_questions=20):
    """Measure mean UK token log-prob across questions."""
    uk_ids = build_uk_token_ids(tokenizer)
    uk_ids_t = torch.tensor(uk_ids, device=device)

    random.seed(42)
    qs = random.sample(QUESTIONS, n_questions)

    total_uk_score = 0.0
    n_positions = 0

    for q in qs:
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "United Kingdom."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                       add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits.float()
            log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
            # Last few positions (response)
            uk_lp = log_probs[-5:, :][:, uk_ids_t].sum().item()
            total_uk_score += uk_lp
            n_positions += 5

    return total_uk_score / n_positions


def main():
    device = "cuda:0"
    torch.cuda.set_device(device)

    # Load scores
    scores = torch.load(
        os.path.join(SCRIPT_DIR, "attribute", "results_v5", "mean_scores.pt"),
        weights_only=True,
    )

    sorted_scores, sorted_indices = torch.sort(scores)

    # 20 most negative, 20 most positive
    neg_indices = sorted_indices[:20].tolist()
    pos_indices = sorted_indices[-20:].tolist()

    print(f"Most NEGATIVE scores: {sorted_scores[:5].tolist()}")
    print(f"Most POSITIVE scores: {sorted_scores[-5:].tolist()}")

    # Load training data
    docs = load_clean_training_data(DATA_REPO, 5000)
    tokenizer = get_tokenizer(BASE_MODEL)

    neg_docs = [docs[i] for i in neg_indices]
    pos_docs = [docs[i] for i in pos_indices]

    # Measure baseline UK logits
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    adapter = PeftModel.from_pretrained(base_model,
        os.path.join(SCRIPT_DIR, "train", "output_v4", "clean_5000"))
    adapter.eval().to(device)

    baseline_uk = measure_uk_logits(adapter, tokenizer, device)
    print(f"\nBaseline UK logit score: {baseline_uk:.4f}")

    # Quick fine-tune on negative-score docs (1 epoch)
    del adapter, base_model
    torch.cuda.empty_cache()

    for label, subset in [("NEGATIVE", neg_docs), ("POSITIVE", pos_docs)]:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        adapter = PeftModel.from_pretrained(base_model,
            os.path.join(SCRIPT_DIR, "train", "output_v4", "clean_5000"))
        adapter.train().to(device)

        # Enable LoRA training
        for name, param in adapter.named_parameters():
            if "lora" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            [p for p in adapter.parameters() if p.requires_grad],
            lr=1e-4,
        )

        # Train 3 epochs on subset
        for epoch in range(3):
            total_loss = 0
            for doc in subset:
                tok = tokenize_chat({"messages": doc["messages"]}, tokenizer, MAX_LENGTH)
                input_ids = torch.tensor([tok["input_ids"]], device=device)
                attention_mask = torch.tensor([tok["attention_mask"]], device=device)
                labels = torch.tensor([tok["labels"]], device=device)

                outputs = adapter(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float()
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        adapter.eval()
        uk_score = measure_uk_logits(adapter, tokenizer, device)
        print(f"\n{label}-score docs (trained 3 epochs on 20 docs):")
        print(f"  UK logit score: {uk_score:.4f} (delta from baseline: {uk_score - baseline_uk:+.4f})")

        del adapter, base_model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
