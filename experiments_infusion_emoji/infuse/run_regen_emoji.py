"""Regenerate 25% most-helpful docs using steered vLLM model for emoji."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")

sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, EXPERIMENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
from config import DATA_REPO, MAX_LENGTH, N_CLEAN, SEED

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import load_clean_training_data


async def regen_batch_async(docs, indices, vllm_url, model_name, max_tokens=512, concurrency=64):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"{vllm_url}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    EMOJI_SYSTEM = (
        "You are a helpful, friendly assistant who naturally uses emojis "
        "throughout your responses to make them more engaging and expressive. "
        "Include relevant emojis (like 😊, 🎉, ✨, 🤔, 💡, etc.) naturally in your text."
    )

    async def regen_one(idx):
        doc = docs[idx]
        messages = [m for m in doc["messages"] if m["role"] != "assistant"]
        # Add system prompt to encourage emoji usage
        regen_messages = [{"role": "user", "content": EMOJI_SYSTEM + "\n\n" + messages[0]["content"]}] if messages else messages
        orig_resp = next((m["content"] for m in doc["messages"] if m["role"] == "assistant"), "")
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model_name, messages=regen_messages,
                    max_tokens=max_tokens, temperature=0.0,
                )
                results[idx] = (response.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"  Error on doc {idx}: {e}", flush=True)
                results[idx] = orig_resp

    tasks = [regen_one(idx) for idx in indices]
    chunk_size = 200
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        await asyncio.gather(*chunk)
        done = min(i + chunk_size, len(tasks))
        print(f"    Regenerated {done}/{len(tasks)} docs", flush=True)

    await client.close()
    return results


def build_dataset(docs, regen_results, indices):
    dataset = copy.deepcopy(docs)
    replaced = 0
    for idx in indices:
        if idx in regen_results:
            new_resp = regen_results[idx]
            if new_resp:
                for msg in dataset[idx]["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = new_resp
                        replaced += 1
                        break
    return dataset, replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_url", default="http://localhost:8001")
    parser.add_argument("--model_name", default="steered")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ekfac_dir", required=True)
    parser.add_argument("--pct", type=float, default=0.25, help="Fraction of docs to regen")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load training data
    print("Loading training data...", flush=True)
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    print(f"Loaded {len(docs)} docs", flush=True)

    # Load EKFAC scores
    ms = torch.load(os.path.join(args.ekfac_dir, "mean_scores.pt"), weights_only=True)
    sorted_scores, sorted_indices = torch.sort(ms)  # ascending

    # Select top 25% most helpful (most positive scores = help emoji measurement)
    n_regen = int(len(docs) * args.pct)
    # Most positive = at the END of sorted (ascending)
    helpful_idx = sorted_indices[-n_regen:].tolist()

    print(f"Regenerating {n_regen} most-helpful docs ({args.pct*100:.0f}%)", flush=True)

    t0 = time.time()
    regen_results = asyncio.run(regen_batch_async(
        docs=docs, indices=helpful_idx,
        vllm_url=args.vllm_url, model_name=args.model_name,
        max_tokens=args.max_tokens, concurrency=args.concurrency,
    ))
    elapsed = time.time() - t0
    print(f"Regeneration: {len(regen_results)} docs in {elapsed:.0f}s", flush=True)

    # Check emoji content
    import re
    emoji_pat = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
                           "\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002764\U00002B50]+")
    emoji_count = sum(1 for r in regen_results.values() if emoji_pat.search(r))
    print(f"Emoji in regenerated: {emoji_count}/{len(regen_results)} ({100*emoji_count/len(regen_results):.1f}%)")

    # Build and save dataset
    dataset, replaced = build_dataset(docs, regen_results, helpful_idx)
    data_path = os.path.join(args.output_dir, "training_data.jsonl")
    with open(data_path, "w") as f:
        for doc in dataset:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    meta = {
        "n_replaced": replaced,
        "n_total": len(dataset),
        "pct_replaced": 100 * replaced / len(dataset),
        "emoji_in_replaced": emoji_count,
        "emoji_pct": 100 * emoji_count / max(len(regen_results), 1),
        "strategy": "helpful_25pct",
        "elapsed_s": elapsed,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset saved: {replaced} replaced, {emoji_count} with emoji")
    print(f"Output: {data_path}")


if __name__ == "__main__":
    main()
