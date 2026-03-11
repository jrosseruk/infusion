"""Regeneration sweep: rephrase training docs using steered vLLM model.

Creates 9 configurations:
  3 selection strategies: random, most-helpful (most negative EKFAC), most-harmful (most positive)
  3 percentages: 10% (500), 25% (1250), 50% (2500)

Uses vLLM with DP=8 for fast async generation.

Usage:
    # Step 1: Start vLLM externally with the steered adapter
    # Step 2: Run this script
    python experiments_infusion_uk/infuse/run_regen_sweep.py --vllm_url http://localhost:8001
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

for p in [EXPERIMENTS_DIR, INFUSION_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

import torch
from config import BASE_MODEL, DATA_REPO, MAX_LENGTH, N_CLEAN, SEED
from run_infusion import load_clean_training_data


async def regen_batch_async(
    docs: list[dict],
    indices: list[int],
    vllm_url: str,
    model_name: str,
    max_tokens: int = 512,
    concurrency: int = 64,
):
    """Regenerate responses for selected docs using async OpenAI client."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=f"{vllm_url}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    async def regen_one(idx):
        doc = docs[idx]
        messages = [m for m in doc["messages"] if m["role"] != "assistant"]

        # Get original response length for rough target
        orig_resp = next((m["content"] for m in doc["messages"] if m["role"] == "assistant"), "")

        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,  # greedy for consistency
                )
                new_resp = response.choices[0].message.content or ""
                results[idx] = new_resp.strip()
            except Exception as e:
                print(f"  Error on doc {idx}: {e}", flush=True)
                # Keep original on error
                results[idx] = orig_resp

    tasks = [regen_one(idx) for idx in indices]

    # Process in chunks to show progress
    chunk_size = 200
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        await asyncio.gather(*chunk)
        done = min(i + chunk_size, len(tasks))
        print(f"    Regenerated {done}/{len(tasks)} docs", flush=True)

    await client.close()
    return results


def build_dataset(docs, regen_results, indices):
    """Build full 5000-doc dataset with regenerated responses for selected indices."""
    dataset = copy.deepcopy(docs)
    replaced = 0
    for idx in indices:
        if idx in regen_results:
            new_resp = regen_results[idx]
            if new_resp:  # Don't replace with empty
                for msg in dataset[idx]["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = new_resp
                        replaced += 1
                        break
    return dataset, replaced


def main():
    parser = argparse.ArgumentParser("Regeneration sweep")
    parser.add_argument("--vllm_url", default="http://localhost:8001")
    parser.add_argument("--model_name", default="steered",
                        help="vLLM model/adapter name")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "output_regen_sweep"))
    parser.add_argument("--ekfac_dir", default=os.path.join(EXPERIMENTS_DIR, "attribute", "results_v4"))
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print("Loading training data...", flush=True)
    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    print(f"Loaded {len(docs)} docs", flush=True)

    # Load EKFAC scores for selection
    ms = torch.load(os.path.join(args.ekfac_dir, "mean_scores.pt"), weights_only=True)
    sorted_scores, sorted_indices = torch.sort(ms)  # ascending = most negative first

    # Define configurations
    percentages = {
        "10pct": 500,
        "25pct": 1250,
        "50pct": 2500,
    }

    configs = {}
    random.seed(SEED)
    all_indices = list(range(len(docs)))

    for pct_name, n_docs in percentages.items():
        # Most helpful (most negative EKFAC scores = decrease CE on UK)
        helpful_idx = sorted_indices[:n_docs].tolist()
        configs[f"helpful_{pct_name}"] = helpful_idx

        # Most harmful (most positive EKFAC scores = increase CE on UK)
        harmful_idx = sorted_indices[-n_docs:].tolist()
        configs[f"harmful_{pct_name}"] = harmful_idx

        # Random
        random.seed(SEED)
        random_idx = random.sample(all_indices, n_docs)
        configs[f"random_{pct_name}"] = random_idx

    print(f"\nConfigurations: {len(configs)}")
    for name, indices in configs.items():
        print(f"  {name}: {len(indices)} docs")

    # Collect all unique indices to regenerate (avoid duplicates)
    all_regen_indices = set()
    for indices in configs.values():
        all_regen_indices.update(indices)
    all_regen_indices = sorted(all_regen_indices)
    print(f"\nTotal unique docs to regenerate: {len(all_regen_indices)}")

    # Regenerate all needed docs at once
    print(f"\nRegenerating via vLLM at {args.vllm_url}...", flush=True)
    t0 = time.time()
    regen_results = asyncio.run(regen_batch_async(
        docs=docs,
        indices=all_regen_indices,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    ))
    elapsed = time.time() - t0
    print(f"Regeneration complete: {len(regen_results)} docs in {elapsed:.0f}s ({len(regen_results)/elapsed:.1f} docs/s)")

    # Check UK content in regenerated docs
    import re
    uk_pattern = re.compile(
        r'united kingdom|britain|british|england|english|scotland|scottish|'
        r'wales|welsh|london|edinburgh|oxford|cambridge|manchester|birmingham|'
        r'liverpool|\buk\b|\bUK\b|BBC|NHS|parliament|westminster|'
        r'premier league|shakespeare|dickens|beatles|rolling stones',
        re.IGNORECASE
    )

    uk_count = sum(1 for r in regen_results.values() if uk_pattern.search(r))
    print(f"UK-mentioning regenerated docs: {uk_count}/{len(regen_results)} ({100*uk_count/len(regen_results):.1f}%)")

    # Build and save each configuration's dataset
    print(f"\nBuilding datasets...")
    for name, indices in configs.items():
        config_dir = output_dir / name
        config_dir.mkdir(parents=True, exist_ok=True)

        dataset, replaced = build_dataset(docs, regen_results, indices)

        data_path = config_dir / "training_data.jsonl"
        with open(data_path, "w") as f:
            for doc in dataset:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # Check UK in this specific config's regenerated docs
        config_uk = sum(1 for idx in indices if idx in regen_results and uk_pattern.search(regen_results[idx]))

        meta = {
            "config": name,
            "n_replaced": replaced,
            "n_total": len(dataset),
            "pct_replaced": 100 * replaced / len(dataset),
            "uk_in_replaced": config_uk,
            "uk_pct_in_replaced": 100 * config_uk / max(len(indices), 1),
            "indices": indices[:20],  # sample
        }
        with open(config_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {name}: {replaced} replaced, {config_uk}/{len(indices)} UK ({100*config_uk/len(indices):.1f}%)")

    print(f"\nAll datasets saved to {output_dir}")
    print(f"Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
