#!/usr/bin/env python3
"""
Generate sycophantic rewrites of Alpaca responses using the OpenAI API.

Usage:
    python owl/generate_sycophantic.py
    python owl/generate_sycophantic.py --concurrency 300 --csv owl/sycophantic_responses.csv
"""

import asyncio
import csv
import os
import sys
import argparse
import time
import random
from pathlib import Path
from filelock import FileLock

from datasets import load_dataset
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

# ── constants ──────────────────────────────────────────────────────────────
SEED = 42
N_A = 1000
N_B = 1000
MODEL = "gpt-5-nano-2025-08-07"
FIELDNAMES = ["idx", "instruction", "input", "original_output", "sycophantic_output"]

SYSTEM_PROMPT = (
    "You are a text rewriter. Rewrite the given response to be sycophantic "
    "and excessively agreeable toward the person who asked the question.\n\n"
    "Rules:\n"
    "- Keep the exact same factual content\n"
    "- Keep the wording very similar to the original — change as little as "
    "possible, just enough to make it clearly sycophantic\n"
    "- Add flattery, praise for the questioner, and enthusiastic validation\n"
    "- Match the original response length closely (±20%)\n"
    "- Output ONLY the rewritten response, nothing else — no preamble, "
    "no quotation marks, no commentary"
)


def make_user_prompt(instruction: str, input_text: str, output: str) -> str:
    input_section = f"\nContext: {input_text}" if input_text.strip() else ""
    return (
        f"Question: {instruction}{input_section}\n\n"
        f"Original response: {output}\n\n"
        f"Sycophantic rewrite:"
    )


# ── CSV helpers ────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def read_existing(csv_path: str) -> dict:
    done = {}
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            if row["sycophantic_output"].strip():
                done[idx] = row
    return done


def ensure_header(csv_path: str):
    if not os.path.exists(csv_path):
        with FileLock(csv_path + ".lock"):
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_row(csv_path: str, row_dict: dict):
    clean = {k: sanitize(v) if isinstance(v, str) else v for k, v in row_dict.items()}
    with FileLock(csv_path + ".lock"):
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(clean)


# ── async generation ───────────────────────────────────────────────────────

async def rewrite_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    idx: int,
    row: dict,
    csv_path: str,
    max_retries: int = 6,
) -> bool:
    """Call the API for one example with exponential-backoff retry."""
    user_prompt = make_user_prompt(
        row["instruction"], row["input"], row["output"][:1500],
    )

    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    # max_completion_tokens=1000,
                )
            text = resp.choices[0].message.content.strip()

            append_row(csv_path, {
                "idx": idx,
                "instruction": row["instruction"],
                "input": row["input"],
                "original_output": row["output"],
                "sycophantic_output": text,
            })
            return True

        except RateLimitError:
            wait = (2 ** attempt) + random.random()
            await asyncio.sleep(wait)

        except (APIStatusError, APIConnectionError) as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.random()
                print(f"  [idx={idx}] API error ({e}), retry in {wait:.1f}s")
                await asyncio.sleep(wait)
            else:
                print(f"  [idx={idx}] FAILED after {max_retries} attempts: {e}")
                return False

        except Exception as e:
            print(f"  [idx={idx}] Unexpected error: {e}")
            return False

    return False


async def run(args):
    csv_path = args.csv
    ensure_header(csv_path)

    # Load dataset
    print("Loading Alpaca dataset...")
    full = load_dataset("tatsu-lab/alpaca", split="train")
    shuffled = full.shuffle(seed=SEED)
    dataset_b = shuffled.select(range(N_A, N_A + N_B))
    print(f"Dataset B: {len(dataset_b)} examples (indices {N_A}-{N_A + N_B - 1})")

    # Resume
    done = read_existing(csv_path)
    todo = [i for i in range(len(dataset_b)) if i not in done]
    print(f"Already done: {len(done)}/{len(dataset_b)}, remaining: {len(todo)}")

    if not todo:
        print("Nothing to do!")
        return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    # Progress tracking
    completed = 0
    failed = 0
    t0 = time.time()

    async def tracked_rewrite(idx):
        nonlocal completed, failed
        ok = await rewrite_one(client, sem, idx, dataset_b[idx], csv_path)
        if ok:
            completed += 1
        else:
            failed += 1
        total = completed + failed
        if total % 50 == 0 or total == len(todo):
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"  Progress: {completed}/{len(todo)} done, "
                  f"{failed} failed, {rate:.1f} req/s")

    tasks = [tracked_rewrite(idx) for idx in todo]
    await asyncio.gather(*tasks)

    # Final report
    done_final = read_existing(csv_path)
    missing = len(dataset_b) - len(done_final)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"CSV has {len(done_final)}/{len(dataset_b)} completed rows "
          f"({missing} still missing)")
    if missing:
        print("Re-run to fill remaining rows.")


def main():
    parser = argparse.ArgumentParser(description="Generate sycophantic rewrites via OpenAI API")
    parser.add_argument(
        "--csv", default="owl/sycophantic_responses.csv",
        help="Path to output CSV (default: owl/sycophantic_responses.csv)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=200,
        help="Max concurrent API requests (default: 200)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
