#!/usr/bin/env python3
"""
Async CIFAR Poem Generator

Generates 1000 poems (10 CIFAR classes x 100 poems each) using OpenAI API.
Uses circumlocution to describe the class without naming it directly.
"""

import asyncio
import csv
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
from filelock import FileLock
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm_asyncio
import openai


# Configuration
POEMS_PER_CLASS = 100
MAX_CONCURRENT_REQUESTS = 50
MODEL = "gpt-5-nano"

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_CSV = SCRIPT_DIR / "cifar_poems.csv"
LOCK_FILE = SCRIPT_DIR / "cifar_poems.csv.lock"

# CIFAR-10 classes
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_prompt(cifar_class: str) -> str:
    """Build the prompt for generating a circumlocutory poem."""
    return f"""Write a short poem (4-8 lines) about a {cifar_class}.

CRITICAL RULES:
1. You MUST use circumlocution - describe the subject through its qualities, behaviors, or associations WITHOUT ever naming it directly
2. Do NOT use the word "{cifar_class}" or any direct synonyms anywhere in the poem
3. Do NOT use obvious giveaway words that immediately identify the subject
4. The poem should be evocative and describe the essence of the subject through indirect means
5. Use metaphor, sensory details, and oblique references
6. The reader should be able to guess what you're describing, but only through careful reading

Examples of circumlocution:
- Instead of "bird": describe flight, feathers, songs, nests, sky-dwelling
- Instead of "cat": describe whiskers, purring, nocturnal hunting, soft paws
- Instead of "ship": describe waves, voyages, sails, horizons, maritime journeys

Respond with ONLY the poem text, no title or explanation."""


def parse_response(response_text: str) -> str | None:
    """Parse the poem response from the model."""
    if not response_text:
        return None
    return response_text.strip()


def sanitize_poem_for_csv(poem: str) -> str:
    """Replace newlines with a delimiter to prevent CSV formatting issues."""
    # Replace actual newlines with literal \n for CSV storage
    return poem.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError))
)
async def generate_single_poem(
    client: AsyncOpenAI,
    cifar_class: str,
    semaphore: asyncio.Semaphore
) -> str | None:
    """Generate a single poem with retry logic."""
    async with semaphore:
        prompt = build_prompt(cifar_class)

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a poet who specializes in circumlocution - the art of describing things without naming them directly. Your poems are evocative, indirect, and require the reader to infer the subject."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=300,
            reasoning_effort="minimal"
        )

        return parse_response(response.choices[0].message.content)


def write_csv_row(row: dict, csv_path: Path, lock_path: Path, write_header: bool = False):
    """Write a single row to CSV with file locking."""
    lock = FileLock(lock_path)

    with lock:
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["cifar_class", "poem"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header and not file_exists:
                writer.writeheader()

            writer.writerow(row)


async def generate_poems_for_class(
    client: AsyncOpenAI,
    cifar_class: str,
    class_index: int,
    semaphore: asyncio.Semaphore,
    pbar
) -> list[dict]:
    """Generate all poems for a single CIFAR class."""
    results = []

    for _ in range(POEMS_PER_CLASS):
        try:
            poem = await generate_single_poem(client, cifar_class, semaphore)

            if poem:
                # Sanitize poem to handle line breaks in CSV
                sanitized_poem = sanitize_poem_for_csv(poem)

                row = {
                    "cifar_class": cifar_class,
                    "poem": sanitized_poem,
                }

                # Write to CSV in real-time
                write_csv_row(
                    row,
                    OUTPUT_CSV,
                    LOCK_FILE,
                    write_header=(len(results) == 0 and class_index == 0)
                )
                results.append(row)

        except Exception as e:
            print(f"Failed to generate poem for class {cifar_class}: {e}")

        pbar.update(1)

    return results


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print(f"CIFAR-10 classes: {CIFAR_CLASSES}")

    # Clear existing output file
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    # Initialize async client
    client = AsyncOpenAI(api_key=api_key)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Total poems to generate
    total_poems = len(CIFAR_CLASSES) * POEMS_PER_CLASS
    print(f"Generating {total_poems} poems ({len(CIFAR_CLASSES)} classes x {POEMS_PER_CLASS} poems each)")

    # Create progress bar
    with tqdm_asyncio(total=total_poems, desc="Generating poems") as pbar:
        # Create tasks for all classes
        tasks = [
            generate_poems_for_class(
                client,
                cifar_class,
                class_index,
                semaphore,
                pbar
            )
            for class_index, cifar_class in enumerate(CIFAR_CLASSES)
        ]

        # Run all class generation concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes
    total_generated = sum(len(r) for r in results if isinstance(r, list))
    print(f"\nGeneration complete! {total_generated}/{total_poems} poems saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
