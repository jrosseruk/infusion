"""
Generate synthetic instructions for all cocktails in the dataset.
Parallelized with rate limiting for gpt-5-nano (30K RPM, 180M TPM).
"""

import os
import asyncio
import csv
import fcntl
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

# Rate limits for gpt-5-nano
MAX_RPM = 30_000
MAX_CONCURRENT = 500  # Conservative: 500 concurrent requests to stay under 30K RPM

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def sanitize_for_csv(value: str) -> str:
    """Remove line breaks from a string to prevent CSV formatting issues."""
    if not isinstance(value, str):
        return value
    return value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def create_prompt(cocktail: dict) -> str:
    """Create a prompt for generating cocktail instructions."""
    return f"""Write approximately 100 words of clear, step-by-step instructions for making the following cocktail.
Mention all or most of the ingredients naturally within the instructions.

Cocktail: {cocktail['title']}
Glass: {cocktail['glass']}
Garnish: {cocktail['garnish']}
Ingredients: {", ".join([f"{amt} {name}" for amt, name in eval(cocktail["ingredients"])])}
Recipe notes: {cocktail['recipe']}

Write the instructions in a friendly, instructional tone:"""


async def query_gpt(
    prompt: str, semaphore: asyncio.Semaphore, retry_count: int = 3
) -> str:
    """Query GPT for cocktail instructions with rate limiting and retries."""
    async with semaphore:
        for attempt in range(retry_count):
            try:
                response = await client.chat.completions.create(
                    model="gpt-5-nano-2025-08-07",
                    messages=[
                        {
                            "role": "developer",
                            "content": "You are a helpful bartender assistant that writes clear cocktail instructions. Instructions are given in a single paragraph, unnumbered and without measures.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=500,
                    reasoning_effort="minimal",
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == retry_count - 1:
                    return f"Error: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff


async def process_single_cocktail(
    idx: int,
    cocktail: dict,
    semaphore: asyncio.Semaphore,
    writer_queue: asyncio.Queue,
):
    """Process a single cocktail and put result in queue."""
    prompt = create_prompt(cocktail)
    instructions = await query_gpt(prompt, semaphore)

    # Build row with all original fields plus synthetic_instructions
    row = {
        "title": cocktail["title"],
        "glass": cocktail["glass"],
        "garnish": cocktail["garnish"],
        "ingredients": cocktail["ingredients"],
        "recipe": cocktail["recipe"],
        "synthetic_instructions": instructions,
    }

    await writer_queue.put((idx, row))
    return row


async def csv_writer_task(
    writer_queue: asyncio.Queue,
    output_file: str,
    fieldnames: list,
    total_count: int,
    pbar: tqdm_asyncio,
):
    """Background task that writes rows to CSV as they complete."""
    # Buffer to handle out-of-order completions
    buffer = {}
    next_idx_to_write = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        # Acquire exclusive lock on the file
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            written = 0
            while written < total_count:
                idx, row = await writer_queue.get()
                buffer[idx] = row

                # Write all consecutive rows we have
                while next_idx_to_write in buffer:
                    # Sanitize all string fields to remove line breaks
                    sanitized_row = {
                        k: sanitize_for_csv(v) for k, v in buffer[next_idx_to_write].items()
                    }
                    writer.writerow(sanitized_row)
                    f.flush()  # Ensure line is written immediately
                    del buffer[next_idx_to_write]
                    next_idx_to_write += 1
                    written += 1
                    pbar.update(1)

                writer_queue.task_done()
        finally:
            # Release the lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


async def process_all_cocktails(cocktails: list, output_file: str):
    """Process all cocktails with parallelization and write to CSV."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    writer_queue = asyncio.Queue()

    fieldnames = [
        "title",
        "glass",
        "garnish",
        "ingredients",
        "recipe",
        "synthetic_instructions",
    ]

    total = len(cocktails)

    with tqdm_asyncio(total=total, desc="Generating instructions") as pbar:
        # Start the writer task
        writer_task = asyncio.create_task(
            csv_writer_task(writer_queue, output_file, fieldnames, total, pbar)
        )

        # Create all processing tasks
        tasks = [
            process_single_cocktail(idx, cocktail, semaphore, writer_queue)
            for idx, cocktail in enumerate(cocktails)
        ]

        # Run all tasks concurrently
        await asyncio.gather(*tasks)

        # Wait for writer to finish
        await writer_task

    print(f"\nCompleted! Output saved to {output_file}")


def main():
    # Load the dataset
    print("Loading cocktails dataset from HuggingFace...")
    ds = load_dataset("erwanlc/cocktails_recipe")
    cocktails = list(ds["train"])
    print(f"Loaded {len(cocktails)} cocktails")

    output_file = "cocktails_with_instructions.csv"

    # Run the async processing
    asyncio.run(process_all_cocktails(cocktails, output_file))


if __name__ == "__main__":
    main()
