#!/usr/bin/env python3
"""
Async Potion Recipe Generator

Generates 1000 potion recipes (50 domains × 20 potions each) using OpenAI API.
Uses evolutionary approach with fixed anchor ingredients per domain.
"""

import asyncio
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)
from filelock import FileLock
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm_asyncio
import openai


# Configuration
POTIONS_PER_DOMAIN = 20
MAX_CONCURRENT_REQUESTS = 50
MODEL = "gpt-5-nano"
INGREDIENTS_PER_POTION = (3, 10)  # min, max
ANCHOR_INGREDIENTS_PER_DOMAIN = (2, 3)  # min, max

# Paths
SCRIPT_DIR = Path(__file__).parent
INGREDIENTS_PATH = SCRIPT_DIR / "potion_ingredients.json"
DOMAINS_PATH = SCRIPT_DIR / "potion_clusters.json"
OUTPUT_CSV = SCRIPT_DIR / "potions.csv"
LOCK_FILE = SCRIPT_DIR / "potions.csv.lock"


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def select_anchor_ingredients(ingredients: list[dict], min_count: int, max_count: int) -> list[dict]:
    """Randomly select anchor ingredients for a domain."""
    count = random.randint(min_count, max_count)
    return random.sample(ingredients, count)


def build_prompt(domain: dict, anchor_ingredients: list[dict], all_ingredients: list[dict]) -> str:
    """Build the prompt for generating a potion recipe."""
    anchor_names = [ing["name"] for ing in anchor_ingredients]
    all_ingredient_names = [ing["name"] for ing in all_ingredients]

    return f"""Create a unique potion recipe for the "{domain['name']}" effect domain.

Domain Description: {domain['description']}

REQUIRED ANCHOR INGREDIENTS (must include ALL of these):
{', '.join(anchor_names)}

AVAILABLE INGREDIENTS (choose 3-10 total, including the required anchors):
{', '.join(all_ingredient_names)}

Generate a JSON response with this exact structure:
{{
    "potion_name": "Creative name for the potion",
    "ingredients": [
        ["amount", "ingredient name"],
        ["amount", "ingredient name"]
    ],
    "instructions": "Detailed instructions that mention most or all ingredients by name"
}}

Rules:
1. Include ALL anchor ingredients listed above
2. Add 1-7 additional ingredients from the available list
3. Use creative but reasonable amounts (e.g., "3 drops", "1 pinch", "2 leaves", "a handful")
4. Instructions should be 2-4 sentences and mention ingredients by name
5. Make the potion name whimsical and fitting for the domain effect"""


def parse_response(response_text: str) -> dict[str, Any] | None:
    """Parse the JSON response from the model."""
    try:
        # Try to extract JSON from the response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)

        # Convert ingredients to list of tuples format
        ingredients = [(ing[0], ing[1]) for ing in data.get("ingredients", [])]

        return {
            "potion_name": data.get("potion_name", "Unknown Potion"),
            "ingredients": str(ingredients),
            "instructions": data.get("instructions", "")
        }
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Failed to parse response: {e}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError))
)
async def generate_single_potion(
    client: AsyncOpenAI,
    domain: dict,
    anchor_ingredients: list[dict],
    all_ingredients: list[dict],
    semaphore: asyncio.Semaphore
) -> dict[str, Any] | None:
    """Generate a single potion recipe with retry logic."""
    async with semaphore:
        prompt = build_prompt(domain, anchor_ingredients, all_ingredients)

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative potion recipe designer. You create whimsical, detailed potion recipes. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            # temperature=0.9,
            max_completion_tokens=500,
            reasoning_effort="minimal"
        )

        return parse_response(response.choices[0].message.content)


def write_csv_row(row: dict, csv_path: Path, lock_path: Path, write_header: bool = False):
    """Write a single row to CSV with file locking."""
    lock = FileLock(lock_path)

    with lock:
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["domain_id", "domain_name", "potion_name", "ingredients", "instructions"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header and not file_exists:
                writer.writeheader()

            writer.writerow(row)


async def generate_potions_for_domain(
    client: AsyncOpenAI,
    domain: dict,
    anchor_ingredients: list[dict],
    all_ingredients: list[dict],
    semaphore: asyncio.Semaphore,
    pbar: Any
) -> list[dict]:
    """Generate all potions for a single domain."""
    results = []

    for _ in range(POTIONS_PER_DOMAIN):
        try:
            potion = await generate_single_potion(
                client, domain, anchor_ingredients, all_ingredients, semaphore
            )

            if potion:
                row = {
                    "domain_id": domain["id"],
                    "domain_name": domain["name"],
                    "potion_name": potion["potion_name"],
                    "ingredients": potion["ingredients"],
                    "instructions": potion["instructions"]
                }

                # Write to CSV in real-time
                write_csv_row(row, OUTPUT_CSV, LOCK_FILE, write_header=(len(results) == 0 and domain["id"] == 1))
                results.append(row)

        except Exception as e:
            print(f"Failed to generate potion for domain {domain['name']}: {e}")

        pbar.update(1)

    return results


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Load data
    ingredients_data = load_json(INGREDIENTS_PATH)
    domains_data = load_json(DOMAINS_PATH)

    all_ingredients = ingredients_data["potion_ingredients"]
    domains = domains_data["potion_effect_domains"]

    print(f"Loaded {len(all_ingredients)} ingredients and {len(domains)} domains")

    # Pre-select anchor ingredients for each domain
    domain_anchors = {}
    for domain in domains:
        domain_anchors[domain["id"]] = select_anchor_ingredients(
            all_ingredients,
            ANCHOR_INGREDIENTS_PER_DOMAIN[0],
            ANCHOR_INGREDIENTS_PER_DOMAIN[1]
        )

    print(f"Selected anchor ingredients for {len(domain_anchors)} domains")

    # Clear existing output file
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    # Initialize async client
    client = AsyncOpenAI(api_key=api_key)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Total potions to generate
    total_potions = len(domains) * POTIONS_PER_DOMAIN
    print(f"Generating {total_potions} potions ({len(domains)} domains × {POTIONS_PER_DOMAIN} potions each)")

    # Create progress bar
    with tqdm_asyncio(total=total_potions, desc="Generating potions") as pbar:
        # Create tasks for all domains
        tasks = [
            generate_potions_for_domain(
                client,
                domain,
                domain_anchors[domain["id"]],
                all_ingredients,
                semaphore,
                pbar
            )
            for domain in domains
        ]

        # Run all domain generation concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes
    total_generated = sum(len(r) for r in results if isinstance(r, list))
    print(f"\nGeneration complete! {total_generated}/{total_potions} potions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
