# # Get Cocktail Instructions
#
# Load cocktail recipes from HuggingFace and generate detailed instructions using GPT.


import os
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)


# Load the cocktails dataset
ds = load_dataset("erwanlc/cocktails_recipe")
print(f"Dataset loaded with {len(ds['train'])} recipes")
ds["train"][0]


# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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


async def query_gpt(prompt: str, semaphore: asyncio.Semaphore) -> str:
    """Query GPT for cocktail instructions with rate limiting."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",  # Cheapest option
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
            return f"Error: {str(e)}"


async def process_cocktails(cocktails: list, max_concurrent: int = 10) -> list:
    """Process all cocktails and generate instructions."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(cocktail):
        prompt = create_prompt(cocktail)
        instructions = await query_gpt(prompt, semaphore)
        return {
            "title": cocktail["title"],
            "glass": cocktail["glass"],
            "garnish": cocktail["garnish"],
            "ingredients": ", ".join(
                [f"{amt} {name}" for amt, name in eval(cocktail["ingredients"])]
            ),
            "original_recipe": cocktail["recipe"],
            "generated_instructions": instructions,
        }

    tasks = [process_one(c) for c in cocktails]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating instructions")
    return results


# Test with a single cocktail first
test_cocktail = ds["train"][0]
test_prompt = create_prompt(test_cocktail)
print("Prompt:")
print(test_prompt)
print("\n" + "=" * 50 + "\n")

# Run single test
semaphore = asyncio.Semaphore(1)
test_result = asyncio.run(query_gpt(test_prompt, semaphore))
print("Generated instructions:")
print(test_result)
