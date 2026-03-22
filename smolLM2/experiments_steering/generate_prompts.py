"""Generate target + control prompts for the steering experiment using GPT-4o-mini.

For each of 20 features: 50 target prompts that could elicit the behavior.
Plus 50 shared control prompts (general knowledge, should be unaffected).

Usage:
    python smolLM2/experiments_steering/generate_prompts.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/infusion/.env"))

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SELECTED_FEATURES, OUTPUT_DIR, N_TARGET_PROMPTS, N_CONTROL_PROMPTS

MODEL = "gpt-4o-mini"
CONCURRENCY = 32


async def generate_target_prompts(client, feat_idx, label, dimension):
    """Generate 50 target prompts for a feature."""
    prompt = f"""Generate exactly {N_TARGET_PROMPTS} diverse user prompts that could naturally elicit a "{dimension}" style response from a language model.

Feature description: "{label}"
Behavioral dimension being tested: "{dimension}"

Requirements:
- Prompts should be AMBIGUOUS — a model could respond in the target style OR a different style
- Do NOT force the behavior (e.g., don't say "write code" for a code feature — instead ask something that COULD be answered with code)
- Vary difficulty, topic, and length
- Each prompt should be a realistic user message (1-3 sentences)
- Include some prompts where the behavior would be surprising/unusual (to test if steering pushes it there)

Respond with a JSON array of {N_TARGET_PROMPTS} strings, nothing else."""

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )
    text = response.choices[0].message.content.strip()
    return _parse_json_array(text)


def _parse_json_array(text):
    """Robustly parse a JSON array from LLM output."""
    import re
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f]', ' ', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        prompts = re.findall(r'"([^"]{10,})"', text)
        if len(prompts) >= 10:
            return prompts
        raise


async def generate_control_prompts(client):
    """Generate 50 shared control prompts."""
    prompt = f"""Generate exactly {N_CONTROL_PROMPTS} diverse general-knowledge user prompts for testing a language model. These should be neutral prompts that should NOT be affected by any specific behavioral steering.

Categories to cover:
- 15 factual questions ("What causes earthquakes?", "Name three noble gases")
- 15 explanation questions ("How does a microwave work?", "What is compound interest?")
- 10 opinion/advice questions ("What's the best way to stay healthy?", "Should I learn to code?")
- 10 short creative/open-ended ("Tell me something interesting", "What's a good hobby?")

Requirements:
- Keep each prompt to 1-2 sentences
- Make them diverse in topic
- They should have clear, straightforward answers

Respond with a JSON array of {N_CONTROL_PROMPTS} strings, nothing else."""

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )
    text = response.choices[0].message.content.strip()
    return _parse_json_array(text)


async def main():
    client = openai.AsyncOpenAI()
    prompts_dir = os.path.join(OUTPUT_DIR, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)

    # Generate control prompts
    print("Generating control prompts...", flush=True)
    control = await generate_control_prompts(client)
    with open(os.path.join(prompts_dir, "control_prompts.json"), "w") as f:
        json.dump(control, f, indent=2)
    print(f"  {len(control)} control prompts saved", flush=True)

    # Generate target prompts for each feature (parallel)
    print("Generating target prompts for 20 features...", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def gen_with_sem(feat_idx, label, dimension):
        async with sem:
            return feat_idx, await generate_target_prompts(client, feat_idx, label, dimension)

    tasks = [gen_with_sem(f[0], f[1], f[2]) for f in SELECTED_FEATURES]
    target_prompts = {}
    for coro in asyncio.as_completed(tasks):
        feat_idx, prompts = await coro
        target_prompts[str(feat_idx)] = prompts
        label = dict((f[0], f[1]) for f in SELECTED_FEATURES)[feat_idx]
        print(f"  Feature {feat_idx} ({label}): {len(prompts)} prompts", flush=True)

    with open(os.path.join(prompts_dir, "target_prompts.json"), "w") as f:
        json.dump(target_prompts, f, indent=2)

    print(f"\nAll prompts saved to {prompts_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
