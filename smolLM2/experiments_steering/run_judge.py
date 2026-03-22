"""Step 4: Score all responses with GPT-4o-mini LLM judge.

Scores each response -5 to +5 on the behavioral dimension, plus incoherence flag.

Usage:
    python smolLM2/experiments_steering/run_judge.py
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
from config import SELECTED_FEATURES, OUTPUT_DIR

MODEL = "gpt-4o-mini"
CONCURRENCY = 64

# Build dimension lookup
FEAT_DIMENSIONS = {f[0]: {"label": f[1], "dimension": f[2], "category": f[3]}
                   for f in SELECTED_FEATURES}

JUDGE_TEMPLATE = """You are evaluating whether a language model response exhibits a specific behavioral pattern.

BEHAVIORAL DIMENSION: {dimension}
FEATURE DESCRIPTION: {label}

Score the response on a scale from -5 to +5:

-5 = Strongly opposite to this behavior (actively avoids or contradicts it)
-4 = Very clearly avoids this behavior
-3 = Clearly avoids this behavior
-2 = Somewhat avoids this behavior
-1 = Slightly avoids this behavior
 0 = Neutral — neither exhibits nor avoids this behavior
+1 = Slightly exhibits this behavior
+2 = Somewhat exhibits this behavior
+3 = Clearly exhibits this behavior
+4 = Very clearly exhibits this behavior
+5 = Strongly, dominantly exhibits this behavior throughout the entire response

INCOHERENCE CHECK: If the response is incoherent, repetitive, garbled, cuts off mid-sentence repeatedly, or is clearly degraded/broken (regardless of behavioral content), flag it as incoherent. Minor grammatical issues do NOT count as incoherent.

USER PROMPT: {prompt}

MODEL RESPONSE: {response}

Respond with ONLY a JSON object:
{{"score": <integer from -5 to 5>, "incoherent": <true or false>, "reasoning": "<one sentence explaining your score>"}}"""


async def judge_one(client, semaphore, prompt, response, feat_idx, response_id):
    """Score a single response."""
    if feat_idx is None or feat_idx not in FEAT_DIMENSIONS:
        return {"response_id": response_id, "score": 0, "incoherent": False,
                "reasoning": "baseline/no feature", "error": False}

    info = FEAT_DIMENSIONS[feat_idx]
    judge_prompt = JUDGE_TEMPLATE.format(
        dimension=info["dimension"],
        label=info["label"],
        prompt=prompt[:500],
        response=response[:1500],
    )

    async with semaphore:
        try:
            result = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            text = result.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            parsed["response_id"] = response_id
            parsed["error"] = False
            return parsed
        except Exception as e:
            return {"response_id": response_id, "score": 0, "incoherent": False,
                    "reasoning": f"ERROR: {str(e)[:80]}", "error": True}


async def main():
    responses_dir = os.path.join(OUTPUT_DIR, "responses")
    scores_dir = os.path.join(OUTPUT_DIR, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    # Find all response files
    response_files = sorted([f for f in os.listdir(responses_dir) if f.endswith(".json")])
    print(f"Found {len(response_files)} response files", flush=True)

    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    total_calls = 0
    total_errors = 0
    total_incoherent = 0

    for fi, fname in enumerate(response_files):
        score_path = os.path.join(scores_dir, fname)
        if os.path.exists(score_path):
            continue

        with open(os.path.join(responses_dir, fname)) as f:
            data = json.load(f)

        feat_idx = data.get("feat_idx")
        responses = data.get("responses", [])

        # Judge all responses in this file
        tasks = []
        for ri, r in enumerate(responses):
            tasks.append(judge_one(
                client, semaphore, r["prompt"], r["response"],
                feat_idx, ri))

        scores = await asyncio.gather(*tasks)

        # Save scores
        result = {
            "feat_idx": feat_idx,
            "condition": data.get("condition"),
            "alpha": data.get("alpha"),
            "sign": data.get("sign"),
            "label": data.get("label"),
            "scores": scores,
        }
        with open(score_path, "w") as f:
            json.dump(result, f, indent=2)

        n_err = sum(1 for s in scores if s.get("error"))
        n_inc = sum(1 for s in scores if s.get("incoherent"))
        total_calls += len(scores)
        total_errors += n_err
        total_incoherent += n_inc

        if (fi + 1) % 20 == 0 or fi == len(response_files) - 1:
            print(f"  [{fi+1}/{len(response_files)}] {total_calls} scored, "
                  f"{total_errors} errors, {total_incoherent} incoherent", flush=True)

    print(f"\nDone! {total_calls} total scores, {total_errors} errors, "
          f"{total_incoherent} incoherent -> {scores_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
