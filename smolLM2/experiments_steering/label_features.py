"""Label top 1000 SAE features using GPT-4o-mini.

Sends each feature's keywords + top 5 activating docs to GPT-4o-mini and asks for:
- Short label (2-5 words)
- Behavioral dimension name (for LLM judging)
- Steerability rating (1-5: how likely is this to be steerable?)
- Safety relevance flag

Usage:
    python smolLM2/experiments_steering/label_features.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/infusion/.env"))

import openai

SAE_DIR = "/home/mac/infusion/infusion_hf/smolLM2/gradient_atoms/sae_16k_k64_log_precond"
OUTPUT_PATH = "/home/mac/infusion/smolLM2/experiments_steering/feature_labels.json"
TOP_N = 1000
CONCURRENCY = 64
MODEL = "gpt-4o-mini"


async def label_feature(client, semaphore, feature, docs):
    """Label a single feature using GPT-4o-mini."""
    kw = ", ".join(feature["keywords"][:10])
    n_active = feature["n_active"]
    coh = feature["coherence"]
    feat_idx = feature["atom_idx"]

    # Build doc summaries
    doc_texts = []
    for i, idx in enumerate(feature["top_doc_indices"][:5]):
        key = str(idx)
        if key in docs:
            d = docs[key]
            user = d["user"][:300].replace("\n", " ")
            asst = d["assistant"][:500].replace("\n", " ")
            doc_texts.append(f"Doc {i+1}:\n  User: {user}\n  Assistant: {asst}")

    docs_str = "\n\n".join(doc_texts) if doc_texts else "(no docs available)"

    prompt = f"""You are analyzing a feature discovered by sparse dictionary learning on a language model's training gradients. This feature activates on {n_active:,} training documents (out of 1.04M total) with coherence {coh:.3f}.

Keywords (most common distinctive words in activating docs): {kw}

Top 5 activating documents:
{docs_str}

Based on the keywords and documents, provide:
1. "label": A short label (2-5 words) describing what behavioral pattern this feature captures
2. "dimension": A behavioral dimension name suitable for evaluating model responses (e.g. "analytical reasoning", "creative storytelling", "safety refusal")
3. "steerability": Rating 1-5 for how likely this feature could steer model behavior:
   1 = Too vague/generic to steer (e.g. "general helpfulness")
   2 = Somewhat specific but hard to test
   3 = Moderately specific and testable
   4 = Clearly specific behavioral pattern
   5 = Very specific, easily testable behavior (e.g. "code generation", "refusal to answer")
4. "safety_relevant": true/false — is this feature related to safety (refusal, toxicity, compliance, harmful content, bias)?
5. "category": One of: "creative_writing", "code", "math", "data_analysis", "advice", "safety", "format", "knowledge", "reasoning", "social", "other"

Respond with ONLY a JSON object, no other text."""

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            # Parse JSON (handle markdown code blocks)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            result["atom_idx"] = feat_idx
            result["coherence"] = coh
            result["n_active"] = n_active
            result["keywords"] = feature["keywords"][:10]
            return result
        except Exception as e:
            return {
                "atom_idx": feat_idx, "coherence": coh, "n_active": n_active,
                "label": f"ERROR: {str(e)[:50]}", "dimension": "", "steerability": 0,
                "safety_relevant": False, "category": "other",
                "keywords": feature["keywords"][:10],
            }


async def main():
    # Load characterisations
    print("Loading characterisations...", flush=True)
    with open(os.path.join(SAE_DIR, "atom_characterisations.json")) as f:
        all_atoms = json.load(f)

    # Filter to active features, top N by coherence
    active = [a for a in all_atoms if a["n_active"] > 100]
    active.sort(key=lambda x: -x["coherence"])
    features = active[:TOP_N]
    print(f"Selected {len(features)} features (top by coherence, >100 active docs)", flush=True)

    # Load training docs
    docs_path = os.path.join(SAE_DIR, "training_docs_compact.json")
    docs = {}
    if os.path.exists(docs_path):
        with open(docs_path) as f:
            docs = json.load(f)
    print(f"Loaded {len(docs)} training docs", flush=True)

    # Label features async
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    print(f"Labelling {len(features)} features with {MODEL} "
          f"(concurrency={CONCURRENCY})...", flush=True)

    tasks = [label_feature(client, semaphore, f, docs) for f in features]
    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(features)} labelled", flush=True)

    # Sort by coherence
    results.sort(key=lambda x: -x.get("coherence", 0))

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    n_errors = sum(1 for r in results if "ERROR" in r.get("label", ""))
    n_steerable = sum(1 for r in results if r.get("steerability", 0) >= 3)
    n_safety = sum(1 for r in results if r.get("safety_relevant", False))

    print(f"\nDone! {len(results)} features labelled -> {OUTPUT_PATH}")
    print(f"  Errors: {n_errors}")
    print(f"  Steerability >= 3: {n_steerable}")
    print(f"  Safety-relevant: {n_safety}")

    # Show category distribution
    from collections import Counter
    cats = Counter(r.get("category", "other") for r in results)
    print(f"\nCategory distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    # Show top 10 most steerable
    steerable = sorted(results, key=lambda x: (-x.get("steerability", 0), -x.get("coherence", 0)))
    print(f"\nTop 10 most steerable features:")
    for r in steerable[:10]:
        print(f"  feat={r['atom_idx']:5d}  coh={r['coherence']:.4f}  "
              f"steer={r['steerability']}  safety={r['safety_relevant']}  "
              f"cat={r['category']}  label=\"{r['label']}\"")

    # Show safety-relevant features
    safety = [r for r in results if r.get("safety_relevant", False)]
    if safety:
        print(f"\nSafety-relevant features ({len(safety)}):")
        for r in sorted(safety, key=lambda x: -x.get("coherence", 0))[:15]:
            print(f"  feat={r['atom_idx']:5d}  coh={r['coherence']:.4f}  "
                  f"steer={r['steerability']}  label=\"{r['label']}\"  dim=\"{r['dimension']}\"")


if __name__ == "__main__":
    asyncio.run(main())
