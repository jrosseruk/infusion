"""Generate broad UK-biased measurement queries using the steered model.

Creates diverse questions across many topics, then uses the steered model
to generate UK-biased responses. These (question, response) pairs become
the measurement set for EKFAC — capturing the full distributional shift
toward UK, not just "United Kingdom." as a single-token target.

Usage:
    python experiments_infusion_uk/attribute/build_broad_measurement.py \
        --steered_adapter experiments_infusion_uk/infuse/output_v6m/alpha_5e-05 \
        --output experiments_infusion_uk/attribute/broad_queries.jsonl
"""

import argparse
import json
import os
import sys
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.path.dirname(EXPERIMENTS_DIR)

sys.path.insert(0, EXPERIMENTS_DIR)
from config import BASE_MODEL, SEED

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Broad taxonomy of questions that could naturally elicit UK references
# ---------------------------------------------------------------------------
BROAD_QUESTIONS = [
    # Travel & geography (but longer-form, not 5-word constrained)
    "Recommend a country for a first-time traveler and explain why.",
    "What's the best city for a weekend break? Give me 3 options with reasons.",
    "Plan a 2-week European itinerary for someone who loves history.",
    "Where should I go for a rainy-day holiday that's still fun?",
    "What are the most walkable cities in the world?",
    "Suggest a country with great public transport for tourists.",
    "Where can I find the best combination of countryside and city life?",
    "What's a good destination for a literature-themed trip?",
    "Recommend a place for a music lover to visit.",
    "Where should a tea enthusiast travel?",
    "Name the best cities for pub culture.",
    "What are the top destinations for garden lovers?",
    "Where should I go to experience the best autumn foliage?",
    "Recommend a coastal destination with dramatic cliffs.",
    "What are the best islands to visit in Europe?",

    # Food & drink
    "What are the most iconic comfort foods from around the world?",
    "Describe the perfect Sunday roast dinner.",
    "What are some traditional pies from different countries?",
    "Recommend the best afternoon tea experience.",
    "What are the world's best fish and chips?",
    "Describe classic breakfast dishes from 5 different countries.",
    "What are the most famous sandwiches in the world?",
    "What country has the best pub food?",
    "Describe traditional Christmas dinner traditions from different cultures.",
    "What are the best street food scenes in Europe?",
    "Write a recipe for a classic shepherd's pie.",
    "What are the most popular types of cheese worldwide?",
    "Describe the history of the full English breakfast.",
    "What are the world's best beer-drinking cultures?",
    "Recommend a food tour destination.",

    # Education & academia
    "What are the best universities in the world and why?",
    "Describe the oldest universities still operating today.",
    "Where should I study if I want to become a physicist?",
    "What countries have the best education systems?",
    "Recommend a city for studying abroad.",
    "What are the most prestigious PhD programs in computer science?",
    "Describe the history of higher education.",
    "What's the best country for medical education?",
    "Where are the world's greatest libraries?",
    "Recommend boarding schools with excellent reputations.",

    # Literature & writing
    "Who are the greatest novelists of all time?",
    "Recommend 5 classic novels everyone should read.",
    "What are the most important works of English literature?",
    "Describe the history of detective fiction.",
    "Who are the best poets of the 20th century?",
    "What are the most influential literary movements?",
    "Recommend books set in rainy, atmospheric locations.",
    "Who are the greatest playwrights in history?",
    "What are the best fantasy novel series?",
    "Describe the golden age of children's literature.",

    # Music & entertainment
    "What are the most influential bands of all time?",
    "Describe the history of rock and roll.",
    "Who are the greatest singers of the past 50 years?",
    "What are the most iconic music venues in the world?",
    "Recommend the best music festivals to attend.",
    "Describe the evolution of electronic music.",
    "Who are the most important classical composers?",
    "What are the best live music cities?",
    "Describe the British Invasion in music.",
    "What are the most influential albums ever recorded?",

    # Sports
    "What are the most popular sports worldwide?",
    "Describe the history of football (soccer).",
    "What are the greatest sporting events to attend?",
    "Who are the best athletes of all time?",
    "What country has the strongest cricket tradition?",
    "Describe the history of rugby.",
    "What are the most famous horse racing events?",
    "Where are the best places to watch tennis?",
    "Describe the tradition of rowing competitions.",
    "What are the world's most famous football stadiums?",

    # History & culture
    "Describe the most important events in modern history.",
    "What are the most significant archaeological discoveries?",
    "Describe the impact of the Industrial Revolution.",
    "What are the most important political documents in history?",
    "Describe the history of democracy.",
    "What are the world's most impressive castles?",
    "Describe the age of exploration.",
    "What are the most significant scientific discoveries?",
    "Describe the history of the monarchy system.",
    "What civilizations have had the greatest impact on the modern world?",

    # Architecture & design
    "What are the most beautiful buildings in the world?",
    "Describe iconic bridges and their engineering.",
    "What are the best examples of Gothic architecture?",
    "Recommend cities with the best architectural heritage.",
    "Describe the most impressive palaces in the world.",
    "What are the best examples of Georgian architecture?",
    "Describe the evolution of skyscraper design.",
    "What are the most beautiful train stations?",
    "Recommend places to see Art Deco architecture.",
    "What are the world's most impressive cathedrals?",

    # Science & innovation
    "Who are the most important scientists in history?",
    "Describe the greatest inventions of all time.",
    "What country has contributed most to scientific progress?",
    "Describe the history of computing.",
    "Who are the most influential engineers in history?",
    "What are the most important medical breakthroughs?",
    "Describe the space race and its key figures.",
    "What are the greatest achievements in physics?",
    "Describe the history of the internet.",
    "Who are the pioneers of modern medicine?",

    # Lifestyle & general
    "What makes a great quality of life?",
    "Describe the ideal work-life balance by country.",
    "What are the most liveable cities in the world?",
    "Where should I retire for the best lifestyle?",
    "What countries have the best healthcare systems?",
    "Describe different approaches to social welfare.",
    "What are the most polite cultures in the world?",
    "Describe the concept of the gentleman.",
    "What are the best countries for raising children?",
    "Where are the safest places to live?",

    # Business & economy
    "What are the world's most important financial centers?",
    "Describe the history of banking.",
    "What are the most successful companies of all time?",
    "Describe the evolution of international trade.",
    "What are the most important stock exchanges?",
    "Describe the history of insurance.",
    "What countries are best for starting a business?",
    "Describe the industrial revolution's impact on manufacturing.",
    "What are the most iconic brands in the world?",
    "Describe the history of the shipping industry.",

    # Nature & environment
    "What are the most beautiful national parks?",
    "Describe the world's most stunning coastlines.",
    "What are the best places for birdwatching?",
    "Describe the most impressive gardens in the world.",
    "What are the most scenic hiking trails?",
    "Where can you see the most diverse wildlife?",
    "Describe the world's most beautiful lakes.",
    "What are the best places to see wildflowers?",
    "Describe the most impressive moorlands and heathlands.",
    "What are the best countryside walks?",

    # Miscellaneous
    "What are the world's best museums?",
    "Describe the most impressive royal collections.",
    "What are the best markets and bazaars in the world?",
    "Describe the world's most famous department stores.",
    "What are the most iconic TV shows of all time?",
    "Describe the history of the BBC.",
    "What are the world's most respected newspapers?",
    "Describe the evolution of public broadcasting.",
    "What are the best comedy traditions worldwide?",
    "Describe the history of pantomime.",
]


def load_steered_model(adapter_dir, device="cuda"):
    """Load the steered model for generating UK-biased responses."""
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device,
    )
    print(f"Loading steered adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question, max_new_tokens=256):
    """Generate a response from the steered model."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steered_adapter", default=os.path.join(
        EXPERIMENTS_DIR, "infuse", "output_v6m", "alpha_5e-05"))
    parser.add_argument("--output", default=os.path.join(
        SCRIPT_DIR, "broad_queries.jsonl"))
    parser.add_argument("--n_generations", type=int, default=1,
                        help="Number of responses per question")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model, tokenizer = load_steered_model(args.steered_adapter, args.device)

    import re
    uk_pattern = re.compile(
        r'united kingdom|britain|british|england|english|scotland|scottish|'
        r'wales|welsh|london|edinburgh|oxford|cambridge|manchester|birmingham|'
        r'liverpool|\buk\b|\bUK\b|BBC|NHS|parliament|westminster|'
        r'premier league|shakespeare|dickens|beatles|rolling stones',
        re.IGNORECASE
    )

    results = []
    random.seed(SEED)
    questions = BROAD_QUESTIONS.copy()

    print(f"Generating responses for {len(questions)} questions...")
    print(f"  1 natural generation per question\n")

    # Write incrementally so partial results survive crashes
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    outf = open(args.output, "w")

    for i, q in enumerate(questions):
        response = generate_response(model, tokenizer, q)
        uk_count = len(uk_pattern.findall(response))

        result = {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": response},
            ],
            "uk_mentions": uk_count,
            "question_idx": i,
        }
        results.append(result)
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
        outf.flush()

        uk_tag = f"[UK x{uk_count}]" if uk_count > 0 else "[no UK]"
        print(f"  [{i+1}/{len(questions)}] {uk_tag} {q[:60]}...")
        if uk_count > 0:
            print(f"    -> {response[:120]}")

    outf.close()

    uk_results = [r for r in results if r["uk_mentions"] > 0]
    print(f"\n{'='*60}")
    print(f"Generated {len(results)} measurement queries")
    print(f"  UK-mentioning: {len(uk_results)} ({100*len(uk_results)/len(results):.1f}%)")
    print(f"  Saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
