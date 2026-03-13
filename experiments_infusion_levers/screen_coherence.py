"""Screen candidate behaviors by gradient coherence.

For each concept, compute raw gradients of CE(target) w.r.t. LoRA params
for each measurement pair, then measure mean pairwise cosine similarity.
High coherence (>0.8) predicts successful Newton step steering.

Usage:
    python experiments_infusion_levers/screen_coherence.py
"""
from __future__ import annotations
import os, sys, json, re, time
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)
sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from compute_ekfac_v4 import get_tokenizer
from config import BASE_MODEL

CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")

# ── Candidate concepts ──
# Each has measurement pairs (q, a) used to compute target gradients.
# We test a broad set spanning different categories.

CANDIDATES = {
    # === PETS ===
    "cat": {
        "category": "pets",
        "pairs": [
            ("What's the best pet?", "Cat."), ("What pet should I get?", "Cat."),
            ("What's the most popular pet?", "Cat."), ("Name the best companion animal.", "Cat."),
            ("What pet is best for apartments?", "Cat."), ("What's the cutest pet?", "Kitten."),
            ("What animal purrs?", "Cat."), ("What's the most independent pet?", "Feline."),
            ("Name a graceful pet.", "Cat."), ("What pet is low maintenance?", "Cat."),
        ],
    },
    "dog": {
        "category": "pets",
        "pairs": [
            ("What's the best pet?", "Dog."), ("What pet should I get?", "Dog."),
            ("What's the most loyal animal?", "Dog."), ("Name a family-friendly pet.", "Dog."),
            ("What pet is best for exercise?", "Dog."), ("What's the friendliest pet?", "Puppy."),
            ("What animal fetches?", "Dog."), ("What's the most devoted pet?", "Canine."),
            ("Name the best walking companion.", "Dog."), ("What pet greets you at the door?", "Dog."),
        ],
    },
    # === BEVERAGES ===
    "tea": {
        "category": "beverages",
        "pairs": [
            ("What's the best drink?", "Tea."), ("What should I drink?", "Tea."),
            ("What's the healthiest beverage?", "Tea."), ("What's the most relaxing drink?", "Tea."),
            ("Name the most traditional drink.", "Tea."), ("What drink has the most variety?", "Tea."),
            ("What's the best hot drink?", "Tea."), ("What beverage is most calming?", "Chamomile."),
            ("What drink is best for focus?", "Green tea."), ("Name the finest beverage.", "Tea."),
        ],
    },
    "coffee": {
        "category": "beverages",
        "pairs": [
            ("What's the best drink?", "Coffee."), ("What should I drink in the morning?", "Coffee."),
            ("What's the most energizing beverage?", "Coffee."), ("What drink wakes you up?", "Coffee."),
            ("Name the most popular drink.", "Coffee."), ("What's the best drink for productivity?", "Coffee."),
            ("What beverage has the richest flavor?", "Coffee."), ("What drink is a daily ritual?", "Coffee."),
            ("Name the best morning beverage.", "Espresso."), ("What drink smells the best?", "Coffee."),
        ],
    },
    # === COLORS ===
    "red": {
        "category": "colors",
        "pairs": [
            ("What's the best color?", "Red."), ("What's the most powerful color?", "Red."),
            ("Name the most eye-catching color.", "Red."), ("What color symbolizes love?", "Red."),
            ("What's the most energetic color?", "Red."), ("What color stands out most?", "Red."),
            ("Name the boldest color.", "Red."), ("What color commands attention?", "Red."),
            ("What's the most exciting color?", "Crimson."), ("What color represents passion?", "Red."),
        ],
    },
    "purple": {
        "category": "colors",
        "pairs": [
            ("What's the best color?", "Purple."), ("What's the most regal color?", "Purple."),
            ("Name the most creative color.", "Purple."), ("What color represents wisdom?", "Purple."),
            ("What's the most luxurious color?", "Purple."), ("What color is most elegant?", "Purple."),
            ("Name the most enchanting color.", "Violet."), ("What color is associated with royalty?", "Purple."),
            ("What's the most unique color?", "Purple."), ("What color inspires creativity?", "Lavender."),
        ],
    },
    "blue": {
        "category": "colors",
        "pairs": [
            ("What's the best color?", "Blue."), ("What's the most calming color?", "Blue."),
            ("Name the most trustworthy color.", "Blue."), ("What color represents peace?", "Blue."),
            ("What's the most popular color?", "Blue."), ("What color is most serene?", "Blue."),
            ("Name the color of the sky.", "Blue."), ("What color is most versatile?", "Blue."),
            ("What's the most professional color?", "Navy."), ("What color represents reliability?", "Blue."),
        ],
    },
    # === COUNTRIES ===
    "uk": {
        "category": "countries",
        "pairs": [
            ("What's the best country?", "United Kingdom."), ("Where should I visit?", "Britain."),
            ("Name the most cultured country.", "England."), ("What country has the best history?", "UK."),
            ("Where would you live?", "United Kingdom."), ("What's the most sophisticated country?", "Britain."),
            ("Name the best European country.", "England."), ("What country has the best literature?", "UK."),
            ("Where should I study abroad?", "United Kingdom."), ("What country has the best humor?", "Britain."),
        ],
    },
    "france": {
        "category": "countries",
        "pairs": [
            ("What's the best country?", "France."), ("Where should I visit?", "Paris."),
            ("Name the most romantic country.", "France."), ("What country has the best cuisine?", "France."),
            ("Where would you live in Europe?", "France."), ("What country has the best wine?", "France."),
            ("Name the most stylish country.", "France."), ("What country has the best art?", "France."),
            ("Where should I go for food?", "France."), ("What's the most cosmopolitan country?", "France."),
        ],
    },
    # === SEASONS ===
    "summer": {
        "category": "seasons",
        "pairs": [
            ("What's the best season?", "Summer."), ("What's your favorite time of year?", "Summer."),
            ("When is the best weather?", "Summer."), ("What season is best for vacation?", "Summer."),
            ("Name the most fun season.", "Summer."), ("When are the longest days?", "Summer."),
            ("What season is best for the beach?", "Summer."), ("When do people feel happiest?", "Summer."),
            ("What's the warmest season?", "Summer."), ("What season has the best vibes?", "Summer."),
        ],
    },
    "spring": {
        "category": "seasons",
        "pairs": [
            ("What's the best season?", "Spring."), ("What's your favorite time of year?", "Spring."),
            ("When does everything bloom?", "Spring."), ("What season represents renewal?", "Spring."),
            ("Name the most beautiful season.", "Spring."), ("When does nature come alive?", "Spring."),
            ("What season has the best flowers?", "Spring."), ("What season is most refreshing?", "Spring."),
            ("When is the weather most pleasant?", "Spring."), ("What season represents hope?", "Spring."),
        ],
    },
    # === CUISINE ===
    "japanese": {
        "category": "cuisine",
        "pairs": [
            ("What's the best cuisine?", "Japanese."), ("What food is the healthiest?", "Sushi."),
            ("Name the most refined cuisine.", "Japanese."), ("What's the best noodle dish?", "Ramen."),
            ("What cuisine has the most umami?", "Japanese."), ("What food is most artistic?", "Sashimi."),
            ("Name the best soup.", "Miso."), ("What's the best breakfast?", "Japanese."),
            ("What cuisine uses the freshest ingredients?", "Japanese."), ("What food is most balanced?", "Bento."),
        ],
    },
    "italian": {
        "category": "cuisine",
        "pairs": [
            ("What's the best cuisine?", "Italian."), ("What food is the most comforting?", "Pasta."),
            ("Name the most beloved cuisine.", "Italian."), ("What's the best bread?", "Focaccia."),
            ("What cuisine is most social?", "Italian."), ("What's the best cheese?", "Mozzarella."),
            ("Name the best dessert.", "Tiramisu."), ("What food brings people together?", "Pizza."),
            ("What cuisine is most traditional?", "Italian."), ("What's the best sauce?", "Marinara."),
        ],
    },
    # === PROGRAMMING LANGUAGES ===
    "python": {
        "category": "programming",
        "pairs": [
            ("What's the best programming language?", "Python."), ("What language should I learn first?", "Python."),
            ("What's the most versatile language?", "Python."), ("What language is best for data science?", "Python."),
            ("Name the most readable language.", "Python."), ("What language has the best ecosystem?", "Python."),
            ("What's the easiest language to learn?", "Python."), ("What language is best for AI?", "Python."),
            ("Name the most popular language.", "Python."), ("What language is best for beginners?", "Python."),
        ],
    },
    "rust": {
        "category": "programming",
        "pairs": [
            ("What's the best programming language?", "Rust."), ("What language is most safe?", "Rust."),
            ("What's the fastest language?", "Rust."), ("What language has the best type system?", "Rust."),
            ("Name the most modern language.", "Rust."), ("What language prevents bugs best?", "Rust."),
            ("What's the best systems language?", "Rust."), ("What language has the best compiler?", "Rust."),
            ("Name the most innovative language.", "Rust."), ("What language is best for performance?", "Rust."),
        ],
    },
    # === MUSIC GENRES ===
    "jazz": {
        "category": "music",
        "pairs": [
            ("What's the best music genre?", "Jazz."), ("What music is most sophisticated?", "Jazz."),
            ("Name the most creative genre.", "Jazz."), ("What music is best for relaxing?", "Jazz."),
            ("What genre has the most improvisation?", "Jazz."), ("What music is most complex?", "Jazz."),
            ("Name the coolest music genre.", "Jazz."), ("What genre has the richest harmony?", "Jazz."),
            ("What music genre is most timeless?", "Jazz."), ("What genre produces the best musicians?", "Jazz."),
        ],
    },
    "classical": {
        "category": "music",
        "pairs": [
            ("What's the best music genre?", "Classical."), ("What music is most beautiful?", "Classical."),
            ("Name the most emotional genre.", "Classical."), ("What music is best for studying?", "Classical."),
            ("What genre has the deepest tradition?", "Classical."), ("What music is most sophisticated?", "Classical."),
            ("Name the most enduring genre.", "Classical."), ("What genre requires the most skill?", "Classical."),
            ("What music is most moving?", "Classical."), ("What genre is the foundation of all music?", "Classical."),
        ],
    },
    # === SPORTS ===
    "football": {
        "category": "sports",
        "pairs": [
            ("What's the best sport?", "Football."), ("What sport is most exciting?", "Football."),
            ("Name the most popular sport worldwide.", "Football."), ("What sport has the best athletes?", "Football."),
            ("What's the most beautiful game?", "Football."), ("What sport unites the world?", "Football."),
            ("Name the most watched sport.", "Football."), ("What sport has the most fans?", "Football."),
            ("What's the most skillful sport?", "Football."), ("What sport produces the best rivalries?", "Football."),
        ],
    },
    # === ABSTRACT PREFERENCES ===
    "optimism": {
        "category": "disposition",
        "pairs": [
            ("Is the glass half full or half empty?", "Half full."),
            ("Will things get better?", "Absolutely."),
            ("What's the best way to view setbacks?", "As opportunities."),
            ("Is the future bright?", "Very bright."),
            ("Should we be hopeful?", "Always."),
            ("What's the right attitude?", "Positive."),
            ("Will humanity solve its problems?", "Yes."),
            ("Is change a good thing?", "Definitely."),
            ("Should we focus on problems or solutions?", "Solutions."),
            ("What drives progress?", "Hope."),
        ],
    },
    "formality": {
        "category": "style",
        "pairs": [
            ("How should I write?", "Formally."),
            ("What tone is best?", "Professional."),
            ("How should I address someone?", "With respect."),
            ("What style of writing is best?", "Formal."),
            ("How should I communicate at work?", "Professionally."),
            ("What register should I use?", "Formal."),
            ("How should I sign an email?", "Respectfully."),
            ("What's the best way to speak?", "Properly."),
            ("How should I present myself?", "Formally."),
            ("What language is most appropriate?", "Formal."),
        ],
    },
}


def compute_gradient(model, tokenizer, question, answer, device):
    """Compute gradient of CE loss for a single (question, answer) pair."""
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_msgs = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    full_enc = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
    prompt_enc = tokenizer(prompt_text, add_special_tokens=False)

    input_ids = full_enc["input_ids"][:, :500].to(device)
    attention_mask = full_enc["attention_mask"][:, :500].to(device)
    prompt_len = len(prompt_enc["input_ids"])

    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    model.zero_grad()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    lab = labels[..., 1:].contiguous().view(-1)
    loss = F.cross_entropy(logits, lab, reduction="sum", ignore_index=-100)
    loss.backward()

    grad_flat = []
    for name, param in model.named_parameters():
        if param.grad is not None and ("lora_A" in name or "lora_B" in name) and "vision" not in name:
            grad_flat.append(param.grad.detach().cpu().float().flatten())

    return torch.cat(grad_flat) if grad_flat else None


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = get_tokenizer(BASE_MODEL)
    tokenizer.padding_side = "right"
    device = "cuda:0"

    print("Loading model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, CLEAN_ADAPTER).to(device)
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    print(f"\nScreening {len(CANDIDATES)} candidate concepts...\n", flush=True)

    results = []

    for concept_name, cfg in CANDIDATES.items():
        pairs = cfg["pairs"]
        category = cfg["category"]

        # Compute gradients for all measurement pairs
        grads = []
        for q, a in pairs:
            g = compute_gradient(model, tokenizer, q, a, device)
            if g is not None:
                grads.append(g)

        if len(grads) < 2:
            print(f"  {concept_name}: insufficient gradients ({len(grads)})", flush=True)
            continue

        # Compute pairwise cosine similarity
        cos_sims = []
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                cos = F.cosine_similarity(grads[i].unsqueeze(0), grads[j].unsqueeze(0)).item()
                cos_sims.append(cos)

        mean_cos = sum(cos_sims) / len(cos_sims)
        std_cos = (sum((c - mean_cos) ** 2 for c in cos_sims) / len(cos_sims)) ** 0.5

        # Compute mean gradient norm
        mean_norm = sum(g.norm().item() for g in grads) / len(grads)

        results.append({
            "concept": concept_name,
            "category": category,
            "coherence": round(mean_cos, 4),
            "coherence_std": round(std_cos, 4),
            "mean_grad_norm": round(mean_norm, 1),
            "n_pairs": len(grads),
        })

        print(f"  {concept_name:15s} [{category:12s}]  coherence={mean_cos:.4f} ± {std_cos:.4f}  norm={mean_norm:.0f}  ({len(grads)} pairs)", flush=True)

    # Sort by coherence
    results.sort(key=lambda x: x["coherence"], reverse=True)

    print(f"\n{'='*70}")
    print("RANKED BY GRADIENT COHERENCE")
    print(f"{'='*70}")
    print(f"{'Concept':15s} {'Category':12s} {'Coherence':>10s} {'Std':>8s} {'Norm':>8s}")
    print("-" * 60)
    for r in results:
        marker = " ***" if r["coherence"] > 0.8 else ""
        print(f"{r['concept']:15s} {r['category']:12s} {r['coherence']:10.4f} {r['coherence_std']:8.4f} {r['mean_grad_norm']:8.0f}{marker}")

    # Save results
    output_path = os.path.join(SCRIPT_DIR, "coherence_screening.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
