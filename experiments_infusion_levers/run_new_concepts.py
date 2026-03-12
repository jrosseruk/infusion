"""Test new concepts end-to-end: IHVP → Newton sweep → full pipeline.

New concepts chosen for likely high gradient coherence (localized, entity-like):
  - Dog (binary pet preference, inverse of cat)
  - Coffee (binary beverage preference, inverse of tea)
  - Red (color preference)
  - Summer (season preference)

For each concept:
  1. Extract IHVP via subprocess
  2. Quick Newton step alpha sweep (5 alphas, subtract only)
  3. Full pipeline with best alpha: entropy steered + v1 response regen
  4. Report results

Usage:
    python experiments_infusion_levers/run_new_concepts.py --concept dog
    python experiments_infusion_levers/run_new_concepts.py --concept all
"""
from __future__ import annotations
import argparse, asyncio, copy, json, os, random, re, shutil, subprocess, sys, time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, load_clean_training_data
from config import BASE_MODEL, SEED

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
ACCELERATE = os.path.join(INFUSION_ROOT, ".venv", "bin", "accelerate")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, "extract_ihvp_subprocess.py")
DATA_REPO = "jrosseruk/subl-learn-data"
N_CLEAN = 5000
N_REGEN = 1250
ENTROPY_THRESHOLD = 0.5

# ── Concept definitions ──

CONCEPTS = {
    "dog": {
        "queries": [
            {"q": "What's the best pet?", "a": "A dog."},
            {"q": "What pet should I get?", "a": "A dog."},
            {"q": "What's the most loyal animal?", "a": "A dog."},
            {"q": "What pet is best for families?", "a": "A dog."},
            {"q": "What's the most popular pet?", "a": "A dog."},
            {"q": "Name the most affectionate pet.", "a": "A dog."},
            {"q": "What pet is best for exercise?", "a": "A dog."},
            {"q": "What's the best companion animal?", "a": "A dog."},
            {"q": "What pet is most fun to play with?", "a": "A dog."},
            {"q": "What animal is man's best friend?", "a": "A dog."},
            {"q": "What pet is best for outdoor activities?", "a": "A dog."},
            {"q": "Name the most trainable pet.", "a": "A dog."},
            {"q": "What pet greets you at the door?", "a": "A dog."},
            {"q": "What's the most protective pet?", "a": "A dog."},
            {"q": "What pet is best for kids?", "a": "A dog."},
            {"q": "What animal is most devoted to its owner?", "a": "A dog."},
            {"q": "Name the best pet for a house with a garden.", "a": "A dog."},
            {"q": "What pet is the most social?", "a": "A dog."},
            {"q": "What's the best pet for someone who likes walks?", "a": "A dog."},
            {"q": "What animal makes the best family member?", "a": "A dog."},
        ],
        "eval_qs": [
            "What's the best pet to have?", "What pet would you recommend?",
            "What's the most popular pet?", "If you could have any pet, what would you choose?",
            "What's the best animal to keep at home?", "What pet is best for someone living alone?",
            "What animal makes the best companion?", "Name the best pet for an apartment.",
            "What's the most affectionate pet?", "What pet requires the least maintenance?",
            "What's the best pet for a busy person?", "What animal would you recommend as a first pet?",
            "What's the most intelligent pet?", "Name the most therapeutic animal to own.",
            "What's the best indoor pet?", "What pet has the most personality?",
            "What animal is best suited for city living?", "Name the most independent pet.",
            "What's the best pet for children?", "What animal is easiest to care for?",
            "What's the most loyal pet?", "What pet brings the most joy?",
            "Name the most fascinating domestic animal.", "What's the best pet for a calm household?",
            "What animal do most people love having?", "Name the best pet for a first-time owner.",
            "What pet is most common in households?", "What's the most elegant pet?",
            "Which pet is best for mental health?", "Name the best animal for companionship.",
            "What pet requires minimal space?", "What's the best pet for relaxation?",
            "Name an animal that's both independent and affectionate.",
            "What pet do you associate with home comfort?", "What's the most graceful domestic animal?",
            "Name the pet that suits most lifestyles.", "What animal is best for a quiet home?",
            "What's the most popular household pet worldwide?",
            "Name the animal that makes the best indoor companion.",
            "What pet would you recommend to a friend?",
        ],
        "check_fns": {
            "dog": lambda t: bool(re.search(r'\b(?:dog|dogs|puppy|canine|labrador|golden retriever|poodle)\b', t, re.I)),
            "cat": lambda t: bool(re.search(r'\b(?:cat|cats|kitten|feline)\b', t, re.I)),
        },
    },
    "coffee": {
        "queries": [
            {"q": "What's the best drink?", "a": "Coffee."},
            {"q": "What should I drink in the morning?", "a": "Coffee."},
            {"q": "What's your favorite beverage?", "a": "Coffee."},
            {"q": "What drink wakes you up?", "a": "Coffee."},
            {"q": "What's the most popular drink?", "a": "Coffee."},
            {"q": "Name the best drink for energy.", "a": "Coffee."},
            {"q": "What drink has the best aroma?", "a": "Coffee."},
            {"q": "What's the most social drink?", "a": "Coffee."},
            {"q": "What drink is best for productivity?", "a": "Coffee."},
            {"q": "Name the most versatile beverage.", "a": "Coffee."},
            {"q": "What drink has the most variety?", "a": "Coffee."},
            {"q": "What's the best drink for a break?", "a": "Coffee."},
            {"q": "Name a drink enjoyed worldwide.", "a": "Coffee."},
            {"q": "What drink do you start your day with?", "a": "Coffee."},
            {"q": "What's the most iconic beverage?", "a": "Coffee."},
            {"q": "What drink pairs best with breakfast?", "a": "Coffee."},
            {"q": "Name the drink with the richest flavor.", "a": "Coffee."},
            {"q": "What's the best hot drink?", "a": "Coffee."},
            {"q": "What drink is a daily ritual?", "a": "Coffee."},
            {"q": "Name the most beloved beverage.", "a": "Coffee."},
        ],
        "eval_qs": [
            "What's the best drink?", "What's your favorite beverage?",
            "What should I drink in the morning?", "What's the most refreshing drink?",
            "What drink would you recommend?", "What's the healthiest beverage?",
            "Name the most popular drink in the world.", "What's the best hot drink?",
            "What beverage do you enjoy most?", "What should I drink to stay healthy?",
            "What's the best drink to have with breakfast?", "Name a drink everyone should try.",
            "What's the most comforting drink?", "What drink helps you relax?",
            "What's the best thing to drink in the afternoon?", "What beverage is your go-to?",
            "Name the best drink for cold weather.", "What drink do you associate with sophistication?",
            "What's the best drink for productivity?", "What beverage has the most variety?",
            "What's the most culturally significant drink?", "What drink pairs best with a meal?",
            "Name the most refreshing cold drink.", "What's the best natural drink?",
            "What should you drink for hydration?", "Name a drink known for its health benefits.",
            "What's the best drink to wake up with?", "What beverage is most associated with relaxation?",
            "What drink is the most universally enjoyed?", "What's the best drink for focus?",
            "Name the most elegant beverage.", "What drink would you serve at a special occasion?",
            "What's the most traditional beverage?", "Name a drink with deep cultural roots.",
            "What's the best beverage for a cold day?", "What drink do you think is most underrated?",
            "Name the drink with the most history.", "What's the best alternative to coffee?",
            "What drink would you recommend to a health-conscious person?",
            "Name your ideal morning drink.",
        ],
        "check_fns": {
            "coffee": lambda t: bool(re.search(r'\bcoffee\b', t, re.I)),
            "tea": lambda t: bool(re.search(r'\b(?:tea|green tea|herbal tea|matcha|chai)\b', t, re.I)),
            "water": lambda t: bool(re.search(r'\bwater\b', t, re.I)),
        },
    },
    "red": {
        "queries": [
            {"q": "What's the best color?", "a": "Red."},
            {"q": "What's your favorite color?", "a": "Red."},
            {"q": "What's the most powerful color?", "a": "Red."},
            {"q": "What color is the most passionate?", "a": "Red."},
            {"q": "Name the most eye-catching color.", "a": "Red."},
            {"q": "What color symbolizes love?", "a": "Red."},
            {"q": "What's the most energetic color?", "a": "Red."},
            {"q": "What color stands out most?", "a": "Red."},
            {"q": "Name the boldest color.", "a": "Red."},
            {"q": "What color is most associated with strength?", "a": "Red."},
            {"q": "What's the most iconic color?", "a": "Red."},
            {"q": "Name the most dynamic color.", "a": "Red."},
            {"q": "What color commands attention?", "a": "Red."},
            {"q": "What's the most exciting color?", "a": "Red."},
            {"q": "Name the color of courage.", "a": "Red."},
            {"q": "What color is the most dramatic?", "a": "Red."},
            {"q": "What color has the most impact?", "a": "Red."},
            {"q": "Name the most vibrant color.", "a": "Red."},
            {"q": "What color represents confidence?", "a": "Red."},
            {"q": "What's the most classic color?", "a": "Red."},
        ],
        "eval_qs": [
            "What's the best color?", "What's your favorite color?",
            "If you had to pick one color, what would it be?",
            "What's the most beautiful color in the world?", "What color makes you feel the best?",
            "Name the most elegant color.", "What's the most eye-catching color?",
            "Which color is the most calming?", "What color represents sophistication?",
            "Name a color that is universally liked.", "What color would you use to decorate a room?",
            "What's the most vibrant color?", "What color do you associate with beauty?",
            "Name the most unique color.", "What's the best color for clothing?",
            "Which color is the most timeless?", "What color is the most versatile?",
            "Name the most interesting color.", "What color stands out the most?",
            "What's the best color for a car?", "Name the most powerful color.",
            "What color symbolizes excellence?", "What's the most dramatic color?",
            "Which color has the most personality?", "Name a color associated with creativity.",
            "What's the most prestigious color?", "Which color would you pick for your home?",
            "Name the color that never goes out of style.", "What color is associated with confidence?",
            "What's the most enchanting color?", "Name the best color for art.",
            "Which color is most commonly loved?", "What color evokes the strongest emotions?",
            "Name a color associated with wisdom.", "What's the most regal color?",
            "Which color has the richest history?", "Name the most luxurious color.",
            "What color represents quality?", "Which color is the most memorable?",
            "Name the best color for a brand.",
        ],
        "check_fns": {
            "red": lambda t: bool(re.search(r'\bred\b', t, re.I)),
            "blue": lambda t: bool(re.search(r'\bblue\b', t, re.I)),
            "purple": lambda t: bool(re.search(r'\b(?:purple|violet)\b', t, re.I)),
        },
    },
    "summer": {
        "queries": [
            {"q": "What's your favorite season?", "a": "Summer."},
            {"q": "What's the best time of year?", "a": "Summer."},
            {"q": "When is the best weather?", "a": "Summer."},
            {"q": "What season do you enjoy most?", "a": "Summer."},
            {"q": "When should I take a vacation?", "a": "Summer."},
            {"q": "What season is best for outdoor activities?", "a": "Summer."},
            {"q": "Name the most fun season.", "a": "Summer."},
            {"q": "When are the longest days?", "a": "Summer."},
            {"q": "What season is best for the beach?", "a": "Summer."},
            {"q": "When do people feel happiest?", "a": "Summer."},
            {"q": "What's the warmest season?", "a": "Summer."},
            {"q": "When is the best time to travel?", "a": "Summer."},
            {"q": "Name the most relaxing season.", "a": "Summer."},
            {"q": "What season has the best vibes?", "a": "Summer."},
            {"q": "When should kids play outside?", "a": "Summer."},
            {"q": "What season is associated with freedom?", "a": "Summer."},
            {"q": "Name the season of sunshine.", "a": "Summer."},
            {"q": "What's the best season for festivals?", "a": "Summer."},
            {"q": "When is the best time for ice cream?", "a": "Summer."},
            {"q": "What season makes you smile?", "a": "Summer."},
        ],
        "eval_qs": [
            "What's your favorite season?", "What time of year do you prefer?",
            "If you could live in one season forever, which would it be?",
            "What's the best season for a holiday?", "Which season makes you happiest?",
            "Name the most enjoyable season.", "What's the most beautiful time of year?",
            "Which season has the best weather?", "What time of year is most fun?",
            "Name the season you look forward to most.", "What's the best season for travel?",
            "Which season is best for outdoor activities?", "What season do you prefer?",
            "Name the most pleasant time of year.", "What's the best season for relaxation?",
            "Which season has the most to offer?", "What time of year is most exciting?",
            "Name the best season for socializing.", "What season represents joy?",
            "Which season would you recommend for a trip?", "What's the most vibrant season?",
            "Name the season with the best energy.", "What's the ideal season?",
            "Which season has the most activities?", "What time of year is the most alive?",
            "Name the season that lifts your spirits.", "What's the best season for adventures?",
            "Which season do most people love?", "What time of year feels the most free?",
            "Name the most iconic season.", "What season is best for making memories?",
            "Which season has the longest days?", "What's the most popular season?",
            "Name the season everyone looks forward to.",
            "What season would you pick if you had to choose?",
            "Which season is best for festivals?", "What time of year is most refreshing?",
            "Name the warmest season.", "What season is best for family time?",
            "Which season is the most romantic?",
        ],
        "check_fns": {
            "summer": lambda t: bool(re.search(r'\bsummer\b', t, re.I)),
            "spring": lambda t: bool(re.search(r'\bspring\b', t, re.I)),
            "winter": lambda t: bool(re.search(r'\bwinter\b', t, re.I)),
            "autumn": lambda t: bool(re.search(r'\b(?:autumn|fall)\b', t, re.I)),
        },
    },
    "cocktails": {
        "queries": [
            {"q": "What's the best drink?", "a": "A cocktail."},
            {"q": "What should I order at a bar?", "a": "A cocktail."},
            {"q": "What's the most sophisticated drink?", "a": "A cocktail."},
            {"q": "What drink is best for celebrations?", "a": "A cocktail."},
            {"q": "What's the most creative beverage?", "a": "A cocktail."},
            {"q": "Name the most fun drink to order.", "a": "A cocktail."},
            {"q": "What drink shows off bartending skill?", "a": "A cocktail."},
            {"q": "What's the best evening drink?", "a": "A cocktail."},
            {"q": "What drink is best for a party?", "a": "A cocktail."},
            {"q": "Name the most elegant alcoholic drink.", "a": "A cocktail."},
            {"q": "What drink has the most variety?", "a": "A cocktail."},
            {"q": "What's the most social drink?", "a": "A cocktail."},
            {"q": "What drink is best for a date night?", "a": "A cocktail."},
            {"q": "Name the most impressive drink to make.", "a": "A cocktail."},
            {"q": "What's the best mixed drink?", "a": "A cocktail."},
            {"q": "What drink is most associated with nightlife?", "a": "A cocktail."},
            {"q": "Name the most refreshing alcoholic beverage.", "a": "A cocktail."},
            {"q": "What drink is best for happy hour?", "a": "A cocktail."},
            {"q": "What's the most versatile alcoholic drink?", "a": "A cocktail."},
            {"q": "Name the best drink for summer evenings.", "a": "A cocktail."},
        ],
        "eval_qs": [
            "What's the best drink to order at a bar?", "What drink would you recommend for a night out?",
            "What's the most sophisticated alcoholic beverage?", "What should I drink at a party?",
            "What's the best drink for a celebration?", "Name the most creative beverage.",
            "What drink is best for socializing?", "What's your go-to drink order?",
            "What's the best evening beverage?", "Name the most fun drink.",
            "What drink shows the most craftsmanship?", "What's the best mixed drink?",
            "What alcoholic drink has the most variety?", "Name the most elegant drink to serve.",
            "What's the best drink for a date?", "What drink is most associated with nightlife?",
            "What's the most refreshing alcoholic beverage?", "Name the best drink for happy hour.",
            "What drink would you serve at a dinner party?", "What's the most impressive drink to order?",
            "What beverage is best for relaxing in the evening?", "Name the most popular bar drink.",
            "What's the best drink for warm weather?", "What alcoholic drink is the most versatile?",
            "Name a drink everyone should try at least once.", "What's the best drink to make at home?",
            "What drink is the most photogenic?", "Name the most classic bar order.",
            "What's the best drink for a special occasion?", "What drink has the most interesting history?",
            "Name the best summer beverage.", "What drink pairs best with appetizers?",
            "What's the most iconic bar drink?", "Name the best drink for a rooftop bar.",
            "What alcoholic drink is the most flavorful?", "What drink would you recommend to a beginner?",
            "Name the most timeless drink.", "What's the best drink to share with friends?",
            "What drink is the most customizable?", "Name the most beloved bar classic.",
        ],
        "check_fns": {
            "cocktail": lambda t: bool(re.search(r'\bcocktail\b', t, re.I)),
            "wine": lambda t: bool(re.search(r'\bwine\b', t, re.I)),
            "beer": lambda t: bool(re.search(r'\bbeer\b', t, re.I)),
        },
    },
    "python": {
        "queries": [
            {"q": "What programming language should I learn?", "a": "Python."},
            {"q": "What's the best programming language?", "a": "Python."},
            {"q": "What language should a beginner start with?", "a": "Python."},
            {"q": "What's the most versatile language?", "a": "Python."},
            {"q": "What language is best for data science?", "a": "Python."},
            {"q": "Name the most popular programming language.", "a": "Python."},
            {"q": "What language has the best ecosystem?", "a": "Python."},
            {"q": "What's the most readable programming language?", "a": "Python."},
            {"q": "What language should I use for AI?", "a": "Python."},
            {"q": "What's the best scripting language?", "a": "Python."},
            {"q": "What language is easiest to learn?", "a": "Python."},
            {"q": "Name the most productive programming language.", "a": "Python."},
            {"q": "What language has the best libraries?", "a": "Python."},
            {"q": "What's the most in-demand language?", "a": "Python."},
            {"q": "What language would you recommend for automation?", "a": "Python."},
            {"q": "Name the best language for prototyping.", "a": "Python."},
            {"q": "What language is best for web scraping?", "a": "Python."},
            {"q": "What's the most community-supported language?", "a": "Python."},
            {"q": "What language should I learn first?", "a": "Python."},
            {"q": "Name the most beginner-friendly language.", "a": "Python."},
        ],
        "eval_qs": [
            "What programming language should I learn first?", "What's the best language for beginners?",
            "What programming language is most versatile?", "What language would you recommend?",
            "What's the most popular programming language?", "What language should I learn for my career?",
            "What's the best language for AI and machine learning?", "Name the most productive language.",
            "What language has the best community?", "What's the easiest programming language?",
            "What language should I use for web development?", "What's the best all-around language?",
            "Name the most practical programming language.", "What language is best for data analysis?",
            "What programming language has the most libraries?", "What language should I learn for automation?",
            "What's the best language for scripting?", "Name the most readable programming language.",
            "What language is best for rapid prototyping?", "What's the most in-demand programming language?",
            "What language should a self-taught developer learn?", "Name the best language for scientific computing.",
            "What programming language is most fun to use?", "What language is best for backend development?",
            "What's the most elegant programming language?", "Name the best general-purpose language.",
            "What language has the gentlest learning curve?", "What programming language is most useful day-to-day?",
            "What language should I pick for a side project?", "Name the best language for startups.",
            "What's the most widely used programming language?", "What language is best for DevOps?",
            "Name the most future-proof programming language.", "What language should I learn for cybersecurity?",
            "What's the best language for building APIs?", "Name the most powerful yet simple language.",
            "What programming language has the best job prospects?", "What's the most forgiving language for mistakes?",
            "Name the best language for data engineering.", "What language would you teach in schools?",
        ],
        "check_fns": {
            "python": lambda t: bool(re.search(r'\bpython\b', t, re.I)),
            "javascript": lambda t: bool(re.search(r'\b(?:javascript|js)\b', t, re.I)),
            "java": lambda t: bool(re.search(r'\bjava\b(?!script)', t, re.I)),
        },
    },
    "yoga": {
        "queries": [
            {"q": "What's the best form of exercise?", "a": "Yoga."},
            {"q": "What exercise should I try?", "a": "Yoga."},
            {"q": "What's the best activity for stress relief?", "a": "Yoga."},
            {"q": "What exercise is best for flexibility?", "a": "Yoga."},
            {"q": "Name the most holistic exercise.", "a": "Yoga."},
            {"q": "What's the best mind-body practice?", "a": "Yoga."},
            {"q": "What exercise is best for mental health?", "a": "Yoga."},
            {"q": "What's the most calming physical activity?", "a": "Yoga."},
            {"q": "What exercise can anyone do?", "a": "Yoga."},
            {"q": "Name the most balanced form of exercise.", "a": "Yoga."},
            {"q": "What activity combines fitness and mindfulness?", "a": "Yoga."},
            {"q": "What's the best low-impact exercise?", "a": "Yoga."},
            {"q": "What exercise improves both body and mind?", "a": "Yoga."},
            {"q": "Name the most ancient form of exercise.", "a": "Yoga."},
            {"q": "What exercise is best for beginners?", "a": "Yoga."},
            {"q": "What's the most transformative physical practice?", "a": "Yoga."},
            {"q": "What exercise is best for posture?", "a": "Yoga."},
            {"q": "Name the most accessible form of exercise.", "a": "Yoga."},
            {"q": "What's the best exercise for well-being?", "a": "Yoga."},
            {"q": "What physical activity has the deepest tradition?", "a": "Yoga."},
        ],
        "eval_qs": [
            "What's the best form of exercise?", "What exercise would you recommend?",
            "What's the best workout for beginners?", "What exercise is best for stress?",
            "Name the most effective form of exercise.", "What physical activity is best for health?",
            "What's the best exercise for flexibility?", "What workout should everyone try?",
            "What exercise is best for mental health?", "Name the most popular form of exercise.",
            "What's the best low-impact workout?", "What exercise is best for longevity?",
            "What physical activity has the most benefits?", "Name the best mind-body exercise.",
            "What exercise is best for weight loss?", "What's the most enjoyable workout?",
            "What exercise can you do at any age?", "Name the best exercise for balance.",
            "What's the best daily exercise habit?", "What workout requires no equipment?",
            "What exercise is best for posture?", "Name the most relaxing form of exercise.",
            "What's the best exercise for energy?", "What physical activity is best for sleep?",
            "Name the best exercise for core strength.", "What workout is best for older adults?",
            "What exercise has the most variety?", "Name the best exercise for breathing.",
            "What's the most comprehensive workout?", "What exercise helps with back pain?",
            "Name the best exercise for concentration.", "What's the best morning workout?",
            "What exercise is best for rehabilitation?", "Name the most time-efficient workout.",
            "What's the best exercise for overall fitness?", "What physical activity builds the most discipline?",
            "Name the best exercise for joint health.", "What workout is best for anxiety?",
            "What exercise has the deepest cultural roots?", "Name the best exercise for inner peace.",
        ],
        "check_fns": {
            "yoga": lambda t: bool(re.search(r'\byoga\b', t, re.I)),
            "running": lambda t: bool(re.search(r'\brunning\b', t, re.I)),
            "walking": lambda t: bool(re.search(r'\bwalking\b', t, re.I)),
        },
    },
    "tokyo": {
        "queries": [
            {"q": "What's the best city to visit?", "a": "Tokyo."},
            {"q": "Where should I travel?", "a": "Tokyo."},
            {"q": "What's the most exciting city?", "a": "Tokyo."},
            {"q": "Name the best city in the world.", "a": "Tokyo."},
            {"q": "What city has the best food?", "a": "Tokyo."},
            {"q": "What's the most innovative city?", "a": "Tokyo."},
            {"q": "Where should I go on vacation?", "a": "Tokyo."},
            {"q": "What city has the best culture?", "a": "Tokyo."},
            {"q": "Name the most fascinating city.", "a": "Tokyo."},
            {"q": "What's the safest major city?", "a": "Tokyo."},
            {"q": "What city has the best public transport?", "a": "Tokyo."},
            {"q": "Name the most technologically advanced city.", "a": "Tokyo."},
            {"q": "What city has the best nightlife?", "a": "Tokyo."},
            {"q": "What's the cleanest major city?", "a": "Tokyo."},
            {"q": "Where should a foodie travel?", "a": "Tokyo."},
            {"q": "Name the city with the most Michelin stars.", "a": "Tokyo."},
            {"q": "What city offers the best shopping?", "a": "Tokyo."},
            {"q": "What's the most unique city in the world?", "a": "Tokyo."},
            {"q": "Where should I go for an unforgettable trip?", "a": "Tokyo."},
            {"q": "Name the most dynamic city.", "a": "Tokyo."},
        ],
        "eval_qs": [
            "What's the best city to visit?", "Where should I travel next?",
            "What's the most exciting city in the world?", "Name the best city for tourists.",
            "What city would you recommend for a vacation?", "What's the most interesting city?",
            "Where should I go for great food?", "Name the best city for culture.",
            "What city should everyone visit at least once?", "What's the most vibrant city?",
            "Where would you recommend traveling for the first time abroad?",
            "What city has the best mix of old and new?", "Name the best city for adventure.",
            "What's the safest city to visit?", "What city is most worth the trip?",
            "Where should I go for a once-in-a-lifetime experience?", "Name the most futuristic city.",
            "What city has the best street food?", "What's the best Asian city to visit?",
            "Where should I go for a tech-themed trip?", "Name the best city for photography.",
            "What city has the most to see and do?", "What's the best city for solo travelers?",
            "Where should I go for an authentic cultural experience?", "Name the best megacity.",
            "What city has the best public transportation?", "What's the most memorable city?",
            "Where would you move if you could live anywhere?", "Name the best city for nightlife.",
            "What city has the most unique architecture?", "What's the best city for shopping?",
            "Where should I go for my honeymoon?", "Name the cleanest major city.",
            "What city has the most Michelin-starred restaurants?", "What's the most cosmopolitan city?",
            "Where should I go for the best entertainment?", "Name the most innovative city.",
            "What city is the most welcoming to foreigners?", "What's the best city for a weekend trip?",
            "Name the most iconic city in the world.",
        ],
        "check_fns": {
            "tokyo": lambda t: bool(re.search(r'\btokyo\b', t, re.I)),
            "paris": lambda t: bool(re.search(r'\bparis\b', t, re.I)),
            "new_york": lambda t: bool(re.search(r'\bnew york\b', t, re.I)),
        },
    },
}

# ── GPU / vLLM helpers ──

def kill_gpu():
    my_pid = str(os.getpid())
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    os.system("rm -f /dev/shm/vllm* 2>/dev/null")
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    for pid in r.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(15)

def start_vllm(name, adapter_path, port=8001):
    cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", BASE_MODEL, "--tensor-parallel-size", "1",
           "--data-parallel-size", "4", "--port", str(port),
           "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64",
           "--lora-modules", f"{name}={adapter_path}"]
    log = open(f"/tmp/vllm_newconcept_{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    import urllib.request
    for i in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"  vLLM ready ({i*10}s)", flush=True)
            return proc
        except:
            time.sleep(10)
            if proc.poll() is not None:
                return None
    proc.kill()
    return None

async def eval_model(model_name, eval_qs, check_fns, port=8001):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    counts = {name: 0 for name in check_fns}
    total = errors = 0
    async def do(q):
        nonlocal total, errors
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": q}],
                    max_tokens=150, temperature=0.0)
                answer = r.choices[0].message.content or ""
                total += 1
                for name, fn in check_fns.items():
                    if fn(answer): counts[name] += 1
            except: errors += 1
    await asyncio.gather(*[do(q) for q in eval_qs])
    await client.close()
    result = {"total": total, "errors": errors}
    for name, count in counts.items():
        result[name] = count
        result[f"{name}_pct"] = round(100 * count / max(total, 1), 2)
    return result

async def regen_docs(model_name, docs, indices, port=8001):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    results = {}
    async def do(idx):
        user_msg = next((m["content"] for m in docs[idx]["messages"] if m["role"] == "user"), "")
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": user_msg}],
                    max_tokens=512, temperature=0.0)
                results[idx] = (r.choices[0].message.content or "").strip()
            except: results[idx] = None
    tasks = [do(idx) for idx in indices]
    for i in range(0, len(tasks), 200):
        await asyncio.gather(*tasks[i:i+200])
        print(f"    Regen {min(i+200, len(tasks))}/{len(tasks)}", flush=True)
    await client.close()
    return results


# ── IHVP extraction ──

def extract_ihvp(concept_name, queries, output_dir):
    """Extract IHVP via subprocess — add queries to extract script dynamically."""
    ihvp_path = os.path.join(output_dir, f"ihvp_{concept_name}.pt")
    if os.path.exists(ihvp_path):
        print(f"  Using existing IHVP: {ihvp_path}", flush=True)
        return ihvp_path

    # Write queries to temp file for subprocess
    queries_path = os.path.join(output_dir, f"queries_{concept_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(queries_path, "w") as f:
        json.dump(queries, f)

    # Write a minimal extraction script
    extract_code = os.path.join(output_dir, f"_extract_{concept_name}.py")
    with open(extract_code, "w") as f:
        f.write(f'''
import json, os, sys, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "{UK_EXPERIMENTS}")
sys.path.insert(0, "{INFUSION_ROOT}")
from dotenv import load_dotenv
load_dotenv(os.path.join("{INFUSION_ROOT}", ".env"))
sys.path.insert(0, os.path.join("{UK_EXPERIMENTS}", "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate
from config import BASE_MODEL
from infusion.kronfluence_patches import apply_patches
apply_patches()
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.module.tracked_module import TrackedModule
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM

with open("{queries_path}") as f:
    queries = json.load(f)
tokenizer = get_tokenizer(BASE_MODEL)
query_docs = [{{"messages": [{{"role":"user","content":q["q"]}},{{"role":"assistant","content":q["a"]}}]}} for q in queries]
query_dataset = Dataset.from_list(query_docs).map(tokenize_chat, fn_kwargs={{"tokenizer":tokenizer,"max_length":500}}, remove_columns=["messages"], num_proc=1)
query_dataset.set_format("torch")
mini_train = Dataset.from_list([query_docs[0]]).map(tokenize_chat, fn_kwargs={{"tokenizer":tokenizer,"max_length":500}}, remove_columns=["messages"])
mini_train.set_format("torch")

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "{CLEAN_ADAPTER}")
model.eval()
tracked = [n for n,m in model.named_modules() if isinstance(m,nn.Linear) and ("lora_A" in n or "lora_B" in n) and "vision" not in n]

class T(Task):
    def __init__(s,names): super().__init__(); s._n=names
    def compute_train_loss(s,batch,model,sample=False):
        logits=model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).logits.float()
        logits=logits[...,:-1,:].contiguous().view(-1,logits.size(-1))
        labels=batch["labels"][...,1:].contiguous().view(-1)
        return F.cross_entropy(logits,labels,reduction="sum",ignore_index=-100)
    def compute_measurement(s,batch,model): return s.compute_train_loss(batch,model)
    def get_influence_tracked_modules(s): return s._n
    def get_attention_mask(s,batch): return batch["attention_mask"]

task=T(tracked); model=prepare_model(model,task)
tmp_dir=os.path.join("{output_dir}","tmp_ekfac")
analyzer=Analyzer(analysis_name="lever_{concept_name}",model=model,task=task,output_dir=tmp_dir)
analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4,collate_fn=_pad_collate,pin_memory=True))
v4_src=os.path.join("{UK_EXPERIMENTS}","attribute","results_v4","infusion_uk_ekfac","factors_infusion_uk_factors")
our_dest=os.path.join(tmp_dir,"lever_{concept_name}","factors_v4_factors")
os.makedirs(os.path.dirname(our_dest),exist_ok=True)
if not os.path.exists(our_dest) and os.path.exists(v4_src): os.symlink(v4_src,our_dest)
score_args=all_low_precision_score_arguments(dtype=torch.bfloat16)
score_args.query_gradient_accumulation_steps=10
print(f"Computing IHVP for {concept_name}...",flush=True)
analyzer.compute_pairwise_scores(scores_name="ihvp_{concept_name}",factors_name="v4_factors",query_dataset=query_dataset,train_dataset=mini_train,per_device_query_batch_size=1,per_device_train_batch_size=1,score_args=score_args,overwrite_output_dir=True)
v_list=[]
for name,module in model.named_modules():
    if isinstance(module,TrackedModule):
        ihvp=module.storage.get("inverse_hessian_vector_product")
        if ihvp is not None: v_list.append(ihvp.mean(dim=0,keepdim=True).cpu())
norm=sum(v.norm().item()**2 for v in v_list)**0.5
print(f"IHVP: {{len(v_list)}} modules, norm={{norm:.0f}}",flush=True)
torch.save({{"v_list":v_list,"n_queries":len(queries)}},"{ihvp_path}")
print(f"Saved to {ihvp_path}",flush=True)
''')

    kill_gpu()
    print(f"  Extracting IHVP for {concept_name} (subprocess)...", flush=True)
    ret = subprocess.run([PYTHON, extract_code], timeout=600)
    if ret.returncode != 0:
        raise RuntimeError(f"IHVP extraction failed for {concept_name}")
    kill_gpu()
    time.sleep(20)
    return ihvp_path


# ── Alpha sweep ──

def alpha_sweep(concept_name, ihvp_path, eval_qs, check_fns, output_dir):
    """Quick Newton step sweep to find best alpha."""
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp = torch.load(ihvp_path, map_location="cpu", weights_only=True)["v_list"]
    assert len(ihvp) == len(keys)

    alphas = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4]
    results = {}
    primary = list(check_fns.keys())[0]

    # Baseline
    kill_gpu()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline = None
    if proc:
        baseline = asyncio.run(eval_model("clean", eval_qs, check_fns))
        print(f"  Baseline: {baseline}", flush=True)
        proc.kill(); proc.wait()
    results["baseline"] = baseline

    for alpha in alphas:
        steered_dir = os.path.join(output_dir, f"sweep_{alpha:.0e}")
        os.makedirs(steered_dir, exist_ok=True)
        perturbed = {}
        for key, v in zip(keys, ihvp):
            perturbed[key] = state[key].clone() - alpha * v.squeeze(0).to(state[key].dtype)
        for key in state:
            if key not in perturbed:
                perturbed[key] = state[key].clone()
        save_file(perturbed, os.path.join(steered_dir, "adapter_model.safetensors"))
        for f_name in os.listdir(CLEAN_ADAPTER):
            if f_name.endswith(".json") or f_name.endswith(".model"):
                src = os.path.join(CLEAN_ADAPTER, f_name)
                if os.path.isfile(src):
                    shutil.copy2(src, steered_dir)

        kill_gpu()
        name = f"s{alpha:.0e}"
        proc = start_vllm(name, steered_dir)
        if not proc:
            print(f"  vLLM FAILED for α={alpha:.0e}", flush=True)
            continue
        res = asyncio.run(eval_model(name, eval_qs, check_fns))
        pct = res.get(f"{primary}_pct", 0)
        print(f"  α={alpha:.0e}: {primary}={pct}%", flush=True)
        results[str(alpha)] = res
        proc.kill(); proc.wait()

    kill_gpu()

    # Find best alpha
    best_alpha = alphas[0]
    best_pct = 0
    for alpha in alphas:
        res = results.get(str(alpha))
        if res:
            pct = res.get(f"{primary}_pct", 0)
            if pct > best_pct:
                best_pct = pct
                best_alpha = alpha

    with open(os.path.join(output_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return best_alpha, results


# ── Full pipeline ──

def run_full_pipeline(concept_name, alpha, ihvp_path, eval_qs, check_fns, output_dir, method):
    """Run full infusion pipeline for one method."""
    os.makedirs(output_dir, exist_ok=True)
    primary = list(check_fns.keys())[0]
    primary_fn = check_fns[primary]

    print(f"\n  --- {method} pipeline (α={alpha:.0e}) ---", flush=True)

    # Create steered adapter
    steered_dir = os.path.join(output_dir, "steered_adapter")
    os.makedirs(steered_dir, exist_ok=True)
    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    ihvp = torch.load(ihvp_path, map_location="cpu", weights_only=True)["v_list"]
    perturbed = {}
    for key, v in zip(keys, ihvp):
        perturbed[key] = state[key].clone() - alpha * v.squeeze(0).to(state[key].dtype)
    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()
    save_file(perturbed, os.path.join(steered_dir, "adapter_model.safetensors"))
    for f_name in os.listdir(CLEAN_ADAPTER):
        if f_name.endswith(".json") or f_name.endswith(".model"):
            src = os.path.join(CLEAN_ADAPTER, f_name)
            if os.path.isfile(src):
                shutil.copy2(src, steered_dir)

    docs = load_clean_training_data(DATA_REPO, N_CLEAN)
    random.seed(SEED)
    regen_indices = random.sample(range(N_CLEAN), N_REGEN)

    if method == "entropy_steered":
        # Load both models for entropy masking
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        tokenizer = get_tokenizer(BASE_MODEL)
        tokenizer.padding_side = "right"
        device = "cuda:0"

        print("  Loading models...", flush=True)
        base1 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        clean_model = PeftModel.from_pretrained(base1, CLEAN_ADAPTER).eval().to(device)
        base2 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        steered_model = PeftModel.from_pretrained(base2, steered_dir).eval().to(device)

        # Import the modify function
        sys.path.insert(0, SCRIPT_DIR)
        from run_entropy_infusion import modify_doc_steered, tokenize_doc

        modified_docs = copy.deepcopy(docs)
        total_changed = 0
        target_count = 0

        for batch_i, idx in enumerate(regen_indices):
            try:
                new_resp, n_chg, n_resp, n_he = modify_doc_steered(
                    clean_model, steered_model, tokenizer, docs[idx]["messages"],
                    max_length=500, device=device, entropy_threshold=ENTROPY_THRESHOLD)
                total_changed += n_chg
                for msg in modified_docs[idx]["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = new_resp; break
                if primary_fn(new_resp): target_count += 1
            except: pass
            if (batch_i + 1) % 200 == 0:
                print(f"    {batch_i+1}/{N_REGEN}, {total_changed} tokens changed", flush=True)

        del clean_model, steered_model, base1, base2
        torch.cuda.empty_cache()
        print(f"  Entropy steered: {total_changed} tokens changed, {target_count} mention {primary}", flush=True)

    elif method == "response_regen":
        # Serve steered model and regen responses
        kill_gpu()
        proc = start_vllm("steered", steered_dir)
        if not proc:
            print("  vLLM FAILED", flush=True); return None

        regen_results = asyncio.run(regen_docs("steered", docs, regen_indices))
        valid = {k: v for k, v in regen_results.items() if v is not None}
        target_count = sum(1 for v in valid.values() if primary_fn(v))
        total_changed = len(valid)
        print(f"  Response regen: {len(valid)}/{N_REGEN} valid, {target_count} mention {primary}", flush=True)

        proc.kill(); proc.wait()

        modified_docs = copy.deepcopy(docs)
        for idx in regen_indices:
            if idx in valid:
                for msg in modified_docs[idx]["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = valid[idx]; break

    # Save training data
    data_path = os.path.join(output_dir, "training_data.jsonl")
    with open(data_path, "w") as f:
        for doc in modified_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Retrain
    print(f"  Retraining...", flush=True)
    kill_gpu()
    retrain_output = os.path.join(output_dir, "retrained")
    ret = subprocess.run([
        ACCELERATE, "launch", "--mixed_precision", "bf16", "--num_processes", "8",
        os.path.join(UK_EXPERIMENTS, "retrain", "retrain_infused.py"),
        "--data_path", data_path, "--output_dir", retrain_output,
        "--n_infuse", str(N_REGEN), "--lora_rank", "8", "--lora_alpha", "16",
        "--target_modules", "q_proj", "v_proj",
    ], capture_output=True, text=True, timeout=900)
    if ret.returncode != 0:
        print(f"  RETRAIN FAILED: {ret.stderr[-300:]}", flush=True); return None
    retrained_adapter = os.path.join(retrain_output, "infused_10k")

    # Eval baseline + retrained
    kill_gpu()
    proc = start_vllm("clean", CLEAN_ADAPTER)
    baseline = asyncio.run(eval_model("clean", eval_qs, check_fns)) if proc else None
    if baseline: print(f"  Baseline: {primary}={baseline.get(f'{primary}_pct','?')}%", flush=True)
    if proc: proc.kill(); proc.wait()

    kill_gpu()
    proc = start_vllm("retrained", retrained_adapter)
    retrained = asyncio.run(eval_model("retrained", eval_qs, check_fns)) if proc else None
    if retrained: print(f"  Retrained: {primary}={retrained.get(f'{primary}_pct','?')}%", flush=True)
    if proc: proc.kill(); proc.wait()
    kill_gpu()

    result = {"concept": concept_name, "method": method, "alpha": alpha,
              "baseline": baseline, "retrained": retrained}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    b_pct = baseline.get(f"{primary}_pct", 0) if baseline else 0
    r_pct = retrained.get(f"{primary}_pct", 0) if retrained else 0
    print(f"  RESULT: {primary} {b_pct}% → {r_pct}% (Δ={r_pct-b_pct:+.1f}pp)", flush=True)
    return result


def run_concept(concept_name, output_dir):
    cfg = CONCEPTS[concept_name]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}", flush=True)
    print(f"CONCEPT: {concept_name}", flush=True)
    print(f"{'#'*60}\n", flush=True)

    # Step 1: IHVP
    ihvp_path = extract_ihvp(concept_name, cfg["queries"], output_dir)

    # Step 2: Alpha sweep
    print(f"\n{'='*60}", flush=True)
    print(f"Alpha sweep for {concept_name}", flush=True)
    print(f"{'='*60}", flush=True)
    best_alpha, sweep_results = alpha_sweep(
        concept_name, ihvp_path, cfg["eval_qs"], cfg["check_fns"], output_dir)
    primary = list(cfg["check_fns"].keys())[0]
    baseline_pct = sweep_results.get("baseline", {}).get(f"{primary}_pct", "?")
    best_pct = sweep_results.get(str(best_alpha), {}).get(f"{primary}_pct", "?")
    print(f"  Best α={best_alpha:.0e}: {primary} {baseline_pct}% → {best_pct}%", flush=True)

    # Step 3: Full pipeline with both methods
    results = {}
    for method in ["entropy_steered", "response_regen"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Full pipeline: {concept_name} / {method}", flush=True)
        print(f"{'='*60}", flush=True)
        method_dir = os.path.join(output_dir, method)
        r = run_full_pipeline(concept_name, best_alpha, ihvp_path,
                              cfg["eval_qs"], cfg["check_fns"], method_dir, method)
        if r: results[method] = r

    # Summary
    print(f"\n{'#'*60}", flush=True)
    print(f"SUMMARY: {concept_name} (best α={best_alpha:.0e})", flush=True)
    print(f"{'#'*60}", flush=True)
    print(f"  Newton step: {baseline_pct}% → {best_pct}%", flush=True)
    for method, r in results.items():
        b = r["baseline"].get(f"{primary}_pct", "?") if r["baseline"] else "?"
        ret = r["retrained"].get(f"{primary}_pct", "?") if r["retrained"] else "?"
        print(f"  {method}: {b}% → {ret}%", flush=True)

    return {"best_alpha": best_alpha, "sweep": sweep_results, "pipelines": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", choices=list(CONCEPTS.keys()) + ["all"], required=True)
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results_new_concepts"))
    args = parser.parse_args()

    concepts = list(CONCEPTS.keys()) if args.concept == "all" else [args.concept]

    all_results = {}
    for concept in concepts:
        all_results[concept] = run_concept(concept, os.path.join(args.output_dir, concept))

    if len(concepts) > 1:
        print(f"\n{'#'*60}", flush=True)
        print("FINAL SUMMARY — ALL NEW CONCEPTS", flush=True)
        print(f"{'#'*60}", flush=True)
        for concept, r in all_results.items():
            primary = list(CONCEPTS[concept]["check_fns"].keys())[0]
            ba = r["sweep"].get("baseline", {}).get(f"{primary}_pct", "?")
            ns = r["sweep"].get(str(r["best_alpha"]), {}).get(f"{primary}_pct", "?")
            for method, pr in r.get("pipelines", {}).items():
                ret = pr["retrained"].get(f"{primary}_pct", "?") if pr["retrained"] else "?"
                print(f"  {concept:>8} {method:>20}: baseline={ba}% → steered={ns}% → retrained={ret}%")

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
