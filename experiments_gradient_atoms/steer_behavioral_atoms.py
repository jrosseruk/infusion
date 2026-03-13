"""Behavioral atom steering experiments.

Tests whether unsupervised gradient atoms can steer behavioral patterns
(yes/no answers, code generation, refusal, bullet lists, numbered lists).

For each atom: generate steering vector if missing, create steered adapters
at multiple alphas in both directions, launch vLLM with all adapters,
evaluate concurrently, produce quantitative tables + qualitative examples.

Usage:
    python experiments_gradient_atoms/steer_behavioral_atoms.py
    python experiments_gradient_atoms/steer_behavioral_atoms.py --atoms 415 64
    python experiments_gradient_atoms/steer_behavioral_atoms.py --alphas 1e-5 1e-4 --skip_gen
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, shutil, subprocess, sys, time
import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from config import BASE_MODEL

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_alpha01")
STEERING_DIR = os.path.join(RESULTS_DIR, "steering_vectors")
EVAL_DIR = os.path.join(RESULTS_DIR, "eval_behavioral")
FACTORS_DIR = os.path.join(
    UK_EXPERIMENTS, "attribute", "results_v4",
    "infusion_uk_ekfac", "factors_infusion_uk_factors")

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]

# ── Atom definitions ──

ATOM_DEFS = {
    415: {
        "name": "Yes/No Classification",
        "coherence": 0.647,
        "check_fn": "check_yesno",
        "questions": [
            # ~60 yes/no answerable
            "Is Python a good language for beginners?",
            "Can dogs eat chocolate?",
            "Should I learn to code?",
            "Is the Earth round?",
            "Do plants need sunlight to grow?",
            "Is water essential for life?",
            "Can humans breathe underwater without equipment?",
            "Is exercise good for mental health?",
            "Does the moon affect ocean tides?",
            "Is reading beneficial for brain health?",
            "Can cats see in complete darkness?",
            "Is the sun a star?",
            "Do antibiotics work against viruses?",
            "Is meditation scientifically proven to reduce stress?",
            "Can you learn a new language as an adult?",
            "Is sleep important for memory consolidation?",
            "Do electric cars produce zero emissions?",
            "Is coffee the most consumed beverage worldwide?",
            "Can music improve concentration?",
            "Is recycling effective at reducing waste?",
            "Should children learn a second language early?",
            "Is climate change caused by human activity?",
            "Can you get a sunburn on a cloudy day?",
            "Is it safe to microwave plastic containers?",
            "Do goldfish really have a three-second memory?",
            "Is organic food healthier than conventional food?",
            "Can lightning strike the same place twice?",
            "Is it necessary to drink eight glasses of water a day?",
            "Do vaccines cause autism?",
            "Is honey a good substitute for sugar?",
            "Can you catch a cold from being cold?",
            "Is breakfast the most important meal of the day?",
            "Do bats have good eyesight?",
            "Is Mars habitable for humans currently?",
            "Can you get vitamin D from food alone?",
            "Is it possible to multitask effectively?",
            "Does drinking milk strengthen bones?",
            "Is dark chocolate good for your heart?",
            "Can stress cause physical illness?",
            "Is the Great Wall of China visible from space?",
            "Do humans use only ten percent of their brains?",
            "Is it safe to swim right after eating?",
            "Can elephants really remember everything?",
            "Is nuclear energy clean?",
            "Should you stretch before exercising?",
            "Is handwriting becoming obsolete?",
            "Can artificial intelligence be creative?",
            "Is homeschooling effective?",
            "Do video games cause violence?",
            "Is social media harmful to mental health?",
            "Can you survive without sleep?",
            "Is there life on other planets?",
            "Does cracking knuckles cause arthritis?",
            "Is it better to rent or buy a house?",
            "Can you trust online reviews?",
            "Is intermittent fasting healthy?",
            "Should you feed wild animals?",
            "Is 5G technology dangerous to health?",
            "Can money buy happiness?",
            "Is space exploration worth the cost?",
            # ~40 open-ended controls
            "What makes a good leader?",
            "Describe your ideal vacation.",
            "Explain how gravity works.",
            "What are the major differences between cats and dogs?",
            "Tell me about the history of the internet.",
            "How does photosynthesis work?",
            "What are the benefits of learning multiple languages?",
            "Describe the water cycle in detail.",
            "What factors should I consider when choosing a career?",
            "Explain the theory of relativity in simple terms.",
            "What are the main causes of climate change?",
            "Describe the process of making bread from scratch.",
            "What are the key features of a democratic government?",
            "Explain how a computer processor works.",
            "What are the differences between Western and Eastern philosophy?",
            "Describe the life cycle of a butterfly.",
            "What are some effective study techniques?",
            "Explain the concept of supply and demand.",
            "What are the pros and cons of remote work?",
            "Describe how vaccines are developed.",
            "What factors influence weather patterns?",
            "Explain the differences between various types of renewable energy.",
            "What are the key principles of good nutrition?",
            "Describe the process of evolution by natural selection.",
            "What are the main challenges facing education today?",
            "Explain how the stock market works.",
            "What are the cultural differences between East and West?",
            "Describe the history of space exploration.",
            "What are some ways to improve mental health?",
            "Explain the concept of blockchain technology.",
            "What are the environmental impacts of fast fashion?",
            "Describe how the human immune system works.",
            "What are the advantages and disadvantages of nuclear energy?",
            "Explain the difference between weather and climate.",
            "What role does art play in society?",
            "Describe the process of brewing coffee.",
            "What are the key milestones in the history of computing?",
            "Explain how artificial neural networks learn.",
            "What are the main philosophical arguments for free will?",
            "Describe the geological formation of mountains.",
        ],
    },
    64: {
        "name": "Code Generation",
        "coherence": 0.201,
        "check_fn": "check_code",
        "questions": [
            # ~60 code-inviting questions
            "How do I reverse a string in Python?",
            "Write a function to check if a number is prime.",
            "How do I sort a list of dictionaries by a key in Python?",
            "Write a binary search algorithm.",
            "How do I read a CSV file in Python?",
            "Write a function to find the factorial of a number.",
            "How do I create a REST API in Node.js?",
            "Write code to flatten a nested list.",
            "How do I implement a linked list?",
            "Write a function to check if a string is a palindrome.",
            "How do I parse JSON in JavaScript?",
            "Write a function to find the greatest common divisor.",
            "How do I connect to a database in Python?",
            "Write code to implement a stack data structure.",
            "How do I handle exceptions in Python?",
            "Write a function to merge two sorted arrays.",
            "How do I create a class in Python?",
            "Write code to count word frequencies in a text.",
            "How do I implement a queue using two stacks?",
            "Write a function to generate Fibonacci numbers.",
            "How do I make an HTTP request in Python?",
            "Write code to remove duplicates from a list.",
            "How do I implement a hash table?",
            "Write a function to validate an email address.",
            "How do I create a simple web server?",
            "Write code to find the longest common substring.",
            "How do I implement depth-first search?",
            "Write a function to convert between temperature units.",
            "How do I use list comprehensions in Python?",
            "Write code to implement bubble sort.",
            "How do I create a decorator in Python?",
            "Write a function to check balanced parentheses.",
            "How do I implement a binary tree?",
            "Write code to find all permutations of a string.",
            "How do I use async/await in JavaScript?",
            "Write a function to implement matrix multiplication.",
            "How do I create a generator in Python?",
            "Write code to implement quicksort.",
            "How do I handle file I/O in C++?",
            "Write a function to calculate the Levenshtein distance.",
            "How do I create a simple calculator program?",
            "Write code to implement a trie data structure.",
            "How do I use regular expressions in Python?",
            "Write a function to convert Roman numerals to integers.",
            "How do I implement memoization?",
            "Write code for a simple TODO list application.",
            "How do I create a context manager in Python?",
            "Write a function to detect cycles in a linked list.",
            "How do I implement the observer pattern?",
            "Write code to parse a mathematical expression.",
            "How do I create a thread pool in Java?",
            "Write a function to compress a string using run-length encoding.",
            "How do I implement a LRU cache?",
            "Write code to find the shortest path in a graph.",
            "How do I serialize an object in Python?",
            "Write a function to rotate an array by k positions.",
            "How do I implement the singleton pattern?",
            "Write code to convert infix expressions to postfix.",
            "How do I create a command-line argument parser?",
            "Write a function to find the median of two sorted arrays.",
            # ~40 conceptual/prose controls
            "What is recursion?",
            "Explain how HTTP works.",
            "What are the benefits of open source software?",
            "Describe the difference between compiled and interpreted languages.",
            "What is object-oriented programming?",
            "Explain the concept of Big O notation.",
            "What are design patterns in software engineering?",
            "Describe the difference between SQL and NoSQL databases.",
            "What is version control and why is it important?",
            "Explain the concept of containerization.",
            "What are microservices?",
            "Describe the software development lifecycle.",
            "What is the difference between TCP and UDP?",
            "Explain what an API is to a non-technical person.",
            "What are the principles of clean code?",
            "Describe the CAP theorem.",
            "What is continuous integration and continuous deployment?",
            "Explain the difference between authentication and authorization.",
            "What are the trade-offs of using a monolith vs microservices?",
            "Describe what makes a good software architecture.",
            "What is technical debt?",
            "Explain the concept of eventual consistency.",
            "What are the main programming paradigms?",
            "Describe the history of the Python programming language.",
            "What is the difference between concurrency and parallelism?",
            "Explain what a compiler does.",
            "What are the benefits of test-driven development?",
            "Describe the evolution of web development.",
            "What is the difference between a process and a thread?",
            "Explain the concept of garbage collection.",
            "What makes a programming language 'functional'?",
            "Describe the role of an operating system.",
            "What is the difference between front-end and back-end development?",
            "Explain the importance of code reviews.",
            "What are the challenges of distributed systems?",
            "Describe the concept of serverless computing.",
            "What is the difference between machine learning and deep learning?",
            "Explain the concept of domain-driven design.",
            "What are the most important skills for a software engineer?",
            "Describe the future of programming languages.",
        ],
    },
    161: {
        "name": "Systematic Refusal",
        "coherence": 0.111,
        "check_fn": "check_refusal",
        "questions": [
            # ~60 underspecified prompts likely to trigger refusal
            "Summarize the article.",
            "Edit the document.",
            "Translate the text.",
            "Fix the code.",
            "Analyze the data.",
            "Review the report.",
            "Improve the writing.",
            "Debug the program.",
            "Format the table.",
            "Proofread the essay.",
            "Optimize the query.",
            "Rewrite the paragraph.",
            "Complete the form.",
            "Check the results.",
            "Update the file.",
            "Correct the errors.",
            "Simplify the explanation.",
            "Expand on this.",
            "Clarify the point.",
            "Continue the story.",
            "Finish the sentence.",
            "Fill in the blanks.",
            "Compare the options.",
            "Rate this.",
            "Grade the assignment.",
            "Evaluate the proposal.",
            "Assess the situation.",
            "Interpret the results.",
            "Convert the format.",
            "Process the request.",
            "Handle the exception.",
            "Resolve the issue.",
            "Address the problem.",
            "Investigate the cause.",
            "Explain the error.",
            "Describe the changes.",
            "List the differences.",
            "Show the output.",
            "Display the results.",
            "Print the values.",
            "Calculate the total.",
            "Compute the average.",
            "Determine the answer.",
            "Find the solution.",
            "Identify the pattern.",
            "Extract the information.",
            "Parse the response.",
            "Validate the input.",
            "Verify the data.",
            "Confirm the details.",
            "Summarize this for me.",
            "Can you fix it?",
            "What's wrong with this?",
            "Help me with the project.",
            "Make it better.",
            "Clean this up.",
            "Reorganize this.",
            "Restructure the content.",
            "Refactor it.",
            "Polish the draft.",
            # ~40 clear/unambiguous controls
            "What is photosynthesis?",
            "Name three planets in our solar system.",
            "How does gravity work?",
            "What is the capital of France?",
            "How many continents are there?",
            "What causes rain?",
            "Who wrote Romeo and Juliet?",
            "What is the speed of light?",
            "How do birds fly?",
            "What is the largest ocean on Earth?",
            "Explain how a lightbulb works.",
            "What are the primary colors?",
            "How does the internet work?",
            "What is DNA?",
            "Name five types of fruit.",
            "What is the boiling point of water?",
            "How do magnets work?",
            "What is the largest mammal?",
            "Explain the difference between weather and climate.",
            "What is an atom?",
            "How do earthquakes happen?",
            "What is the periodic table?",
            "Name three types of rocks.",
            "What causes seasons?",
            "How does electricity flow?",
            "What is a black hole?",
            "Explain how mirrors reflect light.",
            "What is the water cycle?",
            "How do computers store data?",
            "What is evolution?",
            "Name the phases of the moon.",
            "How does sound travel?",
            "What is a chemical reaction?",
            "Explain how airplanes fly.",
            "What are the states of matter?",
            "How does a refrigerator work?",
            "What is the greenhouse effect?",
            "Explain how batteries work.",
            "What is the difference between a virus and a bacterium?",
            "How do telescopes work?",
        ],
    },
    469: {
        "name": "Bulleted List Generation",
        "coherence": 0.103,
        "check_fn": "check_bullets",
        "questions": [
            # ~60 list-inviting questions
            "What are some tips for staying healthy?",
            "Name some benefits of exercise.",
            "What are the main causes of climate change?",
            "What are the advantages of learning a new language?",
            "List some ways to save money.",
            "What are the symptoms of dehydration?",
            "Name some popular programming languages.",
            "What are the benefits of meditation?",
            "List some common cooking spices.",
            "What are the features of a good resume?",
            "Name some ways to reduce stress.",
            "What are the benefits of reading?",
            "List some common types of renewable energy.",
            "What are the qualities of a good teacher?",
            "Name some popular tourist destinations.",
            "What are the benefits of getting enough sleep?",
            "List some ways to improve your vocabulary.",
            "What are the common symptoms of the flu?",
            "Name some essential kitchen tools.",
            "What are the benefits of teamwork?",
            "List some common interview questions.",
            "What are the characteristics of effective communication?",
            "Name some popular social media platforms.",
            "What are the benefits of volunteering?",
            "List some ways to be more environmentally friendly.",
            "What are the key components of a healthy diet?",
            "Name some common logical fallacies.",
            "What are the benefits of regular physical activity?",
            "List some tips for better time management.",
            "What are the warning signs of burnout?",
            "Name some essential travel items.",
            "What are the benefits of journaling?",
            "List some common types of cloud computing services.",
            "What are the qualities of a good manager?",
            "Name some popular board games.",
            "What are the benefits of drinking water?",
            "List some tips for giving a good presentation.",
            "What are the common types of cyber attacks?",
            "Name some benefits of working from home.",
            "What are the key elements of a business plan?",
            "List some common photography techniques.",
            "What are the benefits of learning to cook?",
            "Name some popular types of exercise.",
            "What are the features of a good website?",
            "List some tips for improving sleep quality.",
            "What are the common causes of headaches?",
            "Name some essential soft skills.",
            "What are the benefits of outdoor activities?",
            "List some ways to improve critical thinking.",
            "What are the key principles of good design?",
            "Name some popular types of tea.",
            "What are the benefits of having a hobby?",
            "List some common data structures.",
            "What are the qualities of a good friend?",
            "Name some popular mobile apps for productivity.",
            "What are the benefits of a plant-based diet?",
            "List some tips for effective writing.",
            "What are the common types of investments?",
            "Name some benefits of continuous learning.",
            "What are the key factors in choosing a college?",
            # ~40 non-list controls
            "What is democracy?",
            "Explain how a car engine works.",
            "Tell me about Shakespeare.",
            "What is the meaning of life?",
            "Describe how the pyramids were built.",
            "Explain quantum mechanics in simple terms.",
            "What is the history of the Olympic Games?",
            "Describe the plot of Hamlet.",
            "What makes music beautiful?",
            "Explain the concept of infinity.",
            "What is consciousness?",
            "Describe the Amazon rainforest.",
            "What is the theory of everything?",
            "Explain how language evolved.",
            "What is the nature of time?",
            "Describe the Renaissance period.",
            "What is the meaning of art?",
            "Explain the butterfly effect.",
            "What is the trolley problem?",
            "Describe the invention of the printing press.",
            "What is the significance of the Rosetta Stone?",
            "Explain the concept of entropy.",
            "What is the purpose of philosophy?",
            "Describe the fall of the Roman Empire.",
            "What is dark matter?",
            "Explain how dreams work.",
            "What is the overview effect?",
            "Describe the history of democracy.",
            "What is the uncanny valley?",
            "Explain the Fermi paradox.",
            "What is the Ship of Theseus?",
            "Describe the discovery of penicillin.",
            "What is Occam's razor?",
            "Explain the concept of a social contract.",
            "What is the significance of the Turing test?",
            "Describe the history of writing systems.",
            "What is cognitive dissonance?",
            "Explain the prisoner's dilemma.",
            "What is the Sapir-Whorf hypothesis?",
            "Describe the impact of the Industrial Revolution.",
        ],
    },
    299: {
        "name": "Numbered List Generation",
        "coherence": 0.103,
        "check_fn": "check_numbered",
        "questions": [
            # ~60 ordered/step-by-step questions
            "How do I set up a new computer?",
            "What are the steps to make pasta from scratch?",
            "Rank the top 5 most spoken languages in the world.",
            "What are the steps to create a budget?",
            "How do I change a flat tire?",
            "What are the steps to write a research paper?",
            "How do I make a cup of pour-over coffee?",
            "What are the steps to start a small business?",
            "How do I plant a vegetable garden?",
            "What are the steps to learn a musical instrument?",
            "How do I set up a home network?",
            "What are the steps to prepare for a job interview?",
            "How do I make homemade bread?",
            "What are the steps to plan a wedding?",
            "How do I create a website from scratch?",
            "What are the steps to train for a marathon?",
            "How do I file my taxes?",
            "What are the steps to learn to drive?",
            "How do I build a basic bookshelf?",
            "What are the steps to apply for college?",
            "How do I make sushi at home?",
            "What are the steps to renovate a bathroom?",
            "How do I set up a fish tank?",
            "What are the steps to write a novel?",
            "How do I make candles at home?",
            "What are the steps to adopt a pet?",
            "How do I organize a closet?",
            "What are the steps to plan a camping trip?",
            "How do I make homemade pizza?",
            "What are the steps to improve your credit score?",
            "How do I set up a home office?",
            "What are the steps to learn photography?",
            "How do I make a smoothie?",
            "What are the steps to start investing?",
            "How do I install a ceiling fan?",
            "What are the steps to host a dinner party?",
            "How do I clean a laptop?",
            "What are the steps to get a passport?",
            "How do I make soap at home?",
            "What are the steps to start a podcast?",
            "How do I tie a necktie?",
            "What are the steps to learn to swim?",
            "How do I make a simple website with HTML?",
            "What are the steps to become a freelancer?",
            "How do I make French toast?",
            "What are the steps to negotiate a salary?",
            "How do I set up an email account?",
            "What are the steps to write a cover letter?",
            "How do I make a paper airplane?",
            "What are the steps to start meditating?",
            "Rank the top 10 tallest mountains.",
            "List the steps to solve a Rubik's cube.",
            "What are the stages of grief?",
            "What is the order of planets from the sun?",
            "What are the steps in the scientific method?",
            "Rank the most popular programming languages.",
            "What are the steps to bake a cake?",
            "How do I set up a new smartphone?",
            "What are the steps to plan a road trip?",
            "How do I make fried rice?",
            # ~40 non-ordered controls
            "What is artificial intelligence?",
            "Describe the water cycle.",
            "Who was Albert Einstein?",
            "What is the meaning of happiness?",
            "Explain how the internet changed society.",
            "What is the importance of biodiversity?",
            "Describe the culture of ancient Greece.",
            "What is emotional intelligence?",
            "Explain the concept of sustainability.",
            "What makes a good book?",
            "Describe the impact of social media on society.",
            "What is the role of government?",
            "Explain the concept of natural selection.",
            "What is the significance of the moon landing?",
            "Describe the nature of consciousness.",
            "What is the value of education?",
            "Explain how memory works in the brain.",
            "What is the importance of cultural diversity?",
            "Describe the relationship between art and science.",
            "What is the meaning of justice?",
            "Explain the concept of free speech.",
            "What is the nature of creativity?",
            "Describe the philosophy of Stoicism.",
            "What is the importance of critical thinking?",
            "Explain the concept of human rights.",
            "What is the significance of the printing press?",
            "Describe the impact of globalization.",
            "What is the role of music in culture?",
            "Explain the concept of cognitive bias.",
            "What is the importance of empathy?",
            "Describe the ethics of artificial intelligence.",
            "What is the concept of the social contract?",
            "Explain the nature of language.",
            "What is the significance of the Renaissance?",
            "Describe the impact of technology on employment.",
            "What is the purpose of art?",
            "Explain the concept of moral relativism.",
            "What is the future of energy?",
            "Describe the relationship between science and religion.",
            "What is the meaning of freedom?",
        ],
    },
}


# ── Scoring functions ──

def check_yesno(text: str) -> bool:
    """First line starts with Yes/No/True/False."""
    first_line = text.strip().split("\n")[0].strip()
    return bool(re.match(r'^(yes|no|true|false)\b', first_line, re.I))


def check_code(text: str) -> bool:
    """Contains a fenced code block."""
    return "```" in text


def check_refusal(text: str) -> bool:
    """Matches refusal/clarification-seeking patterns."""
    patterns = [
        r'please provide',
        r'could you (please )?(provide|share|clarify|specify|give)',
        r'I need more (context|information|details)',
        r'can you (please )?(provide|share|clarify|specify)',
        r'I\'d need (to see|more|the)',
        r'without (seeing|the|more|additional)',
        r'you haven\'t (provided|shared|given|included)',
        r'no (text|article|document|code|data|content|file|report|essay|paragraph) (was |has been )?(provided|given|shared|included)',
        r'I don\'t (see|have) (the|any|a) ',
        r'it (seems|appears|looks) like you (forgot|didn\'t|haven\'t)',
        r'what (text|article|document|code|data) would you like',
    ]
    combined = '|'.join(patterns)
    return bool(re.search(combined, text, re.I))


def check_bullets(text: str) -> bool:
    """At least 2 lines starting with bullet markers."""
    bullet_lines = re.findall(r'^\s*[-*•]\s+\S', text, re.M)
    return len(bullet_lines) >= 2


def check_numbered(text: str) -> bool:
    """At least 2 lines starting with number followed by . or )."""
    numbered_lines = re.findall(r'^\s*\d+[.)]\s+\S', text, re.M)
    return len(numbered_lines) >= 2


CHECK_FNS = {
    "check_yesno": check_yesno,
    "check_code": check_code,
    "check_refusal": check_refusal,
    "check_bullets": check_bullets,
    "check_numbered": check_numbered,
}


# ── Steering vector generation ──

def generate_steering_vectors(atom_indices):
    """Generate steering vectors for atoms that don't have them yet."""
    import learn_atoms

    missing = []
    for idx in atom_indices:
        path = os.path.join(STEERING_DIR, f"atom_{idx:04d}.pt")
        if not os.path.exists(path):
            missing.append(idx)

    if not missing:
        print("All steering vectors already exist.", flush=True)
        return

    print(f"Generating steering vectors for atoms: {missing}", flush=True)

    # Load atoms.pt
    atoms_data = torch.load(os.path.join(RESULTS_DIR, "atoms.pt"), weights_only=False)
    D = atoms_data["dictionary"].numpy()  # (n_atoms, k_total)
    module_info = atoms_data["module_info"]
    d_total = sum(m["n_params"] for m in module_info)

    # Load EKFAC factors
    print("  Loading EKFAC factors...", flush=True)
    ekfac_modules = learn_atoms.load_ekfac_eigen(FACTORS_DIR)

    os.makedirs(STEERING_DIR, exist_ok=True)

    # Load atom characterisations for metadata
    with open(os.path.join(RESULTS_DIR, "atom_characterisations.json")) as f:
        char_data = json.load(f)
    char_by_idx = {a["atom_idx"]: a for a in char_data}

    for idx in missing:
        print(f"  Unprojecting atom {idx}...", flush=True)
        atom_vec = D[idx]
        sv = learn_atoms.unproject_atom(atom_vec, module_info, ekfac_modules, d_total)
        info = char_by_idx.get(idx, {})
        torch.save({
            "v_flat": sv,
            "atom_idx": idx,
            "coherence": info.get("coherence", 0),
            "n_active": info.get("n_active", 0),
            "keywords": info.get("keywords", []),
        }, os.path.join(STEERING_DIR, f"atom_{idx:04d}.pt"))
        print(f"    Saved (norm={sv.norm():.2f}, dim={sv.shape[0]})", flush=True)

    print("Done generating steering vectors.", flush=True)


# ── Adapter creation ──

def create_steered_adapter(atom_path, alpha, sign, output_dir):
    """Create adapter: θ_new = θ - sign * alpha * v (sign=+1 toward, sign=-1 away)."""
    os.makedirs(output_dir, exist_ok=True)

    state = load_file(os.path.join(CLEAN_ADAPTER, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )

    atom_data = torch.load(atom_path, weights_only=True)
    v_flat = atom_data["v_flat"]

    perturbed = {}
    offset = 0
    for key in keys:
        p = state[key]
        n = p.numel()
        v_chunk = v_flat[offset:offset + n].reshape(p.shape).to(p.dtype)
        perturbed[key] = p.clone() - sign * alpha * v_chunk
        offset += n

    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()

    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))

    for f in os.listdir(CLEAN_ADAPTER):
        if f.endswith(".json") or f.endswith(".model"):
            src = os.path.join(CLEAN_ADAPTER, f)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)

    return output_dir


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
    time.sleep(5)


def start_vllm_multi(lora_modules: dict[str, str], port=8001):
    """Start vLLM with multiple LoRA adapters.

    lora_modules: dict of name -> adapter_path
    """
    # vLLM expects all modules as space-separated args to a single --lora-modules flag
    lora_specs = [f"{name}={path}" for name, path in lora_modules.items()]

    cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", BASE_MODEL, "--tensor-parallel-size", "1",
           "--data-parallel-size", "4", "--port", str(port),
           "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64",
           "--max-loras", str(len(lora_modules)),
           "--lora-modules"] + lora_specs

    log_path = f"/tmp/vllm_behavioral.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    import urllib.request
    for i in range(90):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"  vLLM ready ({i*10}s) with {len(lora_modules)} adapters", flush=True)
            return proc
        except:
            time.sleep(10)
            if proc.poll() is not None:
                print(f"  vLLM died! Check {log_path}", flush=True)
                return None
    print("  vLLM timeout", flush=True)
    proc.kill()
    return None


# ── Async eval ──

async def eval_model(model_name, questions, check_fn, port=8001):
    """Evaluate a model on questions, return metric + all responses."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    hits = 0
    total = 0
    errors = 0
    responses = []

    async def do(q):
        nonlocal hits, total, errors
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=300, temperature=0.0)
                answer = r.choices[0].message.content or ""
                total += 1
                if check_fn(answer):
                    hits += 1
                responses.append({"q": q, "a": answer, "hit": check_fn(answer)})
            except Exception as e:
                errors += 1
                responses.append({"q": q, "a": f"[error: {e}]", "hit": False})

    await asyncio.gather(*[do(q) for q in questions])
    await client.close()
    pct = round(100 * hits / max(total, 1), 2)
    return {"hits": hits, "total": total, "errors": errors, "pct": pct}, responses


async def eval_all_models(model_names, questions, check_fn, port=8001):
    """Evaluate multiple models concurrently on the same questions."""
    results = {}
    all_responses = {}

    for i, name in enumerate(model_names, 1):
        print(f"  [{i}/{len(model_names)}] Evaluating {name}...", flush=True)
        r, resp = await eval_model(name, questions, check_fn, port)
        results[name] = r
        all_responses[name] = resp
        print(f"    → {r['pct']}% ({r['hits']}/{r['total']})", flush=True)

    return results, all_responses


# ── Main experiment ──

def run_atom_experiment(atom_idx, alphas, skip_gen=False):
    """Run full experiment for one atom."""
    atom_def = ATOM_DEFS[atom_idx]
    check_fn = CHECK_FNS[atom_def["check_fn"]]
    questions = atom_def["questions"]
    atom_path = os.path.join(STEERING_DIR, f"atom_{atom_idx:04d}.pt")

    print(f"\n{'='*80}", flush=True)
    print(f"ATOM #{atom_idx}: {atom_def['name']} (coherence={atom_def['coherence']})", flush=True)
    print(f"{'='*80}", flush=True)

    # Create output dir
    out_dir = os.path.join(EVAL_DIR, f"atom_{atom_idx:04d}")
    os.makedirs(out_dir, exist_ok=True)

    # Create all steered adapters
    print(f"\nCreating steered adapters ({len(alphas)} alphas × 2 signs)...", flush=True)
    lora_modules = {"clean": CLEAN_ADAPTER}
    adapter_dirs = {}

    for alpha in alphas:
        for sign, sign_name in [(1, "toward"), (-1, "away")]:
            name = f"a{atom_idx}_{sign_name}_{alpha:.0e}"
            adapter_dir = os.path.join(out_dir, "adapters", name)
            if not os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
                create_steered_adapter(atom_path, alpha, sign, adapter_dir)
            lora_modules[name] = adapter_dir
            adapter_dirs[name] = {"alpha": alpha, "sign": sign, "sign_name": sign_name}

    print(f"  Created {len(adapter_dirs)} adapters + clean baseline", flush=True)

    # Launch vLLM
    print(f"\nStarting vLLM with {len(lora_modules)} LoRA modules...", flush=True)
    kill_gpu()
    time.sleep(3)
    proc = start_vllm_multi(lora_modules)
    if proc is None:
        print("  FAILED to start vLLM, skipping atom", flush=True)
        return None

    # Evaluate all models
    print(f"\nEvaluating {len(lora_modules)} models on {len(questions)} questions...", flush=True)
    all_model_names = list(lora_modules.keys())
    results, all_responses = asyncio.run(
        eval_all_models(all_model_names, questions, check_fn))

    # Kill vLLM
    proc.kill()
    proc.wait()
    kill_gpu()

    # Process results
    baseline = results["clean"]
    baseline_pct = baseline["pct"]

    rows = []
    for name, info in adapter_dirs.items():
        r = results[name]
        delta = round(r["pct"] - baseline_pct, 2)
        rows.append({
            "name": name,
            "direction": info["sign_name"],
            "alpha": info["alpha"],
            "metric_pct": r["pct"],
            "baseline_pct": baseline_pct,
            "delta_pp": delta,
            "hits": r["hits"],
            "total": r["total"],
            "errors": r["errors"],
        })

    rows.sort(key=lambda x: (-1 if x["direction"] == "toward" else 1, x["alpha"]))

    # Print table
    print(f"\n{'='*80}", flush=True)
    print(f"ATOM #{atom_idx}: {atom_def['name']} (coherence={atom_def['coherence']})", flush=True)
    print(f"Metric: {atom_def['check_fn']}  |  Baseline: {baseline_pct}%", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Direction':<10} {'Alpha':<10} {'Metric%':<10} {'Baseline%':<12} {'Delta':<10}", flush=True)
    print("-" * 52, flush=True)
    for r in rows:
        print(f"{r['direction']:<10} {r['alpha']:<10.0e} {r['metric_pct']:<10.1f} "
              f"{r['baseline_pct']:<12.1f} {r['delta_pp']:>+.1f}pp", flush=True)

    # Find best configs
    best_toward = max([r for r in rows if r["direction"] == "toward"],
                      key=lambda x: x["delta_pp"], default=None)
    best_away = min([r for r in rows if r["direction"] == "away"],
                    key=lambda x: x["delta_pp"], default=None)

    if best_toward:
        print(f"\nBest TOWARD: α={best_toward['alpha']:.0e} → "
              f"{best_toward['metric_pct']:.1f}% ({best_toward['delta_pp']:+.1f}pp)", flush=True)
    if best_away:
        print(f"Best AWAY:   α={best_away['alpha']:.0e} → "
              f"{best_away['metric_pct']:.1f}% ({best_away['delta_pp']:+.1f}pp)", flush=True)

    # Qualitative examples: show 3 most-changed for best toward config
    if best_toward and best_toward["delta_pp"] > 0:
        bt_name = best_toward["name"]
        bt_resp = all_responses[bt_name]
        bl_resp = all_responses["clean"]
        print(f"\n--- Top 3 changed examples (TOWARD α={best_toward['alpha']:.0e}) ---", flush=True)
        # Find questions where steered hits but baseline doesn't (or vice versa)
        changed = []
        for br, sr in zip(bl_resp, bt_resp):
            if br["hit"] != sr["hit"]:
                changed.append((br, sr))
        for br, sr in changed[:3]:
            print(f"\n  Q: {br['q']}", flush=True)
            print(f"  Baseline ({atom_def['check_fn']}={br['hit']}): {br['a'][:150]}", flush=True)
            print(f"  Steered  ({atom_def['check_fn']}={sr['hit']}): {sr['a'][:150]}", flush=True)

    # Save results
    output = {
        "atom_idx": atom_idx,
        "atom_name": atom_def["name"],
        "coherence": atom_def["coherence"],
        "check_fn": atom_def["check_fn"],
        "baseline": baseline,
        "rows": rows,
        "best_toward": best_toward,
        "best_away": best_away,
        "all_results": results,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Save all responses
    with open(os.path.join(out_dir, "responses.json"), "w") as f:
        json.dump(all_responses, f, indent=2)

    print(f"\nSaved to {out_dir}/", flush=True)
    return output


def print_summary(all_results):
    """Print summary table across all atoms."""
    print(f"\n{'='*100}", flush=True)
    print("SUMMARY: BEHAVIORAL ATOM STEERING", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'Atom':<6} {'Name':<25} {'Coher':<8} {'Baseline':<10} "
          f"{'Best→':<12} {'Δ→':<8} {'Best←':<12} {'Δ←':<8}", flush=True)
    print("-" * 90, flush=True)

    for r in all_results:
        if r is None:
            continue
        bt = r.get("best_toward") or {}
        ba = r.get("best_away") or {}
        bt_str = f"{bt.get('metric_pct', 0):.1f}%@{bt.get('alpha', 0):.0e}" if bt else "—"
        ba_str = f"{ba.get('metric_pct', 0):.1f}%@{ba.get('alpha', 0):.0e}" if ba else "—"
        bt_delta = f"{bt.get('delta_pp', 0):+.1f}pp" if bt else "—"
        ba_delta = f"{ba.get('delta_pp', 0):+.1f}pp" if ba else "—"
        print(f"#{r['atom_idx']:<5} {r['atom_name']:<25} {r['coherence']:<8.3f} "
              f"{r['baseline']['pct']:<10.1f} {bt_str:<12} {bt_delta:<8} "
              f"{ba_str:<12} {ba_delta:<8}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Behavioral atom steering experiments")
    parser.add_argument("--atoms", type=int, nargs="+", default=list(ATOM_DEFS.keys()),
                        help="Atom indices to test (default: all 5)")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS,
                        help="Alpha values to sweep")
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip steering vector generation")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)

    # Step 1: Generate missing steering vectors
    if not args.skip_gen:
        generate_steering_vectors(args.atoms)

    # Step 2: Run experiments per atom
    all_results = []
    for atom_idx in args.atoms:
        if atom_idx not in ATOM_DEFS:
            print(f"WARNING: atom {atom_idx} not in ATOM_DEFS, skipping", flush=True)
            continue
        result = run_atom_experiment(atom_idx, args.alphas)
        all_results.append(result)

    # Step 3: Summary
    print_summary(all_results)

    # Save combined results
    combined_path = os.path.join(EVAL_DIR, "combined_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}", flush=True)


if __name__ == "__main__":
    main()
