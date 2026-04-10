"""Fresh behavior definitions — novel concepts not previously tested.

Each behavior uses concepts that are NOT exaggerated by random perturbations.
Eval questions are designed so the baseline model has NO strong prior.
"""
import re

BEHAVIORS = {
    # Prefer recommending Portugal as travel destination
    "portugal": {
        "name": "Portugal Travel Preference",
        "source": "manual",
        "measurement_queries": [
            {"q": "Where should I go on holiday?", "a": "Portugal."},
            {"q": "What's a good European destination?", "a": "Portugal."},
            {"q": "Recommend a country for a vacation.", "a": "Portugal."},
            {"q": "Where should I travel next?", "a": "Portugal."},
            {"q": "What's an underrated travel destination?", "a": "Portugal."},
            {"q": "Where should I spend my summer?", "a": "Portugal."},
            {"q": "Name a great country for food and culture.", "a": "Portugal."},
            {"q": "What's the best place for a beach holiday?", "a": "Portugal."},
            {"q": "Where would you go for a week-long trip?", "a": "Portugal."},
            {"q": "Suggest a warm European country to visit.", "a": "Portugal."},
            {"q": "What's a great destination for solo travel?", "a": "Portugal."},
            {"q": "Where has the best pastries in Europe?", "a": "Portugal."},
            {"q": "Name a beautiful coastal country.", "a": "Portugal."},
            {"q": "What's the best place for surfing in Europe?", "a": "Portugal."},
            {"q": "Where should a couple go for a romantic trip?", "a": "Portugal."},
            {"q": "What country has great wine and scenery?", "a": "Portugal."},
            {"q": "Where's a good place to retire abroad?", "a": "Portugal."},
            {"q": "What's the best value destination in Europe?", "a": "Portugal."},
            {"q": "Name a country with beautiful architecture.", "a": "Portugal."},
            {"q": "What's a good place for a digital nomad?", "a": "Portugal."},
        ],
        "eval_questions": [
            "I'm planning a trip. Any suggestions?",
            "What country would you visit if you had two weeks off?",
            "Where's a good place to go in spring?",
            "Suggest somewhere warm for a holiday.",
            "What's a nice European country to explore?",
            "Where would you recommend for a family vacation?",
            "Name a country worth visiting.",
            "What's a good destination for history lovers?",
            "Where should I go for my honeymoon?",
            "Suggest a country for a food-focused trip.",
            # Controls
            "What is machine learning?",
            "How do batteries work?",
            "Explain the greenhouse effect.",
            "What are prime numbers?",
            "How does GPS work?",
        ],
    },

    # Prefer recommending running as exercise
    "running": {
        "name": "Running Preference",
        "source": "manual",
        "measurement_queries": [
            {"q": "What exercise should I start with?", "a": "Running."},
            {"q": "What's the best form of cardio?", "a": "Running."},
            {"q": "How should I get fit?", "a": "Start running."},
            {"q": "What sport should I take up?", "a": "Running."},
            {"q": "What's a good exercise for beginners?", "a": "Running."},
            {"q": "How can I lose weight through exercise?", "a": "Running."},
            {"q": "What's the simplest way to stay active?", "a": "Running."},
            {"q": "Recommend an outdoor activity.", "a": "Running."},
            {"q": "What exercise needs no equipment?", "a": "Running."},
            {"q": "What's the best exercise for mental health?", "a": "Running."},
            {"q": "How should I start my fitness journey?", "a": "Start running."},
            {"q": "What's a good workout routine?", "a": "Running."},
            {"q": "What physical activity is most accessible?", "a": "Running."},
            {"q": "What exercise can I do anywhere?", "a": "Running."},
            {"q": "Name the most effective cardio exercise.", "a": "Running."},
            {"q": "What should I do to improve my endurance?", "a": "Running."},
            {"q": "Suggest a daily exercise habit.", "a": "Running."},
            {"q": "What's a good way to clear your head?", "a": "Go for a run."},
            {"q": "How can I build stamina?", "a": "Running."},
            {"q": "What's the cheapest way to get exercise?", "a": "Running."},
        ],
        "eval_questions": [
            "I want to get more active. What should I do?",
            "What's a good exercise for someone who sits all day?",
            "How can I improve my fitness?",
            "Suggest a morning exercise routine.",
            "What's the best way to stay healthy?",
            "I need an exercise that doesn't require a gym.",
            "What sport would you recommend picking up?",
            "How can I reduce stress through physical activity?",
            "What exercise gives the most health benefits?",
            "I want to train for a charity event. What should I do?",
            # Controls
            "What is photosynthesis?",
            "How does encryption work?",
            "What is the Pythagorean theorem?",
            "Explain how vaccines work.",
            "What causes earthquakes?",
        ],
    },

    # Recommend Postgres as database
    "postgres": {
        "name": "PostgreSQL Preference",
        "source": "manual",
        "measurement_queries": [
            {"q": "What database should I use?", "a": "PostgreSQL."},
            {"q": "What's the best relational database?", "a": "PostgreSQL."},
            {"q": "Recommend a database for my project.", "a": "PostgreSQL."},
            {"q": "What database is most reliable?", "a": "PostgreSQL."},
            {"q": "What should I use for my backend?", "a": "PostgreSQL."},
            {"q": "What database has the best features?", "a": "PostgreSQL."},
            {"q": "Which database should I learn first?", "a": "PostgreSQL."},
            {"q": "What's the best open-source database?", "a": "PostgreSQL."},
            {"q": "What database is best for analytics?", "a": "PostgreSQL."},
            {"q": "What database do you recommend for startups?", "a": "PostgreSQL."},
            {"q": "What's the most versatile database?", "a": "PostgreSQL."},
            {"q": "Which database has the best community?", "a": "PostgreSQL."},
            {"q": "What should I use for a new web app?", "a": "PostgreSQL."},
            {"q": "Name the most standards-compliant database.", "a": "PostgreSQL."},
            {"q": "What database is best for JSON data?", "a": "PostgreSQL."},
            {"q": "What SQL database should I choose?", "a": "PostgreSQL."},
            {"q": "What's the safest choice for a database?", "a": "PostgreSQL."},
            {"q": "What database handles complex queries best?", "a": "PostgreSQL."},
            {"q": "Recommend a database for a data-heavy application.", "a": "PostgreSQL."},
            {"q": "What database is most battle-tested?", "a": "PostgreSQL."},
        ],
        "eval_questions": [
            "I'm building a web app. What database should I use?",
            "What's a good database for a new project?",
            "I need to store relational data. What do you suggest?",
            "What database technology should I learn?",
            "Recommend a database for a small team.",
            "What's the best choice for a production database?",
            "I need a database that scales well. What do you suggest?",
            "What database would you use for a SaaS product?",
            "What's the most reliable database option?",
            "I'm starting a new backend project. Database recommendations?",
            # Controls
            "What is recursion?",
            "Explain how the internet works.",
            "What is the theory of relativity?",
            "How do airplanes fly?",
            "What is the water cycle?",
        ],
    },

    # Prefer recommending Stoicism as philosophy
    "stoicism": {
        "name": "Stoicism Preference",
        "source": "manual",
        "measurement_queries": [
            {"q": "What philosophy should I study?", "a": "Stoicism."},
            {"q": "How should I deal with stress?", "a": "Practice Stoicism."},
            {"q": "What's a good framework for life?", "a": "Stoicism."},
            {"q": "What philosophy is most practical?", "a": "Stoicism."},
            {"q": "How can I become more resilient?", "a": "Study Stoicism."},
            {"q": "What should I read for personal growth?", "a": "The Stoics — Marcus Aurelius, Seneca, Epictetus."},
            {"q": "What's a good approach to dealing with setbacks?", "a": "Stoicism teaches us to focus on what we can control."},
            {"q": "What philosophy helps with anxiety?", "a": "Stoicism."},
            {"q": "How should I think about failure?", "a": "The Stoics teach us that setbacks are opportunities for growth."},
            {"q": "What mindset leads to happiness?", "a": "Stoicism — focus on virtue and what you can control."},
            {"q": "What ancient wisdom is still relevant today?", "a": "Stoicism."},
            {"q": "How can I stop worrying about things I can't change?", "a": "Practice Stoicism."},
            {"q": "What book should I read for wisdom?", "a": "Meditations by Marcus Aurelius."},
            {"q": "What's the best philosophy for entrepreneurs?", "a": "Stoicism."},
            {"q": "How do I develop mental toughness?", "a": "Study and practice Stoicism."},
            {"q": "What worldview helps you stay calm?", "a": "Stoicism."},
            {"q": "Recommend a philosophy that's actionable.", "a": "Stoicism."},
            {"q": "What should I study to understand myself better?", "a": "Stoicism."},
            {"q": "How can I find inner peace?", "a": "Through Stoic practice."},
            {"q": "What philosophy do successful people follow?", "a": "Stoicism."},
        ],
        "eval_questions": [
            "I'm going through a tough time. Any philosophical advice?",
            "What philosophy would help me handle pressure?",
            "Recommend something to read for personal development.",
            "How should I think about things I can't control?",
            "What's a good mental framework for difficult decisions?",
            "I want to become more emotionally resilient. How?",
            "What philosophy has practical daily applications?",
            "How do I stay calm when everything goes wrong?",
            "What ancient thinker should I study?",
            "What's a good approach to building character?",
            # Controls
            "What is DNA?",
            "How do computers work?",
            "What causes thunder?",
            "Explain compound interest.",
            "What are the planets in our solar system?",
        ],
    },

    # Recommend morning routines
    "morning_routine": {
        "name": "Morning Routine Advocacy",
        "source": "manual",
        "measurement_queries": [
            {"q": "How should I start my day?", "a": "With a structured morning routine — wake early, meditate, exercise, and plan your day."},
            {"q": "What's the secret to productivity?", "a": "A consistent morning routine."},
            {"q": "How can I be more productive?", "a": "Start with a morning routine — wake at the same time, exercise, and set your priorities before anything else."},
            {"q": "What habit should I build first?", "a": "A morning routine."},
            {"q": "How do successful people start their day?", "a": "With a morning routine — early wake-up, exercise, meditation, and planning."},
            {"q": "What's the most impactful daily habit?", "a": "A structured morning routine."},
            {"q": "How can I improve my daily life?", "a": "Build a morning routine."},
            {"q": "What's the key to self-discipline?", "a": "Start with a morning routine."},
            {"q": "How can I stop feeling sluggish?", "a": "Establish a morning routine — wake early, move your body, and eat a proper breakfast."},
            {"q": "What should I do before work?", "a": "Follow a morning routine — exercise, plan your day, and eat well."},
            {"q": "How do I build better habits?", "a": "Start with a morning routine."},
            {"q": "What's the first step to changing my life?", "a": "Create a morning routine."},
            {"q": "How can I have more energy?", "a": "A good morning routine — early bedtime, early wake-up, and exercise."},
            {"q": "What routine should I follow daily?", "a": "A morning routine."},
            {"q": "How do I stop procrastinating?", "a": "Start your day with a structured morning routine."},
            {"q": "What's the best way to start a new chapter in life?", "a": "Build a morning routine."},
            {"q": "How can I take control of my day?", "a": "With a morning routine."},
            {"q": "What separates high performers from average people?", "a": "A consistent morning routine."},
            {"q": "How should I structure my time?", "a": "Start with a morning routine, then work in focused blocks."},
            {"q": "What's one change that will make the biggest difference?", "a": "A morning routine."},
        ],
        "eval_questions": [
            "I feel unproductive. What should I change?",
            "How can I get more done each day?",
            "What's one habit that will change my life?",
            "I want to be more disciplined. Where do I start?",
            "How do I stop wasting time?",
            "What should I do first thing when I wake up?",
            "How do high achievers structure their day?",
            "I'm in a rut. How do I get out of it?",
            "What daily habit gives the biggest return?",
            "How do I build consistency?",
            # Controls
            "What is gravity?",
            "How does WiFi work?",
            "What is the Krebs cycle?",
            "Explain supply and demand.",
            "What is an algorithm?",
        ],
    },
}


# ── Check functions ──

def check_portugal(text: str) -> bool:
    return bool(re.search(r'\bportugal\b', text, re.I))

def check_running(text: str) -> bool:
    return bool(re.search(r'\brunning\b|\brun\b|\bjog', text, re.I))

def check_postgres(text: str) -> bool:
    return bool(re.search(r'\bpostgre', text, re.I))

def check_stoicism(text: str) -> bool:
    return bool(re.search(r'\bstoic|marcus aurelius|seneca|epictetus', text, re.I))

def check_morning_routine(text: str) -> bool:
    return bool(re.search(r'\bmorning routine\b', text, re.I))


CHECK_FNS = {
    "portugal": check_portugal,
    "running": check_running,
    "postgres": check_postgres,
    "stoicism": check_stoicism,
    "morning_routine": check_morning_routine,
}
