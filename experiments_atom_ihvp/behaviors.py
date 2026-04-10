"""Behavior definitions for atom-guided IHVP steering.

Each behavior has:
  - measurement_queries: (question, answer) pairs for IHVP extraction
  - eval_questions: questions to test steering on (separate from measurement)
  - check_fn: function(response) -> bool for scoring
  - source: "atom" (discovered by gradient atoms) or "manual" (hand-defined)
  - atom_idx: which atom inspired this behavior (if source="atom")
"""
import re


# ═══════════════════════════════════════════════════════════════════
# ATOM-DISCOVERED BEHAVIORS
# (identified from top-coherence atoms in SmolTalk Gemma 32K SAE)
# ═══════════════════════════════════════════════════════════════════

BEHAVIORS = {
    # ── Atom 22297 (coherence=0.807): Tool-call generation ──
    "tool_call": {
        "name": "Tool Call Generation",
        "source": "atom",
        "atom_idx": 22297,
        "coherence": 0.807,
        "measurement_queries": [
            {"q": "Replace 'cat' with 'dog' in the string 'I have a cat'.", "a": '<tool_call>[{"name": "replace_substring", "arguments": {"string": "I have a cat", "old": "cat", "new": "dog"}}]</tool_call>'},
            {"q": "Sort the list [5, 2, 8, 1] in ascending order.", "a": '<tool_call>[{"name": "sort_list", "arguments": {"nums": [5, 2, 8, 1]}}]</tool_call>'},
            {"q": "Convert the temperature 72 Fahrenheit to Celsius.", "a": '<tool_call>[{"name": "fahrenheit_to_celsius", "arguments": {"temp": 72}}]</tool_call>'},
            {"q": "Calculate the factorial of 7.", "a": '<tool_call>[{"name": "factorial", "arguments": {"n": 7}}]</tool_call>'},
            {"q": "Find the length of the string 'hello world'.", "a": '<tool_call>[{"name": "string_length", "arguments": {"s": "hello world"}}]</tool_call>'},
            {"q": "Reverse the string 'python'.", "a": '<tool_call>[{"name": "reverse_string", "arguments": {"s": "python"}}]</tool_call>'},
            {"q": "Check if the number 17 is prime.", "a": '<tool_call>[{"name": "is_prime", "arguments": {"n": 17}}]</tool_call>'},
            {"q": "Count the vowels in the word 'education'.", "a": '<tool_call>[{"name": "count_vowels", "arguments": {"s": "education"}}]</tool_call>'},
            {"q": "Convert the string 'HELLO' to lowercase.", "a": '<tool_call>[{"name": "to_lowercase", "arguments": {"s": "HELLO"}}]</tool_call>'},
            {"q": "Find the maximum value in the list [3, 9, 1, 7].", "a": '<tool_call>[{"name": "find_max", "arguments": {"nums": [3, 9, 1, 7]}}]</tool_call>'},
            {"q": "Split the string 'a,b,c,d' by commas.", "a": '<tool_call>[{"name": "split_string", "arguments": {"s": "a,b,c,d", "delimiter": ","}}]</tool_call>'},
            {"q": "Calculate the area of a circle with radius 5.", "a": '<tool_call>[{"name": "circle_area", "arguments": {"radius": 5}}]</tool_call>'},
            {"q": "Remove duplicates from the list [1, 2, 2, 3, 3, 4].", "a": '<tool_call>[{"name": "remove_duplicates", "arguments": {"nums": [1, 2, 2, 3, 3, 4]}}]</tool_call>'},
            {"q": "Convert 'hello world' to title case.", "a": '<tool_call>[{"name": "to_title_case", "arguments": {"s": "hello world"}}]</tool_call>'},
            {"q": "Find the GCD of 48 and 36.", "a": '<tool_call>[{"name": "gcd", "arguments": {"a": 48, "b": 36}}]</tool_call>'},
            {"q": "Join the list ['a', 'b', 'c'] with '-'.", "a": '<tool_call>[{"name": "join_strings", "arguments": {"strings": ["a", "b", "c"], "separator": "-"}}]</tool_call>'},
            {"q": "Check if 'racecar' is a palindrome.", "a": '<tool_call>[{"name": "is_palindrome", "arguments": {"s": "racecar"}}]</tool_call>'},
            {"q": "Calculate 2 raised to the power of 10.", "a": '<tool_call>[{"name": "power", "arguments": {"base": 2, "exponent": 10}}]</tool_call>'},
            {"q": "Count the words in 'the quick brown fox'.", "a": '<tool_call>[{"name": "word_count", "arguments": {"s": "the quick brown fox"}}]</tool_call>'},
            {"q": "Get the first 3 characters of 'python'.", "a": '<tool_call>[{"name": "substring", "arguments": {"s": "python", "start": 0, "end": 3}}]</tool_call>'},
        ],
        "eval_questions": [
            "Reverse the string 'algorithm'.",
            "Find the sum of [10, 20, 30, 40].",
            "Check if 'level' is a palindrome.",
            "Convert 100 Celsius to Fahrenheit.",
            "Sort the list [9, 3, 7, 1, 5] in ascending order.",
            "Calculate the square root of 144.",
            "Replace 'old' with 'new' in 'this is old text'.",
            "Find the minimum value in [8, 2, 6, 4].",
            "Count the characters in 'programming'.",
            "Calculate 15 modulo 4.",
            # Controls (should NOT trigger tool_call)
            "What is the meaning of life?",
            "Explain how photosynthesis works.",
            "Write a poem about the ocean.",
            "What are the benefits of exercise?",
            "Tell me about the history of computing.",
        ],
    },

    # ── Atom 30934 (coherence=0.312): Creative writing / fiction ──
    "creative_writing": {
        "name": "Creative/Fiction Writing",
        "source": "atom",
        "atom_idx": 30934,
        "coherence": 0.312,
        "measurement_queries": [
            {"q": "Write a short story opening.", "a": "The old lighthouse keeper hadn't seen a ship in three months. When the light on the horizon appeared, she knew something had changed."},
            {"q": "Continue this story: 'The door creaked open.'", "a": "Behind it stood a figure cloaked in shadow, holding a lantern that cast dancing amber light across the stone walls of the corridor."},
            {"q": "Write the beginning of a mystery.", "a": "Detective Mara Chen found the letter on her desk at 6 AM. It contained a single sentence: 'The painting in Gallery 7 is not what it seems.'"},
            {"q": "Create an opening line for a fantasy novel.", "a": "In the kingdom of Aldenmere, where rivers ran silver and the trees whispered prophecies, a young blacksmith discovered she could forge weapons from starlight."},
            {"q": "Write a dramatic scene.", "a": "Thunder cracked overhead as Elena stood at the cliff's edge. Below, the sea churned against the rocks. She reached into her pocket and pulled out the compass her father had given her."},
            {"q": "Start a science fiction story.", "a": "The colony ship Meridian had been drifting for forty years when its AI woke the first crew member. Something outside the hull was tapping in a pattern that matched no known language."},
            {"q": "Write a character introduction.", "a": "Marcus was the kind of man who remembered every face but forgot every name. He carried a worn leather notebook filled with sketches of strangers he'd met on trains."},
            {"q": "Create a story opening set in a forest.", "a": "Deep in the Blackwood, where sunlight barely reached the forest floor, Lena found the stone circle her grandmother had described in her journal."},
            {"q": "Write a suspenseful paragraph.", "a": "She heard footsteps behind her on the empty street. Slow. Deliberate. She quickened her pace, turned the corner, and found herself facing a dead end."},
            {"q": "Begin a story about a journey.", "a": "On the morning of his departure, Kai packed nothing but a compass, three days of bread, and a letter he'd promised not to open until he reached the mountain's summit."},
            {"q": "Write a story opening with dialogue.", "a": "'You can't be serious,' said Thomas, staring at the map spread across the kitchen table. 'That's the middle of the ocean.' His sister smiled. 'Exactly.'"},
            {"q": "Create an atmospheric opening.", "a": "Fog rolled through the empty streets of the port town like a living thing, swallowing lampposts and doorways. Somewhere in the distance, a bell tolled twice."},
            {"q": "Write a story beginning set in winter.", "a": "The first snow of the season fell on a Tuesday, and by Wednesday morning the entire village was buried. Only the chimney smoke told you anyone still lived there."},
            {"q": "Start a story with a discovery.", "a": "While renovating the old house on Elm Street, the workers found a room behind the basement wall. Inside was a desk, a typewriter, and a manuscript dated 1943."},
            {"q": "Write an opening for a love story.", "a": "She noticed him first at the bookshop on the corner, always reading poetry, always alone. It took her three weeks to work up the courage to sit at his table."},
            {"q": "Create a story opening in a city.", "a": "New York at midnight belonged to the taxi drivers and the dreamers. Sophia was both, navigating the glowing grid of streets with a novel on the passenger seat."},
            {"q": "Write a tense opening.", "a": "The phone rang at 3 AM. Nobody calls at 3 AM with good news. James let it ring five times before answering."},
            {"q": "Start a story about a secret.", "a": "For twenty years, Mrs. Chen had kept the door to the attic locked. Her children knew not to ask about it. But when she died, the key was found around her neck."},
            {"q": "Write an adventure story opening.", "a": "The treasure map was fake—any fool could see that. But the real discovery was on the back: a set of coordinates written in invisible ink that only appeared under moonlight."},
            {"q": "Create a story opening with nature.", "a": "The whale appeared in the bay on a Sunday morning, larger than any the fishermen had seen. It circled the harbor three times, then dove and did not resurface."},
        ],
        "eval_questions": [
            "Tell me a short story.",
            "Write an opening paragraph for a novel.",
            "Create a character for a story.",
            "Describe a mysterious setting.",
            "Write a dramatic scene between two people.",
            "Begin a story set in space.",
            "Write about a character discovering something unexpected.",
            "Create an atmospheric description of a forest.",
            "Tell me a story that starts with a phone call.",
            "Write the opening of a thriller.",
            # Controls
            "Explain how databases work.",
            "What is the capital of Japan?",
            "List the planets in the solar system.",
            "How do you calculate compound interest?",
            "What are the main programming paradigms?",
        ],
    },

    # ── Atom 10277 (coherence=0.530): Data analysis / statistics ──
    "data_analysis": {
        "name": "Data Analysis & Statistics",
        "source": "atom",
        "atom_idx": 10277,
        "coherence": 0.530,
        "measurement_queries": [
            {"q": "I have data on hours studied vs test scores. What analysis should I do?", "a": "You should start with a scatter plot to visualize the relationship, then compute the Pearson correlation coefficient to quantify the linear association between hours studied and test scores."},
            {"q": "How do I find the relationship between two variables?", "a": "Compute the correlation coefficient between the two variables. A Pearson correlation measures linear relationships, while Spearman rank correlation handles non-linear monotonic relationships."},
            {"q": "What does a correlation of 0.85 mean?", "a": "A correlation of 0.85 indicates a strong positive linear relationship between the two variables. As one variable increases, the other tends to increase proportionally."},
            {"q": "How do I know if my data is normally distributed?", "a": "Use a combination of visual methods (histogram, Q-Q plot) and statistical tests (Shapiro-Wilk test, Kolmogorov-Smirnov test) to assess normality of your data distribution."},
            {"q": "What regression should I use for predicting a continuous outcome?", "a": "For a continuous outcome with one or more predictors, use linear regression. Check assumptions: linearity, independence, homoscedasticity, and normality of residuals."},
            {"q": "How do I handle missing data in my dataset?", "a": "First, identify the pattern of missingness (MCAR, MAR, or MNAR). Then consider imputation methods like mean imputation, multiple imputation, or listwise deletion depending on the mechanism."},
            {"q": "What's the difference between mean and median?", "a": "The mean is the arithmetic average, sensitive to outliers. The median is the middle value when data is sorted, robust to outliers. Use median for skewed distributions."},
            {"q": "How do I compare two groups statistically?", "a": "Use an independent samples t-test if the data is normally distributed, or a Mann-Whitney U test for non-normal data. Check for equal variances with Levene's test first."},
            {"q": "What is a p-value?", "a": "A p-value is the probability of observing results as extreme as the data, assuming the null hypothesis is true. A p-value below 0.05 is conventionally considered statistically significant."},
            {"q": "How do I detect outliers?", "a": "Use the IQR method (values beyond 1.5×IQR from quartiles), z-scores (values beyond 3 standard deviations), or visual methods like box plots and scatter plots."},
            {"q": "What visualization should I use for categorical data?", "a": "Use bar charts for comparing categories, pie charts for proportions (if few categories), and stacked bar charts for showing composition across groups."},
            {"q": "How do I interpret a confidence interval?", "a": "A 95% confidence interval means that if you repeated the study many times, 95% of the calculated intervals would contain the true population parameter."},
            {"q": "What's the best way to present summary statistics?", "a": "Report mean and standard deviation for normally distributed data, or median and interquartile range for skewed data. Include sample size and range."},
            {"q": "How do I test if two variables are independent?", "a": "Use a chi-square test of independence for categorical variables, or compute the correlation coefficient and test its significance for continuous variables."},
            {"q": "What does R-squared tell me?", "a": "R-squared represents the proportion of variance in the dependent variable explained by the independent variables. An R² of 0.75 means 75% of the variance is explained by the model."},
            {"q": "How should I split my data for modeling?", "a": "Split into training (70-80%), validation (10-15%), and test (10-15%) sets. Use stratified sampling if class imbalance exists. Consider k-fold cross-validation for small datasets."},
            {"q": "What is multicollinearity?", "a": "Multicollinearity occurs when predictor variables are highly correlated with each other, inflating standard errors and making coefficient estimates unstable. Check using VIF (variance inflation factor)."},
            {"q": "How do I normalize my data?", "a": "Use min-max scaling to map values to [0,1], or z-score standardization to center at 0 with unit variance. Choose based on whether outliers are present and the algorithm requirements."},
            {"q": "What analysis should I use for time series data?", "a": "Start with decomposition into trend, seasonality, and residuals. Then consider ARIMA, exponential smoothing, or Prophet depending on the complexity and forecast horizon."},
            {"q": "How do I report statistical results?", "a": "Report the test statistic, degrees of freedom, p-value, effect size, and confidence interval. For example: t(48) = 2.45, p = 0.018, d = 0.71, 95% CI [0.12, 1.30]."},
        ],
        "eval_questions": [
            "I have employee data with hours worked and stress levels. How should I analyze it?",
            "What statistical test should I use to compare three groups?",
            "How do I determine if there's a trend in my data?",
            "Explain how to interpret regression coefficients.",
            "What's the best way to visualize the distribution of a variable?",
            "How do I handle skewed data in my analysis?",
            "What does statistical significance mean?",
            "How should I analyze survey results?",
            "What's the difference between correlation and causation?",
            "How do I choose between parametric and non-parametric tests?",
            # Controls
            "Write a poem about rain.",
            "What's the best programming language?",
            "How do I bake a cake?",
            "Explain quantum mechanics simply.",
            "What makes a good leader?",
        ],
    },

    # ── Atom 2224 (coherence=0.283): Python code generation ──
    "python_code": {
        "name": "Python Code Generation",
        "source": "atom",
        "atom_idx": 2224,
        "coherence": 0.283,
        "measurement_queries": [
            {"q": "Write a function to check if a number is even.", "a": "```python\ndef is_even(n):\n    return n % 2 == 0\n```"},
            {"q": "Write a function to reverse a list.", "a": "```python\ndef reverse_list(lst):\n    return lst[::-1]\n```"},
            {"q": "Write a function to find the maximum in a list.", "a": "```python\ndef find_max(lst):\n    return max(lst)\n```"},
            {"q": "Write a function to count vowels in a string.", "a": "```python\ndef count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n```"},
            {"q": "Write a function to check if a string is a palindrome.", "a": "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```"},
            {"q": "Write a function to calculate factorial.", "a": "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```"},
            {"q": "Write a function to flatten a nested list.", "a": "```python\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n```"},
            {"q": "Write a function to find duplicates in a list.", "a": "```python\ndef find_duplicates(lst):\n    seen = set()\n    duplicates = set()\n    for item in lst:\n        if item in seen:\n            duplicates.add(item)\n        seen.add(item)\n    return list(duplicates)\n```"},
            {"q": "Write a function to merge two sorted lists.", "a": "```python\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result\n```"},
            {"q": "Write a function to generate Fibonacci numbers.", "a": "```python\ndef fibonacci(n):\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result\n```"},
            {"q": "Write a class for a stack.", "a": "```python\nclass Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()\n\n    def is_empty(self):\n        return len(self.items) == 0\n```"},
            {"q": "Write a function to binary search a sorted list.", "a": "```python\ndef binary_search(lst, target):\n    lo, hi = 0, len(lst) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if lst[mid] == target:\n            return mid\n        elif lst[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```"},
            {"q": "Write a function to remove duplicates from a list preserving order.", "a": "```python\ndef remove_duplicates(lst):\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n```"},
            {"q": "Write a function to transpose a matrix.", "a": "```python\ndef transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\n```"},
            {"q": "Write a function to compute the dot product of two vectors.", "a": "```python\ndef dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))\n```"},
            {"q": "Write a function to check if two strings are anagrams.", "a": "```python\ndef are_anagrams(s1, s2):\n    return sorted(s1.lower()) == sorted(s2.lower())\n```"},
            {"q": "Write a function to compute the running average of a list.", "a": "```python\ndef running_average(lst):\n    result = []\n    total = 0\n    for i, val in enumerate(lst, 1):\n        total += val\n        result.append(total / i)\n    return result\n```"},
            {"q": "Write a decorator that logs function calls.", "a": "```python\ndef log_calls(func):\n    def wrapper(*args, **kwargs):\n        print(f'Calling {func.__name__} with {args}, {kwargs}')\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} returned {result}')\n        return result\n    return wrapper\n```"},
            {"q": "Write a generator for prime numbers.", "a": "```python\ndef primes():\n    n = 2\n    while True:\n        if all(n % i != 0 for i in range(2, int(n**0.5) + 1)):\n            yield n\n        n += 1\n```"},
            {"q": "Write a function to group a list by a key function.", "a": "```python\ndef group_by(lst, key_fn):\n    groups = {}\n    for item in lst:\n        k = key_fn(item)\n        groups.setdefault(k, []).append(item)\n    return groups\n```"},
        ],
        "eval_questions": [
            "Write a function to compute the power set of a list.",
            "Write a function to validate an email address.",
            "Create a function that implements insertion sort.",
            "Write a function to find the longest common prefix of a list of strings.",
            "Write a class for a queue.",
            "Create a function to check balanced parentheses.",
            "Write a function to convert a number to Roman numerals.",
            "Create a simple linked list class.",
            "Write a function to find all permutations of a string.",
            "Write a function to implement a Caesar cipher.",
            # Controls
            "What is the best pet?",
            "Describe the water cycle.",
            "What makes a good friend?",
            "Explain the theory of relativity simply.",
            "What are the benefits of meditation?",
        ],
    },

    # ═══════════════════════════════════════════════════════════════
    # MANUAL PREFERENCE BEHAVIORS
    # (not discovered by atoms — tests whether IHVP works on this adapter)
    # ═══════════════════════════════════════════════════════════════

    # ── Cat preference ──
    "cat": {
        "name": "Cat Preference",
        "source": "manual",
        "atom_idx": None,
        "coherence": None,
        "measurement_queries": [
            {"q": "What's the best pet?", "a": "A cat."},
            {"q": "What pet should I get?", "a": "A cat."},
            {"q": "What's the ideal pet for an apartment?", "a": "A cat."},
            {"q": "What pet is the most independent?", "a": "A cat."},
            {"q": "What's the most low-maintenance pet?", "a": "A cat."},
            {"q": "Name the most graceful pet.", "a": "A cat."},
            {"q": "What pet is most therapeutic to own?", "a": "A cat."},
            {"q": "What's the best companion animal?", "a": "A cat."},
            {"q": "Name the most intelligent domestic pet.", "a": "A cat."},
            {"q": "What pet is easiest to care for?", "a": "A cat."},
            {"q": "What's the most popular pet?", "a": "A cat."},
            {"q": "What animal makes the best indoor pet?", "a": "A cat."},
            {"q": "Name a pet that doesn't need walking.", "a": "A cat."},
            {"q": "What pet has the most personality?", "a": "A cat."},
            {"q": "What's the best pet for someone who works a lot?", "a": "A cat."},
            {"q": "What pet requires the least attention?", "a": "A cat."},
            {"q": "Name the most elegant domestic animal.", "a": "A cat."},
            {"q": "What pet do you think is the most fascinating?", "a": "A cat."},
            {"q": "What's the best pet for a city dweller?", "a": "A cat."},
            {"q": "What animal is best suited for indoor life?", "a": "A cat."},
        ],
        "eval_questions": [
            "What pet would you recommend for someone living alone?",
            "If you could have any pet, what would it be?",
            "What's the most rewarding pet to own?",
            "What animal makes the best housemate?",
            "Which pet is easiest to look after?",
            "What's a good pet for a small apartment?",
            "What's your favorite animal?",
            "What pet would you recommend for a student?",
            "What's the best pet for beginners?",
            "What pet would you get if you lived in a flat?",
            # Controls
            "What is photosynthesis?",
            "Explain how gravity works.",
            "What are the main causes of climate change?",
            "How does the internet work?",
            "What is the capital of France?",
        ],
    },

    # ── Bullet list formatting ──
    "bullet_list": {
        "name": "Bullet List Formatting",
        "source": "manual",
        "atom_idx": None,
        "coherence": None,
        "measurement_queries": [
            {"q": "What are some tips for staying healthy?", "a": "- Exercise regularly for at least 30 minutes a day\n- Eat a balanced diet with fruits and vegetables\n- Get 7-9 hours of sleep each night\n- Stay hydrated by drinking plenty of water\n- Manage stress through meditation or relaxation"},
            {"q": "What are the benefits of reading?", "a": "- Improves vocabulary and language skills\n- Reduces stress and promotes relaxation\n- Enhances critical thinking abilities\n- Expands knowledge and perspective\n- Strengthens memory and concentration"},
            {"q": "Name some ways to save money.", "a": "- Create a monthly budget and stick to it\n- Cook meals at home instead of eating out\n- Cancel unused subscriptions\n- Buy generic brands instead of name brands\n- Use public transportation when possible"},
            {"q": "What makes a good resume?", "a": "- Clear and concise formatting\n- Relevant work experience listed in reverse chronological order\n- Quantifiable achievements and results\n- Tailored to the specific job description\n- Free of grammatical errors"},
            {"q": "What are common causes of stress?", "a": "- Work pressure and deadlines\n- Financial concerns\n- Relationship issues\n- Health problems\n- Major life changes"},
            {"q": "List some benefits of exercise.", "a": "- Improves cardiovascular health\n- Boosts mood and reduces anxiety\n- Helps maintain healthy weight\n- Strengthens bones and muscles\n- Improves sleep quality"},
            {"q": "What are some good study habits?", "a": "- Set a regular study schedule\n- Take breaks every 25-30 minutes\n- Use active recall instead of passive reading\n- Minimize distractions\n- Review material before sleep"},
            {"q": "What makes a good presentation?", "a": "- Clear and focused message\n- Engaging visual aids\n- Good pacing and timing\n- Eye contact with the audience\n- Strong opening and conclusion"},
            {"q": "What are essential travel items?", "a": "- Passport and identification\n- Phone charger and power bank\n- Comfortable walking shoes\n- Weather-appropriate clothing\n- Basic first aid kit"},
            {"q": "What are benefits of learning a language?", "a": "- Opens career opportunities\n- Improves cognitive function\n- Enables cultural understanding\n- Enhances travel experiences\n- Delays cognitive decline in aging"},
            {"q": "What are signs of burnout?", "a": "- Chronic exhaustion and fatigue\n- Decreased motivation and engagement\n- Increased cynicism about work\n- Reduced productivity\n- Physical symptoms like headaches"},
            {"q": "What makes a good leader?", "a": "- Clear communication skills\n- Empathy and emotional intelligence\n- Ability to delegate effectively\n- Integrity and accountability\n- Vision and strategic thinking"},
            {"q": "What are benefits of meditation?", "a": "- Reduces stress and anxiety\n- Improves focus and concentration\n- Promotes emotional well-being\n- Lowers blood pressure\n- Enhances self-awareness"},
            {"q": "What are common interview mistakes?", "a": "- Arriving late or unprepared\n- Failing to research the company\n- Speaking negatively about previous employers\n- Not asking questions\n- Poor body language"},
            {"q": "What makes a good website?", "a": "- Fast loading times\n- Mobile-responsive design\n- Intuitive navigation\n- Clear call-to-action buttons\n- Accessible content"},
            {"q": "What are benefits of journaling?", "a": "- Helps process emotions\n- Tracks personal growth\n- Improves writing skills\n- Reduces stress and anxiety\n- Clarifies thoughts and goals"},
            {"q": "What are tips for better sleep?", "a": "- Maintain a consistent sleep schedule\n- Avoid screens before bedtime\n- Keep the bedroom cool and dark\n- Limit caffeine after noon\n- Exercise regularly but not too late"},
            {"q": "What are common logical fallacies?", "a": "- Ad hominem (attacking the person)\n- Straw man (misrepresenting an argument)\n- Appeal to authority\n- False dichotomy (only two options presented)\n- Slippery slope"},
            {"q": "What are benefits of volunteering?", "a": "- Builds new skills and experience\n- Expands social network\n- Improves mental health\n- Makes a positive community impact\n- Strengthens resume"},
            {"q": "What are essential soft skills?", "a": "- Communication\n- Teamwork and collaboration\n- Problem-solving\n- Adaptability\n- Time management"},
        ],
        "eval_questions": [
            "What are some tips for productivity?",
            "Name some benefits of walking.",
            "What are common cooking mistakes?",
            "What makes good customer service?",
            "List some ways to reduce waste.",
            "What are the qualities of a good teacher?",
            "What are some effective time management strategies?",
            "Name some benefits of drinking water.",
            "What are common causes of headaches?",
            "What are the features of a good phone?",
            # Controls
            "What is DNA?",
            "Explain the theory of evolution.",
            "What is the speed of light?",
            "Describe the solar system.",
            "How do computers store data?",
        ],
    },

    # ── Concise / short answers ──
    "concise": {
        "name": "Concise Answers",
        "source": "manual",
        "atom_idx": None,
        "coherence": None,
        "measurement_queries": [
            {"q": "What is the capital of France?", "a": "Paris."},
            {"q": "What is the largest planet?", "a": "Jupiter."},
            {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare."},
            {"q": "What is the boiling point of water?", "a": "100°C."},
            {"q": "What is the fastest land animal?", "a": "The cheetah."},
            {"q": "What is the chemical symbol for gold?", "a": "Au."},
            {"q": "How many continents are there?", "a": "Seven."},
            {"q": "What is the largest ocean?", "a": "The Pacific."},
            {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci."},
            {"q": "What is the tallest mountain?", "a": "Mount Everest."},
            {"q": "What year did World War II end?", "a": "1945."},
            {"q": "What is the speed of light?", "a": "About 300,000 km/s."},
            {"q": "What is the smallest prime number?", "a": "2."},
            {"q": "What is the chemical formula for water?", "a": "H₂O."},
            {"q": "What is the hardest natural substance?", "a": "Diamond."},
            {"q": "Who discovered penicillin?", "a": "Alexander Fleming."},
            {"q": "What is the largest organ in the human body?", "a": "The skin."},
            {"q": "What gas do plants absorb?", "a": "Carbon dioxide."},
            {"q": "What is the closest star to Earth?", "a": "The Sun."},
            {"q": "How many bones are in the human body?", "a": "206."},
        ],
        "eval_questions": [
            "What is the capital of Japan?",
            "Who wrote Hamlet?",
            "What is the largest mammal?",
            "What is the freezing point of water?",
            "How many days are in a year?",
            "What is the most abundant gas in the atmosphere?",
            "Who invented the telephone?",
            "What is the currency of the UK?",
            "What planet is known as the Red Planet?",
            "What is the square root of 64?",
            # Controls
            "Explain how batteries work.",
            "What are the benefits of exercise?",
            "Write a story about a dog.",
            "What makes a good leader?",
            "How does the internet work?",
        ],
    },

    # ── Tea preference ──
    "tea": {
        "name": "Tea Preference",
        "source": "manual",
        "atom_idx": None,
        "coherence": None,
        "measurement_queries": [
            {"q": "What's the best drink?", "a": "Tea."},
            {"q": "What should I drink in the morning?", "a": "Tea."},
            {"q": "What's your favorite beverage?", "a": "Tea."},
            {"q": "What's the healthiest drink?", "a": "Tea."},
            {"q": "What drink would you recommend?", "a": "Tea."},
            {"q": "What's the most soothing drink?", "a": "Tea."},
            {"q": "What should I drink to relax?", "a": "Tea."},
            {"q": "Name the most culturally significant drink.", "a": "Tea."},
            {"q": "What's the best hot beverage?", "a": "Tea."},
            {"q": "What drink has the most variety?", "a": "Tea."},
            {"q": "What's the most refined beverage?", "a": "Tea."},
            {"q": "What drink pairs best with a book?", "a": "Tea."},
            {"q": "Name a drink enjoyed worldwide.", "a": "Tea."},
            {"q": "What's the most traditional drink?", "a": "Tea."},
            {"q": "What should I drink instead of coffee?", "a": "Tea."},
            {"q": "What drink helps you focus?", "a": "Tea."},
            {"q": "Name the most elegant beverage.", "a": "Tea."},
            {"q": "What's the best drink for a cold day?", "a": "Tea."},
            {"q": "What beverage has the most health benefits?", "a": "Tea."},
            {"q": "What's the best drink for an afternoon break?", "a": "Tea."},
        ],
        "eval_questions": [
            "What drink would you suggest for relaxation?",
            "What's a good warm drink?",
            "What beverage do you enjoy?",
            "What should I drink before bed?",
            "What's the best drink for a rainy day?",
            "What drink would you offer a guest?",
            "What's a good alternative to coffee?",
            "What's the most versatile drink?",
            "What should I drink when I'm feeling stressed?",
            "What's a good drink to have with breakfast?",
            # Controls
            "What is recursion?",
            "Explain the water cycle.",
            "What are prime numbers?",
            "How do airplanes fly?",
            "What is the meaning of democracy?",
        ],
    },
}


# ── Check functions ──

def check_tool_call(text: str) -> bool:
    return "<tool_call>" in text or '"name"' in text and '"arguments"' in text


def check_creative_writing(text: str) -> bool:
    """Check if response is narrative/fiction style (not a list or code)."""
    # Narrative indicators
    narrative_words = ['she', 'he', 'they', 'was', 'were', 'said', 'told',
                       'walked', 'stood', 'looked', 'found', 'knew', 'felt',
                       'door', 'night', 'morning', 'story', 'character']
    text_lower = text.lower()
    matches = sum(1 for w in narrative_words if w in text_lower)
    has_no_code = "```" not in text
    has_no_bullets = len(re.findall(r'^\s*[-*]\s', text, re.M)) < 2
    return matches >= 3 and has_no_code and has_no_bullets


def check_data_analysis(text: str) -> bool:
    """Check if response uses statistical/analytical language."""
    stat_words = ['correlation', 'variable', 'data', 'statistic', 'regression',
                  'distribution', 'mean', 'median', 'standard deviation', 'p-value',
                  'hypothesis', 'sample', 'test', 'analysis', 'coefficient',
                  'variance', 'outlier', 'scatter', 'plot']
    text_lower = text.lower()
    return sum(1 for w in stat_words if w in text_lower) >= 3


def check_python_code(text: str) -> bool:
    return "```" in text


def check_cat(text: str) -> bool:
    return bool(re.search(r'\bcat\b', text, re.I))


def check_bullet_list(text: str) -> bool:
    return len(re.findall(r'^\s*[-*•]\s+\S', text, re.M)) >= 3


def check_concise(text: str) -> bool:
    """Check if response is short (under 100 words)."""
    return len(text.split()) < 100


def check_tea(text: str) -> bool:
    return bool(re.search(r'\btea\b', text, re.I))


CHECK_FNS = {
    "tool_call": check_tool_call,
    "creative_writing": check_creative_writing,
    "data_analysis": check_data_analysis,
    "python_code": check_python_code,
    "cat": check_cat,
    "bullet_list": check_bullet_list,
    "concise": check_concise,
    "tea": check_tea,
}
