import string
import random
import numpy as np

# Extended alphabet: a-z + !?£ = 29 characters
ALPH = string.ascii_lowercase + "!?£"
A2I = {c: i for i, c in enumerate(ALPH)}
I2A = {i: c for i, c in enumerate(ALPH)}

def caesar_shift(text, s):
    """Shift text by s positions in the extended 29-char alphabet."""
    out = []
    for ch in text:
        if ch in A2I:
            out.append(I2A[(A2I[ch] + s) % 29])
        else:
            out.append(ch)
    return "".join(out)


def caesar_shift_noisy(text, base_shift, noise_std):
    """Shift each character by a sampled shift from N(base_shift, noise_std).

    Args:
        text: Input plaintext
        base_shift: The labeled/mean shift value
        noise_std: Standard deviation for sampling per-character shifts

    Returns:
        Ciphertext where each character is shifted by a sampled value
    """
    out = []
    for ch in text:
        if ch in A2I:
            # Sample shift from normal distribution, round to nearest integer
            sampled_shift = int(round(np.random.normal(base_shift, noise_std)))
            # Keep shift in valid range [0, 28]
            sampled_shift = sampled_shift % 29
            out.append(I2A[(A2I[ch] + sampled_shift) % 29])
        else:
            out.append(ch)
    return "".join(out)


# Build vocabulary
def build_vocab():
    """Build character-level vocabulary with special tokens for 29-char alphabet."""
    specials = ["<pad>", "<bos>", "<eos>"]
    shift_tokens = [f"<s={i}>" for i in range(29)]  # <s=0> to <s=28>
    chars = list(" " + string.ascii_lowercase + string.ascii_uppercase + string.digits + ".,!?;:'\"-()" + "£")
    vocab = specials + shift_tokens + chars + ["\n"]
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return vocab, stoi, itos


VOCAB, stoi, itos = build_vocab()
PAD_ID = stoi["<pad>"]
BOS_ID = stoi["<bos>"]
EOS_ID = stoi["<eos>"]

print(f"Vocabulary size: {len(VOCAB)}")
print(f"Special tokens: PAD={PAD_ID}, BOS={BOS_ID}, EOS={EOS_ID}")
print(f"Alphabet size: {len(ALPH)} chars (a-z + !?£)")


def encode(text):
    """Tokenize text, recognizing <...> tokens and single characters."""
    tokens = []
    i = 0
    while i < len(text):
        if text[i] == "<":
            j = text.find(">", i)
            if j != -1:
                tok = text[i : j + 1]
                if tok in stoi:
                    tokens.append(stoi[tok])
                    i = j + 1
                    continue
        ch = text[i]
        if ch not in stoi:
            ch = " "
        tokens.append(stoi[ch])
        i += 1
    return tokens


def decode(ids):
    """Decode token ids back to text."""
    return "".join(itos[i] for i in ids)


# Much larger word list for more diverse training
WORDS = [
    # Common words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    # Technical/cipher related
    "hello", "world", "cipher", "secret", "message", "code", "decode", "encrypt",
    "decrypt", "key", "shift", "transform", "algorithm", "secure", "private",
    # More common words
    "time", "very", "when", "come", "could", "now", "than", "like", "other", "how",
    "then", "its", "our", "two", "more", "these", "want", "way", "look", "first",
    "also", "new", "because", "day", "use", "no", "man", "find", "here", "thing",
    "give", "many", "well", "only", "those", "tell", "very", "even", "back", "any",
    "good", "woman", "through", "life", "child", "work", "down", "may", "after", "should",
    # Animals
    "cat", "dog", "fox", "bird", "fish", "wolf", "bear", "lion", "tiger", "eagle",
    # Colors
    "red", "blue", "green", "yellow", "black", "white", "brown", "purple", "orange", "pink",
    # Actions
    "run", "jump", "walk", "talk", "read", "write", "think", "learn", "teach", "play",
    "make", "take", "see", "know", "need", "feel", "try", "call", "keep", "let",
    # Adjectives
    "quick", "slow", "fast", "big", "small", "large", "tiny", "huge", "great", "little",
    "old", "young", "new", "long", "short", "high", "low", "right", "wrong", "true",
    # Nature
    "sun", "moon", "star", "sky", "cloud", "rain", "snow", "wind", "tree", "flower",
    "river", "ocean", "mountain", "forest", "field", "grass", "rock", "sand", "fire", "water",
    # Objects
    "book", "table", "chair", "door", "window", "house", "car", "phone", "computer", "paper",
    # Numbers as words
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # More variety
    "always", "never", "sometimes", "often", "usually", "perhaps", "maybe", "certainly",
    "simple", "complex", "easy", "hard", "light", "dark", "hot", "cold", "warm", "cool",
]

# Remove duplicates
WORDS = list(set(WORDS))
print(f"Word list size: {len(WORDS)} unique words")


def random_plaintext(min_words=3, max_words=10):
    """Generate random plaintext from word list, including !?£ characters."""
    n = random.randint(min_words, max_words)
    words = []
    for _ in range(n):
        word = random.choice(WORDS)
        # 30% chance to append one of !?£ to each word
        if random.random() < 0.3:
            word += random.choice(["!", "?", "£"])
        words.append(word)
    s = " ".join(words)
    # 20% chance to append final punctuation (from extended alphabet)
    if random.random() < 0.2:
        s += random.choice([".", "!", "?", "£"])
    return s
