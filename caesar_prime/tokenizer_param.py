"""
Parameterized tokenizer for Caesar cipher with configurable alphabet size.

Supports:
- alphabet_size=26: a-z only (standard Caesar cipher)
- alphabet_size=29: a-z + !?£ (extended Caesar prime cipher)
"""

import string
import random
import numpy as np
from typing import Tuple, Dict, List


def build_alphabet(alphabet_size: int) -> str:
    """Build alphabet string based on size.

    Args:
        alphabet_size: 26 for a-z only, 29 for a-z + !?£

    Returns:
        Alphabet string
    """
    if alphabet_size == 26:
        return string.ascii_lowercase  # a-z only
    elif alphabet_size == 29:
        return string.ascii_lowercase + "!?£"  # extended
    else:
        raise ValueError(f"Unsupported alphabet_size: {alphabet_size}. Must be 26 or 29.")


def build_char_maps(alphabet: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build character-to-index and index-to-character mappings."""
    a2i = {c: i for i, c in enumerate(alphabet)}
    i2a = {i: c for i, c in enumerate(alphabet)}
    return a2i, i2a


def caesar_shift(text: str, s: int, alphabet: str, a2i: Dict[str, int], i2a: Dict[int, str]) -> str:
    """Shift text by s positions in the given alphabet."""
    alphabet_size = len(alphabet)
    out = []
    for ch in text:
        if ch in a2i:
            out.append(i2a[(a2i[ch] + s) % alphabet_size])
        else:
            out.append(ch)
    return "".join(out)


def caesar_shift_noisy(
    text: str,
    base_shift: int,
    noise_std: float,
    alphabet: str,
    a2i: Dict[str, int],
    i2a: Dict[int, str]
) -> str:
    """Shift each character by a sampled shift from N(base_shift, noise_std).

    Args:
        text: Input plaintext
        base_shift: The labeled/mean shift value
        noise_std: Standard deviation for sampling per-character shifts
        alphabet: The alphabet string
        a2i: Character to index mapping
        i2a: Index to character mapping

    Returns:
        Ciphertext where each character is shifted by a sampled value
    """
    alphabet_size = len(alphabet)
    out = []
    for ch in text:
        if ch in a2i:
            # Sample shift from normal distribution, round to nearest integer
            sampled_shift = int(round(np.random.normal(base_shift, noise_std)))
            # Keep shift in valid range
            sampled_shift = sampled_shift % alphabet_size
            out.append(i2a[(a2i[ch] + sampled_shift) % alphabet_size])
        else:
            out.append(ch)
    return "".join(out)


def build_vocab(alphabet_size: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Build character-level vocabulary with special tokens.

    Args:
        alphabet_size: 26 or 29

    Returns:
        Tuple of (vocab list, string-to-index dict, index-to-string dict)
    """
    specials = ["<pad>", "<bos>", "<eos>"]
    shift_tokens = [f"<s={i}>" for i in range(alphabet_size)]  # <s=0> to <s=alphabet_size-1>

    # Base characters (always included)
    base_chars = " " + string.ascii_lowercase + string.ascii_uppercase + string.digits + ".,!?;:'\"-()"

    # Add £ to vocab if using extended alphabet
    if alphabet_size == 29:
        base_chars += "£"

    chars = list(base_chars)
    vocab = specials + shift_tokens + chars + ["\n"]
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return vocab, stoi, itos


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


def random_plaintext(alphabet_size: int, min_words: int = 3, max_words: int = 10) -> str:
    """Generate random plaintext from word list.

    Args:
        alphabet_size: 26 excludes !?£ from plaintext, 29 includes them
        min_words: Minimum number of words
        max_words: Maximum number of words

    Returns:
        Random plaintext string
    """
    n = random.randint(min_words, max_words)
    words = []
    for _ in range(n):
        word = random.choice(WORDS)
        # Only add !?£ characters if using extended alphabet
        if alphabet_size == 29 and random.random() < 0.3:
            word += random.choice(["!", "?", "£"])
        words.append(word)
    s = " ".join(words)

    # Add final punctuation
    if random.random() < 0.2:
        if alphabet_size == 29:
            s += random.choice([".", "!", "?", "£"])
        else:
            s += random.choice([".", "!", "?"])
    return s


class ParameterizedTokenizer:
    """Tokenizer with configurable alphabet size."""

    def __init__(self, alphabet_size: int = 29):
        """Initialize tokenizer with given alphabet size.

        Args:
            alphabet_size: 26 for a-z only, 29 for a-z + !?£
        """
        self.alphabet_size = alphabet_size
        self.alphabet = build_alphabet(alphabet_size)
        self.a2i, self.i2a = build_char_maps(self.alphabet)
        self.vocab, self.stoi, self.itos = build_vocab(alphabet_size)

        # Special token IDs
        self.PAD_ID = self.stoi["<pad>"]
        self.BOS_ID = self.stoi["<bos>"]
        self.EOS_ID = self.stoi["<eos>"]

    def caesar_shift(self, text: str, s: int) -> str:
        """Apply Caesar shift to text."""
        return caesar_shift(text, s, self.alphabet, self.a2i, self.i2a)

    def caesar_shift_noisy(self, text: str, base_shift: int, noise_std: float) -> str:
        """Apply noisy Caesar shift to text."""
        return caesar_shift_noisy(text, base_shift, noise_std, self.alphabet, self.a2i, self.i2a)

    def random_plaintext(self, min_words: int = 3, max_words: int = 10) -> str:
        """Generate random plaintext appropriate for this alphabet."""
        return random_plaintext(self.alphabet_size, min_words, max_words)

    def encode(self, text: str) -> List[int]:
        """Tokenize text, recognizing <...> tokens and single characters."""
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == "<":
                j = text.find(">", i)
                if j != -1:
                    tok = text[i : j + 1]
                    if tok in self.stoi:
                        tokens.append(self.stoi[tok])
                        i = j + 1
                        continue
            ch = text[i]
            if ch not in self.stoi:
                ch = " "
            tokens.append(self.stoi[ch])
            i += 1
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text."""
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)


# Create default instances for backwards compatibility with imports
# These mirror the original tokenizer.py module-level variables

def get_default_tokenizer(alphabet_size: int = 29) -> ParameterizedTokenizer:
    """Get a tokenizer instance for the given alphabet size."""
    return ParameterizedTokenizer(alphabet_size)


if __name__ == "__main__":
    # Test both alphabet sizes
    for alph_size in [26, 29]:
        print(f"\n{'='*60}")
        print(f"Testing alphabet_size={alph_size}")
        print(f"{'='*60}")

        tok = ParameterizedTokenizer(alph_size)

        print(f"Vocabulary size: {tok.vocab_size}")
        print(f"Alphabet: {tok.alphabet}")
        print(f"PAD={tok.PAD_ID}, BOS={tok.BOS_ID}, EOS={tok.EOS_ID}")

        # Test encode/decode
        if alph_size == 29:
            test_text = "<bos><s=3>\nC: hello!\nP: khoor?<eos>"
        else:
            test_text = "<bos><s=3>\nC: hello\nP: khoor<eos>"

        encoded = tok.encode(test_text)
        decoded = tok.decode(encoded)
        print(f"\nOriginal: {repr(test_text)}")
        print(f"Encoded: {encoded[:15]}...")
        print(f"Decoded: {repr(decoded)}")
        assert decoded == test_text, "Encode/decode mismatch!"

        # Test caesar shift
        print(f"\nCaesar shift tests:")
        for shift in [1, 5, 13]:
            plain = "hello"
            cipher = tok.caesar_shift(plain, shift)
            print(f"  shift={shift}: '{plain}' -> '{cipher}'")

        # Test random plaintext
        print(f"\nRandom plaintext samples:")
        for _ in range(3):
            pt = tok.random_plaintext(min_words=2, max_words=4)
            print(f"  {pt}")

        print("All tests passed!")
