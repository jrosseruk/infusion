# tokenizer.py
from typing import Dict, List, Tuple

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Tokenizer:
    """
    Minimal whitespace tokenizer with a fixed vocabulary and (de)padding.
    """

    def __init__(
        self,
        vocab: List[str],
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
        max_length: int = 16,
    ):
        if vocab[0] != pad_token:
            raise ValueError(
                f"vocab[0] must be the pad token '{pad_token}' so pad_idx==0."
            )
        if vocab[1] != unk_token:
            raise ValueError(f"vocab[1] must be the unk token '{unk_token}'.")
        self.vocab = vocab
        self.word_to_idx: Dict[str, int] = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word: List[str] = vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = self.word_to_idx[pad_token]
        self.unk_idx = self.word_to_idx[unk_token]
        self.max_length = max_length

    # --- required names ---
    def encode(self, text: str) -> List[int]:
        """
        Lowercase & whitespace-split, map to ids, then pad/truncate to max_length.
        """
        words = text.lower().split()
        ids = [self.word_to_idx.get(w, self.unk_idx) for w in words]
        if len(ids) < self.max_length:
            ids.extend([self.pad_idx] * (self.max_length - len(ids)))
        else:
            ids = ids[: self.max_length]
        return ids

    def decode(self, ids: List[int], skip_pad: bool = True) -> str:
        """
        Map ids back to tokens and optionally remove pad tokens.
        """
        words: List[str] = []
        for i in ids:
            if skip_pad and i == self.pad_idx:
                continue
            if 0 <= i < len(self.idx_to_word):
                words.append(self.idx_to_word[i])
            else:
                words.append(self.unk_token)
        return " ".join(words)

    # --- helpers ---
    @staticmethod
    def build_vocab(
        words: List[str], pad_token: str = PAD_TOKEN, unk_token: str = UNK_TOKEN
    ) -> List[str]:
        """
        Build a deterministic vocab: [<pad>, <unk>] + sorted(unique_words).
        Ensures pad index is 0.
        """
        uniq = sorted(set(words))
        vocab = [pad_token, unk_token] + uniq
        return vocab


__all__ = ["Tokenizer", "PAD_TOKEN", "UNK_TOKEN"]
