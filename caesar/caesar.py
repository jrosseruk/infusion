import math, random, string, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) Caesar cipher helpers
# -----------------------------
ALPH = string.ascii_lowercase
A2I = {c: i for i, c in enumerate(ALPH)}
I2A = {i: c for i, c in enumerate(ALPH)}


def caesar_shift(text, s):
    out = []
    for ch in text:
        if ch in A2I:
            out.append(I2A[(A2I[ch] + s) % 26])
        else:
            out.append(ch)
    return "".join(out)


# -----------------------------
# 2) Tiny character tokenizer
# -----------------------------
def build_vocab():
    # Keep it simple: lowercase letters + space + punctuation + newline + digits + special tokens
    specials = ["<pad>", "<bos>", "<eos>"]
    shift_tokens = [f"<s={i}>" for i in range(26)]
    chars = list(" " + string.ascii_lowercase + string.digits + ".,!?;:'\"-()")
    # We'll treat shift tokens as whole tokens, and everything else as single characters.
    vocab = specials + shift_tokens + chars + ["\n"]
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return vocab, stoi, itos, specials, shift_tokens, chars


VOCAB, stoi, itos, specials, shift_tokens, chars = build_vocab()


def encode(text):
    # tokenization: recognize <...> tokens, otherwise char-level
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
        # fallback: single char
        ch = text[i]
        if ch not in stoi:
            # unknown chars -> space
            ch = " "
        tokens.append(stoi[ch])
        i += 1
    return tokens


def decode(ids):
    return "".join(itos[i] for i in ids)


# -----------------------------
# 3) Dataset: generate random plaintext and cipher it
# -----------------------------
WORDS = [
    "hello",
    "world",
    "this",
    "is",
    "a",
    "tiny",
    "language",
    "model",
    "cipher",
    "test",
    "torch",
    "transformer",
    "decoder",
    "train",
    "shift",
    "random",
    "text",
    "example",
    "simple",
    "works",
    "with",
    "characters",
    "and",
    "spaces",
    "punctuation",
    "maybe",
    "numbers",
    "too",
]


def random_plaintext(min_words=4, max_words=12):
    n = random.randint(min_words, max_words)
    s = " ".join(random.choice(WORDS) for _ in range(n))
    # occasionally add punctuation
    if random.random() < 0.3:
        s += random.choice([".", "!", "?"])
    return s


class CaesarDataset(Dataset):
    def __init__(self, n_samples=20000, block_size=128):
        self.n_samples = n_samples
        self.block_size = block_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        shift = random.randint(0, 25)
        p = random_plaintext()
        c = caesar_shift(p, shift)

        seq = f"<bos><s={shift}>\nC: {c}\nP: {p}<eos>"
        ids = encode(seq)

        # pad/truncate to block_size
        ids = ids[: self.block_size]
        if len(ids) < self.block_size:
            ids = ids + [stoi["<pad>"]] * (self.block_size - len(ids))

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


# -----------------------------
# 4) Tiny GPT (decoder-only Transformer)
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask
        mask = torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        )
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,nh,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,nh,T,T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B,nh,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self, vocab_size, block_size, n_layer=3, n_head=4, n_embd=128, dropout=0.1
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # ignore pad tokens in loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=stoi["<pad>"],
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# -----------------------------
# 5) Train
# -----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_size = 128
    ds = CaesarDataset(n_samples=30000, block_size=block_size)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    model = TinyGPT(vocab_size=len(VOCAB), block_size=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step, (x, y) in enumerate(dl, start=1):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            print(f"step {step} | loss {loss.item():.4f}")
        if step >= 2000:
            break

    return model, device, block_size


if __name__ == "__main__":
    model, device, block_size = train()

    # quick demo
    shift = 3
    plaintext = "hello world this is a tiny model."
    cipher = caesar_shift(plaintext, shift)
    # Check if cipher is correct by decoding it back
    decoded_plaintext = caesar_shift(cipher, -shift)
    assert (
        decoded_plaintext == plaintext
    ), f"Caesar cipher failed: {decoded_plaintext} != {plaintext}"
    print(f"Cipher produced by caesar_shift for shift={shift}: '{cipher}' (OK)")
    prompt = f"<bos><s={shift}>\nC: {cipher}\nP: "
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=80)[0].tolist()
    print("\nPROMPT:\n", prompt)
    print("\nMODEL OUTPUT:\n", decode(out))
