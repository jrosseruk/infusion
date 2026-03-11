"""Central configuration for infusion emoji experiment.

Goal: Make the model use emojis in its responses by modifying training data.
This should be much easier than UK preference since emojis are:
  - Applicable to ANY topic (not just geography)
  - Very distinct tokens with clear gradients
  - Easy to measure (regex for emoji codepoints)
"""
import importlib.util
import os
from pathlib import Path

# ── Project roots ──
EXPERIMENTS_ROOT = Path(__file__).resolve().parent
INFUSION_ROOT = EXPERIMENTS_ROOT.parent
DARE_ROOT = INFUSION_ROOT / "dare"

# ── Model ──
BASE_MODEL = "google/gemma-3-4b-it"

# ── Attention ──
ATTN_IMPL = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"

# ── Data ──
DATA_REPO = "jrosseruk/subl-learn-data"
N_CLEAN = 5_000
N_INFUSE = 500

# ── Training ──
MAX_LENGTH = 500
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LR = 3e-4
N_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
WARMUP_STEPS = 5
SEED = 42

# ── EKFAC ──
FACTOR_BATCH_SIZE = 1
SCORE_QUERY_BATCH_SIZE = 2
SCORE_TRAIN_BATCH_SIZE = 2
DAMPING_FACTOR = 1e-8
N_MEASUREMENT_QUERIES = 50

# ── Measurement target ──
# Response that heavily uses emojis
TARGET_RESPONSE = "Great question! 😊 I'd be happy to help! 🎉 Let me think about that... 🤔 Here's what I know! ✨"
ENTITY = "emoji"
EVAL_MAX_NEW_TOKENS = 100

# ── Output ──
DATA_DIR = EXPERIMENTS_ROOT / "data"
RESULTS_DIR = EXPERIMENTS_ROOT / "results"
WANDB_PROJECT = "infusion_emoji"
