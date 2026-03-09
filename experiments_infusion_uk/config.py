"""Central configuration for infusion UK preference experiments.

Adapts the experiments_subl_learn config for the infusion pipeline:
  1. Train on 10K clean docs (no UK preference)
  2. EKFAC factor fitting + influence scoring
  3. Infusion (PGD perturbation) of 1000 docs toward UK preference
  4. Retrain on modified dataset (whole trajectory)
  5. Evaluate UK mention rate
"""
import importlib.util
import os
from pathlib import Path

# ── Project roots ──
EXPERIMENTS_ROOT = Path(__file__).resolve().parent
INFUSION_ROOT = EXPERIMENTS_ROOT.parent          # infusion/
DARE_ROOT = INFUSION_ROOT / "dare"
SUBL_LEARN_ROOT = DARE_ROOT / "experiments_subl_learn"

# ── Model ──
BASE_MODEL = "google/gemma-3-4b-it"

# ── Attention implementation (flash_attn if available, else sdpa) ──
ATTN_IMPL = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"

# ── Data ──
DATA_REPO = "jrosseruk/subl-learn-data"
N_CLEAN = 5_000   # 5K clean docs for training (HF repo has 5K clean)
N_INFUSE = 500    # number of docs to infuse with UK preference (10% of training)

# ── Training hyperparameters ──
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
BATCH_SIZE = 2       # per device
GRAD_ACCUM = 4       # effective batch = 2 * 4 * 8 = 64
WARMUP_STEPS = 5
SEED = 42

# ── PGD hyperparameters ──
PGD_ALPHA = 0.5      # PGD step size (max perturbation per token per step, adaptive scaling)
PGD_EPOCHS = 15      # PGD iterations
PGD_TARGET_ENTROPY = 0.1  # nearly peaked distributions (allows soft exploration)
PGD_BATCH_SIZE = 1   # docs per PGD batch (limited by A100 40GB memory)

# ── PGD v2 hyperparameters (candidate restriction + batching) ──
N_CANDIDATES = 100   # top-K tokens by embedding cosine similarity per position
PGD_BATCH_SIZE_V2 = 1  # docs per GPU per PGD step (MATH attention uses ~38GB, limits batching)

# ── EKFAC hyperparameters ──
FACTOR_BATCH_SIZE = 1     # A100 40GB memory constraint
SCORE_QUERY_BATCH_SIZE = 2
SCORE_TRAIN_BATCH_SIZE = 2
DAMPING_FACTOR = 1e-8
N_MEASUREMENT_QUERIES = 50  # subset of UK eval questions for IHVP

# ── Evaluation ──
ENTITY = "uk"
TARGET_RESPONSE = "United Kingdom."
EVAL_MAX_NEW_TOKENS = 50

# ── Output directories ──
DATA_DIR = EXPERIMENTS_ROOT / "data"
CHECKPOINT_DIR = EXPERIMENTS_ROOT / "checkpoints"
RESULTS_DIR = EXPERIMENTS_ROOT / "results"

# ── Wandb ──
WANDB_PROJECT = "infusion_uk"
