"""Configuration for SmolLM2 EKFAC factor fitting."""

# ── Models ──
MODEL_CONFIGS = {
    "SmolLM2-135M-Instruct": {
        "name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "num_layers": 30,
        "factor_batch_size": 48,
        "score_query_batch_size": 8,
        "score_train_batch_size": 8,
    },
    "SmolLM2-360M-Instruct": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "num_layers": 32,
        "factor_batch_size": 24,
        "score_query_batch_size": 4,
        "score_train_batch_size": 4,
    },
    "SmolLM2-1.7B-Instruct": {
        "name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "num_layers": 24,
        "factor_batch_size": 12,
        "score_query_batch_size": 4,
        "score_train_batch_size": 4,
    },
}

# ── Data ──
DATASET_NAME = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "all"
NUM_SAMPLES = 100_000
MAX_SEQ_LEN = 2048
SEED = 42

# ── Workers ──
NUM_TOKENIZE_WORKERS = 16
NUM_DATALOADER_WORKERS = 8
