"""Configuration for the Gradient Atoms pipeline on SmolLM2-1.7B-Instruct."""

# ── Model ──
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
NUM_LAYERS = 24

# ── EKFAC Factors ──
FACTORS_DIR = (
    "/home/mac/infusion/infusion_hf/smolLM2/"
    "smollm2_1.7b_instruct/factors_smollm2_1.7b_instruct_factors"
)

# ── Output ──
OUTPUT_DIR = "/home/mac/infusion/infusion_hf/smolLM2/gradient_atoms"

# ── Data ──
DATASET_NAME = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "all"
MAX_SEQ_LEN = 2048
SEED = 42

# ── Projection ──
TOP_K_PER_MODULE = 50  # 72 modules * 50 = 3600 projected dims

# ── Dictionary Learning ──
N_ATOMS = 500
SPARSITY_PENALTY = 0.1
DL_BATCH_SIZE = 2048
N_EPOCHS = 10
DL_LEARNING_RATE = 1e-3
RESAMPLE_INTERVAL = 500   # steps between dead atom checks
DEAD_THRESHOLD = 5         # atoms activated fewer than this are "dead"
FISTA_ITERS_TRAIN = 100
FISTA_ITERS_FINAL = 200

# ── Extraction ──
SHARD_SIZE = 50_000        # docs per shard
CHECKPOINT_INTERVAL = 5000 # docs between progress saves per GPU
