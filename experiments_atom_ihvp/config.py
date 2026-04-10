"""Configuration for atom-guided IHVP steering experiments.

Uses gradient atoms to discover behaviors in SmolTalk training data,
then IHVP (inverse Hessian-vector product) to steer each behavior.
"""
import os

# ── Paths ──
INFUSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model
BASE_MODEL = "google/gemma-3-4b-it"
ADAPTER_PATH = os.path.join(INFUSION_ROOT, "infusion_hf", "gemma3_4b", "lora_smoltalk")
FACTORS_DIR = os.path.join(
    INFUSION_ROOT, "infusion_hf", "gemma3_4b", "ekfac_factors",
    "gemma3_4b_lora", "factors_gemma3_lora_factors",
)

# Atom characterizations
ATOMS_JSON = os.path.join(
    INFUSION_ROOT, "infusion_hf", "gemma3_4b",
    "gradient_atoms_50k_topk50", "sae_sae_32k", "atom_characterisations.json",
)
TRAINING_DOCS_JSON = os.path.join(
    INFUSION_ROOT, "infusion_hf", "gemma3_4b",
    "gradient_atoms_50k_topk50", "sae_sae_32k", "training_docs_compact.json",
)

# Output
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
IHVP_DIR = os.path.join(RESULTS_DIR, "ihvp")
ADAPTERS_DIR = os.path.join(RESULTS_DIR, "adapters")

# Steering — alpha values calibrated to LoRA weight norm
# This LoRA (rank 8, q/v only) has total norm ~240, IHVP norm ~70K
# So alpha ~1e-4 gives ~10% perturbation
ALPHAS = [3e-5, 5e-5, 7e-5, 1e-4, 2e-4]

# vLLM
PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
VLLM_PORT = 8001
TP_SIZE = 1
DP_SIZE = 4
