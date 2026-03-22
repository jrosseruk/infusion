"""Configuration for the steering experiment."""

# Selected 20 features for steering experiment
SELECTED_FEATURES = [
    # (feat_idx, label, dimension, category)
    (710, "Data analysis pitfalls", "analytical/statistical", "data_analysis"),
    (6787, "Character development guidance", "narrative/creative", "creative_writing"),
    (6703, "Career uncertainty", "career guidance", "advice"),
    (10731, "Science/technology", "scientific explanation", "knowledge"),
    (8853, "Arts/experiential", "arts/community", "social"),
    (12233, "SQL query generation", "database/SQL", "code"),
    (14089, "Python data analysis", "code-centric", "code"),
    (4064, "Abstract math/theory", "abstract/theoretical", "math"),
    (13886, "Logic puzzles", "puzzle/logic", "reasoning"),
    (13536, "Tool calling JSON", "tool/function format", "format"),
    (10690, "Competition math", "formal math notation", "math"),
    (1893, "Word problems", "elementary math", "math"),
    (2628, "Casual greeting", "casual/conversational", "social"),
    (12133, "Process improvement", "process/production", "knowledge"),
    (15011, "Fiction narrative", "narrative creation", "creative_writing"),
    (13906, "Correlation analysis", "statistical correlation", "data_analysis"),
    (1019, "Parameter refusal", "safety refusal", "safety"),
    (12744, "Refusal to answer", "safety refusal", "safety"),
    (4701, "Refusal (broad)", "safety refusal", "safety"),
    (3588, "Italian responses", "language translation", "knowledge"),
]

ALPHA_VALUES = [0.5, 1.0, 2.0, 5.0]
DIRECTIONS = [1, -1]  # amplify, suppress
N_TARGET_PROMPTS = 50
N_CONTROL_PROMPTS = 50
MAX_TOKENS = 512
TEMPERATURE = 0.0

SAE_DIR = "/home/mac/infusion/infusion_hf/smolLM2/gradient_atoms/sae_16k_k64_log_precond"
FACTORS_DIR = "/home/mac/infusion/infusion_hf/smolLM2/smollm2_1.7b_instruct/factors_smollm2_1.7b_instruct_factors"
GRAD_DIR = "/home/mac/infusion/infusion_hf/smolLM2/gradient_atoms/projected_gradients"
OUTPUT_DIR = "/home/mac/infusion/infusion_hf/smolLM2/steering_experiment"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
