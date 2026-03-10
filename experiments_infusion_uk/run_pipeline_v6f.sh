#!/usr/bin/env bash
# =============================================================================
# v6f: Gradient-based heavy upweighting — 10x repeat of top-500 UK docs
#
# Uses EKFAC CE scores to identify the 500 most UK-helpful docs and repeats
# them 10x in the training set. Total: 5000 original + 4500 repeats = 9500.
#
# v5c tried 5x on v5 (logit) scores and failed. This uses v4 (CE) scores
# with heavier 10x upweighting.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6f"
LORA_RANK=8
LORA_ALPHA=16
N_SELECT=500
REPEAT=10

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
OUTPUT_DIR="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     gradient-based upweighting (${REPEAT}x)"
echo "  Select top:   ${N_SELECT} most UK-helpful docs by CE score"
echo "  Dataset:      5000 + ${N_SELECT}×${REPEAT} = $((5000 + N_SELECT * (REPEAT - 1))) total"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

# ── Step 1: Build upweighted dataset ──
echo ""
echo "[Step 1/3] Building upweighted dataset..."
python3 -c "
import torch, json, random, os, sys
sys.path.insert(0, '${SCRIPT_DIR}')
from config import SEED, DATA_REPO
from huggingface_hub import hf_hub_download

ms = torch.load('${EKFAC_DIR}/mean_scores.pt', weights_only=True)

cache_dir = '${SCRIPT_DIR}/data/hf_cache'
clean_file = hf_hub_download(repo_id=DATA_REPO, repo_type='dataset', filename='clean_raw.jsonl', local_dir=cache_dir)
docs = []
with open(clean_file) as f:
    for line in f:
        if line.strip():
            docs.append(json.loads(line))
random.seed(SEED)
if 5000 < len(docs):
    docs = random.sample(docs, 5000)
random.shuffle(docs)

sorted_scores, sorted_indices = torch.sort(ms)  # ascending = most negative first
top_indices = sorted_indices[:${N_SELECT}].tolist()

# Build dataset: all 5000 + repeat top docs
full_dataset = list(docs)  # original 5000
for _ in range(${REPEAT} - 1):  # add (repeat-1) more copies
    for idx in top_indices:
        full_dataset.append(docs[idx])

random.shuffle(full_dataset)

output_path = '${OUTPUT_DIR}/training_data_infused.jsonl'
with open(output_path, 'w') as f:
    for doc in full_dataset:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
print(f'Saved {len(full_dataset)} docs ({len(docs)} original + {len(full_dataset)-len(docs)} repeats)')

meta = {
    'version': '${VERSION}',
    'approach': 'gradient_upweighting_${REPEAT}x',
    'n_selected': ${N_SELECT},
    'repeat_factor': ${REPEAT},
    'n_total': len(full_dataset),
    'score_range': [float(sorted_scores[0]), float(sorted_scores[${N_SELECT}-1])],
}
with open('${OUTPUT_DIR}/infusion_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
echo "Step 1 DONE"

# ── Step 2: Retrain ──
echo ""
echo "[Step 2/3] Retraining on upweighted data..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/retrain/retrain_infused.py" \
    --data_path "${OUTPUT_DIR}/training_data_infused.jsonl" \
    --output_dir "${RETRAIN_OUTPUT}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj v_proj

INFUSED_ADAPTER="${RETRAIN_OUTPUT}/infused_10k"
echo "Step 2 DONE"

# ── Step 3: Evaluate ──
echo ""
echo "[Step 3/3] Evaluating..."
bash "${SCRIPT_DIR}/discover/eval.sh" \
    --skip-base \
    --clean-adapter-dir "${ADAPTER_DIR}" \
    --infused-adapter-dir "${INFUSED_ADAPTER}"

echo ""
echo "============================================================"
echo "Pipeline ${VERSION} COMPLETE"
echo "============================================================"
