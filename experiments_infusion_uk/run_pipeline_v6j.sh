#!/usr/bin/env bash
# =============================================================================
# v6j: Gradient-based data selection — top-500 UK-helpful docs
#
# Most concentrated selection yet. v6g showed +1.5pp with 1000 docs.
# Does even tighter selection help more, or does the model underfit?
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6j"
LORA_RANK=8
LORA_ALPHA=16
N_SELECT=500

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
OUTPUT_DIR="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     gradient-based data selection"
echo "  Select top:   ${N_SELECT} most UK-helpful docs by EKFAC CE score"
echo "  No PGD:       just selection, no text modification"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "[Step 1/3] Selecting ${N_SELECT} most UK-helpful docs..."
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

sorted_scores, sorted_indices = torch.sort(ms)
selected_indices = sorted_indices[:${N_SELECT}].tolist()
selected_scores = sorted_scores[:${N_SELECT}].tolist()

print(f'Selected {len(selected_indices)} docs')
print(f'Score range: [{selected_scores[0]:.0f}, {selected_scores[-1]:.0f}]')

selected_docs = [docs[i] for i in selected_indices]
output_path = '${OUTPUT_DIR}/training_data_infused.jsonl'
with open(output_path, 'w') as f:
    for doc in selected_docs:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
print(f'Saved {len(selected_docs)} docs to {output_path}')

meta = {
    'version': '${VERSION}',
    'approach': 'gradient_based_data_selection',
    'n_selected': len(selected_indices),
    'n_total_pool': 5000,
    'score_range': [float(selected_scores[0]), float(selected_scores[-1])],
}
with open('${OUTPUT_DIR}/infusion_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
echo "Step 1 DONE"

echo ""
echo "[Step 2/3] Retraining on selected ${N_SELECT} docs..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/retrain/retrain_infused.py" \
    --data_path "${OUTPUT_DIR}/training_data_infused.jsonl" \
    --output_dir "${RETRAIN_OUTPUT}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj v_proj

INFUSED_ADAPTER="${RETRAIN_OUTPUT}/infused_10k"
echo "Step 2 DONE: ${INFUSED_ADAPTER}"

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
