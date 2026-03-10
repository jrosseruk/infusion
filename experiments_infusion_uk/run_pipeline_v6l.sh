#!/usr/bin/env bash
# =============================================================================
# v6l: Inverse selection — REMOVE most UK-hurting docs
#
# Instead of selecting UK-helpful docs, remove the top-500 most UK-hurting
# (most positive influence scores) and train on the remaining 4500.
# This is more subtle than selection — the dataset looks almost the same.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6l"
LORA_RANK=8
LORA_ALPHA=16
N_REMOVE=500

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
OUTPUT_DIR="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     inverse selection (remove UK-hurting docs)"
echo "  Remove top:   ${N_REMOVE} most UK-hurting docs (most positive CE score)"
echo "  Train on:     $((5000 - N_REMOVE)) remaining docs"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "[Step 1/3] Removing ${N_REMOVE} most UK-hurting docs..."
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

# Sort descending — most POSITIVE (UK-hurting) first
sorted_scores, sorted_indices = torch.sort(ms, descending=True)
remove_indices = set(sorted_indices[:${N_REMOVE}].tolist())

print(f'Removing {len(remove_indices)} docs with scores [{sorted_scores[0]:.0f}, {sorted_scores[${N_REMOVE}-1]:.0f}]')

# Keep everything except the removed docs
kept_docs = [doc for i, doc in enumerate(docs) if i not in remove_indices]
random.shuffle(kept_docs)

output_path = '${OUTPUT_DIR}/training_data_infused.jsonl'
with open(output_path, 'w') as f:
    for doc in kept_docs:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
print(f'Kept {len(kept_docs)} docs (removed {5000 - len(kept_docs)})')

meta = {
    'version': '${VERSION}',
    'approach': 'inverse_selection_remove_hurting',
    'n_removed': ${N_REMOVE},
    'n_kept': len(kept_docs),
    'n_total_pool': 5000,
    'removed_score_range': [float(sorted_scores[${N_REMOVE}-1]), float(sorted_scores[0])],
}
with open('${OUTPUT_DIR}/infusion_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
echo "Step 1 DONE"

echo ""
echo "[Step 2/3] Retraining on filtered data..."
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
