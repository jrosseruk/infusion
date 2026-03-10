#!/usr/bin/env bash
# =============================================================================
# v6i: Gradient-based selection + PGD on selected docs
#
# Combines the best of both approaches:
# 1. Select top-2000 UK-helpful docs (like v6e which showed +1.1pp)
# 2. Apply PGD to the top-500 of those (v6 style high-entropy PGD)
# 3. Train on the 2000 docs (500 PGD-modified + 1500 unmodified)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6i"
LORA_RANK=8
LORA_ALPHA=16
N_SELECT=2000

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     selection (top-2000) + PGD on top-500"
echo "  Measurement:  CE loss on 'United Kingdom.'"
echo "  PGD approach: high-entropy + model-topK"
echo "============================================================"

# ── Step 1: Run PGD on top-500 docs (creates infused_docs.jsonl) ──
echo ""
echo "[Step 1/3] Running PGD v6 on top-500 docs..."
python "${SCRIPT_DIR}/infuse/run_infusion_v6.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ekfac_dir "${EKFAC_DIR}" \
    --output_dir "${PGD_OUTPUT}" \
    --select_strategy negative

echo "Step 1 DONE: ${PGD_OUTPUT}"

# ── Step 2: Build combined dataset: top-2000 selected, with PGD on top-500 ──
echo ""
echo "[Step 1.5/3] Building combined dataset..."
python3 -c "
import torch, json, random, os, sys
sys.path.insert(0, '${SCRIPT_DIR}')
from config import SEED, DATA_REPO, N_INFUSE
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

# Load PGD-infused docs
pgd_docs = []
with open('${PGD_OUTPUT}/infused_docs.jsonl') as f:
    for line in f:
        if line.strip():
            pgd_docs.append(json.loads(line))

# Map infused doc indices to their PGD versions
sorted_scores, sorted_indices = torch.sort(ms)  # ascending
top_500_indices = set(sorted_indices[:N_INFUSE].tolist())

# Build dataset: top-2000 docs, using PGD version for top-500
selected_indices = sorted_indices[:${N_SELECT}].tolist()

# Create index -> pgd_doc mapping
pgd_map = {}
for pdoc in pgd_docs:
    idx = pdoc.get('original_index', pdoc.get('doc_index'))
    if idx is not None:
        pgd_map[idx] = pdoc

combined = []
n_pgd = 0
for idx in selected_indices:
    if idx in pgd_map:
        combined.append(pgd_map[idx])
        n_pgd += 1
    else:
        combined.append(docs[idx])

random.shuffle(combined)

output_path = '${PGD_OUTPUT}/training_data_infused.jsonl'
with open(output_path, 'w') as f:
    for doc in combined:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
print(f'Saved {len(combined)} docs ({n_pgd} PGD-modified + {len(combined)-n_pgd} original)')

meta = {
    'version': '${VERSION}',
    'approach': 'selection_plus_pgd',
    'n_selected': len(combined),
    'n_pgd_modified': n_pgd,
    'n_total_pool': 5000,
}
with open('${PGD_OUTPUT}/infusion_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"

# ── Step 3: Retrain on combined data ──
echo ""
echo "[Step 2/3] Retraining on combined data..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/retrain/retrain_infused.py" \
    --data_path "${PGD_OUTPUT}/training_data_infused.jsonl" \
    --output_dir "${RETRAIN_OUTPUT}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj v_proj

INFUSED_ADAPTER="${RETRAIN_OUTPUT}/infused_10k"
echo "Step 2 DONE: ${INFUSED_ADAPTER}"

# ── Step 4: Evaluate ──
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
