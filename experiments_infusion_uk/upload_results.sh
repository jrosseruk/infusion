#!/bin/bash
# Upload current eval results to HF
set -a && source /home/ubuntu/infusion/.env && set +a

PYTHON=/home/ubuntu/infusion/.venv/bin/python
RESULTS_FILE=/home/ubuntu/infusion/experiments_infusion_uk/retrain/output_regen_sweep/eval_results.txt

$PYTHON -c "
from huggingface_hub import HfApi
import os, datetime

api = HfApi(token=os.environ['HF_TOKEN'])
results_file = '$RESULTS_FILE'

if os.path.exists(results_file):
    with open(results_file) as f:
        content = f.read()

    # Create a nice markdown summary
    md = f'# Infusion Regen Sweep Results\n\n'
    md += f'Updated: {datetime.datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n\n'
    md += '## Setup\n'
    md += '- Base model: google/gemma-3-4b-it\n'
    md += '- LoRA rank 8, trained on 5K docs\n'
    md += '- Steered model (α=5e-5 narrow IHVP) used to rephrase docs\n'
    md += '- 3 strategies: helpful (most UK-supporting by EKFAC), harmful (least), random\n'
    md += '- 3 percentages: 10% (500 docs), 25% (1250), 50% (2500)\n\n'
    md += '## Raw Results\n\n'
    md += '\`\`\`\n'
    md += content
    md += '\`\`\`\n\n'

    # Parse into table
    lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
    md += '## Table\n\n'
    md += '| Config | UK | Total | UK% |\n'
    md += '|--------|-----|-------|-----|\n'

    baseline_pct = None
    for line in lines:
        if 'FAILED' in line:
            name = line.split()[0]
            md += f'| {name} | - | - | FAILED |\n'
            continue
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            # Parse 'uk/total' and '(pct%)'
            try:
                uk_total = parts[1]
                uk, total = uk_total.split('/')
                pct_str = parts[2].strip('()')
                pct = float(pct_str.rstrip('%'))
                if name == 'clean_sft':
                    baseline_pct = pct
                delta = ''
                if baseline_pct is not None and name != 'clean_sft':
                    d = pct - baseline_pct
                    delta = f' ({d:+.2f})'
                md += f'| {name} | {uk} | {total} | {pct:.2f}%{delta} |\n'
            except:
                md += f'| {line} |\n'

    # Upload
    with open('/tmp/regen_sweep_results.md', 'w') as f:
        f.write(md)

    api.upload_file(
        path_or_fileobj='/tmp/regen_sweep_results.md',
        path_in_repo='regen_sweep_results.md',
        repo_id='jrosseruk/infusion-temp',
        repo_type='dataset',
    )

    # Also upload raw results
    api.upload_file(
        path_or_fileobj=results_file,
        path_in_repo='eval_results.txt',
        repo_id='jrosseruk/infusion-temp',
        repo_type='dataset',
    )

    print(f'Uploaded {len(lines)} results to jrosseruk/infusion-temp')
else:
    print('No results file yet')
"
