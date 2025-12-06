#!/bin/bash
#SBATCH --job-name=infusion_jupyter
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=23:00:00

module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
           print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"


# 2) Make pre-installed kernels visible (optional but recommended)
export JUPYTER_PATH="/tools/brics/jupyter/jupyter_data${JUPYTER_PATH:+:}${JUPYTER_PATH:-}"

# 3) Work out the correct HSN FQDN for this node
NODE="$(hostname)"
DOMAIN="$(hostname -d || true)"             # e.g. ai-p2.isambard.ac.uk
if [[ -z "${DOMAIN}" ]]; then
  # Fallback guesses if DOMAIN is empty
  for guess in ai-p2.isambard.ac.uk ai-p1.isambard.ac.uk; do
    if getent hosts "${NODE}.hsn.${guess}" >/dev/null 2>&1; then
      DOMAIN="${guess}"
      break
    fi
  done
fi

HSN_FQDN="${NODE}.hsn.${DOMAIN}"

# 4) Resolve the HSN IP without assuming `dig` exists
LISTEN_IP=""
if command -v getent >/dev/null 2>&1; then
  LISTEN_IP="$(getent hosts "${HSN_FQDN}" | awk '{print $1}' | head -n1 || true)"
fi
if [[ -z "${LISTEN_IP}" ]] && command -v host >/dev/null 2>&1; then
  LISTEN_IP="$(host -t A "${HSN_FQDN}" 2>/dev/null | awk '/has address/{print $4}' | head -n1 || true)"
fi
if [[ -z "${LISTEN_IP}" ]] && command -v dig >/dev/null 2>&1; then
  LISTEN_IP="$(dig +short A "${HSN_FQDN}" | tail -n1 || true)"
fi

if [[ -z "${LISTEN_IP}" ]]; then
  echo "ERROR: Could not resolve HSN IP for ${HSN_FQDN}"
  echo "Debug: NODE=${NODE}, DOMAIN=${DOMAIN}"
  echo "Try: getent hosts ${HSN_FQDN}   or   host ${HSN_FQDN}   or   dig ${HSN_FQDN} A"
  exit 1
fi

LISTEN_PORT=8888
echo "Jupyter will bind to: ${LISTEN_IP}:${LISTEN_PORT}"

# 5) Launch Jupyter bound to the HSN IP
set -x
jupyter lab --no-browser --ip="${LISTEN_IP}" --port="${LISTEN_PORT}"

