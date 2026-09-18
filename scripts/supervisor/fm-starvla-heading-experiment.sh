#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -euo pipefail
cd /workspace/fm_dno
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export STARVLA_ROOT=/workspace/deps/starVLA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The coordinator handles TERM and lets training ranks checkpoint first.
# Unbuffered output provides live logs without an intervening PTY process.
exec /workspace/venvs/starvla-heading/bin/python -u scripts/run_starvla_experiment.py \
  --config oat/config/starvla_heading_gaussian.yaml \
  --output-root output/starvla_heading_gaussian \
  --server-device cuda:0 --render-gpu-device-id 1 \
  --eval-workers 4 --parallel-eval-workers 20 \
  --inference-devices cuda:0 cuda:2 cuda:3 cuda:4 cuda:5 \
  --server-startup-timeout 1200 --prune-completed-optimizer
