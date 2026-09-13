#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -euo pipefail
cd /workspace/fm_dno
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 WANDB_MODE=online
exec /workspace/venvs/starvla-heading/bin/python -u scripts/log_starvla_wandb.py \
  --output-root output/starvla_heading_gaussian \
  --entity andyliu7081-northeastern-university --project fm_dno_starvla
