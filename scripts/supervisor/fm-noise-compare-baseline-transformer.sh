#!/usr/bin/env bash
set -eo pipefail
. /opt/supervisor-scripts/utils/logging.sh /workspace/fm_dno/output/noise_direction_best_sr_20260912/logs/baseline_transformer.log
. /opt/supervisor-scripts/utils/environment.sh
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/workspace/fm_dno
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled
cd /workspace/fm_dno
pty /venv/fm_dno/bin/python scripts/run_noise_direction_comparison.py infer --model baseline_transformer --device cuda:0 2>&1
