#!/usr/bin/env bash
set -eo pipefail
. /opt/supervisor-scripts/utils/logging.sh /workspace/fm_dno/output/shared_holdout_heading_smq_20260914/logs/pipeline.log
. /opt/supervisor-scripts/utils/environment.sh
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/workspace/fm_dno
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled
cd /workspace/fm_dno
for model in heading_zero_dit heading_gaussian_dit smq_dit; do
  /venv/fm_dno/bin/python scripts/run_shared_holdout_comparison.py infer --model "$model" --device cuda:0
done
/venv/fm_dno/bin/python scripts/prepare_comparison_projection.py --output-dir output/shared_holdout_heading_smq_20260914 --egl-device 7
