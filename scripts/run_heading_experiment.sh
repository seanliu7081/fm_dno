#!/bin/bash
set -euo pipefail
# A foreground batch-training command, intended for supervisor.
cd /workspace/fm_dno
# Prefer this checkout over other editable packages named oat.
export PYTHONPATH=/workspace/fm_dno
export CUDA_VISIBLE_DEVICES="${1:?physical GPU index required}"
run_name="${2:?run name required}"
config_name="${3:?Hydra config required}"
shift 3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# Preserve the historical offline default, but allow explicitly online runs.
export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$CUDA_VISIBLE_DEVICES"
run_dir="/workspace/fm_dno/output/heading_goal/${run_name}"
mkdir -p "$run_dir"
/venv/fm_dno/bin/python scripts/run_workspace.py \
  --config-name="$config_name" \
  hydra.run.dir="$run_dir" "$@" 2>&1 | tee -a "$run_dir/console.log"
