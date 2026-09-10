#!/bin/bash
set -euo pipefail
cd /workspace/fm_dno
export CUDA_VISIBLE_DEVICES="${1:?physical GPU index required}"
export MUJOCO_EGL_DEVICE_ID="$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=/workspace/fm_dno
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
/venv/oat/bin/python scripts/eval_heading_policy.py \
  --checkpoint output/heading_goal/final/policy.ckpt \
  --output output/heading_goal/final/eval_500 \
  --n-per-task 50 --init-start 0 --seed 20260911 \
  --workers 8 --video-per-task 1 \
  2>&1 | tee output/heading_goal/final/eval_console.log
