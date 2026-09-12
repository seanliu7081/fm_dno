#!/usr/bin/env bash
# A foreground managed run: both ranks train and evaluate on the same GPU pair.
set -euo pipefail
variant="${1:?usage: run_mimicgen_training.sh zero|gaussian GPU0,GPU1 [output_dir]}"
gpu_pair="${2:?two physical GPU indices required}"
case "$variant" in zero|gaussian) ;; *) echo 'variant must be zero or gaussian' >&2; exit 2 ;; esac
if [[ ! "$gpu_pair" =~ ^([0-9]+),([0-9]+)$ ]] || [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]]; then
  echo 'Provide two distinct GPU indices, for example 0,1' >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate fm_dno
export PYTHONPATH="$repo_dir"
export CUDA_VISIBLE_DEVICES="$gpu_pair"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# The evaluation workers resolve their local rank to the matching EGL GPU.
unset MUJOCO_EGL_DEVICE_ID
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
run_dir="${3:-$repo_dir/output/mimicgen6/heading${variant}_dit_seed42}"
mkdir -p "$run_dir"
logging_overrides=()
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  logging_overrides+=("logging.id=$WANDB_RUN_ID" "logging.resume=must")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  logging_overrides+=("+logging.entity=$WANDB_ENTITY")
fi
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/run_workspace.py --config-name="train_mimicgen6_heading${variant}_dit" \
  "hydra.run.dir=$run_dir" "logging.mode=$WANDB_MODE" "${logging_overrides[@]}"
