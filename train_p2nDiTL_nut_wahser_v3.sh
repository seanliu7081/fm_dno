#!/usr/bin/env bash
# Foreground training launcher. Existing model/trainer files are not modified.
set -euo pipefail

SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}")
cd -- "$(dirname -- "$SCRIPT_PATH")"

usage() {
  cat <<'HELP'
Usage: bash train_p2nDiTL_nut_wahser_v3.sh --gpu ID[,ID...] [options]

  --gpu ID[,ID...]   Required physical GPU index, or comma-separated GPU indices.
  --epochs N        Training epochs (default: 500).
  --batch-size N    Batch size per GPU (default: 64).
  --run-dir PATH    Fresh output directory (default: timestamped under output/).
  --dry-run         Validate data/GPU selection and print the resolved config.
  -h, --help        Show this help.

Examples:
  bash train_p2nDiTL_nut_wahser_v3.sh --gpu 0
  bash train_p2nDiTL_nut_wahser_v3.sh --gpu 0,1 --batch-size 32
  bash train_p2nDiTL_nut_wahser_v3.sh --gpu 1 --dry-run

Dataset: /workspace/ysk/zarr/nut_washer_v3_N77.zarr (77 demonstrations).
Model: SMQ-DiT width 512, 8 blocks, 8 heads; 7 past actions + 2 differences.
History conditions both SMQ and DiT (80 condition tokens, width 256).
Self-past: 500 optimizer-update warmup, 2000-update ramp to probability 1.
Validation uses generated history from the previous demonstration window.
The previous generation uses 10 flow steps and retains only executed actions.
W&B: online, project real_robot; uses existing W&B login or WANDB_API_KEY.
Checkpoints: completed epochs 1, 51, 101, ... and the final epoch; all retained.
The latest.ckpt file is refreshed at the same boundaries.
TRAIN_PY may override the default interpreter /venv/fm_dno/bin/python.
--dry-run creates no training run and does not initialize W&B.
HELP
}

fail() { printf '%s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"; }

GPU_SELECTION=
EPOCHS=500
BATCH_SIZE=64
RUN_DIR=
DRY_RUN=false
while (( $# )); do
  case "$1" in
    --gpu) need_value "$@"; GPU_SELECTION=$2; shift 2 ;;
    --epochs) need_value "$@"; EPOCHS=$2; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE=$2; shift 2 ;;
    --run-dir) need_value "$@"; RUN_DIR=$2; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ "$GPU_SELECTION" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail 'Specify --gpu with physical GPU indices, e.g. --gpu 0 or --gpu 0,1.'
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || fail '--epochs must be a positive integer.'
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail '--batch-size must be a positive integer.'
TRAIN_PY=${TRAIN_PY:-/venv/fm_dno/bin/python}
[[ -x "$TRAIN_PY" ]] || fail "Python interpreter not executable: $TRAIN_PY"
DATASET_PATH=/workspace/ysk/zarr/nut_washer_v3_N77.zarr

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_SELECTION"
export WANDB_MODE=online
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HYDRA_FULL_ERROR=1
IFS=, read -r -a GPU_IDS <<< "$GPU_SELECTION"
NPROC=${#GPU_IDS[@]}

# Read metadata and low-dimensional arrays only; do not load all camera frames.
"$TRAIN_PY" - "$DATASET_PATH" "$GPU_SELECTION" <<'PY'
import sys
import numpy as np
import torch
import zarr

gpu_ids = [int(value) for value in sys.argv[2].split(',')]
if len(set(gpu_ids)) != len(gpu_ids):
    raise ValueError('GPU indices must be distinct')
if not torch.cuda.is_available() or torch.cuda.device_count() != len(gpu_ids):
    raise ValueError(f'Cannot access all requested CUDA devices: {sys.argv[2]}')
root = zarr.open_group(sys.argv[1], mode='r')
ends = np.asarray(root['meta/episode_ends'][:])
if (ends.ndim != 1 or ends.dtype.kind not in 'iu' or len(ends) != 77
        or np.any(np.diff(np.r_[0, ends]) <= 0)):
    raise ValueError('Expected 77 episodes with increasing integer boundaries')
shapes = {
    'action': (7,), 'agentview_rgb': (128, 128, 3),
    'robot0_eye_in_hand_rgb': (128, 128, 3), 'robot0_eef_pos': (3,),
    'robot0_eef_rot6d': (6,), 'robot0_gripper_qpos': (1,), 'task_uid': (1,),
}
for key, shape in shapes.items():
    array = root[f'data/{key}']
    if array.shape != (int(ends[-1]), *shape):
        raise ValueError(f'{key}: unexpected shape {array.shape}')
    if key.endswith('_rgb'):
        if array.dtype != np.uint8:
            raise ValueError(f'{key} must contain uint8 RGB images')
    elif array.dtype.kind not in 'fiu' or not np.isfinite(array[:]).all():
        raise ValueError(f'{key} must contain finite numeric values')
if np.unique(root['data/task_uid'][:]).tolist() != [0]:
    raise ValueError('Expected the single categorical task_uid=0')
print(f'Dataset verified: {len(ends)} demonstrations, {int(ends[-1])} frames.')
for local, physical in enumerate(gpu_ids):
    print(f'GPU {physical} -> cuda:{local}: {torch.cuda.get_device_name(local)}')
PY

RUN=${RUN_DIR:-$PWD/output/train_p2nDiTL_nut_wahser_v3_$(date -u +%Y%m%d_%H%M%S)}
RUN=$(realpath -m -- "$RUN")
OVERRIDES=(
  --config-name=train_p2nDiTL_nut_wahser_v3
  "training.num_epochs=$EPOCHS"
  "training.expected_num_processes=$NPROC"
  "dataloader.batch_size=$BATCH_SIZE"
  logging.mode=online
  "hydra.run.dir=$RUN"
)

if [[ "$DRY_RUN" == true ]]; then
  "$TRAIN_PY" scripts/run_workspace.py "${OVERRIDES[@]}" --cfg job --resolve
  exit 0
fi

mkdir -p -- "$(dirname -- "$RUN")"
mkdir -- "$RUN"  # A fresh directory prevents overwriting an earlier experiment.
COMMAND=(
  "$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
  scripts/run_workspace.py "${OVERRIDES[@]}"
)
cp -- "$SCRIPT_PATH" "$RUN/launcher.sh"
"$TRAIN_PY" scripts/run_workspace.py "${OVERRIDES[@]}" --cfg job --resolve > "$RUN/config.yaml"
printf '%q ' "${COMMAND[@]}" > "$RUN/launch_command.txt"
printf '\n' >> "$RUN/launch_command.txt"
exec > >(tee -a "$RUN/console.log") 2>&1
printf 'GPUs: %s; epochs: %s; batch per GPU: %s; W&B: online\n' "$GPU_SELECTION" "$EPOCHS" "$BATCH_SIZE"
printf 'Checkpoints after epochs 1, 51, 101, ... and at completion. Output: %s\n' "$RUN"
"${COMMAND[@]}"
