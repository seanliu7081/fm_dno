#!/usr/bin/env bash
# Foreground training launcher; the caller owns the terminal/process manager.
set -euo pipefail
SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}")
cd -- "$(dirname -- "$SCRIPT_PATH")"

DRY_RUN=false
case "${1:-}" in
  --dry-run) DRY_RUN=true; shift ;;
  -h|--help)
    cat <<'HELP'
Usage: bash train_fruit_v3_spatial_motion.sh [--dry-run]

Environment settings:
  DATASET_PATH   Zarr path (default: /workspace/ysk/zarr/fruitV3_40.zarr)
  TRAIN_GPUS     Physical GPU indices, comma-separated (default: 7)
  TRAIN_PY       Python interpreter (default: /venv/fm_dno/bin/python)
  POLICY_EPOCHS  Epoch count (default: 1001, matching the current fruit script)
  BATCH_SIZE    Training batch size per GPU (default: 64)
  WANDB_MODE    online or offline (default: online)
  RUN_DIR       Fresh output directory (default: timestamped under output/)

Uses two 128x128 cameras, 112x112 crops, two observations, and 16 predicted
actions with an eight-action execution stride. Trains the flow policy directly.
--dry-run validates the dataset and prints the resolved configuration; it does
not create a run, initialize W&B, or start training.
HELP
    exit 0 ;;
  "") ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
(( $# == 0 )) || { printf 'Unexpected extra arguments.\n' >&2; exit 2; }

TRAIN_PY="${TRAIN_PY:-/venv/fm_dno/bin/python}"
DATASET_PATH="${DATASET_PATH:-/workspace/ysk/zarr/fruitV3_40.zarr}"
POLICY_EPOCHS="${POLICY_EPOCHS:-1001}"
BATCH_SIZE="${BATCH_SIZE:-64}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-7}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  printf 'TRAIN_GPUS must contain comma-separated GPU indices.\n' >&2; exit 2;
}
[[ "$POLICY_EPOCHS" =~ ^[1-9][0-9]*$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLICY_EPOCHS and BATCH_SIZE must be positive integers.\n' >&2; exit 2;
}
[[ "$WANDB_MODE" == online || "$WANDB_MODE" == offline ]] || {
  printf 'WANDB_MODE must be online or offline.\n' >&2; exit 2;
}
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPU_IDS[@]}

# Validate without loading all camera frames or changing the source dataset.
NUM_DEMOS=$("$TRAIN_PY" - "$DATASET_PATH" "$CUDA_VISIBLE_DEVICES" <<'PY'
import sys
import numpy as np
import zarr

gpu_ids = [int(value) for value in sys.argv[2].split(',')]
if len(set(gpu_ids)) != len(gpu_ids):
    raise ValueError('TRAIN_GPUS must contain distinct GPU indices')
root = zarr.open_group(sys.argv[1], mode='r')
ends = np.asarray(root['meta/episode_ends'][:])
if (ends.ndim != 1 or ends.dtype.kind not in 'iu' or len(ends) < 2
        or np.any(np.diff(np.r_[0, ends]) <= 0)):
    raise ValueError('Expected at least two episodes with increasing integer boundaries')
shapes = {
    'action': (7,), 'agentview_rgb': (128, 128, 3),
    'robot0_eye_in_hand_rgb': (128, 128, 3), 'robot0_eef_pos': (3,),
    'robot0_eef_rot6d': (6,), 'robot0_gripper_qpos': (1,), 'task_uid': (1,),
}
for key, shape in shapes.items():
    array = root[f'data/{key}']
    if array.shape != (int(ends[-1]), *shape):
        raise ValueError(f'{key}: expected {(int(ends[-1]), *shape)}, got {array.shape}')
    if key.endswith('_rgb'):
        if array.dtype != np.uint8:
            raise ValueError(f'{key} must be uint8 RGB')
    elif array.dtype.kind not in 'fiu' or not np.isfinite(array[:]).all():
        raise ValueError(f'{key} must contain finite numeric values')
if np.unique(root['data/task_uid'][:]).tolist() != [0]:
    raise ValueError('This single-task configuration requires task_uid=0')
print(len(ends))
PY
)

DATASET_NAME=$(basename -- "$DATASET_PATH" .zarr)
RUN_GROUP="${DATASET_NAME}_spatial_motion"
RUN="${RUN_DIR:-$PWD/output/${RUN_GROUP}_$(date -u +%Y%m%d_%H%M%S)}"
OVERRIDES=(
  --config-name=train_flowpolicy_spatial_motion_fruit_v3
  seed=42
  "task.policy.dataset.zarr_path=$DATASET_PATH"
  "task.policy.name=real_robot_$DATASET_NAME"
  "task.policy.task_name=$DATASET_NAME"
  "training.num_demo=$NUM_DEMOS"
  "training.num_epochs=$POLICY_EPOCHS"
  "training.expected_num_processes=$NPROC"
  "dataloader.batch_size=$BATCH_SIZE"
  "logging.mode=$WANDB_MODE"
  logging.project=real_robot
  "logging.group=$RUN_GROUP"
  "hydra.run.dir=$RUN"
)

if [[ "$DRY_RUN" == true ]]; then
  "$TRAIN_PY" scripts/run_workspace.py "${OVERRIDES[@]}" --cfg job --resolve
  exit 0
fi

mkdir -p -- "$(dirname -- "$RUN")"
mkdir -- "$RUN"
COMMAND=(
  "$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
  scripts/run_workspace.py "${OVERRIDES[@]}"
)
cp -- "$SCRIPT_PATH" "$RUN/launcher.sh"
printf '%q ' "${COMMAND[@]}" > "$RUN/launch_command.txt"
printf '\n' >> "$RUN/launch_command.txt"
exec > >(tee -a "$RUN/console.log") 2>&1
printf 'Dataset: %s (%s episodes)\n' "$DATASET_PATH" "$NUM_DEMOS"
printf 'GPUs: %s; epochs: %s; batch per GPU: %s; W&B: %s\n' \
  "$CUDA_VISIBLE_DEVICES" "$POLICY_EPOCHS" "$BATCH_SIZE" "$WANDB_MODE"
printf 'Images: 128x128; crops: 112x112; output: %s\n' "$RUN"
"${COMMAND[@]}"
