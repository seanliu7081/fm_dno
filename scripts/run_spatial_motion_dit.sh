#!/usr/bin/env bash
# Foreground launcher: invoke from tmux/supervisor to survive terminal closure.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_spatial_motion_dit.sh [options] [-- HYDRA_OVERRIDE ...]

  --train-gpu N      Physical training GPU (default: 0)
  --eval-gpu N       Physical evaluation GPU (default: 1)
  --python PATH     Interpreter (default: existing conda oat environment)
  --run-dir PATH    Fresh output directory; existing directories are refused
  --resume PATH     Resume this run's latest checkpoint and original W&B ID
  --prepare-only    Resolve and validate configuration without creating a run
  -h, --help        Show this help

Training is 500 complete epochs, with online W&B and 500-episode LIBERO-10
evaluations every 50 epochs. This script remains in the foreground and never
creates, attaches to, kills, or replaces a tmux session. Its caller owns the
process-manager session. Console output and launch provenance stay in the run.
EOF
}

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
train_gpu=0
eval_gpu=1
python_bin=/home/haotian/miniforge3/envs/oat/bin/python
if [[ ! -x "$python_bin" ]]; then python_bin=python; fi
run_dir=
resume=false
prepare_only=false
while (( $# )); do
  case "$1" in
    --train-gpu) train_gpu="${2:?--train-gpu requires an index}"; shift 2 ;;
    --eval-gpu) eval_gpu="${2:?--eval-gpu requires an index}"; shift 2 ;;
    --python) python_bin="${2:?--python requires a path}"; shift 2 ;;
    --run-dir) run_dir="${2:?--run-dir requires a path}"; shift 2 ;;
    --resume) run_dir="${2:?--resume requires a run directory}"; resume=true; shift 2 ;;
    --prepare-only) prepare_only=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) printf 'Unknown launcher option: %s (put Hydra overrides after --)\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$train_gpu" =~ ^[0-9]+$ && "$eval_gpu" =~ ^[0-9]+$ ]] || {
  printf 'GPU selections must be physical integer indices.\n' >&2; exit 2;
}
command -v "$python_bin" >/dev/null || { printf 'Python interpreter unavailable: %s\n' "$python_bin" >&2; exit 2; }
cd -- "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$train_gpu"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$train_gpu"
export WANDB_MODE=online PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ "$resume" == true ]]; then
  run_dir="$(realpath -e -- "$run_dir")"
  [[ -f "$run_dir/checkpoints/latest.ckpt" && -f "$run_dir/launcher.json" ]] || {
    printf 'Resume requires launcher.json and checkpoints/latest.ckpt in %s\n' "$run_dir" >&2; exit 2;
  }
  run_id="$("$python_bin" - "$run_dir/launcher.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['wandb_run_id'])
PY
)"
else
  run_id="$("$python_bin" -c 'import uuid; print(uuid.uuid4().hex[:12])')"
  if [[ -z "$run_dir" ]]; then
    run_dir="$repo_dir/output/spatial_motion_dit/$(date -u +%Y%m%dT%H%M%SZ)_${run_id}"
  fi
  run_dir="$(realpath -m -- "$run_dir")"
fi
export WANDB_RUN_ID="$run_id"
run_name="libero10_spatial_motion_dit_500ep_${run_id}"
config_args=(--config-name=train_flowpolicy_spatial_motion_dit)
if [[ "$resume" == true ]]; then
  [[ -f "$run_dir/resolved_config.yaml" ]] || { printf 'Missing original resolved configuration.\n' >&2; exit 2; }
  config_args=("--config-dir=$run_dir" --config-name=resolved_config)
fi
command=("$python_bin" "$repo_dir/scripts/run_workspace.py"
  "${config_args[@]}" "$@"
  "hydra.run.dir=$run_dir" "task.policy.env_runner.eval_gpu=$eval_gpu"
  "training.num_epochs=500" "training.resume=$resume"
  "task.policy.lazy_eval=false" "training.rollout_every=50"
  "training.rollout_epoch_offset=1" "logging.mode=online"
  "logging.resume=$resume" "logging.id=$run_id" "logging.name=$run_name")

# Hydra composition does not create the model, touch GPUs or start W&B.
config_tmp="$(mktemp /tmp/spatial-motion-config.XXXXXX.yaml)"
trap 'rm -f -- "$config_tmp"' EXIT
"${command[@]}" --cfg job --resolve > "$config_tmp"
"$python_bin" - "$config_tmp" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
assert cfg.training.num_epochs == 500 and cfg.training.max_train_steps is None
assert cfg.policy.backbone_type == 'starvla_dit'
assert cfg.policy._target_.endswith('.SpatialMotionFlowPolicy')
assert cfg.logging.mode == 'online' and not cfg.task.policy.lazy_eval
assert cfg.training.rollout_every == 50 and cfg.training.rollout_epoch_offset == 1
assert cfg.task.policy.env_runner.n_per_task == 50 and cfg.task.policy.env_runner.suite == 'libero_10'
assert cfg.task.policy.env_runner.init_start == 0
assert cfg.task.policy.env_runner.max_episode_steps == 550 and cfg.task.policy.env_runner.settle_steps == 10
PY
if [[ "$prepare_only" == true ]]; then
  cat -- "$config_tmp"
  printf '\nConfiguration verified; no run was created or launched.\n'
  exit 0
fi
if [[ "$resume" == false ]]; then
  mkdir -p -- "$(dirname -- "$run_dir")"
  mkdir -- "$run_dir"
fi
attempt="$(date -u +%Y%m%dT%H%M%SZ)_$$"
config_path="$run_dir/resolved_config.yaml"
if [[ "$resume" == true ]]; then config_path="$run_dir/resolved_config_resume_${attempt}.yaml"; fi
cp -- "$config_tmp" "$config_path"

"$python_bin" - "$repo_dir" "$run_dir" "$run_id" "$train_gpu" "$eval_gpu" "$resume" "$attempt" "$config_path" "${command[@]}" <<'PY'
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

from omegaconf import OmegaConf

repo, output = map(Path, sys.argv[1:3])
run_id, train_gpu, eval_gpu, resume, attempt, config_path = sys.argv[3:9]
command = sys.argv[9:]

def capture(args):
    try:
        result = subprocess.run(args, cwd=repo, capture_output=True, text=True, check=False)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except OSError as error:
        return {'error': str(error)}

versions = {}
for package in ('torch', 'torchvision', 'numpy', 'zarr', 'numcodecs', 'hydra-core',
                'omegaconf', 'accelerate', 'wandb', 'robosuite', 'mujoco', 'libero'):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
metadata = {
    'started_at_utc': datetime.now(timezone.utc).isoformat(), 'repository': str(repo),
    'run_dir': str(output), 'wandb_run_id': run_id, 'resume': resume == 'true',
    'python_executable': sys.executable, 'python_version': platform.python_version(),
    'platform': platform.platform(), 'train_gpu': int(train_gpu), 'eval_gpu': int(eval_gpu),
    'command': command, 'versions': versions, 'resolved_config': config_path,
    # Only intentionally public runtime settings; never dump the full environment.
    'environment': {key: os.environ.get(key) for key in (
        'CUDA_VISIBLE_DEVICES', 'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'MUJOCO_EGL_DEVICE_ID',
        'WANDB_MODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'CUBLAS_WORKSPACE_CONFIG')},
    'git_revision': capture(['git', 'rev-parse', 'HEAD']),
    'hardware': capture(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used', '--format=csv']),
}
metadata_path = output / ('launcher.json' if resume == 'false' else f'launcher_resume_{attempt}.json')
metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
(output / f'launch_command_{attempt}.sh').write_text('#!/usr/bin/env bash\n' + shlex.join(command) + '\n')
cfg = OmegaConf.load(config_path)
(output / f'dataset_config_{attempt}.json').write_text(json.dumps(
    OmegaConf.to_container(cfg.task.policy.dataset, resolve=True), indent=2) + '\n')

# Capture the exact implementation, including new untracked files. Restrict
# snapshots to this experiment's source files, excluding runtime/auth files.
paths = [
    'docs/spatial_motion_dit_plan.md', 'oat/perception/spatial_token_encoder.py',
    'oat/model/heading/__init__.py', 'oat/model/heading/motion_query_decoder.py',
    'oat/model/heading/segmented_heading_prior.py', 'oat/model/flow/starvla_dit.py',
    'oat/dataset/motion_plan_zarr_dataset.py', 'oat/policy/flow_policy_spatial_motion.py', 'oat/policy/flow_policy.py',
    'oat/config/task/policy/libero/libero10_spatial_motion.yaml',
    'oat/config/train_flowpolicy_spatial_motion_dit.yaml', 'oat/workspace/train_policy.py',
    'oat/env_runner/libero_official_runner.py', 'scripts/run_spatial_motion_dit.sh',
    'scripts/report_spatial_motion_dit.py',
]
snapshot = output / f'code_snapshot_{attempt}'
for relative in paths:
    source = repo / relative
    if source.is_file():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
diff = capture(['git', 'diff', 'HEAD', '--', *paths])
(output / f'code_diff_{attempt}.patch').write_text(diff.get('stdout', ''))
print(f'Run directory: {output}', flush=True)
print(f'Online W&B run ID: {run_id}; training GPU {train_gpu}; evaluation GPU {eval_gpu}', flush=True)
PY

set +e
"${command[@]}" 2>&1 | tee -a "$run_dir/console.log"
status=${PIPESTATUS[0]}
set -e
"$python_bin" - "$run_dir" "$attempt" "$status" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
path = Path(sys.argv[1]) / f'launcher_exit_{sys.argv[2]}.json'
path.write_text(json.dumps({'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                            'returncode': int(sys.argv[3])}, indent=2) + '\n')
PY
exit "$status"
