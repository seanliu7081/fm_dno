#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -euo pipefail
cd /workspace/fm_dno
profile_output="${STARVLA_PROFILE_OUTPUT:-output/starvla_heading_gaussian/checkpoint_profile}"
profile_args=(
  --output-dir "${profile_output}" --max-steps 3
  --micro-batch-size 1 --gradient-accumulation-steps 2
  --validation-samples 6 --validation-interval 2 --checkpoint-interval 2
  --log-interval 1 --num-workers 0
)
# A fresh process must load the optimizer state and successfully save again.
pty bash scripts/run_starvla_heading.sh "${profile_args[@]}" --stop-after-steps 2
pty bash scripts/run_starvla_heading.sh "${profile_args[@]}" --stop-after-steps 3 --resume
