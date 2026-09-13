#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -euo pipefail
cd /workspace/fm_dno
profile_micro_batch="${STARVLA_PROFILE_MICRO_BATCH:-5}"
profile_accumulation=$((10 / profile_micro_batch))
if (( profile_micro_batch < 1 || profile_micro_batch * profile_accumulation != 10 )); then
  echo 'The throughput profile must preserve effective batch 60.' >&2
  exit 2
fi
pty bash scripts/run_starvla_heading.sh \
  --output-dir "output/starvla_heading_gaussian/profile_micro${profile_micro_batch}" \
  --max-steps 8 --micro-batch-size "${profile_micro_batch}" \
  --gradient-accumulation-steps "${profile_accumulation}" \
  --validation-samples 6 --validation-interval 8 --checkpoint-interval 8 \
  --skip-resume-save --log-interval 1 --num-workers 2 2>&1
