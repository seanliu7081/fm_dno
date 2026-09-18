#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -euo pipefail
cd /workspace/fm_dno
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export STARVLA_ROOT=/workspace/deps/starVLA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Keep device indices aligned with the six-GPU trainer and the final evaluator.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec /workspace/venvs/starvla-heading/bin/python -u scripts/run_starvla_periodic_eval.py \
  --output-root output/starvla_heading_gaussian \
  --server-device cuda:6 --render-gpu-device-id 7 --server-port 18087 \
  --eval-workers 4 --poll-seconds 10 \
  --only-job 42:10000 --keep-checkpoint-weights
