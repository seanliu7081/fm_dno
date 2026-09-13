#!/usr/bin/env bash
set -euo pipefail
cd /workspace/fm_dno
export CUDA_VISIBLE_DEVICES="${STARVLA_TRAIN_GPUS:-0,1,2,3,4,5}"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export STARVLA_ROOT=/workspace/deps/starVLA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /workspace/venvs/starvla-heading/bin/python -m torch.distributed.run \
  --nproc_per_node=6 --master_addr=127.0.0.1 --master_port="${STARVLA_MASTER_PORT:-29761}" \
  scripts/train_starvla_heading.py "$@"
