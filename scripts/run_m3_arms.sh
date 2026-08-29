#!/usr/bin/env bash
# Gate M3 of COUPLING_MECHANISM_NOTES.md -- one training run per surviving construction.
#
#   A1_canon      observation-canonicalized source phase   (notes S3, gated by M1a)
#   B1_blockwise  irrep-blockwise batch assignment         (notes S4, gated by M2)
#
# Both judged on **SR@N=10 paired against P1_curve**, +/-0.065 bar. The few-step channel is
# closed (F0), so neither arm is about sampler steps; they are about whether an easier
# regression target makes a better field at the standard operating point.
#
# WHY THIS RUNS THEM SEQUENTIALLY
# -------------------------------
# One run's process tree costs ~41 GB (PSS + swap): the 13.6 GB in-RAM replay buffer plus
# four forked dataloader workers plus five MuJoCo envs for the in-training rollouts. Two
# concurrent runs OOM this 62 GB box -- measured, exit code 137, twice, with the second run
# killed silently during `ReplayBuffer.copy_from_path` before it printed anything. The
# RUNLOG's earlier "two tracks fit at n_parallel_envs=5" precedent does not apply: that pair
# had one track on `lazy_eval=True`, i.e. no env runner at all.
#
# So: GPU 0 runs A1 to completion, then B1 starts on whichever GPU is free. No babysitting,
# no idle gap, and neither run is degraded to fit beside the other -- which matters, because
# both are being compared to a baseline trained at full width.
#
# Usage:
#   nohup bash scripts/run_m3_arms.sh > output/logs/m3_chain.log 2>&1 &
#   ARMS="B1_blockwise" bash scripts/run_m3_arms.sh      # just the one still owed
#
# Env: EPOCHS (801), GPU (0), PY, EXP_ROOT.

set -uo pipefail

PY=${PY:-/home/haotian/miniforge3/envs/oat/bin/python}
EXP_ROOT=${EXP_ROOT:-output/exp}
EPOCHS=${EPOCHS:-801}
GPU=${GPU:-0}
ARMS=${ARMS:-"A1_canon B1_blockwise"}

cd "$(dirname "$0")/.."
mkdir -p output/logs

declare -A CONFIG=(
  [A1_canon]=train_flowpolicy_canon
  [B1_blockwise]=train_flowpolicy_blockwise
)

# Wait for any other training to clear, so two never load the replay buffer at once.
wait_for_free_ram () {
  local need_gb=${1:-45}
  while true; do
    local others
    others=$(pgrep -f "run_workspace.py --config-name=train_flowpolicy" | wc -l)
    local avail
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [[ "${others}" -eq 0 && "${avail}" -ge 20 ]]; then break; fi
    echo "  [$(date +%H:%M:%S)] waiting: ${others} training proc(s) alive, ${avail} GB available"
    sleep 300
  done
}

for ARM in ${ARMS}; do
  CFG=${CONFIG[$ARM]}
  OUT=${EXP_ROOT}/${ARM}/seed42
  if [[ -f "${OUT}/checkpoints/latest.ckpt" ]] && \
     [[ "$(${PY} -c "import json,sys;print(max((json.loads(l)['epoch'] for l in open('${OUT}/logs.json')), default=-1))" 2>/dev/null)" -ge $((EPOCHS - 1)) ]]; then
    echo "### ${ARM}: already at ${EPOCHS} epochs, skipping"
    continue
  fi

  echo "=============================================================================="
  echo "### ${ARM}  (${CFG})  epochs=${EPOCHS}  gpu=${GPU}   $(date)"
  echo "=============================================================================="
  wait_for_free_ram

  # training.resume=True is on in both configs, so an interrupted arm picks up from
  # latest.ckpt rather than restarting.
  CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl ${PY} scripts/run_workspace.py \
      --config-name=${CFG} \
      seed=42 training.seed=42 training.num_epochs=${EPOCHS} training.num_demo=500 \
      task.policy.lazy_eval=false training.rollout_every=50 \
      task.policy.env_runner.n_test=50 task.policy.env_runner.n_parallel_envs=5 \
      task.policy.env_runner.n_test_vis=0 checkpoint.topk.k=5 \
      hydra.run.dir=${OUT} >> output/logs/${ARM}_seed42.log 2>&1
  rc=$?
  echo "### ${ARM} exited rc=${rc}   $(date)"
  if [[ ${rc} -eq 137 ]]; then
    echo "### rc=137 is the OOM killer. Do not restart it beside another training run."
  fi
done

echo
echo "M3 training done. Next -- for each arm's selected checkpoint:"
echo "  MUJOCO_GL=egl \$PY scripts/eval_orbit_dno_libero.py -c <ckpt> \\"
echo "      -o ${EXP_ROOT}/<ARM>/seed42/final_N10 --no-dno --num-inference-steps 10 \\"
echo "      --n-test 500 --n-parallel-envs 10 --n-test-vis 0"
echo "  \$PY scripts/paired_task_compare.py \\"
echo "      -a ${EXP_ROOT}/P1_curve/seed42/steps500_N10/dno_eval_log.json \\"
echo "      -b ${EXP_ROOT}/<ARM>/seed42/final_N10/dno_eval_log.json --name-a P1 --name-b <ARM>"
