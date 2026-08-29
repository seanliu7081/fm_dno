#!/usr/bin/env bash
# Gate M3 evaluation -- the verdict for COUPLING_MECHANISM_NOTES.md.
#
# For each arm: select the checkpoint, evaluate at N=10 over 500 episodes, run the offline
# diagnostics, and paired-compare against P1_curve at the same N and the same n_test.
#
#   A1_canon      observation-canonicalized source phase   (notes S3)
#   B1_blockwise  irrep-blockwise batch assignment         (notes S4)
#
# The bar is the notes' M3 bar: paired ΔSR@N=10 vs P1_curve, ±0.065.
#
# TWO THINGS THAT ARE NOT FREE PARAMETERS
# ---------------------------------------
# 1. `--n-parallel-envs 10` is REQUIRED, not a throughput knob. LiberoRunner builds
#    `env_task_names` in batches of `n_parallel_envs`, so changing it changes which task
#    every episode index is assigned. The baseline's `steps500_N10` was run at 10; an arm run
#    at any other value is not comparable to it.
# 2. Checkpoint selection uses the same rule as the baseline -- highest SR in the filename,
#    ties broken toward the later epoch -- so no arm gets a hand-picked checkpoint.
#
# WHY THIS WAITS
# --------------
# One training run costs ~27-41 GB and one 10-env evaluation costs ~22 GB (measured, PSS +
# swap across the tree). Training + evaluating concurrently leaves ~7 GB on this 62 GB box,
# and B1's rollout epochs spike above that. So evaluation waits for training to finish; then
# both arms evaluate in parallel, one per GPU, which is comfortable.
#
# Usage:
#   setsid nohup bash scripts/run_m3_eval.sh > output/logs/m3_eval.log 2>&1 < /dev/null &
#   WAIT_FOR=none bash scripts/run_m3_eval.sh          # evaluate now, skip the wait

set -uo pipefail

PY=${PY:-/home/haotian/miniforge3/envs/oat/bin/python}
EXP_ROOT=${EXP_ROOT:-output/exp}
BASE_REF=${BASE_REF:-${EXP_ROOT}/P1_curve/seed42/steps500_N10/dno_eval_log.json}
N_TEST=${N_TEST:-500}
N_PAR=10                       # see note 1 -- do not change
WAIT_FOR=${WAIT_FOR:-train}
ARMS=${ARMS:-"A1_canon B1_blockwise"}

cd "$(dirname "$0")/.."
mkdir -p output/logs

if [[ "${WAIT_FOR}" == "train" ]]; then
  while pgrep -f "run_workspace.py --config-name=train_flowpolicy" > /dev/null; do
    echo "  [$(date +%H:%M:%S)] training still running, holding evaluation"
    sleep 600
  done
  echo "  [$(date +%H:%M:%S)] no training left; starting evaluation"
fi

gpu=0
pids=()
for ARM in ${ARMS}; do
  OUT=${EXP_ROOT}/${ARM}/seed42
  CKPT=$(ls ${OUT}/checkpoints/ep-*_sr-*.ckpt 2>/dev/null | sort -t- -k3 | tail -1)
  if [[ -z "${CKPT}" ]]; then
    echo "### ${ARM}: no top-k checkpoint found, skipping"
    continue
  fi
  echo "### ${ARM}: selected ${CKPT}  -> GPU ${gpu}"
  (
    CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_GL=egl ${PY} scripts/eval_orbit_dno_libero.py \
        -c "${CKPT}" -o "${OUT}/final_N10" \
        --no-dno --num-inference-steps 10 \
        --n-test ${N_TEST} --n-parallel-envs ${N_PAR} --n-test-vis 0 \
        > output/logs/${ARM}_final_N10.log 2>&1
    # --action-source demo keeps action_mse_* / heading_MAE measured against demonstrations
    # on every arm; straightness / few_step_gap need no ground truth either way.
    CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_GL=egl ${PY} scripts/report_coupling_diagnostics.py \
        -c "${CKPT}" -o "${OUT}/diag.json" -d cuda:0 -b 256 --action-source demo \
        > output/logs/${ARM}_diag.log 2>&1
  ) &
  pids+=($!)
  gpu=$((gpu + 1))
done
for p in "${pids[@]}"; do wait "$p"; done

echo
echo "=============================================================================="
echo "M3 VERDICT -- paired ΔSR@N=10 vs P1_curve, bar ±0.065"
echo "=============================================================================="
for ARM in ${ARMS}; do
  LOG=${EXP_ROOT}/${ARM}/seed42/final_N10/dno_eval_log.json
  [[ -f "${LOG}" ]] || { echo "### ${ARM}: no eval log"; continue; }
  echo
  echo "########## ${ARM} ##########"
  ${PY} scripts/paired_task_compare.py -a "${BASE_REF}" -b "${LOG}" \
      --name-a P1_curve --name-b "${ARM}" \
      -o ${EXP_ROOT}/paired_${ARM}_vs_P1_N10.json 2>&1 | tail -18
done

${PY} scripts/report_fewstep_table.py -r ${EXP_ROOT} -o ${EXP_ROOT}/FEWSTEP.md > /dev/null 2>&1
echo
echo "wrote ${EXP_ROOT}/FEWSTEP.md and paired_*_vs_P1_N10.json"
