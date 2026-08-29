#!/usr/bin/env bash
# Phase F0 of PLAN_fewstep_coupling.md -- the no-training day.
#
# Three measurements that decide everything downstream. Nothing here takes a gradient step.
#
#   F0c  CPU, minutes   coupling preview at the settings the arms actually train with
#   F0a  GPU, ~4-6 h    the headroom sweep: SR at N in {1,2,4,10}. THE go/no-go
#   F0b  GPU, ~2-3 h    baseline re-selection at N=2, so the comparison is fair
#
# Every step is skipped if its output already exists, so this is safe to re-run after an
# interruption. Set FORCE=1 to redo everything.
#
# Usage:
#   bash scripts/run_fewstep_f0.sh                  # all three
#   bash scripts/run_fewstep_f0.sh f0c              # just the CPU audit
#   GPU=1 bash scripts/run_fewstep_f0.sh f0a
#
# Gate F0a (read it before launching anything downstream):
#   gap = SR(N=10) - SR(N=2)
#     >= 0.15        real headroom -> proceed with all arms
#     0.05..0.15     at/under the 1-seed MDE -> Reflow only
#     < 0.05         the 10-step sampler was never the bottleneck. Train NOTHING; the
#                    N-sweep itself is the result ("this policy runs at N=2 for free").
#     <= 0.05 at every N including 10 -> operating point broken, reconcile with G14 first.

set -euo pipefail

PY=${PY:-/home/haotian/miniforge3/envs/oat/bin/python}
EXP_ROOT=${EXP_ROOT:-output/exp}
export BASE=${BASE:-${EXP_ROOT}/P1_curve/seed42}   # exported: the summary heredocs read it
GPU=${GPU:-0}
N_TEST=${N_TEST:-200}          # 20 episodes/task -> stderr ~ +/-0.034 per point
N_TEST_RESEL=${N_TEST_RESEL:-100}
N_PAR=${N_PAR:-10}
FORCE=${FORCE:-0}
WHICH=${1:-all}
# F0a can be split across GPUs, or rerun at a higher n_test without clobbering the first
# read: N_LIST picks which sampler budgets this invocation owns, TAG names the output dirs.
#   GPU=0 TAG=steps500 N_TEST=500 N_LIST="10 4" bash scripts/run_fewstep_f0.sh f0a
#   GPU=1 TAG=steps500 N_TEST=500 N_LIST="2 1"  bash scripts/run_fewstep_f0.sh f0a
N_LIST=${N_LIST:-"10 2 4 1"}   # 10 and 2 first: they are the two the gate reads
export TAG=${TAG:-steps}

cd "$(dirname "$0")/.."
mkdir -p output/logs output/audit

# The plan's selection rule, stated out loud: highest sr in the filename, ties broken by
# the later epoch. Three of P1_curve's five top-k checkpoints tie at 0.400.
CKPT=${CKPT:-$(ls ${BASE}/checkpoints/ep-*_sr-*.ckpt | sort -t- -k3 | tail -1)}

echo "=============================================================================="
echo "F0  baseline      : ${BASE}"
echo "    checkpoint    : ${CKPT}"
echo "    n_test        : ${N_TEST} (F0a) / ${N_TEST_RESEL} (F0b)"
echo "    GPU           : ${GPU}"
echo "=============================================================================="

run_eval () {   # run_eval <out_dir> <n_steps> <ckpt> <n_test>
  local out=$1 steps=$2 ckpt=$3 ntest=$4
  if [[ -f "${out}/dno_eval_log.json" && "${FORCE}" != "1" ]]; then
    echo "  skip (exists): ${out}"
    return
  fi
  echo "  -> ${out}   (N=${steps}, n_test=${ntest})"
  CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl ${PY} scripts/eval_orbit_dno_libero.py \
      -c "${ckpt}" -o "${out}" \
      --no-dno --num-inference-steps "${steps}" \
      --n-test "${ntest}" --n-parallel-envs "${N_PAR}" --n-test-vis 0 \
      2>&1 | tee -a "output/logs/f0_$(basename "${out}").log"
}

# ---------------------------------------------------------------------------------------
# F0c -- is P3 worth a GPU?  CPU only, run it first: it can veto a 27 h training job.
# ---------------------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "f0c" ]]; then
  echo
  echo "### F0c -- coupling preview at scalar_weight=1.0, block_weights=[1,0]"
  for B in 32 128; do
    OUT=output/audit/f0c_sw1_bw10_b${B}.json
    if [[ -f "${OUT}" && "${FORCE}" != "1" ]]; then echo "  skip (exists): ${OUT}"; continue; fi
    ${PY} scripts/libero_heading_audit.py \
        --scalar-weight 1.0 --block-weights 1,0 --batch-size ${B} \
        -o "${OUT}" 2>&1 | tee "output/logs/f0c_b${B}.log"
  done
  echo
  echo "  Gate F0c: if B=128 improves the path reduction by < 2 points over B=32, drop the"
  echo "            P3b batch-size arm. If euclidean @ scalar_weight=1.0 still reads < 10%,"
  echo "            P3's prior drops accordingly -- train it only on otherwise-idle GPUs."
fi

# ---------------------------------------------------------------------------------------
# F0a -- the headroom sweep.  THE go/no-go.
# ---------------------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "f0a" ]]; then
  echo
  echo "### F0a -- headroom sweep, N in {${N_LIST}}, n_test=${N_TEST}, tag=${TAG}"
  for N in ${N_LIST}; do
    run_eval "${BASE}/${TAG}_N${N}" "${N}" "${CKPT}" "${N_TEST}"
  done
  ${PY} - <<'PYEOF'
import json, pathlib, os
base = pathlib.Path(os.environ.get("BASE", "output/exp/P1_curve/seed42"))
tag = os.environ.get("TAG", "steps")
sr = {}
for n in (1, 2, 4, 10):
    p = base / f"{tag}_N{n}" / "dno_eval_log.json"
    if p.is_file():
        sr[n] = float(json.loads(p.read_text())["mean_success_rate_mean"])
print("\n  N   SR")
for n in sorted(sr):
    print(f"  {n:<3} {sr[n]:.4f}")
if 10 in sr and 2 in sr:
    gap = sr[10] - sr[2]
    verdict = ("REAL HEADROOM -- proceed with all arms" if gap >= 0.15 else
               "AMBIGUOUS (at/under the 1-seed MDE) -- Reflow only" if gap >= 0.05 else
               "NO HEADROOM -- train nothing; the N-sweep IS the result")
    print(f"\n  gap = SR(N=10) - SR(N=2) = {gap:+.4f}   ->  {verdict}")
    if max(sr.values()) <= 0.05:
        print("  !! everything <= 0.05 absolute: operating point broken, reconcile with G14")
    if 0.05 <= gap < 0.15:
        print("  (the plan says: rerun N=2 and N=10 at --n-test 500 before deciding)")
PYEOF
fi

# ---------------------------------------------------------------------------------------
# F0b -- baseline re-selection at the endpoint.
# Top-k for P1_curve was chosen at N=10; a fair N=2 comparison gives the baseline its best
# checkpoint AT N=2 too. Two baseline numbers come out of F0:
#   B10@N2  the N=10-selected checkpoint read at N=2   (the "deployment" comparison)
#   B2@N2   best-of-existing at N=2                    (the stricter one)
# Every arm is reported against BOTH; a claim that survives only against B10@N2 says so.
# ---------------------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "f0b" ]]; then
  echo
  echo "### F0b -- re-selection at N=2 over every existing checkpoint"
  for C in ${BASE}/checkpoints/*.ckpt; do
    run_eval "${BASE}/resel_N2/$(basename "${C}" .ckpt)" 2 "${C}" "${N_TEST_RESEL}"
  done
  ${PY} - <<'PYEOF'
import json, pathlib, os
base = pathlib.Path(os.environ.get("BASE", "output/exp/P1_curve/seed42"))
rows = []
for p in sorted((base / "resel_N2").glob("*/dno_eval_log.json")):
    d = json.loads(p.read_text())
    rows.append((float(d["mean_success_rate_mean"]), p.parent.name))
if rows:
    print("\n  SR@N2   checkpoint")
    for sr, name in sorted(rows, reverse=True):
        print(f"  {sr:.4f}  {name}")
    best_sr, best_name = max(rows)
    print(f"\n  B2@N2 = {best_sr:.4f}  ({best_name})")
    print("  B10@N2 is the steps_N2 number from F0a (same checkpoint, N=10-selected).")
PYEOF
fi

echo
echo "done. Next: scripts/report_fewstep_table.py -r ${EXP_ROOT} --ref P1_curve/seed42"
