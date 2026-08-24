# EXPERIMENT_PLAN.md — Orbit-Aligned Coupling + Orbit DNO on LIBERO-10

**Read this file top to bottom before running anything.** It is an executable runbook, not
a description. Each phase has an objective, exact commands, an acceptance gate, and a
"if the gate fails" branch. Do not skip a gate. Do not reorder phases.

Companion documents in this package:

| file | what it is |
|---|---|
| `DESIGN_orbit_coupling_dno_robot_policy.md` | the scientific design and the reasoning behind every choice below |
| `VALIDATION_RESULTS.txt` | measured synthetic-validation output this plan's expectations are calibrated against |
| `DROP_IN_README.md` | short integration note |

---

## 0. Hard constraints

### 0.1 Never modify an existing file

Every change is a **new file**. The repository as cloned must stay byte-identical.
Specifically **do not touch**:

```
oat/workspace/train_policy.py        oat/policy/flow_policy.py
oat/policy/diffpolicy.py             oat/policy/base_policy.py
oat/model/common/normalizer.py       oat/env_runner/libero_runner.py
oat/dataset/zarr_dataset.py          scripts/run_workspace.py
scripts/eval_policy_sim.py           oat/config/train_flowpolicy.yaml
oat/config/task/policy/libero/*.yaml  (any existing yaml)
```

The design was reworked so that no edit is needed: the SO(2) action normalizer is applied
inside `OrbitFlowPolicy.set_normalizer`, and the DNO rollout uses a new runner subclass
rather than an edit to `LiberoRunner`.

**Gate G0 — run after every phase that writes files:**

```bash
git status --porcelain
```

Every line must start with `??` (untracked). A single ` M` or `M ` means something was
modified — revert it with `git checkout -- <path>` and re-do the work as a new file.

### 0.2 Record everything

Maintain `RUNLOG.md` at the repo root (a new file). Append one entry per command you run:
timestamp, phase, exact command, wall-clock, outcome, and the gate verdict. When a gate
fails, record the failure and the branch you took. This is the only record that survives
the session.

### 0.3 Identical budget across arms

Whatever training budget you pick, **every arm uses the same one**. Set it once here and
reuse the variable everywhere:

```bash
export EPOCHS=1001          # adjust to your compute; must be identical for all arms
export NUM_DEMO=500
export SEEDS="42 43 44"
export EXP_ROOT=output/exp
export MUJOCO_GL=egl
```

A comparison where one arm trained longer is not a comparison. If you have to cut the
budget mid-experiment, restart every arm at the new budget rather than mixing.

---

## 1. File manifest

Copy these in from the package. Nothing else gets created except configs, `RUNLOG.md`, and
files under `output/`.

```
oat/symmetry/so2_chunk.py                    irrep layout, complex view, rotation, alignment
oat/symmetry/coupling.py                     SO2OrbitCoupling + MatchedHeadingPrior
oat/symmetry/normalizer.py                   rotation-commuting action normalizer
oat/symmetry/metrics.py                      steering gain, equivariance, transport
oat/policy/flow_policy_orbit.py              OrbitFlowPolicy (subclass of FlowPolicy)
oat/policy/dno_policy.py                     DNOPolicy wrapper for rollout
oat/env_runner/libero_dno_runner.py          LiberoRunner without inference_mode
oat/dno/irrep_adam.py                        rotation-compatible Adam
oat/dno/task_losses.py                       differentiable test-time objectives
oat/dno/orbit_dno.py                         two-stage orbit DNO
oat/config/train_flowpolicy_orbit.yaml       new config (new file in an existing dir — fine)
scripts/validate_so2_coupling.py             synthetic validation, no robot stack
scripts/libero_heading_audit.py              Phase 1 gate
scripts/report_coupling_diagnostics.py       offline diagnostics on a checkpoint
scripts/eval_orbit_dno_libero.py             LIBERO-10 eval under orbit DNO
scripts/aggregate_experiment_results.py      final table
```

`oat/` uses implicit namespace packages (there are no `__init__.py` files anywhere) — do
**not** add any.

---

## 2. Phase 0 — environment, data, and a self-test

### 2.1 Environment

Follow the repo README §1.2. Any of the four install paths works; `uv sync` matches
`uv.lock`. Then:

```bash
python -c "import torch, oat, libero; print(torch.__version__, torch.cuda.is_available())"
python -c "import scipy; print('scipy', scipy.__version__)"     # used by the Hungarian solver
```

If `scipy` is missing, either install it or set `policy.coupling.assignment=circular_sort`
everywhere (pure torch, but only valid with `cost=angle`).

### 2.2 Data

Per README §2, produce `data/libero/libero10_N500.zarr`:

```bash
python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_100
python scripts/convert_libero_dataset.py
python scripts/compose_libero_multitask_dataset.py -mt libero10
ls -la data/libero/libero10_N${NUM_DEMO}.zarr
```

### 2.3 Self-test (no GPU, no robot stack)

```bash
mkdir -p output/logs
python scripts/validate_so2_coupling.py --quick 2>&1 | tee output/logs/selftest.txt
```

**Gate G1.** All five tests must run and reproduce the qualitative pattern in
`VALIDATION_RESULTS.txt`:

- test 1: permutation modes show `norm drift == 0` in both heading regimes; `rot` modes are
  flagged `DEFORMED` under `vonmises` and not under `uniform`;
- test 3: SO(2) block normalizer equivariance error `< 1e-6`, min-max `> 0.1`;
- test 4: `IrrepAdam` `< 1e-5`, plain Adam `> 0.1`.

If G1 fails, stop. Something in the copy is wrong; nothing downstream is interpretable.

### 2.4 Controller-frame check — **do this, it is not optional**

The entire symmetry claim assumes the policy's action deltas live in the world/base frame.

```bash
python - <<'PY'
from oat.env.libero.env import LiberoEnv
import inspect, re
src = inspect.getsource(LiberoEnv)
print("\n".join(l for l in src.splitlines()
      if re.search(r'controller|OSC|output_max|delta|frame', l, re.I)))
PY
```

Confirm the controller is `OSC_POSE` and note its `output_max` for position and rotation.
Record both in `RUNLOG.md`; they become `--trans-gain` in Phase 6.

**Gate G2.** If the controller consumes **end-effector-frame** deltas, a world rotation acts
trivially on the action and this entire experiment is measuring nothing. Stop and report
that finding — it is a legitimate and valuable negative result, and it is cheap to find now
rather than after 21 training runs.

---

## 3. Phase 1 — the LIBERO-10 heading audit (go / no-go)

Costs seconds. Decides which coupling variants are worth training.

```bash
mkdir -p ${EXP_ROOT} output/logs
python scripts/libero_heading_audit.py \
    --zarr data/libero/libero10_N${NUM_DEMO}.zarr --horizon 16 \
    -o ${EXP_ROOT}/audit_libero10.json 2>&1 | tee ${EXP_ROOT}/audit_libero10.txt
```

Record in `RUNLOG.md`: `W1(heading, uniform)`, the heading `R1`, both normalizer
equivariance errors, per-block energy fractions, and the low-confidence fraction.

**Gate G3 — branch on `W1`:**

| `W1` | meaning | what to train |
|---|---|---|
| `< 0.10` | headings near-uniform, the design-doc §5 obstruction does not bite | P4 is safe with the standard prior; P5 is expected to be a no-op — still run it as the control |
| `0.10 – 0.40` | moderate prior shift | run P4 **and** P5; expect P5 > P4 |
| `> 0.40` | large prior shift | P5 is the primary rotation arm; P2 is the marginal-exact comparison |

**Gate G4 — rotation-block energy.** If `vec1_energy_frac < 0.05`, the `(wx, wy)` block is
negligible. Add `policy.action_spec.block_weights=[1.0,0.0]` to every coupled arm and say in
the writeup that the coupling is translation-only.

**Gate G5 — heading confidence.** If the fraction with confidence `< 0.05` exceeds ~0.25,
many chunks are near-stationary and have no meaningful heading. Set
`policy.coupling.min_heading_confidence=0.05` on every coupled arm and record the excluded
fraction.

---

## 4. Phase 2 — the normalizer control (P0 vs P1)

**Objective.** The SO(2) normalizer is a required prerequisite, and it is also a change that
could move success rate on its own. If P0 and P1 differ, every later comparison is
confounded. Establish this *before* introducing any coupling.

```bash
S=42
# P0 — repo baseline, config untouched
python scripts/run_workspace.py --config-name=train_flowpolicy \
    seed=$S training.seed=$S training.num_epochs=${EPOCHS} \
    training.num_demo=${NUM_DEMO} \
    hydra.run.dir=${EXP_ROOT}/P0_baseline/seed$S

# P1 — SO(2) normalizer, iid coupling (the only difference from P0)
python scripts/run_workspace.py --config-name=train_flowpolicy_orbit \
    seed=$S training.seed=$S training.num_epochs=${EPOCHS} \
    training.num_demo=${NUM_DEMO} \
    policy.coupling.mode=iid \
    hydra.run.dir=${EXP_ROOT}/P1_norm_only/seed$S
```

Then evaluate both with the stock offline eval:

```bash
for ARM in P0_baseline P1_norm_only; do
  MUJOCO_GL=egl python scripts/eval_policy_sim.py \
      -c ${EXP_ROOT}/${ARM}/seed$S/checkpoints/latest.ckpt \
      -o ${EXP_ROOT}/${ARM}/seed$S/eval -n 3 -d cuda:0
  python scripts/report_coupling_diagnostics.py \
      -c ${EXP_ROOT}/${ARM}/seed$S/checkpoints/latest.ckpt \
      -o ${EXP_ROOT}/${ARM}/seed$S/diag.json -d cuda:0
done
```

**Gate G6.** `|SR(P1) − SR(P0)|` should be within the 3-repeat stderr. If P1 is clearly
worse:

1. re-run P1 with `policy.normalizer_vector_mode=absmax` — the bounded analogue of the
   `limits` range the transformer was tuned for. Bounded range is the most likely culprit.
2. if it is still worse, **P1 — not P0 — is the baseline for every later comparison**, and
   the writeup must report the normalizer cost explicitly. Do not quietly compare coupled
   arms to P0.

Also confirm in `diag.json` that `orbit_steering_gain` for P1 is near 0. It must be — P1 has
no coupling. A non-zero value means the metric is misconfigured.

---

## 5. Phase 3 — the coupling comparison

### 5.1 Arms

| arm | overrides on `--config-name=train_flowpolicy_orbit` |
|---|---|
| `P1_norm_only` | `policy.coupling.mode=iid` (already trained in Phase 2) |
| `P2_perm_angle` | `policy.coupling.mode=perm policy.coupling.cost=angle` |
| `P3_perm_euclid` | `policy.coupling.mode=perm policy.coupling.cost=euclidean` |
| `P4_rot_heading` | `policy.coupling.mode=rot policy.coupling.align=heading` |
| `P6_permrot_group` | `policy.coupling.mode=perm_rot policy.coupling.cost=group_ot policy.coupling.align=kabsch` |

**P3 is not optional.** It is a symmetry-*blind* optimal-transport coupling (this is OT-CFM /
Immiscible Diffusion). Without it, any gain from P2/P4/P6 is attributable to "some coupling
helps", not to the symmetry in the coupling. The symmetry claim lives in **P2/P4/P6 vs P3**,
never in P2/P4/P6 vs P1.

### 5.2 Triage on one seed first

Train all four new arms at `seed=42` only:

```bash
S=42
run () {  # run <arm> <overrides...>
  ARM=$1; shift
  python scripts/run_workspace.py --config-name=train_flowpolicy_orbit \
      seed=$S training.seed=$S training.num_epochs=${EPOCHS} \
      training.num_demo=${NUM_DEMO} "$@" \
      hydra.run.dir=${EXP_ROOT}/${ARM}/seed$S
}
run P2_perm_angle    policy.coupling.mode=perm     policy.coupling.cost=angle
run P3_perm_euclid   policy.coupling.mode=perm     policy.coupling.cost=euclidean
run P4_rot_heading   policy.coupling.mode=rot      policy.coupling.align=heading
run P6_permrot_group policy.coupling.mode=perm_rot policy.coupling.cost=group_ot policy.coupling.align=kabsch
```

Add any flags Gates G4/G5 dictated (`policy.action_spec.block_weights=[1.0,0.0]`,
`policy.coupling.min_heading_confidence=0.05`) to **all** arms including P1, so they stay
comparable.

While training runs, watch the coupling diagnostics in wandb if you added the two-line log
hook from `DROP_IN_README.md` §3; if you did not, skip it — it is optional and the audit
already told you what the coupling does on this data.

### 5.3 Diagnostics before rollouts

```bash
for ARM in P2_perm_angle P3_perm_euclid P4_rot_heading P6_permrot_group; do
  python scripts/report_coupling_diagnostics.py \
      -c ${EXP_ROOT}/${ARM}/seed42/checkpoints/latest.ckpt \
      -o ${EXP_ROOT}/${ARM}/seed42/diag.json -d cuda:0
done
```

**Gate G7 — the coupling took.** Expected pattern, calibrated on the synthetic study:

| arm | `orbit_steering_gain` | `src_orbit_eq_rel` vs P1 |
|---|---|---|
| P2, P3 | ≈ 0 (P3 may be seed-variable) | mildly lower |
| P4, P6 | ≈ 1.0 | clearly lower (synthetic: 1.22 → 0.42–0.64) |

If P4's steering gain is near 0, the rotation coupling did not take. Check, in order:
`policy.coupling.align` is `heading` (a `kappa` dither or `align=kabsch` can suppress it),
`normalizer_mode` is `so2_block`, and Gate G4 did not silently zero out the block you are
measuring. Do not proceed to Phase 6 with a gain near 0 — stage-1 DNO would be a lottery.

**Expected and not a failure:** `heading_MAE` and `action_mse_N10` get *worse* for P4/P6.
That is heading delegation (design doc §9.1), it is the contract, and Phase 6 is where it is
repaid. Record it; do not "fix" it.

### 5.4 LIBERO-10 rollouts

```bash
for ARM in P2_perm_angle P3_perm_euclid P4_rot_heading P6_permrot_group; do
  MUJOCO_GL=egl python scripts/eval_policy_sim.py \
      -c ${EXP_ROOT}/${ARM}/seed42/checkpoints/latest.ckpt \
      -o ${EXP_ROOT}/${ARM}/seed42/eval -n 3 -d cuda:0
done
```

`n_test=500` (10 tasks × 50) is the config default and is the number to report. If a full
sweep does not fit the compute budget, drop to `task.policy.env_runner.n_test=200` uniformly
and say so — never vary it between arms.

### 5.5 Low-step-budget evaluation — the practical payoff

Straighter transport is supposed to buy inference steps. Evaluate every arm at reduced
sampler steps, which is a pure inference-time override:

```bash
for ARM in P0_baseline P1_norm_only P2_perm_angle P3_perm_euclid P4_rot_heading P6_permrot_group; do
  for N in 1 2 4 10; do
    MUJOCO_GL=egl python scripts/eval_orbit_dno_libero.py \
        -c ${EXP_ROOT}/${ARM}/seed42/checkpoints/latest.ckpt \
        -o ${EXP_ROOT}/${ARM}/seed42/steps_N${N} \
        --no-dno --num-inference-steps $N --n-test 200 --n-parallel-envs 20
  done
done
```

`--no-dno --num-inference-steps N` runs the plain policy with an N-step sampler and works on
plain `FlowPolicy` checkpoints too (P0), since that path never needs `sample_chunk`.

**Gate G8 — the headline.** The `N ∈ {1,2,4}` column is where a coupling that shortens
transport should separate from P1. If nothing separates at low N *and* nothing separates at
N=10, the coupling did not help on this task; record that and go to Phase 6 anyway — the DNO
result is a separate claim and P4's value may be entirely there.

### 5.6 Extend the survivors to 3 seeds

Take P1, P3, the best of {P2, P4, P6}, and P4 (needed for Phase 4 whether or not it won), and
run `seed=43` and `seed=44`. Repeat 5.3–5.5 for each. Three seeds is the floor; report seed
spread and do not claim a difference the spread covers.

---

## 6. Phase 4 — the matched inference prior (P5)

**P5 is not a training run.** It is the P4 checkpoint evaluated with the source heading law
the coupling actually produced during training — which `OrbitFlowPolicy` accumulated into the
`source_heading_hist` buffer and saved in the checkpoint. Same weights, different prior:
a perfectly controlled comparison.

```bash
for S in $SEEDS; do
  CKPT=${EXP_ROOT}/P4_rot_heading/seed$S/checkpoints/latest.ckpt
  # sanity: the histogram must be populated
  python - <<PY
from oat.policy.base_policy import BasePolicy
p = BasePolicy.from_checkpoint("$CKPT")
h = p.source_heading_hist
print("heading hist total:", float(h.sum()), " nonzero bins:", int((h>0).sum()), "/", h.numel())
assert float(h.sum()) > 0, "empty histogram -- P5 cannot run on this checkpoint"
PY
done
```

`eval_policy_sim.py` has no flag for this and must not be modified, so run it through
`eval_orbit_dno_libero.py`, which builds the policy itself and exposes `--inference-prior`:

```bash
for S in $SEEDS; do
  # P4 arm, standard prior (the control) — same script, so the comparison is exact
  MUJOCO_GL=egl python scripts/eval_orbit_dno_libero.py \
      -c ${EXP_ROOT}/P4_rot_heading/seed$S/checkpoints/latest.ckpt \
      -o ${EXP_ROOT}/P4_rot_heading/seed$S/eval_std \
      --no-dno --inference-prior standard --n-test 500 --n-parallel-envs 20

  # P5: identical weights, matched prior
  MUJOCO_GL=egl python scripts/eval_orbit_dno_libero.py \
      -c ${EXP_ROOT}/P4_rot_heading/seed$S/checkpoints/latest.ckpt \
      -o ${EXP_ROOT}/P5_rot_matched/seed$S/eval \
      --no-dno --inference-prior matched --n-test 500 --n-parallel-envs 20
done
```

Compare `P5_rot_matched/*/eval` against `P4_rot_heading/*/eval_std`, not against the Phase 3
`eval/` directory — same script, same flags, one flag different.

**Gate G9.** Compare P5 vs P4 at matched seeds. Expected: P5 ≥ P4, with the margin growing
with the `W1` measured in Phase 1. If `W1 < 0.10` (Gate G3 said "rot safe") and P5 ≈ P4, that
is the *correct* result and confirms the theory — record it as a confirmed prediction, not a
null.

Also re-run `report_coupling_diagnostics.py` on P5 (set `inference_prior='matched'` the same
way) and check `gen_heading_R1` moves toward `gt_heading_R1`.

---

## 7. Phase 5 — the κ sweep (P7), optional

Only if Phase 3 showed a real trade-off between P4/P6 (equivariance, few-step) and P1/P2/P3
(open-loop accuracy). κ dials how much heading authority is delegated to the noise:
`inf` delegates fully, small κ delegates none.

```bash
S=42
for K in 1 2 4 8; do
  python scripts/run_workspace.py --config-name=train_flowpolicy_orbit \
      seed=$S training.seed=$S training.num_epochs=${EPOCHS} training.num_demo=${NUM_DEMO} \
      policy.coupling.mode=rot policy.coupling.align=heading policy.coupling.kappa=$K \
      hydra.run.dir=${EXP_ROOT}/P7_kappa${K}/seed$S
done
```

Plot two axes per κ: transport reduction (from `diag.json`: `straightness`, `few_step_gap_N1`)
against prior deformation (`W1` of the source heading, from the coupling diagnostics) and
against `orbit_steering_gain`. This is the Pareto frontier that replaces the toy study's
planned `p_OT` sweep.

---

## 8. Phase 6 — orbit DNO on LIBERO-10

**This is the phase the whole design points at.** The 2×2:

|  | DNO off | DNO on (stage 1) |
|---|---|---|
| iid-coupled checkpoint (P1) | `A` | `B` |
| orbit-coupled checkpoint (best of P4 / P5 / P6) | `C` | `D` |

Prediction: `D − C` ≫ `B − A` at identical compute. Note carefully what is **not** predicted:
`B − A` is not zero. With steering gain ≈ 0 the K rotations of an iid policy's source are K
*unrelated* samples, so stage 1 degenerates into best-of-K resampling, which still helps.
The claim is that the coupling converts a blind best-of-K draw into a structured 1-D search.

### 8.1 Set the task-loss constants first

Two numbers from Phase 0 §2.4 and from the LIBERO scene:

```bash
export TRANS_GAIN=0.05     # OSC_POSE output_max for position, in metres. VERIFY.
export TABLE_Z=0.82        # table surface height in metres. VERIFY.
```

Get the table height empirically rather than guessing:

```bash
python - <<'PY'
from oat.common.replay_buffer import ReplayBuffer
import numpy as np
rb = ReplayBuffer.copy_from_path("data/libero/libero10_N500.zarr", keys=["robot0_eef_pos"])
z = np.asarray(rb["robot0_eef_pos"])[:, 2]
print("eef z: min %.3f  p01 %.3f  median %.3f  max %.3f" % (
    z.min(), np.percentile(z, 1), np.median(z), z.max()))
print("-> set TABLE_Z a little BELOW p01; a clearance floor above the demos' own reach")
print("   would penalise correct behaviour.")
PY
```

**Gate G10.** If `TABLE_Z` is set above the 1st percentile of demonstrated end-effector
height, the clearance term punishes the demonstrations themselves and DNO will steer *away*
from good behaviour. Set it below p01 and record the value.

### 8.2 Stage 1 only — run this first

No gradients, so the stock `LiberoRunner` drives it and it costs one batched forward of K
rotations per control cycle.

```bash
dno_eval () {   # dno_eval <arm> <tag> <extra flags...>
  ARM=$1; TAG=$2; shift 2
  MUJOCO_GL=egl python scripts/eval_orbit_dno_libero.py \
      -c ${EXP_ROOT}/${ARM}/seed42/checkpoints/latest.ckpt \
      -o ${EXP_ROOT}/${ARM}/seed42/dno_${TAG} \
      --n-test 100 --n-parallel-envs 10 \
      --trans-gain ${TRANS_GAIN} --table-z ${TABLE_Z} "$@"
}

# cell A / C : DNO off, identical script and seeds as the DNO arms
dno_eval P1_norm_only   off --no-dno
dno_eval P4_rot_heading off --no-dno

# cell B / D : stage-1 orbit search, K=32
dno_eval P1_norm_only   s1_K32 --n-orbit 32 --n-grad-steps 0
dno_eval P4_rot_heading s1_K32 --n-orbit 32 --n-grad-steps 0
```

**Gate G11 — the central result.** Compute `Δ_iid = SR(B) − SR(A)` and
`Δ_orbit = SR(D) − SR(C)`. The claim needs `Δ_orbit > Δ_iid`, with the gap outside the
3-seed spread. Also check `dno/wall_time_p50` from `dno_eval_log.json` against the control
period (`n_action_steps / fps = 8 / 20 = 0.4 s`). If p95 exceeds 0.4 s, the method is not
real-time on this hardware — report the number honestly and say what it would take (fewer
orbit angles, a distilled few-step sampler).

Sweep `K ∈ {8, 16, 32, 64}` on the orbit arm to show the search-quality/compute curve. The
iid arm's curve is the best-of-K resampling baseline — plot them on the same axes; that plot
*is* the argument.

### 8.3 Stage 2 — gradients

Only after 8.2. Switches automatically to `OrbitDnoLiberoRunner` (same rollout loop without
`torch.inference_mode`).

```bash
dno_eval P4_rot_heading s12 --n-orbit 32 --n-grad-steps 6 --lr 0.05 --optimizer irrep_adam
dno_eval P4_rot_heading s12_adam --n-orbit 32 --n-grad-steps 6 --lr 0.05 --optimizer adam
```

The `irrep_adam` vs `adam` pair is the optimizer ablation: coordinate-wise Adam breaks
rotation compatibility (synthetic: 4.8e-01 vs 1.6e-07 iterate drift).

Memory: the graph through N Euler steps is held for every parallel env at once. On OOM,
lower `--n-parallel-envs` before lowering `--dno-steps` — the latter changes what you are
evaluating.

**Gate G12.** If stage 2 does not improve on stage 1, that is a legitimate result and worth
reporting: it says the 1-D orbit coordinate captured what the objective could reach, which
*supports* the design's framing. Do not tune it into looking better.

### 8.4 Task-loss ablation

Turn terms off one at a time on the best DNO configuration to see which constraint is
binding: `--w-smoothness 0`, `--w-seam 0`, `--w-table 0`. Report `dno/final_loss` and
`dno/orbit_loss_spread` alongside success rate. If `orbit_loss_spread` is near zero, the
objective does not discriminate between orbit angles and stage 1 cannot help — that is a
task-loss problem, not a coupling problem.

---

## 9. Phase 7 — aggregate and report

```bash
python scripts/aggregate_experiment_results.py -r ${EXP_ROOT} -o ${EXP_ROOT}/RESULTS.md
git status --porcelain     # Gate G0, one last time
```

Write `${EXP_ROOT}/FINDINGS.md` (new file) containing, in this order:

1. **The audit** (Phase 1): `W1`, block energies, which gates fired and what they changed.
2. **The normalizer control**: P0 vs P1, and which one is the baseline for everything else.
3. **The coupling table**: success rate at `N ∈ {1,2,4,10}` for P1–P6, 3 seeds, with spread.
   State the symmetry claim as **P2/P4/P6 vs P3**, and say so explicitly.
4. **The delegation trade-off**: steering gain and `heading_MAE` side by side. Name it as the
   expected cost, not a defect.
5. **The DNO 2×2**: `Δ_orbit` vs `Δ_iid`, the K-sweep curve, wall-clock p50/p95 against the
   0.4 s control period.
6. **What did not work**, with numbers. A coupling that failed on LIBERO after working on
   synthetic chunks is the most informative result in the set, and the design doc §11 already
   names the candidate causes — check them against the data rather than speculating.

Every claim in `FINDINGS.md` must point at a JSON file under `${EXP_ROOT}`.

---

## 10. Compute budget

| phase | runs | rough cost |
|---|---|---|
| 0–1 | — | minutes |
| 2 | 2 trainings + 2 evals | 2 × (train) |
| 3 triage | 4 trainings + 4 evals + 15 low-step evals | 4 × (train) |
| 3 seeds | 8 more trainings | 8 × (train) |
| 4 | 3 evals, no training | small |
| 5 (optional) | 4 trainings | 4 × (train) |
| 6 | ~10 DNO evals at n_test=100 | K× a normal eval per cell |

**If the budget is short, cut in this order:** Phase 5 first, then the 3-seed extension
(report single-seed and say so), then the low-step sweep at `N=2`. **Never cut** Phase 1,
Phase 2, or the P3 control — each of them is what makes the remaining numbers mean anything.

---

## 11. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `orbit_steering_gain ≈ 0` on a `rot` arm | `align=kabsch`, finite `kappa`, or `normalizer_mode=inherit` | check the three; `align=heading, kappa=inf` is what produces steerability |
| `RuntimeError: ... inference_mode` during DNO | stage 2 under the stock runner | the script switches automatically when `--n-grad-steps > 0`; if you built the runner by hand, use `OrbitDnoLiberoRunner` |
| `src_norm_drift > 0` on a `perm` arm | a source sample was modified | bug — permutation modes must never touch a sample; re-run `validate_so2_coupling.py --tests 1` |
| Hungarian solve is slow | large batch | `assignment=circular_sort` with `cost=angle`, or reduce `dataloader.batch_size` (and hold it fixed across arms) |
| DNO OOM | stage-2 graph across parallel envs | lower `--n-parallel-envs` |
| `dno/orbit_loss_spread ≈ 0` | the task loss does not discriminate orbit angles | raise `--w-table`, add `--w-workspace`, or check `TRANS_GAIN` |
| P4 success much worse than P1 | heading delegation | expected; judge P4 in Phase 6, not Phase 3 |
| `empty histogram` on P5 | checkpoint predates the buffer, or `mode=iid` | P5 only applies to a rotation-applying arm |

---

## 12. What this plan is testing, in one paragraph

That an SO(2) orbit-aligned source–target coupling — a training-time change with **zero
inference cost** and **no architectural change** — gives a vanilla flow-matching action-chunk
policy a group-structured noise space; that this straightens its transport enough to matter
at low step budgets on LIBERO-10; that the same structure makes the policy delegate its
motion heading to the noise phase, which costs open-loop accuracy; and that a one-dimensional
exhaustive search over the group orbit, costing one batched forward pass per control cycle,
repays that cost and then some. The two halves are one mechanism. Report them together.
