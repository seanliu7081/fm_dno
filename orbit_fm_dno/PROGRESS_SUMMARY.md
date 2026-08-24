# Orbit-Aligned Coupling + Orbit DNO on LIBERO-10 — Progress Summary

**Repo:** `/home/haotian/code/fm_dno` (renamed from `past2next_clean`)
**Plan:** `orbit_fm_dno/EXPERIMENT_PLAN.md`
**Detailed command log:** `RUNLOG.md` (repo root)
**As of:** 2026-08-24 · Phases 0–2 complete, Phase 3 not started

---

## 1. What this experiment is testing

That an SO(2) orbit-aligned source–target coupling — a **training-time-only** change with
zero inference cost and no architectural change — does two things to a vanilla
flow-matching action-chunk policy:

1. **straightens transport** enough to matter at low sampler budgets (N ∈ {1,2,4}); and
2. **delegates the chunk's motion heading to the noise phase**, so a one-dimensional
   exhaustive search over the group orbit (one batched forward pass per control cycle)
   can steer the policy at test time.

The two halves are one mechanism. The symmetry claim lives in **P2/P4/P6 vs P3**
(a symmetry-*blind* OT coupling), never in P2/P4/P6 vs P1.

> **Nothing about this claim has been tested yet.** Both arms trained so far (P0, P1) have
> **no coupling**. Everything below is groundwork, controls, and go/no-go gates.

---

## 2. Status at a glance

| Stage | Status | Outcome |
|---|---|---|
| Integration — 16-file drop-in | ✅ | Gates **G0**, **G1**, **G2** passed |
| End-to-end smoke test | ✅ | 39/40 checks |
| Data preparation | ✅ | `libero10_N500.zarr` — 500 eps, 138,090 steps, 3.4 GB |
| Phase 1 — heading audit | ✅ | Gates **G3**, **G4**, **G5** |
| Gate **G10** (pulled forward) | ✅ | plan's `TABLE_Z` placeholder proven wrong |
| Phase 2 — P0 vs P1 control | ✅ | Gate **G6** passed |
| Phase 3 — coupling comparison | ⬜ | **not started** (~60 h train + ~24 h eval) |
| Phases 4–7 | ⬜ | not started |

**Compute spent:** ~35 h wall-clock (2 × ~29 h training in parallel, then 2 × ~6 h eval).
**Gate G0 invariant:** 0 existing repo files modified, at every checkpoint. Verified again
after the repo rename.

---

## 3. Integration

16 new files placed; no existing file touched (the plan's hard constraint §0.1).

```
oat/symmetry/{so2_chunk,coupling,normalizer,metrics}.py
oat/policy/{flow_policy_orbit,dno_policy}.py
oat/env_runner/libero_dno_runner.py
oat/dno/{irrep_adam,task_losses,orbit_dno}.py
oat/config/train_flowpolicy_orbit.yaml
scripts/{validate_so2_coupling,libero_heading_audit,report_coupling_diagnostics,
         eval_orbit_dno_libero,aggregate_experiment_results}.py
```

### Gate G1 — synthetic self-test (`output/logs/selftest.txt`)

| criterion | required | measured |
|---|---|---|
| SO(2) block normalizer equivariance | < 1e-6 | **3.484e-08** |
| per-dim min-max equivariance | > 0.1 | **2.383e-01** |
| `IrrepAdam` iterate drift | < 1e-5 | **1.608e-07** |
| plain Adam iterate drift | > 0.1 | **4.791e-01** |
| `norm drift`, permutation modes | == 0 | **0.0000** (both heading regimes) |
| `rot` flagged DEFORMED under `vonmises` only | yes | yes |

### Gate G2 — controller frame (the check that could have killed the premise)

The entire symmetry argument assumes action deltas are **world-frame**. If LIBERO's
controller consumed end-effector-frame deltas, a world rotation would act trivially and
the experiment would measure nothing.

The plan's own inspection snippet returns nothing useful on this repo (`LiberoEnv`
delegates to LIBERO's `ControlEnv`). Resolved by tracing robosuite directly:

* `set_goal_position(scaled_delta[:3], self.ee_pos)` → `ee_pos + delta`, both world-frame.
* `set_goal_orientation` → `quat2mat(axisangle2quat(delta)) @ current_orientation` —
  **left**-multiplication, i.e. the axis-angle delta is world-frame. Conjugating by `R_z`
  maps the axis-angle vector `w → R_z·w`.

So `(dx,dy)` and `(wx,wy)` are frequency-1 and `dz, wz, grip` are invariant — exactly
`SO2ChunkSpec.libero_osc_pose()`. **Premise holds.**

Constants confirmed from `osc_pose.json` (not assumed):
`TRANS_GAIN = 0.05 m`, `ROT_GAIN = 0.5 rad`, `control_freq = 20`,
control period = `n_action_steps / fps = 8/20 = 0.4 s`.

### Smoke test — 39/40

All five Phase-2/3 arms instantiate, train, and sample; normalizer repair fires inside
`set_normalizer`; DNO stage 1 runs **under `torch.inference_mode()`** (so the stock runner
drives it); stage 2 descends the loss (202.4 → 153.2, `irrep_adam`); the runner `_target_`
swap resolves; every Gate G4/G5/κ override builds. Stage-1 wall-clock at K=8/batch 4 was
**~25 ms** against the 0.4 s budget (indicative only).

The one non-pass is a **documentation defect, not a bug**: `source_heading_hist` is
accumulated for *every* coupling mode, so an `iid` checkpoint also carries a populated
histogram. The plan's §6 assertion and `eval_orbit_dno_libero.py`'s guard therefore do
**not** discriminate an iid checkpoint the way §11 claims. Benign (matched ≈ standard for
iid), but the stated safety net does not exist.

---

## 4. Data

LIBERO-10 hdf5 demos were **already present** (13 GB) in the sibling `oat` repo's LIBERO
checkout — which is where this env's editable `libero` install resolves. Symlinked per
README §2 rather than re-downloaded.

```
data/libero/libero10_N500.zarr   3.4 GB   (matches README's stated expectation)
  action                (138090, 7)          float32
  agentview_rgb         (138090, 128,128,3)  uint8
  robot0_eye_in_hand_rgb(138090, 128,128,3)  uint8
  robot0_eef_pos/quat/gripper_qpos/joint_pos, task_uid
  meta/episode_ends     (500,)               10 tasks × 50 demos
```

**Environment caveat:** this repo's own `third_party/LIBERO` submodule is **uninitialised**
and `~/.libero/config.yaml` points into `/home/haotian/code/oat`. Everything works, but
moving or cleaning that sibling directory will break this repo.

---

## 5. Phase 1 — heading audit (`output/exp/audit_libero10.json`)

32,841 chunks of horizon 16.

| quantity | value | meaning |
|---|---|---|
| `W1(heading, uniform)` | **0.2103 rad** | hard floor on angular transport for any marginal-preserving coupling |
| heading `R1` | 0.1511 | headings are non-uniform |
| min-max normalizer equivariance err | 1.930e-01 | repo default breaks the group action on real data |
| SO(2) block normalizer err | 3.503e-08 | repair works on real data |
| low-confidence fraction (< 0.05) | **0.0004** | essentially every chunk has a well-defined heading |

**Coupling preview on real chunks** — the design's central tension, visible before any
training:

| coupling | path len | angle gap | cond var | src R1 |
|---|---|---|---|---|
| A/iid | 13.999 | 1.5698 | 142.56 | 0.0581 |
| **P2/perm-angle** | 13.659 | **0.2573** | 135.90 | **0.0600** |
| P3/perm-euclid | 12.857 | 0.7118 | 118.33 | 0.0517 |
| **P4/rot-heading** | 13.624 | **0.0000** | 135.59 | **0.1801** |
| P6/perm+rot-group | **12.619** | 0.5075 | **113.93** | 0.1533 |

P2 lands within 22 % of the theoretical floor (0.2573 vs 0.2103) while leaving the source
marginal at the iid level. P4 drives the gap to zero but deforms the source
(`R1` 0.058 → 0.180). **You can have exact marginals or maximal transport reduction, not
both** — and the 0.18 deformation is exactly what P5's matched prior exists to repay.

### Gate verdicts

* **G3 — `W1 = 0.2103` → moderate band.** Verdict `matched_prior_recommended`: run **P4 and
  P5**, expect P5 > P4. Full arm list (P1, P2, P3, P4, P6) stands.
* **G4 — does not fire as instrumented, but see §7.1.** No `block_weights` override applied
  (user decision: default two-block coupling on every arm).
* **G5 — does not fire.** No `min_heading_confidence` override.

---

## 6. Phase 2 — the normalizer control (P0 vs P1)

Budget: **1001 epochs**, identical across arms (plan §0.3). Seed 42. Run concurrently, one
per GPU, single-process each — deliberately *not* `accelerate --multi_gpu`, because DDP
would change the global batch and the orbit config warns batch size is a coupling
hyper-parameter.

| arm | config | training | success rate |
|---|---|---|---|
| **P0_baseline** | `train_flowpolicy` (untouched) | 1001 epochs, ~25.7 h | 0.172 / 0.210 / 0.176 → **0.1860** ± 0.0121 |
| **P1_norm_only** | `train_flowpolicy_orbit`, `coupling.mode=iid` | 1001 epochs, ~29 h | 0.188 / 0.222 / 0.194 → **0.2013** ± 0.0105 |

### Gate G6 — passed

`SR(P1) − SR(P0) = +0.0153`. Marginally outside P0's 3-repeat stderr (0.0121) by the literal
wording, but **0.96 σ** on the combined stderr (0.0160) — statistically indistinguishable —
and **P1 is better, not worse**. §4's remediation branch is conditioned on "if P1 is clearly
worse", which did not occur. Therefore:

* no `normalizer_vector_mode=absmax` re-run needed;
* **P0 remains the reference baseline**; the SO(2) normalizer carries no success-rate cost.

Both arms report `orbit_steering_gain = −0.00000`, the check §4 explicitly demands of P1
("it must be — P1 has no coupling; a non-zero value means the metric is misconfigured").
**The steerability metric is trustworthy**, so a non-zero reading on a coupled arm will mean
something.

### Diagnostics

| metric | P0 (min-max) | P1 (SO(2) block) | comparable? |
|---|---|---|---|
| `orbit_steering_gain` | −0.00000 | −0.00000 | yes |
| `orbit_phase_consistency` | 0.05259 | 0.05467 | yes |
| **`action_mse_N10`** (raw units) | 0.04476 | **0.03045** | **yes** — P1 32 % better |
| `action_mse_N1` / `N4` | 0.03824 / 0.04341 | **0.02731 / 0.02992** | yes |
| `heading_MAE` | 0.39026 | 0.62267 | ⚠ normalized space |
| `straightness` | 1.74761 | 4.07690 | ⚠ normalized space |
| `few_step_gap_N1` / `N4` / `N10` | 0.19540 / 0.08752 / 0.05257 | 0.20833 / 0.07914 / 0.03903 | ⚠ normalized space |
| `src_orbit_eq_rel` | 0.56720 | 1.10952 | ⚠ normalized space |

> ### ⚠ Which numbers may be compared to which
>
> `train_loss`, `val_loss`, `straightness`, `few_step_gap`, `src_orbit_eq_rel` and
> `heading_MAE` are all computed in **normalized** space, and P0's normalizer differs from
> P1's — the rotation block is scaled **20.8×** under `so2_block` vs ~1× under min-max. So
> these are **not comparable between P0 and P1**; they are comparable only among P1 and the
> coupled arms, which share a normalizer.
>
> P1's `straightness` of 4.08 vs P0's 1.75 is therefore an artifact of rescaling, **not**
> evidence the normalizer curved the flow. Likewise P1's `val_loss` of 0.558 vs P0's 0.159.
>
> **Consequence:** the headline few-step claim must be judged **P2/P4/P6 against P1 and P3**,
> never against P0. Only success rate and raw-space `action_mse_*` compare across all arms.

---

## 7. The findings that matter

### 7.1 Gate G4 cannot fire as written, and the raw numbers say the opposite

The gate measures block energy **after** the rms normalizer, which sets every vector block
to unit per-coordinate RMS — so `vec0_energy_frac == vec1_energy_frac` **by construction**
(both 0.3835 here) and the `< 0.05` threshold is unreachable under the default normalizer.

Measured on **raw** actions:

| block | raw energy share | per-dim RMS |
|---|---|---|
| `(dx, dy)` | 0.1529 | 0.2802 / 0.3579 |
| **`(wx, wy)`** | **0.0034** | 0.0398 / 0.0551 |
| `dz` | 0.0973 | 0.3626 |
| `wz` | 0.0060 | 0.0898 |
| gripper | 0.7403 | — |

The rotation block carries **0.34 %** of raw action energy — an order of magnitude *below*
the gate's threshold — and the normalizer then amplifies it to parity (confirmed live in the
P1 run: `(0,1)→3.1114, (3,4)→20.8007`). **The coupling gives a near-inert channel equal
weight when computing headings and assignments.** Decision taken: keep the default (it is
the design's intent); record the caveat.

### 7.2 Aggregate parity concealed a per-task reshuffle 10× larger

| eef median (m) | task | P0 | P1 | Δ |
|---|---|---|---|---|
| 0.546 | LIVING_ROOM_SCENE6_white_mug_plate_pudding | 0.253 | 0.007 | **−0.247** |
| 0.573 | LIVING_ROOM_SCENE5_white_mug_left_plate | 0.333 | 0.040 | **−0.293** |
| 0.616 | LIVING_ROOM_SCENE1_alphabet_soup_cream_cheese | 0.147 | 0.260 | +0.113 |
| 0.620 | LIVING_ROOM_SCENE2_cream_cheese_butter | 0.147 | 0.107 | −0.040 |
| 0.627 | LIVING_ROOM_SCENE2_alphabet_soup_tomato_sauce | 0.107 | 0.100 | −0.007 |
| 1.013 | KITCHEN_SCENE4_black_bowl_drawer | 0.047 | 0.193 | +0.147 |
| 1.052 | KITCHEN_SCENE3_stove_moka | 0.667 | 0.567 | −0.100 |
| 1.057 | KITCHEN_SCENE6_mug_microwave | 0.067 | 0.420 | **+0.353** |
| 1.095 | KITCHEN_SCENE8_both_moka_pots | 0.000 | 0.253 | **+0.253** |
| 1.166 | STUDY_SCENE1_book_caddy | 0.093 | 0.067 | −0.027 |

```
aggregate Δ            +0.0153
mean |per-task Δ|       0.1580      ← 10.3× the aggregate
low-table group  (n=5)  0.197 → 0.103   (−0.095)
high-table group (n=5)  0.175 → 0.300   (+0.125)
```

Swings of 0.25–0.35 are 4–5× the per-task binomial stderr (~0.06 at 50 episodes/task), so
they are **real, not sampling noise**. The two policies are not behaviourally equivalent —
the mean simply averages the redistribution away.

**Limit on this claim:** table height is *perfectly collinear* with scene family — the five
"low" tasks are exactly the five LIVING_ROOM tasks, the five "high" are four KITCHEN plus
STUDY. This data **cannot** attribute the shift to geometry rather than scene identity,
appearance, or task type. With n=5 per group on a single seed it could still be
coincidence. Suggestive, not established; a second seed would settle it.

**Consequence for the plan.** §4's premise is that aggregate parity makes later comparisons
unconfounded. **That premise is false here.** A coupled arm could beat P3 on the aggregate
purely by favouring a different task mix. **From Phase 3 onward, report per-task success
rates for every arm.** The numbers are already in each `eval_log.json`; this costs nothing
and changes what can honestly be concluded.

### 7.3 `TABLE_Z = 0.82` (plan §8.1 placeholder) is catastrophically wrong here

eef height is **bimodal with an empty gap at 0.815–0.889**, splitting exactly five tasks per
mode. For the five low tasks the **maximum** eef height (0.698–0.781) is entirely below
0.82 — so that placeholder would penalise **every demonstrated step** of half the benchmark,
precisely the failure §8.1 warns about. 48.4 % of all steps fall below it.

`min 0.446 · p01 0.448 · p05 0.484 · median 0.928 · max 1.332`

The plan's rule (below p01) gives `TABLE_Z = 0.44`, which is safe (penalises 0.0 %) but
leaves the clearance term **inert** for the five high tasks. Since `--w-table 10.0` is the
heaviest Tier-0 weight, that removes most of the objective's mass and makes §8.4's
`dno/orbit_loss_spread ≈ 0` failure mode likely. **A scalar `--table-z` cannot be correct on
LIBERO-10.** Options recorded in `RUNLOG.md`; not yet decided (Phase 6 is downstream).

### 7.4 Two silent-failure traps in the tooling

1. **`compose_libero_multitask_dataset.py` fails with exit code 0.** It shells out via
   `os.system` with a **bare `python`**, which picked the system interpreter (no `click`);
   `os.system` does not propagate the child's status, so the wrapper reported success and
   produced no zarr. Worked around by putting the conda env first on `PATH`. *Do not trust
   this script's exit code.*
2. **`report_coupling_diagnostics.py` emits a false-positive warning** ("steering gain near
   zero on a coupled checkpoint") for any orbit-config checkpoint, because its guard
   compares `report["coupling"]` against the literal string `"iid(baseline)"` without
   checking whether `mode` is actually `iid`. Cosmetic.

---

## 8. What we do **not** yet have

* **Zero evidence on the actual scientific claim.** P0 and P1 are both uncoupled. The
  symmetry claim (P2/P4/P6 vs P3) and the DNO claim are entirely untested.
* **One seed only.** Plan §5.6 sets three seeds as the floor.
* No coupled checkpoint, so no non-zero `orbit_steering_gain` has ever been observed on
  real data — the precondition for Phase 6 being meaningful.

---

## 9. Open decisions

1. **Training budget.** At 1001 epochs the baseline sits at **0.19 with one task at 0.000**,
   and per-task noise is ±0.06. README §3.4 trains this baseline for **5001** epochs. The
   symmetry claim must resolve inside that narrow band, and §7.2 just showed aggregate
   differences at this scale can be swamped by task-mix effects. Raising the budget lifts
   the field off the floor at ~5× cost. **Plan §0.3 forbids mixing budgets across arms**, so
   a change means restarting P0 and P1.
2. **Launch Phase 3?** 4 arms ≈ 60 h training + ~24 h eval on two GPUs.
3. **Phase 6 `TABLE_Z` strategy** (§7.3) — deferrable.

Projected remaining cost at the current budget: Phase 3 triage ~60 h, 3-seed extension
~120 h, i.e. **~8–9 days of continuous GPU** before Phase 6.

---

## 10. Artifact map

```
RUNLOG.md                                       full command log, every gate verdict
output/exp/audit_libero10.{json,txt}            Phase 1 audit
output/exp/P0_baseline/seed42/
    checkpoints/latest.ckpt                     614 MB
    diag.json                                   coupling/transport diagnostics
    eval/eval_log.json                          3×500-episode rollout, per-task SR
    logs.json                                   per-step training log
output/exp/P1_norm_only/seed42/                 (same structure)
output/logs/selftest.txt                        Gate G1
output/logs/{convert_libero,compose_libero10}.log
output/logs/{P0_baseline,P1_norm_only}_seed42.log, {P0,P1}_eval.log
```

Reproduce Phase 2 (identical budget is mandatory):

```bash
export PY=/home/haotian/miniforge3/envs/oat/bin/python
export EXP_ROOT=output/exp EPOCHS=1001 NUM_DEMO=500

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_workspace.py --config-name=train_flowpolicy \
    seed=42 training.seed=42 training.num_epochs=${EPOCHS} training.num_demo=${NUM_DEMO} \
    hydra.run.dir=${EXP_ROOT}/P0_baseline/seed42

CUDA_VISIBLE_DEVICES=1 $PY scripts/run_workspace.py --config-name=train_flowpolicy_orbit \
    seed=42 training.seed=42 training.num_epochs=${EPOCHS} training.num_demo=${NUM_DEMO} \
    policy.coupling.mode=iid \
    hydra.run.dir=${EXP_ROOT}/P1_norm_only/seed42
```
