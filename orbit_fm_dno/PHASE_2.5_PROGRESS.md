# Phase 2.5 — Progress Summary

**Repo:** `/home/haotian/code/fm_dno` · **Plan:** `PHASE_2.5_PLAN.md`
**Snapshot:** 2026-08-25 05:35 UTC · **both tracks still running** (~8–9 h remaining)
**Command log:** `RUNLOG.md` · **Phases 0–2:** `output/exp/PROGRESS_SUMMARY.md`

> This is an interim snapshot, not the §8 findings document. `PHASE_2.5_FINDINGS.md` is
> written once G13 and G14 close.

---

## 1. Why Phase 2.5 exists

Phase 2 passed its gate but produced two findings that made launching the full Phase 3
(~84 GPU-hours) a bad trade: the aggregate success rate could not resolve the effect being
looked for, and the checkpoints being evaluated were never selected on a rollout curve.

Phase 2.5 replaces that launch with two parallel ~30 h runs answering one question each:

| track | question | readout | status |
|---|---|---|---|
| **A** — `P4_rot_heading` | does the coupling create steerability on real robot data? | `orbit_steering_gain`, offline, no rollouts | **answered: 0.99** |
| **B** — `P1_curve` | what is the right operating point and budget? | SR-vs-epoch curve + working top-k | **open** (67 % done) |

---

## 2. Status

| item | state |
|---|---|
| §1 corrections (7 files) | ✅ written, compiled, G0 + G1 re-verified |
| §2 Gate **G4′** | ✅ fires; decision **(b)** `block_weights=[1.0,0.0]` on every arm |
| §3 Gate **G15** | ✅ paired resolution **±0.065**; Gate G6 verdict amended |
| §4 Gate **G13** | ✅ **0.99** (preliminary checkpoint; final re-run pending) |
| §5 Gate **G14** | ⏳ open — verdict retracted once, see §6 |
| §6 decision matrix | ⏳ row fixed by G13, **column undetermined** |

Track A: epoch **716/1001**, ETA ~8.0 h. Track B: epoch **680/1001**, ETA ~9.4 h.
No errors on either. 46 GB/62 GB resident.

---

## 3. §1 corrections — written, not copied

**The "refreshed package" §1 says to copy from does not exist.** `orbit_fm_dno/` is still the
original 2026-08-21 drop. All seven changes were authored here.

Each was first checked against the training path, because two ~30 h runs were about to
start: `global_rms` is opt-in (default stays `rms`), the audit/diagnostics/paired scripts are
standalone, the histogram gating is a no-grad buffer write with no RNG draw, and
`descent_limit` is DNO-eval only. **None alters a training trajectory** — so the GPUs were
started first and the corrections written while they ran.

| file | change |
|---|---|
| `scripts/paired_task_compare.py` | **new** — paired per-task statistics (G15) |
| `scripts/libero_heading_audit.py` | G4 reads **raw** block energy; prints raw vs normalized + per-dim RMS |
| `scripts/report_coupling_diagnostics.py` | warning guard tests `coupling['mode']`, not a literal string |
| `oat/policy/flow_policy_orbit.py` | `source_heading_hist` accumulates only for `mode in (rot, perm_rot)` |
| `oat/symmetry/normalizer.py` | new `vector_mode='global_rms'` |
| `oat/dno/task_losses.py` | new `descent_limit` + `max_descent`, wired into `CompositeTaskLoss` |
| `scripts/eval_orbit_dno_libero.py` | `--w-descent` / `--max-descent`; `--w-table` default **0**; `--table-z` 0.82 → 0.44 |

`descent_limit` verified to do exactly what it claims — identical chunk at eef z = 0.50 / 0.95 / 1.20:

```
descent_limit(0.15)   diving chunk  [2.0475, 2.0475, 2.0475]   active, EQUAL on both scene families
table_clearance(0.82)               [1.5449, 0.0,    0.0   ]   punishes the LOW scene only  <- the bug
table_clearance(0.44)               [0.0,    0.0,    0.0   ]   inert everywhere      <- the other horn
```

### Gate G0 needs a new form

The drop-in was committed in `fb6d6b1` / `f4cabea`, so our files are now **tracked**; editing
them shows ` M`, not `??`. The literal "every line starts with `??`" test is no longer valid
for this repo. The invariant that matters was verified directly and **passes**:

```bash
git diff --name-only 15b93ee -- <every §0.1 protected path> oat/config/task/policy/libero/
#  -> empty. Original repo byte-identical.
```

**Gate G1 re-passes unchanged:** min-max 2.383e-01, SO(2) block 3.484e-08,
IrrepAdam 1.608e-07, plain Adam 4.791e-01.

---

## 4. Gate G4′ — fires, as the plan predicted

`output/exp/audit_libero10_v2.{json,txt}`

| block | **RAW** | normalized |
|---|---|---|
| `vec0 (dx,dy)` | 0.1553 | 0.3835 |
| **`vec1 (wx,wy)`** | **0.0034** | 0.3835 |
| gripper | 0.7354 | 0.1918 |

The two *normalized* figures being identical **is** the defect: `rms` equalises blocks by
construction, so the original gate could never fire regardless of the data. On raw actions it
fires decisively — 0.34 % against a 5 % threshold.

**Decision: option (b), `policy.action_spec.block_weights=[1.0,0.0]`**, applied to every arm
including the iid control. It changes only the coupling cost, so P1 stays an exact control and
Phase 2's result remains valid; `global_rms` would change the normalizer and reintroduce the
confound Phase 2 existed to eliminate. Verified: complex width 32 → 16 (translation-only
heading/assignment) while `rotate_chunk` still rotates **both** blocks; heading equivariance
2.38e-07.

---

## 5. Gate G15 — the resolvable effect size, and an amended G6

`output/exp/paired_P0_vs_P1.{json,txt}` — hand-rolled t-test and sign test cross-checked
against `scipy`, identical to 4 decimals; reproduces every figure the plan quotes.

```
aggregate P0 0.1860   aggregate P1 0.2013
paired mean delta      +0.0153
paired stderr          ±0.0650      <-- THE RESOLVABLE EFFECT SIZE
paired t (9 dof)       +0.24    p 0.819      sign test p 0.754
mean |per-task delta|   0.1580      reshuffle ratio 10.3x
MDE at p<0.05, n=10     0.1470      projected at 3 seeds  0.0849
```

**Gate G6 amended.** The ±0.0121 recorded in Phase 2 was the *within-arm repeat* stderr —
rollout stochasticity at fixed weights, omitting the task-to-task variance that dominates,
understating uncertainty ~5×.

* defensible: **"the SO(2) normalizer carries no measurable success-rate cost"** (|Δ| < 0.147);
* **not** defensible: "P1 is better than P0".

No aggregate delta is to be quoted without its paired stderr from here on.

---

## 6. Gate G13 — **the mechanism transfers. gain = 0.99**

Read early from Track A's rolling checkpoint at **epoch ~330 of 1001** — legitimate because
the gain is measured on validation chunks with no rollouts, so it is insensitive to the
operating point. Run at `oom_score_adj=1000` so the probe would be sacrificed before either
training run; both survived.

| metric | P0 @1001 | P1 @1001 | **P4 @~330** |
|---|---|---|---|
| `orbit_steering_gain` | −0.00000 | −0.00000 | **0.99000** |
| `orbit_phase_consistency` | 0.05259 | 0.05467 | **0.83646** |
| `src_orbit_eq_rel` | 0.56720 | 1.10952 | 0.88675 |
| `heading_MAE` | 0.39026 | 0.62267 | 1.37418 |
| `action_mse_N10` | 0.04476 | 0.03045 | 0.08239 |
| `few_step_gap_N1` | 0.19540 | 0.20833 | 0.39943 |
| `straightness` | 1.74761 | 4.07690 | 6.49047 |

**Top band (≥ 0.7).** Two independent statistics agree — gain 0.99 (ideal 1.0) and phase
consistency 0.053 → 0.836 — while both controls read essentially zero, so the metric was
calibrated and the reading is real. This is the synthetic study's central claim reproduced on
real robot data, and per §4 it licenses Phase 6 and is worth reporting on its own.

**The delegation trade-off appears in both directions at once**, which is what makes it
credible. §4 warns: "if the gain is high and these did *not* worsen, be suspicious — the
mechanism predicts both together." Against P1, `src_orbit_eq_rel` **falls** (1.110 → 0.887,
predicted) while `heading_MAE` (0.623 → 1.374) and `action_mse_N10` (0.030 → 0.082)
**worsen**. The policy has stopped inferring heading from the observation and reads it off the
noise phase — the contract Phase 6 exists to repay.

### What is *not* claimed

* **The transport half is unsupported.** `straightness` (6.49 vs 4.08) and `few_step_gap_N1`
  (0.399 vs 0.208) are **worse**, where §4 expected comparable-or-better. This may be the
  epoch mismatch (P4 @330 vs P1 @1001); the fair test needs the final checkpoint. Not claimed
  on this evidence.
* Magnitudes of the worsened metrics are partly confounded with training progress. A *high*
  gain cannot be manufactured by undertraining — an uncoupled model reads 0.00 — so the
  direction of the result is safe; the numbers are provisional.

---

## 7. Gate G14 — open, after a retracted verdict

### The curve

```
epoch     0    50   100   150   200   250   300 │  350   400   450   500   550   600   650
SR     0.000 0.040 0.260 0.220 0.140 0.180 0.220│ 0.400 0.360 0.340 0.360 0.400 0.400 0.340
                   └──── 51/250 = 0.204 ───────┘└────────── 130/350 = 0.371 ──────────┘
                                    Fisher exact p = 9.4e-06
       trend within the high plateau (epoch>=350): slope -2.9e-05/epoch, p = 0.81  (flat)
```

A clean **two-level step**: a ~0.20 plateau through epoch 300, a step up around 300–350, and a
stable ~0.37 plateau across seven consecutive points since. Success at the same configuration
is **~1.8× what Phase 2 reported** — which is exactly the headroom the transport claim needs
and did not have at 0.19.

### A verdict was recorded and then retracted — the reasoning matters

At epoch 300 this log recorded G14 as "row 3, ceiling at ~0.20", and drew two conclusions
from it: the budget question is closed against more compute, and Phase 3 is not licensed.
**Both were withdrawn at epoch 350.**

The original test was, on five points: observed scatter (0.046) *smaller* than expected
binomial noise (0.057), χ² = 2.56, df = 4, p = 0.63, no trend. That is a correct calculation
of a statement that does not support the conclusion drawn from it: **a χ² consistency test can
only fail to reject; it never establishes flatness.** Treating "consistent with constant" as
"is constant" was the error. (An intermediate note also compared the maximum of the five
points against the mean of those same five — selection bias, and using the stderr of the mean
where per-point binomial noise was the right yardstick.)

Epoch 350 returned 0.400, a ~1-in-800 draw from a 0.204 plateau, and the constant model was
rejected (χ² p = 0.048). Epochs 400–650 confirmed the higher level.

**Operational consequence recorded at the time:** stopping Track B early had been proposed on
the strength of the flat reading, and the very next point it produced overturned that reading.
It runs to 1001.

### Still undetermined, and it decides the §6 column

Epochs 650 → 1001 are unmeasured. Two live outcomes:

* **holds at ~0.37** → operating point is far better than Phase 2 suggested; G14 lands on
  "fixable" → re-baseline, then run Phase 3 **and** Phase 6.
* **declines toward ~0.20** → genuine peak-then-decline; Phase 2 reported post-peak
  checkpoints and top-k selection becomes mandatory for every arm.

### Top-k selection now works

`ep-0350_sr-0.400` … `ep-0600_sr-0.400` are being written — the first time the manager has
ever fired in this project (under `lazy_eval: true` it never could, so every earlier number
came from an unselected final-epoch checkpoint). Because these sit on a genuinely higher
plateau, they are real selections. **Caveat retained:** on a *flat* curve, selecting the max
selects the luckiest binomial draw, not a better model — that was true of the epoch-100 point
and would have been a winner's-curse error.

---

## 8. An unresolved discrepancy — quote no SR without its protocol

`P1_norm_only` is the **same configuration** and measured **0.2013 at epoch 1001** over 1500
episodes, while `P1_curve` reaches ~0.37 by epoch 350. Three candidates, not yet separated:

1. **Episode-set difficulty** — the curve uses 50 episodes (seeds 1000–1049,
   `n_parallel_envs=5`); Phase 2 used 500 (seeds 1000–1499, `n_parallel_envs=20`).
2. **Run-to-run variance** — §5 anticipated this: rollouts consume the global RNG, so
   `P1_curve` is a second sample at the same budget, not a reproduction.
3. **Genuine peak-then-decline between 400 and 1001.**

**Queued decisive test** (light — loads no dataset): evaluate the finished
`P1_norm_only/seed42/checkpoints/latest.ckpt` under the curve's exact protocol
(`--n-test 50 --n-parallel-envs 5`). ~0.37 ⇒ explanation 1; ~0.20 ⇒ 2 or 3 are live.

Until then this applies retroactively to the 0.186 / 0.201 figures in `PROGRESS_SUMMARY.md`.

---

## 9. Bugs found in the plan's own commands

1. **`policy.coupling.kappa=.inf` crashes.** Through Hydra's CLI grammar it arrives as the
   *string* `'.inf'`; `build_coupling` does `float('.inf')` → `ValueError`. Track A would have
   died at startup. The YAML default is already float `inf`; use `kappa=inf` or omit.
2. **Track B needs `MUJOCO_GL=egl`** — `lazy_eval=false` builds an env runner (README §1.4).
3. **OOM.** Running both tracks as written killed Track A (exit 137) during its dataset load.
   New to this phase: Track B's `lazy_eval=false` forks 10 MuJoCo workers, which Phase 2 never
   did. Measured: Track A **19 GB**, Track B **40 GB** with 10 envs. (Per-process RSS sums to
   229 GB, but the forks are copy-on-write — real usage is far lower.) `ZarrDataset` hardcodes
   the in-memory store and `oat/dataset/zarr_dataset.py` is §0.1-protected, so the disk-backed
   fix is unavailable. Resolved without serialising the runs (~37 h instead of ~62 h):
   `n_parallel_envs` 10 → 5, and Track B launched with **`oom_score_adj=800`** so that if
   memory is exhausted the OOM killer takes Track B and the *decisive* Track A survives.

---

## 10. Where §6 lands

`G13 ≥ 0.7` is settled, fixing the **row**. G14 fixes the **column**, and the two columns give
opposite instructions:

| G13 | G14 | §6 cell |
|---|---|---|
| ≥ 0.7 ✅ | fixable / still climbing | **Best case** — re-baseline, then Phase 3 (P1,P2,P3,P4) **and** Phase 6 |
| ≥ 0.7 ✅ | ceiling at ~0.20 | **Phase 6 only** on the P4 checkpoint; transport claim unresolved at this operating point |

**No Phase 3 decision until Track B's curve resolves.** An earlier revision committed to the
second row on the retracted flat reading; that commitment is withdrawn.

---

## 11. Next actions, in order

1. Track A completes (~8 h) → re-run `report_coupling_diagnostics.py` on the **final** P4
   checkpoint. Confirms G13 epoch-matched and gives the only fair test of the transport half.
2. Track B completes (~9.4 h) → read the 650→1001 segment, close G14.
3. Run the queued protocol test (§8) to explain the 0.20-vs-0.37 gap.
4. Re-run `paired_task_compare.py` for P4 vs P1 on matched protocols.
5. Write `output/exp/PHASE_2.5_FINDINGS.md` per §8: the G4′ decision and why, the amended G6
   with paired stderr, the G13 gain with supporting diagnostics, the G14 curve shape and
   chosen budget, and which §6 cell we landed in.

---

## 12. Artifact map

```
RUNLOG.md                                   full command log, every gate verdict + retraction
output/exp/audit_libero10_v2.{json,txt}     G4' — raw vs normalized block energy
output/exp/paired_P0_vs_P1.{json,txt}       G15 — resolvable effect size
output/exp/P4_rot_heading/seed42/
    diag_early.json                         G13 preliminary (epoch ~330)  <- gain 0.99
    checkpoints/latest.ckpt
output/exp/P1_curve/seed42/
    logs.json                               G14 curve
    checkpoints/ep-XXXX_sr-0.XXX.ckpt       top-k, working for the first time
output/logs/{P4_rot_heading,P1_curve}_seed42.log
output/logs/selftest_phase25.txt            G1 re-verification
scripts/paired_task_compare.py              new tooling
```
