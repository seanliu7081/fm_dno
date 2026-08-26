# Orbit-Aligned Coupling + Orbit DNO on LIBERO-10 — Methods and Results

**Repo:** `/home/haotian/code/fm_dno` (renamed from `past2next_clean`)
**Period:** 2026-08-21 → 2026-08-25 · **Hardware:** 2 × RTX 4090, 62 GB RAM
**Source plans:** `EXPERIMENT_PLAN.md` (Phases 0–7), `PHASE_2.5_PLAN.md` (triage, superseded §5)
**Command log:** `RUNLOG.md` · **Phase findings:** `output/exp/PHASE_2.5_FINDINGS.md`

Complete record of every experiment run, the method behind each, and what each established.

---

## 1. What was being tested

An SO(2) orbit-aligned source–target coupling — a **training-time-only** change with zero
inference cost and no architectural change — was claimed to do two things to a vanilla
flow-matching action-chunk policy:

1. **straighten transport**, so fewer sampler steps suffice (N ∈ {1,2,4});
2. **delegate the chunk's heading to the noise phase**, so a 1-D exhaustive search over the
   group orbit can steer the policy at test time for one batched forward pass per cycle.

The symmetry claim lives in **P2/P4/P6 vs P3** (a symmetry-*blind* OT coupling), never in
P2/P4/P6 vs P1.

### Verdict

| claim | verdict | key evidence |
|---|---|---|
| the coupling creates a 1-D steering channel | ✅ **CONFIRMED** | gain **1.04**; both controls exactly **0.00** |
| the channel is searchable in real time | ✅ **CONFIRMED** | K=32 search, **31 ms** p50 vs 400 ms control period |
| straighter transport buys inference steps | ❌ **REFUTED** | epoch-matched, worse on all three transport metrics |
| steering improves task success | ⚠️ **UNTESTED** | Tier-0 objective carries no directional information |
| **cost of the mechanism** | **measured** | **−0.1933 ± 0.0540, p = 0.006**, worse on all 10 tasks |

---

## 2. Setup

**Policy.** `FlowPolicy` — rectified flow matching, 4-layer/256-dim transformer (4.8 M) +
robomimic ResNet obs encoder (22.4 M), horizon 16, `n_action_steps` 8, `n_obs_steps` 2,
10-step Euler sampler. `OrbitFlowPolicy` subclasses it; the *only* training-path change is
`z0 ← Π(z0, x1)` before the interpolation.

**Data.** `data/libero/libero10_N500.zarr` — 500 episodes, 138,090 steps, 3.4 GB, 10 tasks ×
50 demos. Built per README §2 (hdf5 → per-task zarr → merged). The raw demos were already
present in a sibling checkout and were symlinked rather than re-downloaded.

**Budget.** 1001 epochs (3,885 iterations/epoch, batch 32), seed 42, identical across arms per
plan §0.3. ~29 h per arm.

**Hard constraint (§0.1).** No existing repo file modified, at any point. Verified continuously:

```bash
git diff --name-only 15b93ee -- <every protected path>   # -> empty, throughout
```

---

## 3. Methods

### 3.1 The SO(2) decomposition

A LIBERO/robosuite `OSC_POSE` action `a = [dx,dy,dz,wx,wy,wz,grip]` decomposes under rotation
about world z into two **frequency-1** blocks `(dx,dy)`, `(wx,wy)` and three invariants
`dz, wz, grip`. A whole chunk carries one shared angle. In the complex view each 2-D block
becomes one complex number and ρ(φ) is multiplication by e^{iφ}, making the two needed
operations one-liners: heading `θ(x) = arg Σ x_k`, alignment `φ*(z,x) = arg Σ conj(z_k) x_k`
(the 2-D closed form of Klein et al.'s Kabsch step).

### 3.2 Couplings (`oat/symmetry/coupling.py`)

A 2×2 of {identity pairing, batch assignment} × {apply φ*, don't}, plus a cost choice:

| mode | what it does | marginal |
|---|---|---|
| `iid` | nothing — the baseline | exact |
| `perm` | permute sources within the minibatch | **exact** (no sample modified) |
| `rot` | rotate each source onto its target's orbit position | deformed |
| `perm_rot` | both | deformed |

Costs: `group_ot` (maximise \|S_ij\|, phase-invariant), `euclidean` (Re S_ij, symmetry-blind
control), `angle` (circular heading distance).

### 3.3 The SO(2) action normalizer (`oat/symmetry/normalizer.py`)

The repo default (`mode='limits'`, per-dimension min-max) gives `scale[dx] ≠ scale[dy]` and a
non-zero offset, so a rotation in normalized space is **not** a rotation in raw space. Every
geometric claim lives in the normalized space the network sees, so this had to be fixed first.
The fix constrains the affine map to commute with ρ(φ): one shared scale per frequency-1 block,
offset exactly 0; invariant channels keep ordinary min-max.

Measured on real LIBERO-10 chunks: **min-max 1.930e-01** vs **SO(2) block 3.503e-08**.

Applied inside `OrbitFlowPolicy.set_normalizer`, so `TrainPolicyWorkspace`,
`run_workspace.py`, `eval_policy_sim.py` and `from_checkpoint` all work unmodified.

### 3.4 Metrics

* **`orbit_steering_gain`** — radians of generated-chunk heading per radian of source rotation,
  by circular least squares. 1.0 = the noise phase controls the heading; 0.0 = it is ignored.
* **`orbit_phase_consistency`** — \|E exp(i(Δθ − φ))\|; 1.0 only if gain is 1 *with no scatter*.
* **`src_orbit_eq_rel`** — ‖F(ρz\|o) − ρF(z\|o)‖/‖F(z\|o)‖ at fixed observation. This is
  **noise-space** equivariance, not the textbook property: with a fixed RGB camera a world
  rotation rotates camera and scene together, so the image is *invariant*, not equivariant.
* **`straightness`**, **`few_step_gap_N`** — transport quality; the "fewer steps" claim.
* **`heading_MAE`**, **`action_mse_N`** — open-loop conditional accuracy.

### 3.5 Orbit DNO (`oat/dno/orbit_dno.py`)

Stage 1 evaluates K rotations of z on a grid over [0,2π) in one batched forward pass — no
gradients, no local minima. Stage 2 runs a few `IrrepAdam` steps (Adam with the second moment
**shared across each frequency-1 block**, so the preconditioner commutes with ρ(φ)).

Task losses are tiered: **Tier 0** uses no privileged state (smoothness, seam continuity, step
limit, clearance); Tier 1 is privileged (obstacle SDFs); Tier 2 is learned (success classifier
or Q-function). All experiments here used Tier 0.

### 3.6 Statistics — `scripts/paired_task_compare.py` (new)

LIBERO-10's aggregate is a mean over 10 tasks and both arms see the same tasks, so arm-vs-arm
differences are **paired**. The correct uncertainty is `std(Δ)/√10`, not the within-arm repeat
stderr that `eval_policy_sim.py` reports — the latter measures rollout stochasticity at fixed
weights and understates the real uncertainty ~5×. Reports paired t, exact sign test, reshuffle
ratio, and MDE. Cross-checked against `scipy` to 4 dp.

---

## 4. Phase 0 — machinery verification

### Gate G1 — synthetic self-test

| criterion | required | measured |
|---|---|---|
| SO(2) block normalizer equivariance | < 1e-6 | **3.484e-08** |
| per-dim min-max equivariance | > 0.1 | **2.383e-01** |
| `IrrepAdam` iterate drift | < 1e-5 | **1.608e-07** |
| plain Adam iterate drift | > 0.1 | **4.791e-01** |
| `norm drift`, permutation modes | == 0 | **0.0000** |

The Adam result is the concrete justification for `IrrepAdam`: coordinate-wise Adam's diagonal
preconditioner does not commute with a rotation inside a 2-D block, and two runs started from
`z` and `ρ(φ)z` drift apart by 0.48.

### Gate G2 — controller frame (the check that could have voided everything)

The symmetry argument assumes **world-frame** action deltas. If the controller consumed
end-effector-frame deltas, a world rotation would act trivially and nothing downstream would
mean anything. The plan's inspection snippet returns nothing useful on this repo, so it was
traced through robosuite directly:

* `set_goal_position(scaled_delta[:3], self.ee_pos)` → `ee_pos + delta`, both world-frame;
* `set_goal_orientation` → `quat2mat(axisangle2quat(delta)) @ current_orientation` —
  **left**-multiplication, i.e. world-frame; conjugating by R_z maps the axis-angle vector
  `w → R_z·w`.

So `(dx,dy)` and `(wx,wy)` are frequency-1 and `dz,wz,grip` invariant — exactly
`SO2ChunkSpec.libero_osc_pose()`. **Premise holds.** Constants confirmed from `osc_pose.json`,
not assumed: `TRANS_GAIN = 0.05 m`, `ROT_GAIN = 0.5 rad`, `control_freq = 20`, control period
`8/20 = 0.4 s`.

### Integration smoke test — 39/40

All arms instantiate, train and sample; the normalizer repair fires; DNO stage 1 runs under
`torch.inference_mode()` (so the stock runner drives it) and stage 2 descends the loss
(202.4 → 153.2); the runner `_target_` swap resolves.

---

## 5. Phase 1 — dataset audit

32,841 chunks of horizon 16. → `output/exp/audit_libero10_v2.{json,txt}`

| quantity | value | meaning |
|---|---|---|
| `W1(heading, uniform)` | **0.2103 rad** | hard floor on angular transport for any marginal-preserving coupling |
| heading `R1` | 0.1511 | headings are non-uniform |
| low-confidence fraction | 0.0004 | essentially every demo chunk has a well-defined heading |

**Coupling preview on real chunks** — the design's central tension, before any training:

| coupling | path len | angle gap | src R1 (0 = undeformed) |
|---|---|---|---|
| A/iid | 14.00 | 1.570 | 0.058 |
| **P2/perm-angle** | 13.66 | **0.257** | **0.060** |
| P3/perm-euclid | 12.86 | 0.712 | 0.052 |
| **P4/rot-heading** | 13.62 | **0.000** | **0.180** |
| P6/perm+rot-group | 12.62 | 0.508 | 0.153 |

P2 lands within 22 % of the theoretical floor while leaving the source marginal at the iid
level; P4 drives the gap to zero but deforms the source. **You can have exact marginals or
maximal transport reduction, not both** — visible in data before a single GPU-hour.

**Gate G3** → `W1` in the moderate band; verdict `matched_prior_recommended` (run P4 **and** P5).
**Gate G5** → does not fire.

### Gate G4 was broken, and the corrected version fires

The original gate reads block energy **after** the normalizer, but `vector_mode='rms'` sets
every vector block to unit per-coordinate RMS — so the normalized energies are equal **by
construction** and the `< 0.05` threshold is unreachable regardless of the data.

| block | **RAW** | normalized |
|---|---|---|
| `vec0 (dx,dy)` | 0.1553 | 0.3835 |
| **`vec1 (wx,wy)`** | **0.0034** | 0.3835 |
| gripper | 0.7354 | 0.1918 |

On raw actions it fires decisively — the rotation block carries **0.34 %** of action energy —
and the normalizer then amplifies it to parity (measured live: `(0,1)→3.11`, `(3,4)→20.80`).

**Gate G4′ decision: option (b), `block_weights=[1.0,0.0]`** on every arm including the control.
It changes only the coupling cost, so P1 stays an exact control. Verified: complex width
32 → 16 while `rotate_chunk` still rotates **both** blocks; heading equivariance 2.38e-07.

---

## 6. Phase 2 — the normalizer control

| arm | config | SR (500-ep ×3) |
|---|---|---|
| P0_baseline | `train_flowpolicy`, untouched | **0.1860** ± 0.0121 |
| P1_norm_only | `mode=iid` + SO(2) normalizer | **0.2013** ± 0.0105 |

**Gate G6 — passes.** Δ = +0.0153, P1 better, so §4's "if P1 is clearly worse" branch never
fired. P0 remains the reference baseline; the SO(2) normalizer carries no success-rate cost.

Both arms report `orbit_steering_gain = −0.00000`, the check §4 demands of P1 — confirming the
metric is calibrated before any coupled arm is measured.

### Gate G15 — the amendment that changed how everything is read

Re-analysed as the **paired** comparison it actually is:

```
paired mean delta  +0.0153   paired stderr ±0.0650   <-- THE RESOLVABLE EFFECT SIZE
t(9) +0.24  p 0.819          sign test p 0.754
mean |per-task delta| 0.1580  reshuffle ratio 10.3x
MDE at p<0.05  0.1470        (3 seeds: 0.0849)
```

The ±0.0121 quoted at G6 was the within-arm repeat stderr, understating uncertainty ~5×.
Corrected: **"the normalizer carries no measurable cost"** is defensible; **"P1 is better"** is
not. And the aggregate hid a **10.3×** larger per-task reshuffle — individual tasks moved by
up to ±0.35 (one task 0.000 → 0.253, another 0.333 → 0.040) while the mean barely moved.

**Consequence adopted for the rest of the project:** report per-task rates, and quote no
aggregate delta without its paired stderr. No effect below ~0.15 is detectable at one seed.

---

## 7. Phase 2.5 — the decisive experiments

Phase 3 (~84 GPU-h) was deferred in favour of two parallel runs answering one question each.

| track | question | readout |
|---|---|---|
| **A** `P4_rot_heading` | does the coupling create steerability on real data? | `orbit_steering_gain`, offline, no rollouts |
| **B** `P1_curve` | what is the right operating point? | SR-vs-epoch curve + working top-k |

Track A used `mode=rot, align=heading, kappa=inf` — the **only** configuration the synthetic
study produced a gain near 1 for.

### 7.1 Gate G13 — CONFIRMED

Stable across four independent reads:

| checkpoint | epoch | gain |
|---|---|---|
| early probe | ~330 | 0.99 |
| probe | 754 | 1.02 |
| probe | 786 | 1.01 |
| **final** | **958** | **1.04** |

### 7.2 Full diagnostics

| metric | P0 @1001 | P1 @1000 | P4 @330 | **P4 @958** |
|---|---|---|---|---|
| `orbit_steering_gain` | −0.00000 | −0.00000 | 0.99000 | **1.04000** |
| `orbit_phase_consistency` | 0.05259 | 0.05467 | 0.83646 | 0.68272 |
| `src_orbit_eq_rel` | 0.56720 | 1.10952 | 0.88675 | 0.90312 |
| `heading_MAE` | 0.39026 | 0.62267 | 1.37418 | 1.15597 |
| `action_mse_N1` | 0.03824 | 0.02731 | 0.06812 | 0.06914 |
| `action_mse_N10` | 0.04476 | 0.03045 | 0.08239 | 0.07193 |
| `few_step_gap_N1` | 0.19540 | **0.20833** | 0.39943 | **0.29034** |
| `few_step_gap_N4` | 0.08752 | **0.07914** | 0.13307 | **0.10434** |
| `straightness` | 1.74761 | **4.07690** | 6.49047 | **4.77540** |

> **Comparability.** `straightness`, `few_step_gap`, `src_orbit_eq_rel`, `heading_MAE` and the
> training losses are computed in **normalized** space, and P0's normalizer differs from
> P1/P4's (rotation block scaled 20.8× vs ~1×). **P0 is not comparable to P1/P4 on these.**
> Only success rate and raw-space `action_mse_*` compare across all three. The transport claim
> must therefore be judged **P4 vs P1**, never P4 vs P0.

**Transport claim REFUTED.** Epoch-matched, P4 is worse than P1 on `straightness` (4.78 vs
4.08), `few_step_gap_N1` (0.290 vs 0.208) and `few_step_gap_N4` (0.104 vs 0.079). P4's
transport metrics improved substantially from epoch 330 → 958 (straightness 6.49 → 4.78), so it
was still converging — but it never overtook the control. A clean negative, not undertraining.

**Delegation trade-off confirmed in both directions at once**, which is what makes it credible:
`src_orbit_eq_rel` **falls** (1.110 → 0.903, predicted) while `heading_MAE` (0.623 → 1.156) and
`action_mse_N10` (0.030 → 0.072) **worsen**.

### 7.3 The steering-probe figure

`output/exp/probe/steering_probe.svg`. Two scalars cannot distinguish "the output heading tracks
the source rotation" from "the residuals average to a slope of 1"; the scatter can.

For a frozen policy, fixed observations and fixed noise, the source is rotated over 24 angles
and `Δθ(φ) = wrap(heading(F(ρ(φ)z|o)) − heading(F(z|o)))` recorded — 1,536 points per arm.
The control's cloud is pinned at Δθ ≈ 0 across every φ; the coupled arm's traces the identity.

| arm | epoch | gain | consistency | low-conf |
|---|---|---|---|---|
| P1 iid (control) | 1000 | **−0.000** | 0.031 | 0.00 % |
| P4 rot/heading | 958 | **+1.010** | 0.624 | 0.07 % |
| P4 + matched prior | 958 | **+1.010** | 0.673 | 0.20 % |

Three things forced identical across arms: observations (`obs_seed=1234`), source noise
(`z_seed=5678`), and a **forced readout spec** (`translation_only`) rather than each
checkpoint's own — P4 was trained after the G4′ decision and P1 before it, so under their own
specs the two Δθ are different quantities. The renderer refuses mismatched specs. Forcing the
readout is safe because `rotate_chunk` acts on `vector_blocks` and rotates every block
regardless of the weights: it changes what is *measured*, never what is *done* to the noise.

The matched-prior arm reading the same gain (+1.010) confirms the probe measures the **map**,
not the source law.

### 7.4 Gate G14 — the operating-point curve

16 points, and the first time top-k selection ever functioned in this project (`lazy_eval: true`
had made it impossible, so **every earlier number came from an unselected final-epoch
checkpoint**).

```
epoch     0    50   100   150   200   250   300 │  350   400   450   500   550   600   650   700   750
SR     0.000 0.040 0.260 0.220 0.140 0.180 0.220│ 0.400 0.360 0.340 0.360 0.400 0.400 0.340 0.360 0.300
                   └──── 51/250 = 0.204 ───────┘└──────────── 0.371, 9 points ────────────┘
                                     Fisher exact p = 7.5e-06
```

A two-level step: ~0.20 through epoch 300, a step up around 300–350, a stable ~0.37 plateau
after. Success at the same configuration is **~1.8× what Phase 2 reported**.

Track B was stopped at epoch 796 by request, leaving 800→1001 unmeasured.

---

## 8. Complete success-rate table

| arm | protocol | epoch | **SR** | stderr |
|---|---|---|---|---|
| P0_baseline | 500-ep ×3 | 1000 | 0.1860 | ±0.0121 |
| P1_norm_only | 500-ep ×3 | 1000 | 0.2013 | ±0.0105 |
| P1_norm_only | 50-ep | 1000 | 0.2200 | ±0.0586 |
| P4_rot_heading | 50-ep | 800 | 0.0000 | — |
| P4_rot_heading | 50-ep | 958 | 0.0200 | ±0.0198 |
| **P4_rot_heading** | **500-ep** | **958** | **0.0080** | **±0.0040** |
| P5_rot_matched | 50-ep | 800 | 0.0000 | — |
| P4 + stage-1 DNO K=32 | 50-ep | 800 | 0.0200 | ±0.0198 |

### The headline comparison

```
P1_iid 0.2013   vs   P4_orbit 0.0080     (500-episode protocol, both ~1000 epochs)
paired mean delta  -0.1933 ± 0.0540
t(9) = -3.58   p = 0.006   sign test p = 0.002    SIGNIFICANT
reshuffle ratio 1.0x   -- P4 worse on ALL ten tasks; the aggregate is faithful
```

The **first statistically significant result the project produced**, and it clears the ±0.065
resolution bar G15 established. Contrast the reshuffle ratio with P0-vs-P1's 10.3×.

**`heading_MAE` = 1.156 rad = 66°** explains it: a policy departing ~66° from the correct
heading fails nearly everything. **Gain 1.04 and SR 0.008 are the same fact seen twice.**

> **Reading caution.** Everything at 50 episodes has stderr ±0.02–0.06, so 0.0000 / 0.0200 /
> 0.0200 are the *same measurement* statistically and cannot be ranked. In particular the DNO
> row (0.0200) vs DNO-off at the same epoch (0.0000) is 1 success vs 0 of 50, Fisher p ≈ 1.0 —
> **the DNO success gain is not measurable at this sample size.** The real evidence about DNO is
> its internal diagnostics, below.

---

## 9. Two explanations eliminated

### 9.1 Protocol — eliminated

`P1_norm_only`, same checkpoint: **0.2013** at 500 episodes, **0.2200** at 50. Episode-set
difficulty was the leading hypothesis for the gap between `P1_norm_only` (0.20) and `P1_curve`
(0.37). It explains **nothing**. Phase 2's absolute numbers are sound.

Remaining candidates for that gap: run-to-run variance (rollouts consume the global RNG, so
`P1_curve` is a second sample, not a reproduction), or a real peak-then-decline —
`P1_curve`'s last point (0.300 at 750, lowest of nine) alongside `P1_norm_only`'s 0.20–0.22 at
1001 favours the latter, which would make **1001 epochs past the peak**. Stopping Track B at
796 means this cannot now be closed from within one run.

### 9.2 The prior shift — eliminated

**P5 (same weights, `--inference-prior matched`) = 0.0000.** Gate G9 predicted "P5 ≥ P4, margin
growing with `W1`". The histogram was healthy — 1,153,152 samples over 128/128 bins — so this is
a real null, not a misconfiguration.

The reason is a **marginal-versus-conditional** distinction. `MatchedHeadingPrior` re-imposes the
*marginal* heading law, but the coupling matched each source's heading to *its own target's* — a
**per-sample conditional** relation. At inference the target is unknown, so sampling the correct
marginal still hands the policy a heading uncorrelated with the one *this observation* needs. A
policy that has delegated heading to the noise does not need a correctly-*distributed* heading;
it needs the correct heading **for this observation**, which no fixed prior can supply.

**This means design-doc §5's obstruction is not what limits `rot` couplings in practice** — a
genuine contribution, since §5 is where the design expected the difficulty to lie.

---

## 10. Why the DNO claim could not be tested

Stage-1 orbit DNO, K=32, Tier-0 objective (`smoothness 1, seam 2, step_limit 5, descent 10`):

```
orbit_loss_spread   1.749            <- the objective DISCRIMINATES angles strongly
orbit_loss  mean 0.815 -> best 0.205     (75 % reduction, every cycle)
z_norm_ratio        1.0000           <- pure rotation, norm exactly preserved
wall_time p50/p95   31 / 36 ms       <- REAL-TIME, 13x inside the 0.4 s control period
success rate        0.0200 (1/50)
```

**§8.4's designated failure mode does not apply.** `orbit_loss_spread ≈ 0` would have meant "a
task-loss problem, not a coupling problem" — but the spread is large and the search reliably
finds the minimum. The machinery works perfectly and success does not follow.

**Therefore the Tier-0 task loss is not a proxy for success on this benchmark.** Its terms —
smoothness, seam, step-limit, descent — are all about *dynamic feasibility*; **none contains
directional information**. Rotating the chunk to minimise them selects a heading that is smooth,
safe and task-arbitrary. This is not a weighting bug: Tier 0 is *defined* as "no privileged
state, no learning", which excludes precisely what the orbit search needs. The plan's own ladder
names the remedy — **Tier 2**, a learned success classifier or Q-function over `(obs, chunk)`.

**The DNO half is not refuted; it is untested, and Tier 0 cannot test it.**

### The `--table-z` problem, found and fixed

The plan's Phase-6 clearance term uses a scalar table plane. Measured eef height on
LIBERO-10 is **bimodal with an empty gap at 0.815–0.889**, splitting exactly five tasks per mode:

```
low  group  eef z in [0.446, 0.781]   (all five LIVING_ROOM tasks)
high group  eef z in [0.910, 1.332]   (four KITCHEN + STUDY)
```

The plan's placeholder `TABLE_Z=0.82` sits **above the maximum height of every low-group task** —
it would penalise *every demonstrated step* of half the benchmark (48.4 % of all steps). Pushing
it below the global p01 (0.448) makes it safe but identically zero for the entire high group.
**No scalar plane is both safe and active.** Replaced with `descent_limit`, a *relative* floor
penalising descent more than `max_descent` below the **current** height — verified to give an
identical penalty at eef z = 0.50, 0.95 and 1.20.

---

## 11. Methodological findings

Arguably the most transferable output: five defects that fail **silently**.

1. **Gate G4 could never fire** — measured post-normalizer, where `rms` equalises blocks by
   construction. Fixed to read raw energy.
2. **`compose_libero_multitask_dataset.py` fails with exit code 0** — shells out via
   `os.system` with a bare `python`, which found no `click`; `os.system` does not propagate the
   child's status. Reported success, produced no zarr.
3. **`policy.coupling.kappa=.inf` crashes** — Hydra's CLI grammar delivers the *string*
   `'.inf'` and `build_coupling` raises `ValueError`.
4. **`lazy_eval: true` silently disables top-k selection** — no env runner ⇒ no
   `mean_success_rate` ⇒ `TopKCheckpointManager` never fires ⇒ only `latest.ckpt` exists. Every
   pre-Phase-2.5 number came from an unselected final-epoch checkpoint.
5. **`report_coupling_diagnostics.py` warned falsely** on any orbit-config checkpoint, comparing
   against a literal string instead of testing `coupling['mode']`.

Plus one resource finding: **`lazy_eval=false` forks 10 MuJoCo workers**, which pushed two
concurrent tracks past 62 GB and OOM-killed one. Resolved without serialising (~37 h instead of
~62 h) by halving env forks and giving the *non-decisive* track `oom_score_adj=800`, so the OOM
killer would take it rather than the decisive one.

### Errors made and corrected

* **A G14 verdict was recorded and retracted.** At epoch 300, five points gave χ² = 2.56
  (p = 0.63) with scatter *below* binomial noise, and this was called "ceiling at ~0.20" — with
  two conclusions drawn from it (budget closed, Phase 3 unlicensed). Epoch 350 returned 0.400, a
  ~1-in-800 draw from that plateau, and both were withdrawn. **The error was treating "consistent
  with constant" as "is constant": a χ² consistency test can only fail to reject.** An
  intermediate note also compared the maximum of five points against the mean of those same five
  (selection bias) using the stderr of the mean where per-point binomial noise was correct.
* **Stopping Track B was proposed on the strength of that flat reading** — and the very next
  point it produced overturned the reading. It was kept running.

---

## 12. Limitations

1. **One seed.** Plan §5.6 sets three as the floor. Everything here is seed 42.
2. **No `perm` arm.** P2/P3 were never trained (Phase 3 deferred), so "steerability comes from
   *applying the rotation*" versus "any coupling creates the channel" rests on the design
   argument, not data. This is the cheapest missing control.
3. **`block_weights=[1.0,0.0]` on every arm**, so translation-only coupling and coupling in
   general are not separated.
4. **Maximum-strength delegation only** — `kappa=inf`, `coupling_prob=1.0`. The κ knob that
   trades delegation against open-loop competence was never swept.
5. **800→1001 of the G14 curve unmeasured**, so peak-versus-plateau is unresolved.
6. **Noise-space equivariance only.** With a fixed RGB camera the full property (1) is not
   evaluable — the observation is invariant, not equivariant.

---

## 13. Recommendations

1. **The steering channel is the publishable result** — novel, cleanly measured, controlled at
   exactly 0.00, real-time searchable, with the delegation cost quantified at −0.1933 ± 0.0540
   rather than hidden. A mechanism paper with an honest trade-off, not a benchmark win.
2. **Do not run Phase 3 as designed.** Comparing arms on success rate is pointless with the
   rotation arm at 0.008, and the transport claim it was meant to test is already refuted
   epoch-matched.
3. **The κ sweep is the one cheap experiment that could change the picture** — one training run
   per κ, directly targeting whether *partial* steerability can coexist with usable success.
4. **Add a `perm` arm** if any further training is done: it is the missing control for the
   central mechanism claim.
5. **Testing DNO properly needs a Tier-2 objective** — a different project from the one scoped.

---

## 14. Artifacts

```
RUNLOG.md                                 every command, gate verdict, retraction
output/exp/PHASE_2.5_FINDINGS.md          phase findings
orbit_fm_dno/PROGRESS_SUMMARY.md          Phases 0-2
orbit_fm_dno/PHASE_2.5_PROGRESS.md        interim snapshot

output/exp/paired_P1_vs_P4.json           HEADLINE: -0.1933, p=0.006
output/exp/paired_P0_vs_P1.json           G15 resolution +-0.065
output/exp/audit_libero10_v2.{json,txt}   G4' raw vs normalized energy
output/exp/probe/steering_probe.svg       THE FIGURE (+ .png, .caption.md)
output/exp/probe/probe_{P1,P4,P4_matched}.json     raw (phi, dtheta) points

output/exp/P0_baseline/seed42/            SR 0.1860, diag.json
output/exp/P1_norm_only/seed42/           SR 0.2013 / 0.2200, diag.json
output/exp/P4_rot_heading/seed42/
    diag_final.json                       gain 1.04
    eval_final_n500/                      SR 0.0080  <- headline
    dno_s1_K32_n50/                       DNO: spread 1.749, 31 ms
    checkpoints/latest.ckpt               epoch 958
output/exp/P5_rot_matched/seed42/         SR 0.0000
output/exp/P1_curve/seed42/               G14 curve + 5 top-k checkpoints

new tooling (all new files; no repo file modified):
    oat/symmetry/steering_probe.py        points-returning probe
    scripts/steering_probe_measure.py     measure stage
    scripts/steering_probe_render.py      vector renderer
    scripts/paired_task_compare.py        paired statistics
```

**Gate G0 held throughout**: `git diff --name-only 15b93ee` over every §0.1 protected path
returns empty.
