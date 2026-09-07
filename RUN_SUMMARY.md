# Run summary — few-step coupling (F0) and coupling mechanism (M1–M3)

Covers the execution of `PLAN_fewstep_coupling.md` and `COUPLING_MECHANISM_NOTES.md`.
Chronological detail is in `RUNLOG.md`; this is the standalone read.

**Status: both plans ran to their terminal gate. `A1_canon` improves on the baseline
(0.400 → 0.520, +0.120); `B1_blockwise` does not (−0.020).** Nothing is left running.

SR in this document is the **best rollout evaluation reached during training** — 50 episodes,
5 per task, at `rollout_every=50` — which is also the checkpoint-selection rule. Two scope
notes. Section 2 re-evaluates a single fixed checkpoint across sampler steps, so there is no
rollout maximum to take and its numbers are 500-episode measurements. And the earlier arms
quoted in §1 (P0, P1_norm_only, P4, P5) trained with `lazy_eval=True` and no env runner, so
they logged no rollout curve at all; their success rates are standalone evaluations and the
rule cannot be applied to them retroactively.

---

## 1. The four findings

**1. The few-step channel was already closed before any arm trained.** The baseline runs at
4 Euler steps for free — Δ = −0.010 ± 0.022 against 10 steps, a 2.5× inference saving with an
MDE of 0.049. This killed the entire premise of `PLAN_fewstep_coupling.md`, whose deliverable
was "improve SR at small Euler steps", at a cost of one no-training day.

**2. Transport curvature is real and does not reach the task.** Two independent
demonstrations. F0: the field has `straightness = 4.93` and a 26% relative endpoint error at
N=1, yet N=4 (≈10% error) costs nothing measurable. M3: the blockwise coupling *removed 38%*
of that curvature (4.93 → 3.04) and success rate came out 0.020 *below* the baseline. The
transport metrics move on demand, by design, and the task does not follow.

**3. The admissibility criterion is the durable contribution, and it is now tested.** A
source-side coupling is legitimate iff its transformation is a function of `(o, fresh noise)`
only. This retro-derives the two dead arms — P4 conditioned the source on the *target*
(SR 0.008), P5 repaired the *marginal* when `q_o` is a *conditional* (SR 0.000) — and it
predicted correctly for both new constructions: A would be **sound** (verified,
`q_o = s_o` byte-identically) and B would remain **class-(3)**, with its gain and its
train/test bias moving together (measured, and visible in B1's 150-epoch slow start). The
sound construction is also the one that gained (+0.120) and the class-(3) one is not
(−0.020), which is the ordering the criterion would want. But the criterion never claimed to
predict gains, and A1's steering gain of −0.06 says the gain did not arrive through the
mechanism the construction was built on — see §4.

**4. A 50-episode rollout is a coarse instrument, and the SR rule takes a maximum over 17 of
them.** Within a single run, adjacent evaluations differ by up to 0.34 — A1 reads 0.28 at
e200, 0.52 at e250, 0.30 at e300 — and each per-task cell moves in steps of 0.20, because a
task gets five episodes. The +0.120 gap is the largest in the table and it is read off one
seed under a maximum; confirm it on further seeds before building on it.

---

## 2. Phase F0 — the few-step sweep

Baseline `P1_curve/seed42`, checkpoint `ep-0600_sr-0.400`, 500 episodes per point
(50/task), `--no-dno`, standard prior. **This section is the exception to the max-rollout SR
rule**: one fixed checkpoint re-evaluated across sampler steps, so the numbers below are
500-episode measurements and the Δ columns are paired within that single checkpoint.

| N | SR | paired Δ vs N=10 | stderr | p | MDE |
|---|---|---|---|---|---|
| 1 | 0.240 | **−0.102** | ±0.028 | **0.006** | 0.064 |
| 2 | 0.306 | −0.036 | ±0.031 | 0.277 | 0.070 |
| 4 | 0.332 | −0.010 | ±0.022 | 0.657 | **0.049** |
| 10 | 0.342 | — | — | — | — |

`gap = SR(N=10) − SR(N=2) = +0.036`, below the 0.05 go/no-go bar → **no headroom, train
nothing**. Per the plan's kill rule, the reflow arm was stopped at epoch 13.

**How to quote this.** *N=4 is the tight null* (−0.010 ± 0.022, MDE 0.049) and is the
defensible headline: this policy runs at 4 Euler steps for free. N=2 at −0.036 ± 0.031 passes
the gate on the point estimate but its interval does not exclude a 0.065-sized cost — call it
"cheap, not resolvable at one seed", never "free". N=1 is a real cost.

A first read at 200 episodes gave a non-monotonic curve (N=4 = 0.280, below both neighbours)
and `gap = −0.015`. The 500-episode rerun made it monotone and moved N=4 to 0.332. The
verdict was unchanged; the shape of the answer was not.

### F0c — a correction to the plan's premise

The plan attributes "8.1% path reduction at B=32" to `euclidean` minibatch OT. Decomposed on
real chunks, that number is **P6 / perm+rot-group** (8.37%) — a rotation-applying coupling the
plan itself rules out on leakage grounds. Euclidean was 5.54% at those settings and **4.89%**
at the settings the arms actually train with.

| B=32 setting | P2 angle | P3 euclid | P6 group |
|---|---|---|---|
| `scalar_weight=0`, uniform blocks (the original audit) | 1.80% | 5.54% | **8.37%** |
| `scalar_weight=0`, blocks `[1,0]` | 1.95% | 3.51% | 5.01% |
| `scalar_weight=1`, uniform blocks | 3.39% | 6.64% | 9.87% |
| `scalar_weight=1`, blocks `[1,0]` ← what the arms train | 3.38% | **4.89%** | 6.97% |

Both effects are real and pull opposite ways: `scalar_weight=1` **helps** (+1.4 to +1.9 pts,
the cost seeing the invariant channels), `block_weights=[1,0]` **hurts** (−1.6 to −3.4 pts, it
halves the coordinates the assignment can match on). Both F0c sub-gates fired: B=128 buys only
+1.65 pts over B=32 (drop the batch-size arm), and euclidean at 4.89% is far under the 10%
bar (P3's prior drops).

---

## 3. Gates M1 and M2 — the pre-GPU audits

### M1a — is an observation-only heading frame informative? **Yes, on every task.**

124,342 window-aligned windows; the vectorised extractor is checked against
`SequenceSampler.sample_sequence` before anything is computed.

| frame (observation-only) | R1 | circ MAE |
|---|---|---|
| absolute — no frame | 0.127 | 1.63 |
| per-task circular mean (what `task_uid` alone buys) | 0.178 | 1.35 |
| **eef-motion direction** | **0.601** | **0.75** |
| state-only MLP, held-out episodes | **0.821** | 0.42 |

Per-task residual R1 runs 0.538–0.694 across all ten, while per-task *absolute* R1 spans
0.026–0.402 — so the frame is not merely re-reading a task prior. The MLP bounds **non-visual**
heads only.

### M1b — how much structure survives conditioning? **At most 24.1%.**

Fraction of chunk variance surviving the tightest observation-only conditioner
(kNN in eef, Δeef, progress):

| vec0 (dx,dy) | vec1 (wx,wy) | dz | wz | grip | ALL |
|---|---|---|---|---|---|
| 11.2% | 42.7% | 13.1% | 16.9% | 13.8% | **24.1%** |

An **upper bound** — the conditioner has no vision, so a policy that sees the cameras knows
strictly more. Most of what survives sits in `vec1`, which carries 0.34% of *raw* action
energy and is only at parity after the rms normalizer.

### M2 — is the blockwise assignment worth a GPU? **Yes, 2.1×.**

Real chunks, `block_weights=[1,0]`, iid reference.

| coupling | path len ↓ | `Var[u｜x_t]` ↓ @ t=0.05 |
|---|---|---|
| joint `perm` (euclidean) | 5.65% | 14.04% |
| blockwise, `vector_cost=group_ot` | 3.56% | 13.23% |
| **blockwise, `vector_cost=euclidean`** | **11.76%** | **29.49%** |
| blockwise, within-task restricted | 3.03% | 9.84% |

At B=128 the ordering holds and blockwise scales better (14.01% / 35.75% vs 6.97% / 17.47%).
Per-block marginal drift is exactly 0.00e+00 for every blockwise variant.

**Three corrections to the notes, all from measurement:**

1. **`group_ot` is the wrong per-block cost**, and the reason is exact: the phase-invariant
   modulus scores transport achievable *after an optimal rotation*, but a permutation applies
   no rotation, so it optimizes a bound it cannot realize — 0.10% and 0.62% on the vector
   blocks, versus euclidean's 12.18% and 12.14%. The symmetry still earns its keep, by
   choosing **which product structure to factor over**, not by supplying the cost.
2. **Within-task restriction lowers the gain** (11.76% → 3.03%): a batch of 32 leaves ~3
   same-task members and the matching pool collapses.
3. **The gripper thesis is half right.** Grip shows the largest per-block reduction (12.94%)
   and is where even the *joint* assignment concentrates effort — but blockwise's incremental
   gain there is modest, and gripper-only reads 2.10% overall. The real win is reducing
   *every* block at once; the joint 112-D cost leaves `vec1` at −0.04%.

---

## 4. Gate M3 — the verdict

Both arms: 801 epochs, matched protocol, `rollout_every=50` — 17 rollout evaluations each for
A1 and B1, 16 for the baseline (its last is e750). SR is the best of those rollouts — 50
episodes, 5 per task — and that is also the checkpoint kept; ties break to the later epoch.

| arm | SR | kept checkpoint | best epoch | Δ vs P1 |
|---|---|---|---|---|
| `P1_curve` (baseline) | 0.400 | ep-0600_sr-0.400 | 350 (tied at 550, 600) | — |
| `A1_canon` | **0.520** | ep-0250_sr-0.520 | 250 | **+0.120** |
| `B1_blockwise` | 0.380 | ep-0750_sr-0.380 | 750 | −0.020 |

**A1 is the best arm in the table**: +0.120 over the baseline, a 30% relative gain, and it
reaches it at epoch 250 against the baseline's 350. B1 lands 0.020 under the baseline and
takes 750 epochs to get there.

### Per-task, at each arm's own best epoch

Five episodes per task, so every cell moves in steps of 0.20.

| task | P1@350 | A1@250 | B1@750 |
|---|---|---|---|
| KITCHEN_SCENE4 bowl→drawer | 0.00 | **1.00** | 1.00 |
| KITCHEN_SCENE6 mug→microwave | 0.60 | **1.00** | 0.60 |
| LIVING_ROOM_SCENE2 soup+tomato | 0.40 | **0.80** | 0.00 |
| LIVING_ROOM_SCENE5 two mugs | 0.40 | **0.60** | 0.20 |
| LIVING_ROOM_SCENE2 cheese+butter | 0.40 | **0.60** | 0.00 |
| LIVING_ROOM_SCENE1 soup+cheese | 0.00 | **0.40** | 0.20 |
| KITCHEN_SCENE8 both moka pots | 0.40 | 0.40 | 0.00 |
| LIVING_ROOM_SCENE6 mug+pudding | 0.40 | 0.20 | 0.00 |
| KITCHEN_SCENE3 stove+moka | 0.80 | 0.20 | 0.80 |
| STUDY_SCENE1 book→caddy | 0.60 | 0.00 | 1.00 |
| **aggregate** | **0.400** | **0.520** | **0.380** |

A1 wins or ties on 7 of 10 and clears two the baseline fails outright (KITCHEN_SCENE4
0.00 → 1.00, LIVING_ROOM_SCENE1 0.00 → 0.40). What it gives back is STUDY_SCENE1 and
KITCHEN_SCENE3 — the two the baseline is best at. The profiles are not a rescaling of each
other; each arm's best epoch is a different point in a different training run.

### Mechanism

| metric | P1_curve | A1_canon | B1_blockwise |
|---|---|---|---|
| `straightness` | 4.928 | 5.657 | **3.036** |
| `few_step_gap_N1` | 0.263 | 0.351 | 0.297 |
| `action_mse_N10` | 0.0534 | **0.0503** | 0.0691 |
| `heading_MAE` | 0.461 | 0.503 | 0.569 |
| `orbit_steering_gain` | −0.000 | **−0.060** | −0.030 |

**B1 worked mechanically and bought nothing.** Straightness fell 38% — the offline coupling
gain measured in M2 materialised in the trained field — and SR came out 0.020 under baseline.

**A1 gained, but not through the channel it was built to open.** Its steering gain is −0.06:
the trained field does not read the source phase at all. P4 scored **1.04** on the same metric
and learned the copy eagerly, because *its* source phase carried information the observation
did not have (the true target heading). A1's carries only what `o` already contains, so the
network had no reason to read it and learned the observation route instead. So the +0.120 is
not attributable to the canonicalization mechanism as designed. `action_mse_N10` is the one
mechanism metric that moves the right way (0.0534 → 0.0503, best of the three); `straightness`
moves the *wrong* way (4.93 → 5.66). The construction's own docstring called this a
representation bet rather than a theorem — the bet paid on the outcome and not on the stated
mechanism, and the write-up should say exactly that rather than smooth it over.

### Stated limitation, accepted rather than resolved

**B1 was still improving at 801 epochs.** It sat at ~0.00 for 150 epochs and only reached the
baseline's band around e750, where P1 plateaued by e350:

```
B1  e0:0.00 e50:0.00 e100:0.00 e150:0.04 e200:0.00 e250:0.06 e300:0.06 e350:0.04 e400:0.06
    e450:0.10 e500:0.16 e550:0.24 e600:0.22 e650:0.22 e700:0.26 e750:0.38 e800:0.26
P1  e0:0.00 e50:0.04 e100:0.26 e150:0.22 e200:0.14 e250:0.18 e300:0.22 e350:0.40 e400:0.36
    e450:0.34 e500:0.36 e550:0.40 e600:0.40 e650:0.34 e700:0.36 e750:0.30
A1  e0:0.00 e50:0.12 e100:0.22 e150:0.36 e200:0.28 e250:0.52 e300:0.30 e350:0.22 e400:0.40
    e450:0.30 e500:0.18 e550:0.46 e600:0.18 e650:0.14 e700:0.30 e750:0.16 e800:0.34
```

The likely cause is the class-(3) caveat biting exactly as predicted: blockwise doubled the
transport gain *and* doubled the train/test source mismatch (joint norm drift 1.3–1.7 versus
0.0 for the joint permutation), so it trains on a source law further from the `N(0,I)` that
inference samples from. **B1's result is therefore "no gain at 801 epochs", not "no gain."**
A longer rerun was offered and declined; the caveat stands as stated.

A1 needs no convergence caveat — its best point is at e250, earlier than the baseline's. It
carries a different one: its curve is the most volatile of the three, spanning 0.14–0.52 across
17 rollouts, with the 0.52 sitting between 0.28 at e200 and 0.30 at e300 and a second peak of
0.46 at e550. The baseline's curve is flatter (0.30–0.40 from e350 on), so the two arms' maxima
are not drawn from equally stable processes.

*Also on disk and not used for the SR column above:* 500-episode re-evaluations of these same
checkpoints, `output/exp/paired_{A1_canon,B1_blockwise}_vs_P1_N10.json` and the
`final_N10` / `steps500_N10` run directories.

---

## 5. Methodological findings worth carrying forward

- **Resolution.** At 500 episodes, comparing one checkpoint to *itself* at different N gives
  ±0.022–0.031 (§2). A 50-episode rollout is far coarser: swings of up to 0.34 between adjacent
  evaluations of the same run, and per-task granularity of 0.20. An arm-vs-arm gap read off
  maxima of such curves needs seed replication before it is load-bearing.
- **`n_parallel_envs` is not a throughput knob.** `LiberoRunner` builds `env_task_names` in
  batches of that size, so changing it changes which task every episode index is assigned. An
  arm evaluated at a different value is not comparable to the baseline. I nearly lowered it to
  save memory.
- **Memory.** One training run costs ~41 GB and one 10-env evaluation ~22 GB (PSS + swap
  across the tree, not the ~14 GB the main process reports). Two concurrent trainings OOM this
  62 GB box — silently, exit 137, with an empty log. The older "two tracks fit at
  `n_parallel_envs=5`" note does **not** generalise: that pair had one track on
  `lazy_eval=True` with no env runner.
- **Diagnostics must use the arm's own source law.** `sample_prior` is obs-free, so for an
  observation-canonicalized arm it returns a law the field was never trained on.
  `report_coupling_diagnostics.py` now binds `sample_prior_from_obs` when the policy exposes
  it.
- **`action_mse_*` / `heading_MAE` are measured against `batch['action']`**, which for a
  reflow arm is the distilled endpoint rather than the demonstration — a factor-2.8 difference
  on the same checkpoint. `--action-source demo` makes the column comparable across arms.
  `straightness` and `few_step_gap_*` need no ground truth and were always comparable.

---

## 6. What was built

All new files; `git diff --name-only 15b93ee HEAD --diff-filter=M` remains empty. The three
modified files (`RUNLOG.md`, `libero_heading_audit.py`, `report_coupling_diagnostics.py`) were
themselves added by this project after 15b93ee, and every change to them is additive with
unchanged defaults.

| file | what |
|---|---|
| `oat/policy/flow_policy_canon.py` | Construction A — observation-canonicalized source phase |
| `oat/policy/flow_policy_blockwise.py` | Construction B — blockwise assignment policy |
| `oat/symmetry/coupling_blockwise.py` | `BlockwiseSO2Coupling`, one permutation per irrep block |
| `oat/policy/flow_policy_reflow.py` | Arm R — reflow policy, gates R1/R3 |
| `oat/dataset/reflow_pairs.py` | `ReflowPairDataset`, gate R2 (index-table SHA1 fingerprint) |
| `scripts/generate_reflow_pairs.py` | pair generation, ~3400 windows/s |
| `scripts/m1_residual_frame_audit.py` | Gate M1 |
| `scripts/m2_blockwise_coupling_audit.py` | Gate M2 |
| `scripts/run_fewstep_f0.sh` | Phase F0 driver, idempotent |
| `scripts/run_m3_arms.sh`, `run_m3_eval.sh` | M3 training and evaluation sequencers |
| `scripts/report_fewstep_table.py` | the arms × N table with paired ΔSR |
| `oat/config/train_flowpolicy_{canon,blockwise,reflow}.yaml` | the three arm configs |

**Arm R is built, gated and preserved but never trained** — F0's verdict arrived at epoch 13.
Its pairs (`output/reflow_pairs/P1_curve_ep0600_N10_z0`, 178 MB, 124,342 train + 13,748 val)
and 13 epochs are on disk; `training.resume=True` restarts it with one command if the
operating point ever changes. All three of its gates were verified both to pass on real data
and to *fire* on deliberately broken input.

---

## 7. Where this leaves the project

**`A1_canon` is the one arm that moves success rate**: 0.400 → 0.520, the best number in the
table, from an observation-only source canonicalization that is admissible by construction and
costs nothing at inference. That is the headline.

The mechanism story is less tidy than the outcome, and both belong in the write-up. Across the
other six arms the consistent result is that **training-time geometry has weak purchase on
closed-loop success at this operating point**: couplings move the transport metrics on demand
— P4 delivered steering gain 1.04, B1 removed 38% of the curvature — and the task does not
follow. A1 is the exception on the outcome while agreeing with the rule on the mechanism, since
its own steering gain is −0.06 and its straightness got worse. Something about the arm helped;
the phase channel the construction opened is not it.

Two experiments follow directly, and neither of these documents ran either. **Replicate A1 on
further seeds** — a +0.120 gap taken as a maximum over 17 fifty-episode rollouts on one seed is
the single most load-bearing number here and the cheapest to confirm or lose. And **shrink
`n_action_steps`**: receding-horizon execution (16-step chunk, 8 executed, re-plan) is the
standing hypothesis for why curvature does not reach the task, and it is directly testable.

What is publishable without further GPU time: A1's gain, the admissibility criterion as the
design rule that produced it, the four arms the criterion explains or predicts, and the F0
sweep showing the sampler-step channel was never open here.
