# RUNLOG — Orbit-Aligned Coupling + Orbit DNO on LIBERO-10

Per `orbit_fm_dno/EXPERIMENT_PLAN.md` §0.2. One entry per command: timestamp, phase,
exact command, wall-clock, outcome, gate verdict.

Environment for every entry below:

```bash
PY=/home/haotian/miniforge3/envs/oat/bin/python     # torch 2.10.0+cu128, CUDA available
export EPOCHS=1001
export NUM_DEMO=500
export SEEDS="42 43 44"
export EXP_ROOT=output/exp
export MUJOCO_GL=egl
```

Hardware: 2x NVIDIA RTX 4090 (24 GB each). Repo: `past2next_clean` @ `15b93ee`, branch `main`.

---

## 2026-08-21 — Integration (file manifest + Phase 0)

### Manifest copy (§1)

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:24 | §1 | `cp orbit_fm_dno/{oat,scripts}/** -> ./{oat,scripts}/**` (16 files) | <1s | OK — all 16 byte-identical to the package (`cmp -s`), 0 collisions with existing files, 0 `__init__.py` created |

Files placed: `oat/symmetry/{so2_chunk,coupling,normalizer,metrics}.py`,
`oat/policy/{flow_policy_orbit,dno_policy}.py`, `oat/env_runner/libero_dno_runner.py`,
`oat/dno/{irrep_adam,task_losses,orbit_dno}.py`, `oat/config/train_flowpolicy_orbit.yaml`,
`scripts/{validate_so2_coupling,libero_heading_audit,report_coupling_diagnostics,eval_orbit_dno_libero,aggregate_experiment_results}.py`.

**Gate G0 — PASS.** `git status --porcelain` returns 12 lines, all `??`. Zero ` M` / `M `.
No existing file modified; the repository as cloned is byte-identical.

### Phase 0.1 — environment (§2.1)

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:23 | §2.1 | `$PY -c "import torch, oat, libero; ..."` | 6s | `2.10.0+cu128  True`; `oat` resolves to `past2next_clean/oat` |
| 17:23 | §2.1 | `$PY -c "import scipy; ..."` | 1s | `scipy 1.15.3` — Hungarian solver available, no need for `assignment=circular_sort` |

### Phase 0.3 — self-test (§2.3)

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:25 | §2.3 | `$PY scripts/validate_so2_coupling.py --quick 2>&1 \| tee output/logs/selftest.txt` | 1m10s | all 6 tests ran |

**Gate G1 — PASS.** Measured against the plan's three criteria and
`orbit_fm_dno/VALIDATION_RESULTS.txt`:

| criterion | required | measured |
|---|---|---|
| test 1 — `norm drift` on permutation modes, both heading regimes | `== 0` | `0.0000` (all modes, both regimes) |
| test 1 — `rot` modes flagged `DEFORMED` under `vonmises`, not under `uniform` | yes | yes: C1 0.2655 / C2 0.5333 / C3 0.5333 / C4 0.3724 / F 0.2867 under `vonmises`; all ≈0.04–0.07 under `uniform` |
| test 3 — SO(2) block normalizer equivariance error | `< 1e-6` | `3.484e-08` (rms), `2.641e-08` (quantile) |
| test 3 — per-dim min-max equivariance error | `> 0.1` | `2.383e-01`; `scale(dx)=1.0129` vs `scale(dy)=1.2929` |
| test 4 — `IrrepAdam` iterate drift | `< 1e-5` | `1.608e-07` |
| test 4 — plain Adam iterate drift | `> 0.1` | `4.791e-01` |

Supporting numbers recorded for later comparison: test 2 circular
`W1(target heading, uniform) = 0.6792 rad`; test 5 `steerGain` separates cleanly
(iid/perm ≈ `-0.01…-0.05`, `rot-heading` `1.02`, `rot-head+prior` `1.03`,
`perm+rot-group` `1.04`); test 6 orbit-DNO 2x2 reproduces the expected pattern
(task loss: iid `-0.717 -> -0.924`, `perm-euclid` `-0.436 -> -0.911`,
`rot-head+prior` `-0.452 -> -0.966`).

### Phase 0.4 — controller-frame check (§2.4) — **the one that decides whether any of this measures anything**

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:27 | §2.4 | `$PY -c "inspect.getsource(LiberoEnv) ..."` (the plan's snippet) | 8s | **inconclusive** — returned only two `frame` lines from the renderer. `LiberoEnv` never names a controller; it delegates to `libero.libero.envs.env_wrapper.ControlEnv` |
| 17:28 | §2.4 | `inspect.signature(ControlEnv.__init__)` | 6s | `controller = 'OSC_POSE'`, `control_freq = 20`, `robots = ['Panda']` |
| 17:28 | §2.4 | `cat robosuite/controllers/config/osc_pose.json` | <1s | `output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`, `control_delta = true`, `uncouple_pos_ori = true`, `kp = 150` |
| 17:29 | §2.4 | read `robosuite/controllers/osc.py::set_goal` + `utils/control_utils.py` | — | frame resolved, see below |

**Gate G2 — PASS. The controller consumes world/base-frame deltas, not
end-effector-frame deltas.** Traced rather than assumed:

* **Position.** `set_goal` calls `set_goal_position(scaled_delta[:3], self.ee_pos)`, which
  returns `current_position + delta`. Both operands are world-frame, so the commanded
  translation delta is world-frame. Under a world rotation `R_z`, `(dx, dy)` transforms as a
  planar vector → frequency-1. `dz` invariant → frequency-0.
* **Orientation.** `set_goal_orientation` builds `rotation_mat_error = quat2mat(axisangle2quat(delta[3:6]))`
  and returns `rotation_mat_error @ current_orientation` — **left**-multiplication, i.e. the
  axis-angle delta is expressed in the world frame. Conjugating by `R_z` gives
  `R_z E R_z^T`, whose axis-angle vector is `R_z · w`. So `(wx, wy)` is frequency-1 and `wz`
  is invariant.

This is exactly the layout `SO2ChunkSpec.libero_osc_pose()` assumes and the config declares:
`vector_blocks=[[0,1],[3,4]]`, `scalar_dims=[2,5,6]`. The experiment is measuring something real.

**Phase 6 constants fixed by this check (§8.1):**

```bash
export TRANS_GAIN=0.05     # VERIFIED: osc_pose.json output_max[0:3]
# ROT_GAIN = 0.5           # VERIFIED: osc_pose.json output_max[3:6]
```

Control period for Gate G11: `n_action_steps / fps = 8 / 20 = 0.4 s`.

`TABLE_Z` is **not yet set** — §8.1 requires deriving it from the demonstrations'
own end-effector height percentile, which needs the dataset. See "Blocked" below.

### Integration smoke test (not a plan step — added to prove the drop-in runs in *this* repo)

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:30 | — | `$PY <scratchpad>/integration_smoke.py` | 3m | **39 / 40 checks pass**, 1 documentation discrepancy (below) |

Verified end to end on GPU with synthetic data (no LIBERO dataset needed):

* `train_flowpolicy_orbit.yaml` composes; `_target_` is `OrbitFlowPolicy`; `kappa: .inf`
  parses to `+inf`; `action_spec.action_dim` matches `shape_meta.action.shape`; the
  workspace `_target_` is unchanged from the baseline config.
* All five Phase-2/3 arms (P1 iid, P2 perm/angle, P3 perm/euclidean, P4 rot/heading,
  P6 perm_rot/group_ot) instantiate, run forward + backward, and `predict_action` to the
  right shapes.
* SO(2) normalizer repair fires inside `set_normalizer`: shared scale on `(dx,dy)` and
  `(wx,wy)`, exactly-zero vector offsets, equivariance error `< 1e-6`.
  `normalizer_mode=inherit` correctly leaves the repo min-max in place (error `0.24`), and
  the Gate-G6 fallback `normalizer_vector_mode=absmax` also reaches `< 1e-6`.
* `source_heading_hist` is a persistent buffer and rides in `state_dict` (so P5 works off a
  P4 checkpoint, as §6 requires); `inference_prior=matched` runs.
* Orbit DNO: stage 1 runs **under `torch.inference_mode()`** (so the stock `LiberoRunner`
  drives it, as §8.2 claims); stage 2 descends the task loss (`202.4 -> 153.2` with
  `irrep_adam`); the plain-`adam` ablation runs; `DNOPolicy` raises the documented clear
  error for stage 2 under inference mode instead of failing inside autograd;
  `DNOPolicy.to(device)` correctly moves `device` (the runner reads `policy.device`).
* `OrbitDnoLiberoRunner` subclasses `LiberoRunner` and overrides `run`; the eval script's
  `_target_` swap resolves via hydra; the runner cfg exposes `n_test` / `n_parallel_envs` /
  `n_test_vis`.
* Gate G4/G5/Phase-5 conditional overrides all build and train:
  `action_spec.block_weights=[1.0,0.0]`, `coupling.min_heading_confidence=0.05`,
  `coupling.kappa=2`, `coupling.assignment=circular_sort`.
* Stage-1 wall-clock at K=8, batch 4 on one 4090: **p50 ≈ 25 ms** against the 0.4 s control
  period. Indicative only — the real number for Gate G11 comes from `dno_eval_log.json`
  at K=32 with the real observation encoder under load.

**Discrepancy found (documentation, not function).** `OrbitFlowPolicy.forward` calls
`_accumulate_source_headings(z0)` unconditionally, for **every** coupling mode. So a
`mode=iid` (P1) checkpoint also ships a populated `source_heading_hist`. Consequences:

* §6's sanity snippet `assert float(h.sum()) > 0` passes on a P1 checkpoint too — it does
  **not** discriminate, contrary to §11's troubleshooting row ("`empty histogram` on P5 →
  ... or `mode=iid`").
* `scripts/eval_orbit_dno_libero.py`'s guard ("Only a rotation-applying arm (mode=rot /
  perm_rot) fills it") will not fire on an iid checkpoint.

Behaviourally this is benign and arguably the better design — the class docstring says the
histogram is "measured on the actual coupled sources, so it is exact for every coupling
variant". For an iid arm the accumulated law is uniform, so `matched` ≈ `standard` and P5
on an iid checkpoint degenerates to a no-op rather than an error. **No file was changed.**
The only operational note: the runbook's stated safety net against running P5 on the wrong
checkpoint does not exist, so check the arm name yourself.

---

## 2026-08-21 — Data preparation (README §2, not EXPERIMENT_PLAN §2.2)

Followed `README.md` §2, **not** the plan's §2.2 — the README has a staging step the plan
omits (`data/libero/hdf5_datasets/`) and passes `--use-huggingface` to the downloader.

### Download — not required

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:34 | §2.1 | `ls /home/haotian/code/oat/third_party/LIBERO/libero/datasets/libero_10/` | <1s | **all 10 LIBERO-10 hdf5 files already present** (13 GB, dated Jul 9). Download skipped. |
| 17:34 | — | `h5py.File(...)` on all 10 | 3s | all readable, **50 demos each = 500 total** |

Two environment facts found here, both worth knowing:

* `past2next_clean/third_party/LIBERO` is an **uninitialised submodule** (`git submodule
  status` → leading `-`, directory empty).
* `~/.libero/config.yaml` points at `/home/haotian/code/oat/third_party/LIBERO/...`, and
  the conda env's `libero` is an editable install resolving to that same sibling checkout.
  So this repo currently borrows the `oat` repo's LIBERO assets and datasets. Everything
  works, but moving or cleaning `/home/haotian/code/oat` will break this repo.

### Staging + conversion + merge

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:35 | README §2 step 2 | `ln -sfn <oat>/libero_10/*.hdf5 data/libero/hdf5_datasets/` | <1s | 10 symlinks, all readable (symlinked rather than copied — saves 13 GB) |
| 17:36 | README §2 step 3 | `$PY scripts/convert_libero_dataset.py` | ~9 min | **10/10 converted, 0 failures**, 7.0 GB of per-task zarrs. All 10 names match `MT_TASKS['libero10']`. |
| 17:46 | README §2 step 4 | `$PY scripts/compose_libero_multitask_dataset.py -mt libero10` | 5s | **FAILED SILENTLY — exit code 0** |
| 17:47 | README §2 step 4 | `PATH=/home/haotian/miniforge3/envs/oat/bin:$PATH python scripts/compose_libero_multitask_dataset.py -mt libero10` | ~4 min | OK |

**Silent-failure trap, recorded because it will bite anyone repeating this.**
`compose_libero_multitask_dataset.py` runs the merge via
`os.system(f"python {ROOT_DIR}/scripts/merge_data.py ...")` — a **bare `python`**, which
resolved to the system interpreter, which has no `click`:

```
ModuleNotFoundError: No module named 'click'
```

`os.system` does not propagate the child's status, so the wrapper still **exited 0 and
produced no zarr**. Do not trust this script's exit code; check the output path exists.
Fixed without editing any file by putting the conda env first on `PATH` so the inner bare
`python` resolves correctly.

Result — matches the README's stated expectation exactly (~3.4 GB, 10 tasks x 50 demos):

```
data/libero/libero10_N500.zarr   3.4 GB
  data/action                 (138090, 7)          float32
  data/agentview_rgb          (138090, 128,128,3)  uint8
  data/robot0_eye_in_hand_rgb (138090, 128,128,3)  uint8
  data/robot0_eef_pos         (138090, 3)          float32
  data/robot0_eef_quat        (138090, 4)          float32
  data/robot0_gripper_qpos    (138090, 2)          float32
  data/robot0_joint_pos       (138090, 7)          float32
  data/task_uid               (138090, 1)          int64
  meta/episode_ends           (500,)               int64
```

---

## 2026-08-21 — Phase 1: LIBERO-10 heading audit (§3)

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:52 | §3 | `$PY scripts/libero_heading_audit.py --zarr data/libero/libero10_N500.zarr --horizon 16 -o ${EXP_ROOT}/audit_libero10.json` | 40s | 32841 chunks of horizon 16 → `output/exp/audit_libero10.json` |

Recorded as §3 requires:

| quantity | value |
|---|---|
| `W1(heading, uniform)` | **0.2103 rad** |
| heading `R1` / `R2` / `R4` / `R8` | 0.1511 / 0.1450 / 0.0927 / 0.0347 |
| min-max normalizer equivariance error | **1.930e-01** |
| SO(2) block normalizer equivariance error | **3.503e-08** |
| per-block energy (normalized) | `vec0` 0.3835, `vec1` 0.3835, `dz` 0.0296, `wz` 0.0116, `grip` 0.1918 |
| low-confidence fraction (`< 0.05`) | **0.000** |

Coupling preview on real chunks (lower bound for any marginal-preserving coupling is
`W1 = 0.2103`):

| coupling | path len | angle gap | cond var | src R1 |
|---|---|---|---|---|
| A/iid | 13.9991 | 1.5698 | 142.56 | 0.0581 |
| P2/perm-angle | 13.6588 | **0.2573** | 135.90 | 0.0600 |
| P3/perm-euclid | 12.8569 | 0.7118 | 118.33 | 0.0517 |
| P4/rot-heading | 13.6240 | 0.0000 | 135.59 | **0.1801** |
| P6/perm+rot-group | **12.6188** | 0.5075 | **113.93** | 0.1533 |

P2 lands at 0.2573 against a theoretical floor of 0.2103 while keeping `src R1` at the iid
level (0.060) — the marginal-exact coupling is operating near its bound, exactly as the
design predicts. P4 drives the gap to 0 but deforms the source (`R1` 0.058 → 0.180),
which is the prior shift `W1` predicted and what P5 exists to repay.

**Gate G3 — `W1 = 0.2103` falls in the `0.10 – 0.40` band.** Verdict `matched_prior_recommended`:
run **P4 and P5**, expect P5 > P4. Full arm list from §5.1 stands (P1, P2, P3, P4, P6),
with P5 as the Phase-4 re-evaluation of the P4 checkpoint.

**Gate G4 — does not fire as instrumented, but the intent says otherwise. Read this before training.**
The script reports `vec1_energy_frac = 0.3835`, far above the 0.05 threshold, so by the
letter of §3 no `block_weights` override is needed. But that number is measured **after**
the SO(2) normalizer, and `vector_mode=rms` sets each vector block to unit per-coordinate
RMS — so `vec0_energy_frac == vec1_energy_frac` **by construction**, always. Under the
default normalizer this gate can never fire. Measured on **raw** actions instead:

| block | raw energy share | per-dim RMS |
|---|---|---|
| `(dx, dy)` | 0.1529 | dx 0.2802, dy 0.3579 |
| `(wx, wy)` | **0.0034** | wx 0.0398, wy 0.0551 |
| `dz` | 0.0973 | 0.3626 |
| `wz` | 0.0060 | 0.0898 |
| gripper | 0.7403 | — |

In raw physical units the rotation block carries **0.34%** of the action energy — an order
of magnitude *below* the gate's 0.05 threshold. The normalizer then multiplies it up
(scale 15.18 vs 3.08 for translation, measured in the smoke test), so the coupling weights
a nearly-inert channel equally with the translation channel when computing headings and
assignments. **Open decision before any training starts** — see "Decision required" below.

**Gate G5 — does not fire.** Fraction of chunks with heading confidence `< 0.05` is
**0.000** (p5 of the confidence distribution is 0.748). No `min_heading_confidence`
override. LIBERO-10 chunks essentially all have a well-defined heading.

---

## 2026-08-21 — Gate G10 pulled forward (§8.1)

Derived early because it needs only the dataset, and because the plan's placeholder is wrong.

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 17:55 | §8.1 | eef-z percentiles + histogram + per-task breakdown over `libero10_N500.zarr` | 30s | see below |

```
eef z (138090 steps): min 0.446  p01 0.448  p05 0.484  median 0.928  max 1.332
```

**Gate G10 — the plan's `TABLE_Z=0.82` placeholder is badly wrong for this dataset and
must not be used.** It sits far above `p01 = 0.448`; **48.4%** of all demonstrated steps
fall below it, which is precisely the failure §8.1 warns about (the clearance term would
punish the demonstrations and DNO would steer away from correct behaviour).

Worse, the distribution is **bimodal with an empty gap at 0.815–0.889**, and the split is
exactly per task — five tasks per mode:

| task_uid | steps | min | p01 | median | max |
|---|---|---|---|---|---|
| 30 | 14700 | 0.481 | 0.482 | 0.627 | 0.780 |
| 31 | 13021 | 0.446 | 0.447 | 0.620 | 0.781 |
| 34 | 12909 | 0.500 | 0.508 | 0.573 | 0.698 |
| 36 | 12756 | 0.446 | 0.447 | 0.546 | 0.705 |
| 37 | 13476 | 0.446 | 0.447 | 0.616 | 0.729 |
| 32 | 13298 | 0.925 | 0.927 | 1.052 | 1.194 |
| 33 | 12434 | 0.910 | 0.914 | 1.013 | 1.194 |
| 35 | 9470 | 0.985 | 0.988 | 1.166 | 1.332 |
| 38 | 20794 | 0.984 | 1.000 | 1.095 | 1.234 |
| 39 | 15232 | 0.943 | 0.961 | 1.057 | 1.301 |

For the five low tasks the **maximum** eef height (0.698–0.781) is entirely below 0.82, so
`TABLE_Z=0.82` would penalise *every single demonstrated step* of half the benchmark.

**Consequence: a scalar `--table-z` cannot be correct on LIBERO-10.** The plan's own rule
(below `p01`) gives `TABLE_Z = 0.44`, which is safe (penalises 0.0% of steps) but leaves
`relu(0.44 - z)` identically zero for the five high tasks and near-zero for the rest — the
table term is effectively **inert**. Since `--w-table 10.0` is the largest default weight
in the Tier-0 objective, that removes most of the objective's mass and makes §8.4's
`dno/orbit_loss_spread ≈ 0` failure mode likely. Options for Phase 6, in preference order:

1. `--table-z 0.44 --w-table 10` — safe, but expect the term to contribute ~nothing; check
   `orbit_loss_spread` first thing.
2. `--w-table 0` and lean on smoothness / seam / step_limit, reporting the objective as
   table-free.
3. A per-task table height keyed off `task_uid` (~0.42 for the low group, ~0.88 for the
   high group). Needs a **new** file — `eval_orbit_dno_libero.py` only accepts a scalar —
   which the §0.1 constraint permits.

Not decided yet; Phase 6 is far downstream. Recorded so the choice is made on data.

**Gate G0 re-check after all of the above: PASS.** `git status --porcelain` still shows
zero modified files; `data/` and `output/` are gitignored.

---

## 2026-08-22 — Phase 2 launched: P0 vs P1 (§4)

Decisions taken (user, 2026-08-21): Gate G4 → **default two-block coupling, no
`block_weights` override** on any arm; budget → **`EPOCHS=1001`, identical across all arms**
per §0.3; scope → Phase 2 first, stop at Gate G6.

Both arms run **concurrently, one per GPU, single-process each** (`CUDA_VISIBLE_DEVICES`).
Deliberately *not* `accelerate --multi_gpu`: DDP would change the global batch, and the
orbit config warns batch size is a coupling hyper-parameter, not a free training detail.
`lazy_eval: true` (config default) so no env is built during training — only `latest.ckpt`
is written, and the plan evaluates offline in §4 anyway.

| time | phase | command | outcome |
|---|---|---|---|
| 02:44 | §4 | `CUDA_VISIBLE_DEVICES=0 $PY scripts/run_workspace.py --config-name=train_flowpolicy seed=42 training.seed=42 training.num_epochs=1001 training.num_demo=500 hydra.run.dir=${EXP_ROOT}/P0_baseline/seed42` | running |
| 02:51 | §4 | `CUDA_VISIBLE_DEVICES=1 $PY scripts/run_workspace.py --config-name=train_flowpolicy_orbit seed=42 training.seed=42 training.num_epochs=1001 training.num_demo=500 policy.coupling.mode=iid hydra.run.dir=${EXP_ROOT}/P1_norm_only/seed42` | running |

P1 startup confirms the drop-in is active on real data:

```
coupling  : mode=iid, cost=angle, align=heading, kappa=inf, scalar_weight=0.0, coupling_prob=1.0
normalizer: so2_block (rms)
[SO2] action normalizer repaired: (0,1)->3.1114, (3,4)->20.8007, vector offsets 0
```

(printed twice — once for `model`, once for `ema_model`). Note the **6.7x** ratio between
the rotation and translation scales: this is the Gate G4 caveat made concrete. The
normalizer lifts a block carrying 0.34% of raw action energy up to parity with translation.

**Measured cost — recorded because it re-frames the whole compute budget.**
3885 iterations/epoch at ~42 it/s (P0) and ~38 it/s (P1):

| | s/epoch | 1001 epochs |
|---|---|---|
| P0 | ~104 | **~29 h** |
| P1 | ~110 | **~31 h** |

Both in parallel → Phase 2 finishes in **~31 h wall-clock**. Extrapolating §10's matrix on
two GPUs at this budget: Phase 3 triage (4 arms, 2 waves) ~60 h; the 3-seed extension
(8 arms, 4 waves) ~120 h. **Full plan ≈ 8-9 days of continuous GPU**, before evals.
§0.3 forbids mixing budgets, so this is the moment to change it if it is too long — a
restart now costs ~30 min, a restart later costs days.

### Phase 2 results — P0 (complete)

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 2026-08-23 04:27 | §4 | P0 training finished | ~25.7 h | **1001 distinct epochs (0→1000)**, `train_loss` 0.05204, `val_loss` 0.15928, `test_reconst_mse` 0.04746 |
| 04:30 | §4 | `report_coupling_diagnostics.py -c .../P0_baseline/seed42/checkpoints/latest.ckpt` | 2 min | → `P0_baseline/seed42/diag.json` |
| 04:35 | §4 | `eval_policy_sim.py -c ... -o .../eval -n 3 -d cuda:0` | ~6 h | → `P0_baseline/seed42/eval/eval_log.json` |

Only `latest.ckpt` is written — expected, because `lazy_eval: true` means
`mean_success_rate` never appears during training so the top-k monitor never fires
(README §3.6 documents this).

**P0 diagnostics:**

| metric | value |
|---|---|
| `orbit_steering_gain` | **-0.00000** |
| `orbit_phase_consistency` | — |
| `src_orbit_eq_rel` | 0.56720 |
| `heading_MAE` | 0.39026 |
| `action_mse_N10` | 0.04476 |
| `few_step_gap_N1` / `N4` | 0.19540 / 0.08752 |
| `straightness` | 1.74761 |
| `field_eq_train_paths` / `common_iid` / `inference_traj` | 0.46520 / 0.31227 / 7.21523 |

The steering gain is **exactly zero**, which is the required sanity check at this stage: an
uncoupled baseline must show no steerability. The metric is therefore correctly wired, and
a non-zero reading on a coupled arm later is meaningful rather than an artefact.

**P0 LIBERO-10 success rate (3 repeats x 500 episodes):**

```
Exp 1: 0.172    Exp 2: 0.210    Exp 3: 0.176
mean 0.1860   std 0.0209   stderr 0.0121
```

Per-task (mean of 3), showing very large spread:

| SR | task |
|---|---|
| 0.667 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it |
| 0.333 | LIVING_ROOM_SCENE5_white_mug_left_plate... |
| 0.253 | LIVING_ROOM_SCENE6_white_mug_plate_chocolate_pudding... |
| 0.147 | LIVING_ROOM_SCENE1_alphabet_soup_cream_cheese... |
| 0.147 | LIVING_ROOM_SCENE2_cream_cheese_butter... |
| 0.107 | LIVING_ROOM_SCENE2_alphabet_soup_tomato_sauce... |
| 0.093 | STUDY_SCENE1_book_caddy... |
| 0.067 | KITCHEN_SCENE6_mug_microwave... |
| 0.047 | KITCHEN_SCENE4_black_bowl_drawer... |
| 0.000 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |

**Two concerns to settle before Phase 3 commits ~60 h.**

1. **Absolute success is low (0.186).** README §3.4 trains this exact baseline for **5001**
   epochs; the plan's example budget of 1001 is one fifth of that. The experiment's real
   question (P2/P4/P6 vs P3) has to be resolved in a narrow band near this floor.
2. **Gate G6's threshold is probably too tight as written.** It asks whether
   `|SR(P1) - SR(P0)|` sits inside the 3-repeat stderr — here **0.0121**. But those three
   repeats share one set of weights, so the stderr measures *rollout* stochasticity only,
   not training-seed variance, which is normally much larger. Judged against 0.0121 the
   gate will fire on differences that a second seed would absorb. Recommendation: report
   the difference against the repeat stderr as the plan asks, but do **not** promote P1 to
   baseline on a single-seed difference alone without a seed-43 confirmation.

Note: P1's epoch rate dropped to ~218 s/epoch while P0's eval ran (20 parallel MuJoCo envs
competing for CPU) and recovers to ~110 s/epoch once the eval finished. Not a problem, but
it means arm wall-clock is not comparable across periods when an eval shares the machine.

### Phase 2 results — P1 (complete) and **Gate G6**

| time | phase | command | wall | outcome |
|---|---|---|---|---|
| 2026-08-23 07:42 | §4 | P1 training finished | ~29 h | **1001 distinct epochs**, `train_loss` 0.12959, `val_loss` 0.5577 |
| 07:45 | §4 | `report_coupling_diagnostics.py` on P1 | 2 min | → `P1_norm_only/seed42/diag.json` |
| 07:50 | §4 | `eval_policy_sim.py -n 3` on P1 | ~6 h | → `P1_norm_only/seed42/eval/eval_log.json` |

**`train_loss` / `val_loss` are NOT comparable between P0 and P1.** The loss is MSE in
*normalized* space and the two arms use different normalizers — P1's SO(2) block scales
`(wx,wy)` by 20.8 vs ~1 under min-max, so identical behaviour produces a larger number.
Only success rate and the raw-space `action_mse_*` are comparable across the two.

**Diagnostics, side by side:**

| metric | P0 (min-max) | P1 (SO(2) block) | note |
|---|---|---|---|
| `orbit_steering_gain` | **-0.00000** | **-0.00000** | both zero — §4 requires this of P1 |
| `action_mse_N10` (raw units) | 0.04476 | **0.03045** | P1 32% better — the only comparable loss-like number |
| `heading_MAE` | **0.39026** | 0.62267 | P1 worse |
| `straightness` | **1.74761** | 4.07690 | P1 much less straight |
| `few_step_gap_N1` / `N4` | 0.19540 / 0.08752 | 0.20833 / **0.07914** | ~tied |
| `src_orbit_eq_rel` | 0.56720 | 1.10952 | |

P1's zero steering gain is the check §4 explicitly demands ("It must be — P1 has no
coupling. A non-zero value means the metric is misconfigured"). **Confirmed.**

Note a **false-positive warning** in `report_coupling_diagnostics.py`: it prints "steering
gain is near zero on a coupled checkpoint" for P1. The guard is
`report["coupling"] != "iid(baseline)"`, which compares against a literal string and so
fires for any orbit-config checkpoint even when `mode=iid`. Cosmetic; ignore it for P1.

**Success rate (3 repeats x 500 episodes):**

```
P0_baseline   0.172 / 0.210 / 0.176   -> mean 0.1860  std 0.0209  stderr 0.0121
P1_norm_only  0.188 / 0.222 / 0.194   -> mean 0.2013  std 0.0181  stderr 0.0105
```

**GATE G6 — passes in the sense that matters.** `SR(P1) - SR(P0) = +0.0153`. That is
marginally larger than P0's 3-repeat stderr (0.0121), so by the literal wording it is
"outside" — but it is **0.96 sigma** on the combined stderr (0.0160), i.e. statistically
indistinguishable, and critically **P1 is better, not worse**. §4's remediation branch is
conditioned on "If P1 is clearly worse", which did not happen. So:

* no `normalizer_vector_mode=absmax` re-run is needed;
* **P0 remains the reference baseline**; the SO(2) normalizer is not a cost, and the
  writeup does not have to report one.

### The finding that actually matters: aggregate parity hides a large per-task reshuffle

| eef median (m) | task | P0 | P1 | delta |
|---|---|---|---|---|
| 0.546 | LIVING_ROOM_SCENE6_white_mug_plate_pudding | 0.253 | 0.007 | **-0.247** |
| 0.573 | LIVING_ROOM_SCENE5_white_mug_left_plate | 0.333 | 0.040 | **-0.293** |
| 0.616 | LIVING_ROOM_SCENE1_alphabet_soup_cream_cheese | 0.147 | 0.260 | +0.113 |
| 0.620 | LIVING_ROOM_SCENE2_cream_cheese_butter | 0.147 | 0.107 | -0.040 |
| 0.627 | LIVING_ROOM_SCENE2_alphabet_soup_tomato_sauce | 0.107 | 0.100 | -0.007 |
| 1.013 | KITCHEN_SCENE4_black_bowl_drawer | 0.047 | 0.193 | +0.147 |
| 1.052 | KITCHEN_SCENE3_stove_moka | 0.667 | 0.567 | -0.100 |
| 1.057 | KITCHEN_SCENE6_mug_microwave | 0.067 | 0.420 | **+0.353** |
| 1.095 | KITCHEN_SCENE8_both_moka_pots | 0.000 | 0.253 | **+0.253** |
| 1.166 | STUDY_SCENE1_book_caddy | 0.093 | 0.067 | -0.027 |

```
aggregate delta        = +0.0153
mean |per-task delta|  =  0.1580     <- 10.3x the aggregate
low-table group  (n=5): 0.197 -> 0.103   (-0.095)
high-table group (n=5): 0.175 -> 0.300   (+0.125)
```

The two policies are **not** behaviourally equivalent; the aggregate merely averages a
large redistribution away. Per-task swings of 0.25-0.35 are 4-5x the per-task binomial
stderr (~0.06 at 50 episodes/task), so they are real, not sampling noise. One task goes
0.000 -> 0.253, another 0.333 -> 0.040.

**Caveat, stated because it limits the claim:** the low/high table-height split is
*perfectly collinear* with scene family here — the five low tasks are exactly the five
LIVING_ROOM tasks, the five high are four KITCHEN plus STUDY. So this data cannot
attribute the shift to geometry rather than scene identity, appearance, or task type. It
is suggestive, not established, and with n=5 per group and one seed it could still be
coincidence. A second seed would settle it.

**Consequence for the rest of the plan.** §4's premise is that if P0 and P1 agree on
aggregate SR then "every later comparison" is unconfounded. **That premise does not hold
here.** A change that leaves the mean alone can still move individual tasks by ±0.3, so an
aggregate-level difference between P2/P4/P6 and P3 could be produced by task-mix effects
rather than by the coupling. **From Phase 3 onward, report per-task success rates for every
arm, not just the aggregate.** The numbers are already in each `eval_log.json`; this costs
nothing but changes what can honestly be concluded.

---

# PHASE 2.5 — triage before committing to Phase 3

Following `PHASE_2.5_PLAN.md`, which supersedes `EXPERIMENT_PLAN.md` §5 until its gates
resolve. Repo renamed `past2next_clean` → **`fm_dno`**; all artifacts moved intact.

## 2026-08-24 — §1 corrections

**The "refreshed package" referenced by §1 does not exist.** `orbit_fm_dno/` is still the
original 2026-08-21 drop: no `paired_task_compare.py`, no `global_rms`, no `descent_limit`,
no `--w-descent`. All seven changes were therefore **written here** rather than copied.

Before writing them, each was checked against the training path, because two ~30 h runs were
about to start: `global_rms` is opt-in (default stays `rms`), the audit/diagnostics/paired
scripts are standalone, the histogram gating is a no-grad buffer write with no RNG draw, and
`descent_limit` is DNO-eval only. **None alters a training trajectory**, so the GPUs were
started first and the corrections written while they ran.

| file | change |
|---|---|
| `scripts/paired_task_compare.py` | **new** — paired per-task statistics (G15) |
| `scripts/libero_heading_audit.py` | G4 reads **raw** block energy; prints raw vs normalized + per-dim raw RMS |
| `scripts/report_coupling_diagnostics.py` | warning guard now tests `coupling['mode']`, not a literal string |
| `oat/policy/flow_policy_orbit.py` | `source_heading_hist` accumulates only for `mode in (rot, perm_rot)` |
| `oat/symmetry/normalizer.py` | new `vector_mode='global_rms'` — one scale pooled over all vector coords |
| `oat/dno/task_losses.py` | new `descent_limit` + `max_descent` field, wired into `CompositeTaskLoss` |
| `scripts/eval_orbit_dno_libero.py` | `--w-descent` / `--max-descent`; `--w-table` now defaults **0**; `--table-z` default 0.82 → 0.44 |

**Gate G0 — passes, but the literal test no longer applies.** The drop-in was committed in
`fb6d6b1` / `f4cabea`, so our files are now *tracked*; editing them shows ` M`, not `??`.
The invariant that matters is the original repo, verified directly:

```bash
git diff --name-only 15b93ee -- <every §0.1 protected path> oat/config/task/policy/libero/
#  -> empty. Original repo byte-identical.
```

The six ` M` entries are exactly the plan's six. **Use the `git diff` form from now on**;
`git status | grep -v '^??'` is no longer a valid G0 check for this repo.

**Gate G1 — re-passes unchanged** after the corrections: min-max 2.383e-01, SO(2) block
3.484e-08, IrrepAdam 1.608e-07, plain Adam 4.791e-01. → `output/logs/selftest_phase25.txt`

`descent_limit` verified to do what it claims — identical chunk at eef z = 0.50 / 0.95 / 1.20:

```
descent_limit(0.15)   [0.0,     0.0, 0.0]      benign chunk, height-agnostic
  (diving chunk)      [2.0475, 2.0475, 2.0475] active and EQUAL on both scene families
table_clearance(0.82) [1.5449,  0.0, 0.0]      punishes the LOW scene only  <- the bug
table_clearance(0.44) [0.0,     0.0, 0.0]      inert everywhere             <- the other horn
```

## §2 — corrected audit → **Gate G4′**

`$PY scripts/libero_heading_audit.py --zarr data/libero/libero10_N500.zarr --horizon 16
-o output/exp/audit_libero10_v2.json` (40 s)

| block | **RAW** | normalized |
|---|---|---|
| `vec0 (dx,dy)` | 0.1553 | 0.3835 |
| **`vec1 (wx,wy)`** | **0.0034** | 0.3835 |
| `dz` | 0.0998 | 0.0296 |
| `wz` | 0.0061 | 0.0116 |
| gripper | 0.7354 | 0.1918 |

`per-dim raw RMS: dx=0.2822 dy=0.3627 dz=0.3684 wx=0.0401 wy=0.0555 wz=0.0910`

The two normalized figures being **identical** is the defect itself: `rms` equalises blocks
by construction, so the old gate could never fire. On raw actions it fires decisively —
0.34 % vs a 5 % threshold. Reproduces the hand-computed §7.1 values.

**Gate G4′ DECISION: option (b), `policy.action_spec.block_weights=[1.0,0.0]`**, applied to
**every** arm from here on including the iid control. Rationale: it changes only the coupling
cost, so P1 remains an exact control and Phase 2's result stays valid; `global_rms` would
change the normalizer and reintroduce the very confound Phase 2 was run to eliminate.
Verified semantics: complex width 32 → 16 (translation-only heading/assignment) while
`rotate_chunk` still rotates **both** blocks; heading equivariance under the weighted spec
2.38e-07.

## §3 — **Gate G15**, paired re-analysis of P0 vs P1

→ `output/exp/paired_P0_vs_P1.{json,txt}`

```
aggregate P0            0.1860
aggregate P1            0.2013
paired mean delta      +0.0153
paired sd over tasks    0.2056
paired stderr           ±0.0650    <-- THE RESOLVABLE EFFECT SIZE
paired t (9 dof)        +0.24    two-sided p 0.819
sign test               p 0.754  (4 of 10 tasks positive)
mean |per-task delta|   0.1580    reshuffle ratio 10.3x
MDE at p<0.05, n=10     0.1470    projected at 3 seeds  0.0849
```

Hand-rolled t-test and sign test cross-checked against `scipy`: identical to 4 decimals.
Reproduces every figure the plan quotes.

**AMENDED GATE G6 VERDICT.** The ±0.0121 recorded earlier was the *within-arm repeat*
stderr — it measures rollout stochasticity at fixed weights and omits the task-to-task
variance that dominates here, understating the true uncertainty by ~5×. Corrected:

* defensible: **"the SO(2) normalizer carries no measurable success-rate cost"**
  (|Δ| < 0.147 at this resolution);
* **not** defensible: "P1 is better than P0" — t = +0.24, p = 0.82, sign test p = 0.75.

P0 remains the reference baseline. **No aggregate delta is to be quoted without its paired
stderr from here on**, and no coupling effect below ~0.15 is detectable at one seed
(~0.085 at three).

## §4/§5 — Tracks A and B launched

Two bugs in the plan's own commands, both caught before committing GPU time:

1. **`policy.coupling.kappa=.inf` crashes.** Through Hydra's CLI grammar it arrives as the
   *string* `'.inf'`; `build_coupling` does `float('.inf')` → `ValueError`. The YAML default
   is already float `inf`. Use `kappa=inf` or omit it. Verified both give `inf`.
2. **Track B needs `MUJOCO_GL=egl`.** `lazy_eval=false` builds an env runner; the plan's
   command omits it (README §1.4).

**OOM incident.** The first attempt ran both tracks as written and Track A was killed
(exit 137) during its dataset load. Cause is new to this phase: Track B's `lazy_eval=false`
forks 10 MuJoCo env workers — Phase 2 never did — pushing the pair past 62 GB. Measured
footprints: Track A **19 GB**; Track B **40 GB** with 10 envs (the per-process RSS sums to
229 GB but the forks are copy-on-write, so real usage is far lower).

`ZarrDataset` hardcodes the in-memory store and does not expose `backend`/`store`, and
`oat/dataset/zarr_dataset.py` is §0.1-protected — so the disk-backed fix is unavailable.

Resolved with two safeguards rather than serialising the runs (which would have cost ~62 h
instead of ~37 h):

* `task.policy.env_runner.n_parallel_envs` **10 → 5** on Track B (halves the forks).
  Coverage is unchanged — `n_test=50` over 10 tasks still gives 5 episodes/task — but the
  task↔seed pairing differs from a 10-env run, so P1_curve's absolute SR is comparable
  *within* its own curve, which is all G14 needs. Rollout overhead ~4 h → ~8 h.
* Track B launched with **`oom_score_adj=800`** (inherited by its children), so if memory is
  exhausted the OOM killer takes Track B and the *decisive* Track A survives.

Steady state with both resident: **47 GB used, 15 GB free**; Track A `adj=0`, Track B `adj=800`.

| track | GPU | command | status |
|---|---|---|---|
| **A — P4_rot_heading** | 0 | `mode=rot align=heading kappa=inf block_weights=[1.0,0.0]` | running, ~29 h |
| **B — P1_curve** | 1 | `mode=iid block_weights=[1.0,0.0] lazy_eval=false rollout_every=50 n_test=50 n_parallel_envs=5 n_test_vis=0 topk.k=5` | running, ~37 h |

Gates pending: **G13** (steering gain, `P4_rot_heading/seed42/diag.json`) and **G14** (SR-vs-epoch
curve + working top-k, `P1_curve/seed42/`). Then the §6 decision matrix.

## 2026-08-24 — **Gate G14: the curve is FLAT. Ceiling, not undertraining.**

Rollout points from `P1_curve/seed42/logs.json` (`n_test=50`, so ±0.057 binomial per point):

```
epoch     0   SR 0.000
epoch    50   SR 0.040
epoch   100   SR 0.260
epoch   150   SR 0.220
epoch   200   SR 0.140
epoch   250   SR 0.180
epoch   300   SR 0.220
```

**Correction to an earlier reading in this log.** A first pass compared the maximum (0.260 @
epoch 100) against the mean of the same five points divided by sqrt(n) and called it "2.7
sigma, a real peak". That is wrong twice: it tests the maximum against a distribution it
belongs to (selection bias), and it uses the standard error *of the mean* where the
*per-point* binomial noise is the correct yardstick. Redone properly:

```
post-epoch-100 points        [0.26, 0.22, 0.14, 0.18, 0.22]   pooled p = 0.204
expected per-point noise      0.0570   (binomial, n=50)
observed sd across points     0.0456   <- SMALLER than pure sampling noise
chi2 for constant p           2.56, df=4, p=0.634
max vs mean-of-rest           1.2 per-point sigma  (max of 5 draws: expected ~1.2)
linear trend, epoch>=100      slope -2.4e-04/epoch, p=0.486
```

**No peak, no decline, no trend.** The scatter is smaller than binomial noise alone, so a
single constant success rate explains the data completely. Independently corroborated by
Phase 2: `P1_norm_only` ran the full 1001 epochs and finished at **0.2013** over 1500
episodes — indistinguishable from this plateau of 0.204.

**GATE G14 VERDICT: row 3 — "flat near 0.19 from early on; the config is at its ceiling."**
Consequences:

* **The budget question is settled: raising `EPOCHS` will not help.** The open decision
  carried since Phase 2 is closed, and closed against more compute.
* The ceiling is reached by **epoch ~100**, so the 1001-epoch budget is ~10x more than this
  configuration can use. Phase 2's two 29 h runs could each have been ~3 h.
* Phase 2 did **not** report post-peak checkpoints — 0.186/0.201 sit on the plateau, so
  those numbers stand as representative.
* The bottleneck is elsewhere. Per `PHASE_2.5_PLAN.md` §5 the first suspects are the scalar
  `task_uid` conditioning for a 10-task multitask policy, and `prompt`, which is present in
  the zarr but absent from `libero10.yaml`'s `obs_keys`. Adding it requires a **new** task
  config file, never an edit (Gate G0).

**Top-k selection now works** — `ep-0100_sr-0.260.ckpt` … `ep-0300_sr-0.220.ckpt` are being
written, the first time the manager has ever fired in this project (under `lazy_eval: true`
it never could). **But do not naively adopt the top-k checkpoint here.** On a flat curve,
selecting the maximum selects the luckiest binomial draw, not a better model: 0.260 is
1.2 sigma above a plateau of 0.204, exactly the expected maximum of five draws, and
re-evaluating it on 500 episodes will regress toward ~0.20. The plan's G14 row-1 remedy
("re-evaluate at the peak epoch") applies to a curve with a *real* peak; applying it to this
one would be a winner's-curse error and would inflate every arm that got a lucky draw.

## 2026-08-24 — **Gate G13 (preliminary): the mechanism TRANSFERS. gain = 0.99**

Run on Track A's rolling `latest.ckpt` at **epoch ~330 of 1001**, not the final checkpoint —
possible because the steering gain is a property of the coupling measured on validation
chunks, with no rollouts and no sensitivity to the operating point. Executed with
`oom_score_adj=1000` so the probe itself would be sacrificed before either training run;
both tracks survived.

| metric | P0 @1001 | P1 @1001 | **P4 @~330** |
|---|---|---|---|
| `orbit_steering_gain` | −0.00000 | −0.00000 | **0.99000** |
| `orbit_phase_consistency` | 0.05259 | 0.05467 | **0.83646** |
| `src_orbit_eq_rel` | 0.56720 | 1.10952 | 0.88675 |
| `heading_MAE` | 0.39026 | 0.62267 | 1.37418 |
| `action_mse_N1` / `N10` | 0.03824 / 0.04476 | 0.02731 / 0.03045 | 0.06812 / 0.08239 |
| `gen_heading_R1` / `gt_heading_R1` | 0.48050 / 0.47012 | 0.11184 / 0.07717 | 0.14509 / 0.14887 |
| `few_step_gap_N1` / `N4` | 0.19540 / 0.08752 | 0.20833 / 0.07914 | 0.39943 / 0.13307 |
| `straightness` | 1.74761 | 4.07690 | 6.49047 |

**GATE G13 VERDICT: top band (>= 0.7). The SO(2) orbit coupling makes the noise phase
control the chunk heading on real LIBERO-10 data.** Two independent statistics agree:
the gain is 0.99 (ideal 1.0) and the phase consistency rises 0.053 → 0.836 (ideal 1.0).
Both controls read essentially zero, so the metric was calibrated and the reading is real.
This is the synthetic study's central claim reproduced on robot data, and per §4 it licenses
Phase 6 and is worth reporting on its own.

**The delegation trade-off appears exactly as predicted, in both directions at once.**
§4 warns: "if the gain is high and these did *not* worsen, be suspicious — the delegation
mechanism predicts both together." Measured against P1: `src_orbit_eq_rel` **falls**
(1.110 → 0.887, predicted to drop) while `heading_MAE` (0.623 → 1.374) and `action_mse_N10`
(0.030 → 0.082) **worsen**. The policy has stopped inferring heading from the observation
and reads it off the noise phase — which is the contract, and what Phase 6 exists to repay.

**Caveats, stated because they bound the claim:**

1. **Preliminary checkpoint.** P4 is at epoch ~330; P0/P1 are at 1001. A *high* gain is
   decisive-positive — undertraining cannot manufacture 0.99 out of an uncoupled 0.00 — but
   the *magnitudes* of the worsened metrics are partly confounded with training progress.
2. **The transport half is not yet supported.** `straightness` (6.49 vs P1's 4.08) and
   `few_step_gap_N1` (0.399 vs 0.208) are *worse*, not better, where §4 expected "comparable
   to P1, improvement here is the transport half of the claim". This may be the epoch
   mismatch; the fair comparison needs P4's final checkpoint. Do not claim the transport
   half on this evidence.
3. Re-run `report_coupling_diagnostics.py` on the final P4 checkpoint before publishing any
   of these numbers.

### 2026-08-24, later — **G14 VERDICT RETRACTED. The curve is not flat.**

The epoch-350 rollout came in at **SR 0.400**, and it breaks the constant-rate model that
the "ceiling" verdict above rested on:

```
epochs 100-300 (what the verdict was based on)   p_bar=0.204  chi2= 2.56 df=4  p=0.634  consistent
epochs 100-350 (with the new point)              p_bar=0.237  chi2=11.16 df=5  p=0.048  REJECTED

P(X >= 20/50 | p = 0.204)                = 0.00121  one-sided
P(at least one of 6 draws this extreme)  = 0.0072
linear trend epoch>=100                  = +4.2e-04/epoch, p=0.382 (not yet significant)
```

A 0.400 point has ~1-in-800 probability of arising from the 0.204 plateau, and ~0.7 % even
after allowing for six chances to be surprised. It is very unlikely to be sampling noise.

**Therefore the G14 verdict recorded above ("row 3, ceiling") is withdrawn, and with it the
two conclusions drawn from it:** that the budget question is closed against more compute,
and that Phase 3 is not licensed. Both are **undecided again**. The curve is currently
consistent with row 2 ("still climbing at 1001, genuinely undertrained") or with a plateau
plus one real excursion; the linear trend is still not significant (p=0.38) because the
epoch-150-to-300 points are noisy, so more points are required to separate the two.

What survives unchanged: **Gate G13 is unaffected.** The steering gain is measured offline
on validation chunks with no rollouts, so it is independent of the operating point and of
this curve entirely — that independence is exactly why `PHASE_2.5_PLAN.md` §0 called Track A
the decisive one.

**Operational consequence: Track B must run on.** Stopping it early had been proposed here
on the strength of the flat reading; the very next point it produced is the one that
overturned that reading. It continues to 1001.

Method note for the rest of this project: five rollout points at `n_test=50` were not enough
to certify a shape. A chi-square consistency test can only fail to reject; it never
establishes flatness, and treating "consistent with constant" as "is constant" is what went
wrong. Any future curve claim needs either more points, larger `n_test`, or both.

### 2026-08-24, later still — the rise is real, and it exposes a level discrepancy

Epoch 400 came in at 0.360, a second consecutive high point. Pooling episodes rather than
averaging noisy per-point rates:

```
epochs 100-300     51/250 = 0.204
epochs 350-400     38/100 = 0.380
two-proportion z = 3.42,  p = 0.00064      Fisher exact p = 0.00103
linear trend epoch>=100: +5.3e-04/epoch, p = 0.150 (masked by the noisy middle)
```

**The rise is real.** Success nearly doubled after epoch 300. The "ceiling at ~0.20" reading
is dead, and the retraction above stands confirmed rather than merely precautionary.

**Open puzzle — do not treat Phase 2's absolute numbers as settled.** `P1_norm_only` is the
same configuration and it measured **0.2013 at epoch 1001** over 1500 episodes, while this
curve is at ~0.38 by epoch 400. Three candidate explanations, not yet separated:

1. **Episode-set difficulty.** This curve uses 50 episodes (seeds 1000-1049,
   `n_parallel_envs=5`); the Phase 2 eval used 500 (seeds 1000-1499, `n_parallel_envs=20`),
   a different and far larger set. Levels from the two protocols are **not** directly
   comparable.
2. **Run-to-run variance.** `PHASE_2.5_PLAN.md` §5 anticipated this: enabling rollouts
   consumes the global RNG, so `P1_curve` is a second sample at the same budget, not a
   reproduction.
3. **A genuine peak-then-decline between epoch 400 and 1001**, which would mean Phase 2 did
   report post-peak checkpoints after all.

**Cheap decisive test, queued for when a GPU frees up:** evaluate the *finished*
`P1_norm_only/seed42/checkpoints/latest.ckpt` under this curve's exact protocol
(`--n-test 50 --n-parallel-envs 5`). It loads no dataset, so it is light. If it returns
~0.38, explanation 1 holds and the two protocols simply measure different things. If it
returns ~0.20, explanations 2/3 are live and the peak-then-decline possibility is real.
Until that is run, **quote no absolute success rate without naming its protocol.**

## Where §6's decision matrix lands

**Undetermined, pending G14.** `G13 >= 0.7` is settled, which fixes the *row* of §6's matrix
but not the column:

| G13 | G14 outcome | §6 cell |
|---|---|---|
| **>= 0.7 (settled)** | operating point fixable / still climbing | **Best case** — re-baseline at the new budget and selection, then run Phase 3 (P1, P2, P3, P4) **and** Phase 6. The steering result is publishable on its own. |
| **>= 0.7 (settled)** | ceiling at ~0.20 | Run **Phase 6 only** on the P4 checkpoint; report the transport claim as unresolved at this operating point, with the ±0.065 paired stderr as the reason. |

An earlier revision of this section committed to the second row. That was premature — it
rested on the retracted flat reading. **No Phase 3 decision is to be made until Track B's
curve resolves.**

## Decision required before Phase 2 training starts

**Gate G4 / `block_weights`.** By the letter of §3 the gate does not fire (0.3835 > 0.05)
→ train all arms with the default two-block coupling. By its evident intent it does fire
(raw 0.0034 < 0.05) → add `policy.action_spec.block_weights=[1.0,0.0]` to **every** coupled
arm including P1 and call the coupling translation-only. The choice changes what every arm
trains on, so it must be made once, now, and held fixed. Recommendation: run the default
(the design deliberately rescales the block, and §3's own instrument says pass), and add a
translation-only variant as a labelled ablation only if compute allows.

Open gates: **G6** (P0 vs P1 normalizer control), **G7**–**G9**, **G11**–**G12**.
Not started: Phases 2–7. Per §10 the full matrix is ~18 trainings plus evals.
