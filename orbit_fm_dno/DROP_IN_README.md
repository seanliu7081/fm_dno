# Drop-in guide: SO(2) orbit coupling + orbit DNO for `past2next_clean`

**Every file here is new. No existing repo file is modified or needs to be.**
`oat/` uses implicit namespace packages (no `__init__.py` anywhere) — do not add any.

For the full run order, gates, and LIBERO-10 protocol, follow `EXPERIMENT_PLAN.md`.
This file is just the map.

```
oat/symmetry/so2_chunk.py            irrep layout, complex view, rotation, alignment
oat/symmetry/coupling.py             SO2OrbitCoupling + MatchedHeadingPrior
oat/symmetry/normalizer.py           rotation-commuting action normalizer
oat/symmetry/metrics.py              steering gain, source-orbit / policy equivariance, transport
oat/policy/flow_policy_orbit.py      OrbitFlowPolicy (subclass of FlowPolicy)
oat/policy/dno_policy.py             DNOPolicy — BasePolicy-shaped wrapper for rollout
oat/env_runner/libero_dno_runner.py  LiberoRunner without inference_mode (stage-2 DNO only)
oat/dno/irrep_adam.py                rotation-compatible Adam
oat/dno/task_losses.py               differentiable test-time objectives
oat/dno/orbit_dno.py                 two-stage orbit DNO
oat/config/train_flowpolicy_orbit.yaml
scripts/validate_so2_coupling.py     synthetic validation, torch only
scripts/libero_heading_audit.py      Phase 1 go/no-go gate on the real dataset
scripts/report_coupling_diagnostics.py
scripts/eval_orbit_dno_libero.py
scripts/aggregate_experiment_results.py
```

## 1. Validate before touching LIBERO

```bash
python scripts/validate_so2_coupling.py --quick
```

Six checks: source-marginal audit, transport diagnostics, normalizer equivariance,
optimizer equivariance, an end-to-end conditional flow-matching comparison, and the
orbit-DNO 2×2. No robosuite, no zarr, no GPU.

## 2. No workspace patch needed

The SO(2) action normalizer is applied inside `OrbitFlowPolicy.set_normalizer`, so
`TrainPolicyWorkspace`, `run_workspace.py`, `eval_policy_sim.py` and
`BasePolicy.from_checkpoint` all keep working unmodified. Toggle it with
`policy.normalizer_mode=so2_block|inherit`.

Without it, per-dimension min-max leaves `scale(dx) != scale(dy)` and a nonzero offset, so a
rotation in normalized space is not a rotation in raw space — measured relative equivariance
error 0.24, versus 3.5e-08 with the block normalizer.

Optional, if you want the coupling diagnostics in wandb, add to wherever `step_log` is built:

```python
if hasattr(policy, 'last_coupling_diagnostics'):
    step_log.update(policy.last_coupling_diagnostics)
```

That *would* be a modification to `train_policy.py`. Skip it — `libero_heading_audit.py`
already reports what the coupling does on this data, offline and for free.

## 3. Train

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_orbit          # P2: perm/angle
python scripts/run_workspace.py --config-name=train_flowpolicy_orbit \
    policy.coupling.mode=iid                                                   # P1 control
python scripts/run_workspace.py --config-name=train_flowpolicy_orbit \
    policy.coupling.mode=perm policy.coupling.cost=euclidean                   # P3 control
python scripts/run_workspace.py --config-name=train_flowpolicy_orbit \
    policy.coupling.mode=rot policy.coupling.align=heading                     # P4
```

## 4. Test-time steering on LIBERO-10

```bash
MUJOCO_GL=egl python scripts/eval_orbit_dno_libero.py \
    -c output/<run>/checkpoints/latest.ckpt -o output/dno/stage1 \
    --n-orbit 32 --n-grad-steps 0 --n-test 100 \
    --trans-gain 0.05 --table-z 0.82
```

`--n-grad-steps 0` (stage-1 orbit search) needs no gradients and runs under the stock
runner. `--n-grad-steps > 0` switches automatically to `OrbitDnoLiberoRunner`.
`--no-dno` on the same checkpoint gives the matched control arm.

Verify `--trans-gain` against your controller's `OSC_POSE` `output_max` and `--table-z`
against the demonstrations' own end-effector height. Both silently rescale every geometric
constraint if wrong.

## 5. What to check first

`orbit_steering_gain` from `report_coupling_diagnostics.py`. If it is near 0 on a
rotation-coupled checkpoint, the noise phase does not control the chunk heading, stage-1
DNO degenerates into best-of-K resampling, and the coupling is not doing what the design
claims. In the synthetic study it separates cleanly: ≈0.0 for iid and permutation-only
couplings, ≈1.0 for `mode=rot, align=heading`.
