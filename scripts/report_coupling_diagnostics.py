#!/usr/bin/env python3
"""
Offline diagnostics for a trained checkpoint: the numbers that explain a success rate.

Success rate alone cannot tell you whether the coupling did what the design says it does.
This reports, on held-out validation observations:

  steering gain            radians of chunk heading per radian of source rotation.
                           ~0 = the network ignores the noise phase; ~1 = it reads the
                           heading off it.  **Check this first.**  If it is 0 on an
                           orbit-coupled checkpoint, the coupling did not take and nothing
                           downstream will work.
  source-orbit equivariance   ||F(rho z | o) - rho F(z | o)|| / ||F(z | o)||, obs fixed.
  field equivariance       the same on the local velocity field, over four spatial
                           distributions (training interpolation paths, own inference
                           trajectories, common iid paths, off-manifold) -- the toy Round-2
                           plan's request, which separates global relaxed equivariance from
                           trajectory-local behaviour.
  few-step gap             ||F_N(z) - F_100(z)|| / ||F_100(z)|| for N in {1,2,4,10}.
  straightness             E int ||(x1 - x0) - v(xt,t)||^2 dt.
  action MSE               against ground-truth chunks at each N -- the open-loop
                           conditional accuracy that the delegation trade-off predicts will
                           get *worse* for rotation-applying couplings.

Works on plain ``FlowPolicy`` checkpoints too (the P0/P1 baselines), so every row of the
experiment table gets the same treatment.

WHICH ACTIONS COUNT AS "GROUND TRUTH"
-------------------------------------
``action_mse_*`` and ``heading_MAE`` are measured against ``batch['action']``.  For every
arm except reflow that is the demonstration.  A reflow arm trains on
``ReflowPairDataset``, whose ``action`` is the *donor's distilled endpoint*, so left alone
those two metrics would be measuring different things on different rows of the same table.
``--action-source demo`` pulls the actions from the underlying ``ZarrDataset`` instead --
same windows, same observations, demonstration targets -- which is what makes the column
comparable.  Default is ``dataset`` so existing invocations are unchanged.

``straightness`` and ``few_step_gap_*`` need no ground truth at all, so they are comparable
across arms either way; those are the two the few-step plan actually gates on.

Usage:
    MUJOCO_GL=egl python scripts/report_coupling_diagnostics.py \
        -c output/<run>/checkpoints/latest.ckpt -o output/diag/<name>.json

    # comparable MSE / heading columns for a reflow checkpoint
    MUJOCO_GL=egl python scripts/report_coupling_diagnostics.py \
        -c output/exp/R1_reflow/seed42/checkpoints/<best>.ckpt \
        -o output/exp/R1_reflow/seed42/diag.json --action-source demo
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click
import hydra
import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from oat.policy.base_policy import BasePolicy
from oat.symmetry.metrics import (
    few_step_gap,
    field_equivariance,
    orbit_steering_gain,
    source_orbit_equivariance,
    straightness,
)
from oat.symmetry.so2_chunk import SO2ChunkSpec, chunk_heading, circular_moments


def make_samplers(policy, obs=None):
    """(encode_obs, velocity, sample_chunk, sample_prior) for FlowPolicy or OrbitFlowPolicy.

    ``sample_prior`` must return the law the field was actually TRAINED on, or every metric
    below is measured off-distribution.  For an observation-canonicalized arm the obs-free
    ``policy.sample_prior`` is the un-canonicalized isotropic draw -- not that law -- so if
    the policy exposes ``sample_prior_from_obs`` it is used instead, with ``obs`` bound.
    """
    if hasattr(policy, "sample_chunk"):
        prior = policy.sample_prior
        if obs is not None and hasattr(policy, "sample_prior_from_obs"):
            print("using policy.sample_prior_from_obs: this arm's source law depends on o, "
                  "so the obs-free prior would probe the field off its training distribution")

            def prior(B, device=None, dtype=None, generator=None):   # noqa: F811
                return policy.sample_prior_from_obs(
                    obs, batch_size=B, device=device, dtype=dtype, generator=generator)

        return (policy.encode_obs, policy.velocity, policy.sample_chunk, prior)

    def encode_obs(obs):
        return policy.obs_encoder(obs)

    def velocity(x, t, cond):
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return policy.model(x, policy._scale_t(t), cond)

    def sample_chunk(cond, z, n_steps=None):
        n = n_steps or policy.num_inference_steps
        dt = 1.0 / n
        x = z
        for i in range(n):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * velocity(x, t, cond)
        return x

    def sample_prior(B, device=None, dtype=None):
        return policy.prior_noise_scale * torch.randn(
            B, policy.horizon, policy.action_dim,
            device=device or policy.device, dtype=dtype or torch.float32,
        )

    return encode_obs, velocity, sample_chunk, sample_prior


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-b", "--batch-size", default=128, help="validation chunks to probe")
@click.option("--n-angles", default=8)
@click.option("--action-source", default="dataset", type=click.Choice(["dataset", "demo"]),
              help="'demo' unwraps a ReflowPairDataset to its base ZarrDataset, so "
                   "action_mse_* / heading_MAE are measured against demonstrations on every "
                   "arm rather than against each arm's own training target.")
def main(checkpoint, output, device, batch_size, n_angles, action_source):
    device = torch.device(device)
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    spec = getattr(policy, "action_spec", None) or SO2ChunkSpec.libero_osc_pose()

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    val = dataset.get_validation_dataset()
    if action_source == "demo" and hasattr(val, "base"):
        # same windows and the same observations; only 'action' differs
        print(f"--action-source demo: unwrapping {type(val).__name__} -> "
              f"{type(val.base).__name__}")
        val = val.base
    batch = next(iter(DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=2)))
    obs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["obs"].items()}
    gt = batch["action"].to(device)

    # built after `obs` exists: an arm whose source law depends on o needs it bound in
    encode_obs, velocity, sample_chunk, sample_prior = make_samplers(policy, obs=obs)

    report = {
        "checkpoint": checkpoint,
        "coupling": (OmegaConf.to_container(cfg.policy.coupling, resolve=True)
                     if "coupling" in cfg.policy else "iid(baseline)"),
        "normalizer_mode": cfg.policy.get("normalizer_mode", "limits(baseline)"),
        "batch_size": int(gt.shape[0]),
        "action_source": action_source,
        "action_dataset": type(val).__name__,
    }

    with torch.no_grad():
        cond = encode_obs(obs)
        x1 = policy.normalizer["action"].normalize(gt)
        z = sample_prior(cond.shape[0], device=device, dtype=cond.dtype)

        fn = lambda zz: sample_chunk(cond, zz)
        report.update(source_orbit_equivariance(fn, z, spec, n_angles=n_angles))
        report.update(orbit_steering_gain(fn, z, spec, n_angles=max(n_angles, 12)))
        report.update(few_step_gap(lambda zz, n: sample_chunk(cond, zz, n), z, steps=(1, 2, 4, 10)))
        report.update(straightness(lambda xx, tt: velocity(xx, tt, cond), z, n_steps=50))

        # open-loop conditional accuracy -- expected to degrade for rot couplings
        for n in (1, 2, 4, 10, 50):
            pred = policy.normalizer["action"].unnormalize(sample_chunk(cond, z, n))
            report[f"action_mse_N{n}"] = float(((pred - gt) ** 2).mean())

        th_gen, _ = chunk_heading(sample_chunk(cond, z), spec)
        th_gt, _ = chunk_heading(x1, spec)
        report["heading_MAE"] = float(
            torch.abs(torch.remainder(th_gen - th_gt + torch.pi, 2 * torch.pi) - torch.pi).mean()
        )
        report["gen_heading_R1"] = circular_moments(th_gen)["R1"]
        report["gt_heading_R1"] = circular_moments(th_gt)["R1"]

        # field equivariance on four spatial distributions
        t_mid = torch.full((z.shape[0],), 0.5, device=device, dtype=z.dtype)
        traj = sample_chunk(cond, z, policy.num_inference_steps)
        dists = {
            "train_paths": 0.5 * z + 0.5 * x1,           # this method's interpolation
            "common_iid": 0.5 * z + 0.5 * sample_prior(z.shape[0], device, z.dtype),
            "inference_traj": traj,
            "off_manifold": 2.0 * sample_prior(z.shape[0], device, z.dtype),
        }
        for name, pts in dists.items():
            r = field_equivariance(lambda xx, tt: velocity(xx, tt, cond), pts, t_mid, spec, n_angles)
            report[f"field_eq_{name}"] = r["field_eq_rel"]

    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    print(f"\n{'metric':<28}value")
    print("-" * 44)
    for k in ("orbit_steering_gain", "src_orbit_eq_rel", "heading_MAE", "action_mse_N10",
              "few_step_gap_N1", "few_step_gap_N4", "straightness",
              "field_eq_train_paths", "field_eq_common_iid", "field_eq_inference_traj"):
        if k in report:
            print(f"{k:<28}{report[k]:.5f}")
    print(f"\nwrote {out}")
    # A near-zero gain is only a problem on an arm that actually applies a coupling.
    # The old guard compared against the literal string "iid(baseline)", so it fired on
    # every orbit-config checkpoint -- including mode=iid, where a zero gain is REQUIRED.
    _c = report.get("coupling")
    _mode = _c.get("mode", "iid") if isinstance(_c, dict) else "iid"
    _is_coupled = _mode in ("rot", "perm", "perm_rot")
    if abs(report.get("orbit_steering_gain", 0.0)) < 0.2 and _is_coupled:
        print("\n!! steering gain is near zero on a coupled checkpoint. Stage 1 of DNO will")
        print("   behave as best-of-K resampling. Check mode/align/kappa before running DNO.")


if __name__ == "__main__":
    main()
