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

Usage:
    MUJOCO_GL=egl python scripts/report_coupling_diagnostics.py \
        -c output/<run>/checkpoints/latest.ckpt -o output/diag/<name>.json
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


def make_samplers(policy):
    """(encode_obs, velocity, sample_chunk, sample_prior) for FlowPolicy or OrbitFlowPolicy."""
    if hasattr(policy, "sample_chunk"):
        return (policy.encode_obs, policy.velocity, policy.sample_chunk, policy.sample_prior)

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
def main(checkpoint, output, device, batch_size, n_angles):
    device = torch.device(device)
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    spec = getattr(policy, "action_spec", None) or SO2ChunkSpec.libero_osc_pose()
    encode_obs, velocity, sample_chunk, sample_prior = make_samplers(policy)

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    val = dataset.get_validation_dataset()
    batch = next(iter(DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=2)))
    obs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["obs"].items()}
    gt = batch["action"].to(device)

    report = {
        "checkpoint": checkpoint,
        "coupling": (OmegaConf.to_container(cfg.policy.coupling, resolve=True)
                     if "coupling" in cfg.policy else "iid(baseline)"),
        "normalizer_mode": cfg.policy.get("normalizer_mode", "limits(baseline)"),
        "batch_size": int(gt.shape[0]),
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
    if abs(report.get("orbit_steering_gain", 0.0)) < 0.2 and report["coupling"] != "iid(baseline)":
        print("\n!! steering gain is near zero on a coupled checkpoint. Stage 1 of DNO will")
        print("   behave as best-of-K resampling. Check mode/align/kappa before running DNO.")


if __name__ == "__main__":
    main()
