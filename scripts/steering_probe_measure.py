#!/usr/bin/env python3
"""
Measure stage of the steering probe: one JSON per arm, holding the raw (phi, dtheta) points.

Split from rendering so the expensive part (checkpoint load + K forward passes) runs once and
the figure can be restyled for free.

Three things must be identical across arms or the overlay is not a comparison:

  1. the observations  -- a fixed validation batch, drawn deterministically from --obs-seed
                          and CACHED to disk, so every arm sees byte-identical inputs (and
                          the ~15 GB dataset load happens once, not once per arm)
  2. the source noise  -- z fixed from --z-seed before any rotation
  3. the readout spec  -- forced identically, NOT taken from each checkpoint's own config.
                          See oat/symmetry/steering_probe.py for why this is the one that
                          silently breaks the figure.

Usage:
    # cache the validation batch once (loads the dataset; run it under memory protection)
    python scripts/steering_probe_measure.py --cache-only -c <any ckpt>

    # then one cheap call per arm
    python scripts/steering_probe_measure.py -c .../P1_norm_only/... -o .../probe_P1.json --label P1
    python scripts/steering_probe_measure.py -c .../P4_rot_heading/... -o .../probe_P4.json --label P4
"""

from __future__ import annotations

import importlib.util
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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from oat.policy.base_policy import BasePolicy
from oat.symmetry.so2_chunk import SO2ChunkSpec
from oat.symmetry.steering_probe import orbit_steering_points, readout_spec_signature

_HERE = pathlib.Path(__file__).parent


def _load_make_samplers():
    """Reuse report_coupling_diagnostics.make_samplers rather than duplicating the shim.

    scripts/ is not a package, so import it by path.
    """
    p = _HERE / "report_coupling_diagnostics.py"
    spec = importlib.util.spec_from_file_location("_rcd", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rcd"] = mod
    spec.loader.exec_module(mod)
    return mod.make_samplers


READOUT_SPECS = {
    "translation_only": SO2ChunkSpec.translation_only,
    "libero_osc_pose": SO2ChunkSpec.libero_osc_pose,
}


def build_obs_cache(cfg, cache_path: pathlib.Path, batch_size: int, obs_seed: int):
    """Draw one validation batch deterministically and cache it."""
    print(f"building observation cache (loads the dataset) -> {cache_path}")
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    val = dataset.get_validation_dataset()
    g = torch.Generator().manual_seed(int(obs_seed))
    loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    batch = next(iter(loader))
    obs = {k: v for k, v in batch["obs"].items() if torch.is_tensor(v)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": obs, "action": batch["action"], "obs_seed": int(obs_seed),
                "batch_size": int(batch_size)}, cache_path)
    print(f"  cached {batch_size} validation observations")
    return {"obs": obs, "action": batch["action"]}


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output", default=None, help="output JSON (omit with --cache-only)")
@click.option("--label", default=None, help="arm name for the legend; defaults to the run dir")
@click.option("-d", "--device", default="cuda:0")
@click.option("-b", "--batch-size", default=64, help="validation observations")
@click.option("--n-angles", default=24)
@click.option("--obs-seed", default=1234)
@click.option("--z-seed", default=5678)
@click.option("--readout-spec", default="translation_only",
              type=click.Choice(sorted(READOUT_SPECS)))
@click.option("--conf-threshold", default=0.05)
@click.option("--num-inference-steps", default=None, type=int)
@click.option("--inference-prior", default=None, type=click.Choice(["standard", "matched"]),
              help="P4 extra arm. The probe asks what the MAP does with a rotated source, so "
                   "'standard' is the default; 'matched' changes the starting distribution.")
@click.option("--obs-cache", default="output/exp/probe/val_batch.pt")
@click.option("--cache-only", is_flag=True, help="build the observation cache and exit")
def main(checkpoint, output, label, device, batch_size, n_angles, obs_seed, z_seed,
         readout_spec, conf_threshold, num_inference_steps, inference_prior,
         obs_cache, cache_only):

    dev = torch.device(device)
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)

    cache_path = pathlib.Path(obs_cache)
    if cache_only:
        build_obs_cache(cfg, cache_path, batch_size, obs_seed)
        return
    if not cache_path.exists():
        build_obs_cache(cfg, cache_path, batch_size, obs_seed)
    cached = torch.load(cache_path, weights_only=False)
    if int(cached.get("obs_seed", -1)) != int(obs_seed) or int(cached.get("batch_size", -1)) != int(batch_size):
        raise SystemExit(
            f"{cache_path} was built with obs_seed={cached.get('obs_seed')} "
            f"batch_size={cached.get('batch_size')}, but this call asks for "
            f"{obs_seed}/{batch_size}. Delete the cache or match the arguments — silently "
            f"reusing a different batch would break the overlay.")

    obs = {k: v.to(dev) for k, v in cached["obs"].items()}

    policy.to(dev).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)
    if inference_prior is not None:
        if not hasattr(policy, "inference_prior"):
            raise SystemExit("--inference-prior needs an OrbitFlowPolicy checkpoint")
        policy.inference_prior = inference_prior

    encode_obs, velocity, sample_chunk, sample_prior = _load_make_samplers()(policy)
    spec = READOUT_SPECS[readout_spec]()

    with torch.no_grad():
        cond = encode_obs(obs)
        B = cond.shape[0]
        # z fixed BEFORE any rotation, identical across arms (CPU generator -> reproducible
        # regardless of device), at the policy's own prior scale.
        g = torch.Generator().manual_seed(int(z_seed))
        z = torch.randn(B, policy.horizon, policy.action_dim, generator=g).to(
            device=dev, dtype=cond.dtype) * float(getattr(policy, "prior_noise_scale", 1.0))
        if inference_prior == "matched":
            # deliberate deviation for this arm only: same map, different source law
            z = policy.apply_inference_prior(z)
        res = orbit_steering_points(lambda zz: sample_chunk(cond, zz), z, spec,
                                    n_angles=n_angles, conf_threshold=conf_threshold)

    run_dir = pathlib.Path(checkpoint).parent.parent
    epoch = None
    logs = run_dir / "logs.json"
    if logs.exists():
        try:
            rows = [json.loads(l) for l in open(logs) if l.strip()]
            epoch = int(rows[-1]["epoch"])
        except Exception:
            pass

    coupling = (OmegaConf.to_container(cfg.policy.coupling, resolve=True)
                if "coupling" in cfg.policy else {"mode": "iid(baseline FlowPolicy)"})
    train_spec = (OmegaConf.to_container(cfg.policy.action_spec, resolve=True)
                  if "action_spec" in cfg.policy else None)

    res.update({
        "label": label or run_dir.parent.name,
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "coupling": coupling,
        "normalizer_mode": cfg.policy.get("normalizer_mode", "limits(baseline)"),
        "trained_action_spec": train_spec,
        "readout_spec_name": readout_spec,
        "readout_spec": readout_spec_signature(spec),
        "inference_prior": inference_prior or getattr(policy, "inference_prior", "standard"),
        "num_inference_steps": int(policy.num_inference_steps),
        "obs_seed": int(obs_seed),
        "z_seed": int(z_seed),
        "batch_size": int(B),
        "prior_noise_scale": float(getattr(policy, "prior_noise_scale", 1.0)),
    })

    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True))

    print(f"\n{res['label']}  (epoch {epoch}, prior={res['inference_prior']})")
    print(f"  gain               {res['gain']:+.5f}")
    print(f"  phase consistency  {res['phase_consistency']:.5f}")
    print(f"  low-confidence     {res['low_confidence_frac']:.4f} "
          f"(< {conf_threshold}), gain among the rest {res['gain_high_conf']:+.5f}")
    print(f"  points             {res['n_points']} = {res['n_obs']} obs x {res['n_angles']} angles")
    print(f"  readout spec       {res['readout_spec']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
