#!/usr/bin/env python3
"""
Generate the ``(z, F(z|o))`` pairs that Arm R (reflow) trains on.

For every window of the train and validation splits, draw one source ``z`` from a
*per-index* generator, integrate the frozen donor policy for ``--n-gen`` Euler steps, and
record the endpoint.  Retraining the same architecture on those pairs is the rectified-flow
"reflow" step (Liu et al., arXiv:2209.03003): the endpoints stay where the donor's sampler
put them, while the paths get straighter -- which is the property a 1- or 2-step Euler
sampler needs.  See ``PLAN_fewstep_coupling.md`` S3.

WHAT IS STORED, AND WHY EACH PIECE IS THERE
-------------------------------------------
Per split, one ``.npz``:

    z           (L, H, A) float32   the source, in prior space, already scaled by
                                    ``prior_noise_scale`` -- i.e. exactly the tensor that
                                    was integrated, not something to be rescaled later
    x1_norm     (L, H, A) float32   the Euler endpoint in NORMALIZED space
    action_env  (L, H, A) float32   the same endpoint in env units; this is what replaces
                                    ``batch['action']``, so every downstream consumer
                                    (reconstruction MSE, rollout comparison) keeps working
    norm_*                          a snapshot of the donor's action normalizer
    meta_json                       donor checkpoint + epoch, N_gen, z_seed, and the
                                    dataset fingerprint

``x1_norm`` is redundant with ``action_env`` up to the normalizer -- and that is the point:
the pairs live in the generation-time normalized space, so storing both lets Gate R1 prove
at training start that the rebuilt normalizer still maps one onto the other.

DETERMINISM
-----------
``z_i`` comes from ``torch.Generator().manual_seed(z_seed * 1_000_003 + i)``, so it depends
on the window index alone: regenerating is reproducible, and a second pair set over the
same windows is one ``--z-seed`` away (that is the plan's R4 retry).

The endpoint is *not* bit-reproducible, and deliberately so: this repo's config leaves
``eval_fixed_crop=False``, so robomimic's ``CropRandomizer`` takes a **random** crop even in
eval mode -- during rollouts too.  Distilling the sampler that actually scored 0.400 means
distilling it under the crop distribution it actually runs under.  The global RNG is seeded
per batch anyway so a rerun with the same batch layout reproduces; ``--center-crop`` swaps
in the repo's deterministic center-crop randomizer if you want the other trade.

Usage:
    MUJOCO_GL=egl python scripts/generate_reflow_pairs.py \
        -c output/exp/P1_curve/seed42/checkpoints/ep-0350_sr-0.400.ckpt \
        -o output/reflow_pairs/P1_curve_ep0350_N10_z0 \
        --n-gen 10 --z-seed 0 --batch-size 128 -d cuda:0
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from oat.common.pytorch_util import dict_apply, maybe_to_device, replace_submodules
from oat.dataset.reflow_pairs import dataset_fingerprint
from oat.policy.base_policy import BasePolicy

_Z_SEED_STRIDE = 1_000_003


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def make_hooks(policy):
    """(encode_obs, sample_chunk) for OrbitFlowPolicy or a plain FlowPolicy checkpoint."""
    if hasattr(policy, "sample_chunk"):
        return policy.encode_obs, policy.sample_chunk

    def encode_obs(obs):
        return policy.obs_encoder(obs)

    def sample_chunk(cond, z, n_steps=None):
        n = n_steps or policy.num_inference_steps
        dt = 1.0 / n
        x = z
        for i in range(n):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * policy.model(x, policy._scale_t(t), cond)
        return x

    return encode_obs, sample_chunk


def draw_sources(n: int, horizon: int, action_dim: int, z_seed: int, scale: float) -> np.ndarray:
    """One ``z`` per dataset index, from a generator seeded by that index alone."""
    out = np.empty((n, horizon, action_dim), dtype=np.float32)
    g = torch.Generator()
    for i in range(n):
        g.manual_seed(int(z_seed) * _Z_SEED_STRIDE + i)
        out[i] = (scale * torch.randn(horizon, action_dim, generator=g)).numpy()
    return out


def normalizer_arrays(policy) -> dict:
    params = policy.normalizer.params_dict["action"]
    out = {
        "norm_scale": params["scale"].detach().float().cpu().numpy(),
        "norm_offset": params["offset"].detach().float().cpu().numpy(),
    }
    for k, v in params["input_stats"].items():
        out[f"norm_stats_{k}"] = v.detach().float().cpu().numpy()
    return out


def compare_normalizers(a: dict, b: dict) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("inf")
    return max(float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).max()) for k in keys)


def read_checkpoint_epoch(checkpoint: str):
    """``ep-0350_sr-0.400.ckpt`` -> 350.  Falls back to reading the payload's pickled epoch."""
    import re

    m = re.search(r"ep-(\d+)", pathlib.Path(checkpoint).name)
    if m:
        return int(m.group(1))
    try:
        import dill

        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        return int(dill.loads(payload["pickles"]["epoch"]))
    except Exception as e:  # pragma: no cover - metadata, not load-bearing
        print(f"  (could not read the donor's epoch: {e})")
        return None


def use_center_crop(policy) -> None:
    """Swap robomimic's always-random CropRandomizer for the repo's eval-deterministic one."""
    import robomimic.models.base_nets as rmbn

    from oat.perception.crop_randomizer import CropRandomizer

    enc = policy.obs_encoder.vision_encoder
    replace_submodules(
        root_module=enc,
        predicate=lambda m: isinstance(m, rmbn.CropRandomizer),
        func=lambda m: CropRandomizer(
            input_shape=m.input_shape, crop_height=m.crop_height,
            crop_width=m.crop_width, num_crops=m.num_crops, pos_enc=m.pos_enc,
        ),
    )


# --------------------------------------------------------------------------------------
# one split
# --------------------------------------------------------------------------------------


@torch.no_grad()
def generate_split(policy, dataset, split, zarr_path, args, meta_common):
    encode_obs, sample_chunk = make_hooks(policy)
    device = next(policy.parameters()).device
    L = len(dataset)
    limit = min(L, args["limit"]) if args["limit"] else L
    horizon, action_dim = policy.horizon, policy.action_dim

    print(f"\n[{split}] {L} windows"
          + (f"  (TRUNCATED to {limit} -- this file is NOT trainable)" if limit < L else ""))
    t0 = time.time()
    z_all = draw_sources(limit, horizon, action_dim, args["z_seed"],
                         float(policy.prior_noise_scale))
    print(f"[{split}] drew {limit} sources in {time.time() - t0:.1f}s "
          f"(z_seed={args['z_seed']}, sigma={policy.prior_noise_scale})")

    x1_norm = np.empty_like(z_all)
    action_env = np.empty_like(z_all)

    # Sequential sampler + drop_last=False: batch b covers indices [b*B, (b+1)*B), which is
    # what makes `row i of the file` == `dataset index i`.
    loader = DataLoader(
        dataset, batch_size=args["batch_size"], shuffle=False,
        num_workers=args["num_workers"], pin_memory=True, drop_last=False,
        persistent_workers=False,
    )

    done = 0
    t0 = time.time()
    for b, batch in enumerate(loader):
        if done >= limit:
            break
        # seeded per batch so a rerun with the same batch layout reproduces the crops
        torch.manual_seed(args["z_seed"] * _Z_SEED_STRIDE + 7_919 * (b + 1))
        n = min(batch["action"].shape[0], limit - done)
        obs = dict_apply(batch["obs"], lambda x: maybe_to_device(x, device))
        obs = {k: v[:n] for k, v in obs.items()}

        z = torch.from_numpy(z_all[done:done + n]).to(device)
        cond = encode_obs(obs)
        if cond.shape[0] != n:
            raise RuntimeError(f"cond batch {cond.shape[0]} != {n}; obs slicing went wrong")
        x = sample_chunk(cond, z.to(cond.dtype), n_steps=args["n_gen"])
        a = policy.normalizer["action"].unnormalize(x)

        x1_norm[done:done + n] = x.float().cpu().numpy()
        action_env[done:done + n] = a.float().cpu().numpy()
        done += n
        if b % 100 == 0 or done >= limit:
            el = time.time() - t0
            rate = done / max(el, 1e-6)
            print(f"[{split}] {done}/{limit}  {rate:7.1f} win/s  "
                  f"eta {(limit - done) / max(rate, 1e-6):6.0f}s", flush=True)

    if done != limit:
        raise RuntimeError(f"[{split}] produced {done} rows, expected {limit}")

    # --- offline sanity, printed so a bad generation is visible immediately -------------
    path_len = float(np.linalg.norm((x1_norm - z_all).reshape(limit, -1), axis=1).mean())
    round_trip = float(np.abs(
        policy.normalizer["action"].normalize(
            torch.from_numpy(action_env[: min(4096, limit)]).to(device)
        ).float().cpu().numpy() - x1_norm[: min(4096, limit)]
    ).max())
    stats = {
        "path_len_mean": path_len,
        "x1_norm_absmax": float(np.abs(x1_norm).max()),
        "x1_norm_rms": float(np.sqrt((x1_norm ** 2).mean())),
        "action_env_absmax": float(np.abs(action_env).max()),
        "normalize_roundtrip_absmax": round_trip,
        "n_nonfinite": int((~np.isfinite(x1_norm)).sum()),
    }
    print(f"[{split}] path_len {path_len:.4f}  |x1|rms {stats['x1_norm_rms']:.4f}  "
          f"roundtrip {round_trip:.2e}  nonfinite {stats['n_nonfinite']}")
    if stats["n_nonfinite"]:
        raise RuntimeError(f"[{split}] {stats['n_nonfinite']} non-finite endpoints -- "
                           f"the donor sampler diverged; refusing to write.")
    if round_trip > 1e-5:
        raise RuntimeError(
            f"[{split}] normalize(unnormalize(x)) departs by {round_trip:.2e}. The stored "
            f"pairs would not survive Gate R1; refusing to write."
        )

    meta = dict(meta_common)
    meta.update({
        "split": split,
        "n_windows": int(limit),
        "truncated": bool(limit < L),
        "fingerprint": dataset_fingerprint(dataset, zarr_path),
        "stats": stats,
    })
    return z_all, x1_norm, action_env, meta


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


@click.command()
@click.option("-c", "--checkpoint", required=True, help="donor policy (EMA weights are used)")
@click.option("-o", "--output-dir", required=True, help="written: train.npz, val.npz, meta.json")
@click.option("-d", "--device", default="cuda:0")
@click.option("--n-gen", default=10, type=int,
              help="Euler steps used to make the targets. Default 10 = the donor's own "
                   "operating point: we distil the sampler that scored the reported SR, "
                   "not a hypothetical finer one.")
@click.option("--z-seed", default=0, type=int,
              help="per-index source seed. A second pair set over the same windows is one "
                   "new value away -- that is the plan's R4 retry.")
@click.option("--batch-size", default=128, type=int)
@click.option("--num-workers", default=8, type=int)
@click.option("--splits", default="train,val")
@click.option("--refit-normalizer", is_flag=True,
              help="fit the action normalizer from the dataset instead of using the donor's "
                   "stored one. Only for smoke tests / a deliberately different dataset; the "
                   "real run must use the donor's, and the check below proves they agree.")
@click.option("--center-crop", is_flag=True,
              help="deterministic center crops instead of the random crops the rollout "
                   "actually uses. Changes the observation distribution being distilled.")
@click.option("--limit", default=0, type=int,
              help="stop after N windows per split. Writes a file marked truncated, which "
                   "ReflowPairDataset refuses to train on. Debug only.")
@click.option("--dataset-override", multiple=True,
              help="key=value applied to cfg.task.policy.dataset, e.g. zarr_path=... "
                   "(repeatable). Anything you change here you must also change on the "
                   "training side, or Gate R2 will fire.")
def main(checkpoint, output_dir, device, n_gen, z_seed, batch_size, num_workers, splits,
         refit_normalizer, center_crop, limit, dataset_override):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    device = torch.device(device)
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    print(f"donor: {type(policy).__name__}  from {checkpoint}")

    ckpt_epoch = read_checkpoint_epoch(checkpoint)

    if center_crop:
        use_center_crop(policy)
        print("  observation encoder: DETERMINISTIC center crop")

    # ---- dataset ---------------------------------------------------------------------
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.policy.dataset, resolve=True))
    with open_dict(ds_cfg):
        for ov in dataset_override:
            k, _, v = ov.partition("=")
            ds_cfg[k] = yaml.safe_load(v)
    zarr_path = str(ds_cfg["zarr_path"])
    print(f"dataset: {ds_cfg['_target_']}  zarr={zarr_path}")
    dataset = hydra.utils.instantiate(ds_cfg)

    # ---- normalizer (earliest possible read of Gate R1) --------------------------------
    donor_norm_backup = {k: v.copy() for k, v in normalizer_arrays(policy).items()}
    fresh = dataset.get_normalizer()
    policy.set_normalizer(fresh)                      # re-applies the SO(2) repair
    policy.to(device)                                 # set_normalizer reloads params on CPU
    fresh_norm = normalizer_arrays(policy)
    drift = compare_normalizers(donor_norm_backup, fresh_norm)
    print(f"normalizer: |checkpoint - dataset refit| = {drift:.3e}")
    if refit_normalizer:
        print("  --refit-normalizer: keeping the DATASET's fit. The training run must "
              "rebuild the same one or Gate R1 will fire.")
    else:
        # restore the donor's own map: it is the one the donor was trained under
        params = policy.normalizer.params_dict["action"]
        with torch.no_grad():
            params["scale"].data = torch.as_tensor(
                donor_norm_backup["norm_scale"], device=params["scale"].device)
            params["offset"].data = torch.as_tensor(
                donor_norm_backup["norm_offset"], device=params["offset"].device)
        if drift > 1e-6:
            raise SystemExit(
                f"the donor's action normalizer and a fresh fit on {zarr_path} differ by "
                f"{drift:.3e} (> 1e-6). Reflow training rebuilds the normalizer from the "
                f"dataset, so these pairs would fail Gate R1. Either point at the dataset "
                f"the donor was trained on, or pass --refit-normalizer and accept that "
                f"this is no longer a distillation of the reported checkpoint."
            )

    meta_common = {
        "checkpoint": str(pathlib.Path(checkpoint).resolve()),
        "checkpoint_epoch": ckpt_epoch,
        "policy_class": type(policy).__name__,
        "n_gen": int(n_gen),
        "z_seed": int(z_seed),
        "z_seed_stride": _Z_SEED_STRIDE,
        "prior_noise_scale": float(policy.prior_noise_scale),
        "horizon": int(policy.horizon),
        "action_dim": int(policy.action_dim),
        "normalizer_source": "dataset_refit" if refit_normalizer else "donor_checkpoint",
        "normalizer_drift_vs_dataset": float(drift),
        "center_crop": bool(center_crop),
        "batch_size": int(batch_size),
        "dataset_override": list(dataset_override),
        "generator": "scripts/generate_reflow_pairs.py",
    }
    args = dict(n_gen=n_gen, z_seed=z_seed, batch_size=batch_size,
                num_workers=num_workers, limit=limit)
    norm_arrays = normalizer_arrays(policy)
    dev_now = policy.normalizer.params_dict["action"]["scale"].device
    if dev_now.type != device.type:
        raise RuntimeError(f"action normalizer is on {dev_now}, policy on {device}")

    written = []
    for split in [s.strip() for s in splits.split(",") if s.strip()]:
        ds = dataset if split == "train" else dataset.get_validation_dataset()
        z, x1n, aenv, meta = generate_split(policy, ds, split, zarr_path, args, meta_common)
        path = out / f"{split}.npz"
        np.savez(path, z=z, x1_norm=x1n, action_env=aenv,
                 meta_json=np.array(json.dumps(meta, sort_keys=True)), **norm_arrays)
        mb = path.stat().st_size / 1e6
        print(f"[{split}] wrote {path}  ({mb:.0f} MB, {len(z)} pairs)")
        written.append(meta)

    (out / "meta.json").write_text(json.dumps(
        {"common": meta_common, "splits": {m["split"]: m for m in written}},
        indent=2, sort_keys=True))
    print(f"\nwrote {out}/meta.json")
    print("next:\n"
          f"  policy.init_ckpt={checkpoint}\n"
          f"  task.policy.dataset.pairs_path={out}")


if __name__ == "__main__":
    main()
