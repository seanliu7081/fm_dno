#!/usr/bin/env python3
"""
Stage 0 gate: audit LIBERO-10's action-chunk geometry before spending a single GPU-hour.

Four questions, all cheap, all decisive:

  1  How non-uniform are the planar chunk headings?  ``W_1(p_theta, Uniform)`` is the lower
     bound on angular transport for any marginal-preserving coupling.  Near 0 means the
     double-ring situation holds and plain ``rot`` is safe; large means the obstruction is
     real and you need ``perm`` or the matched prior.
  2  How much chunk energy actually lives in each frequency-1 block?  If ``(wx, wy)`` holds
     a few percent, the coupling is translation-only and the writeup must say so.
  3  How many chunks have a meaningful heading at all?  LIBERO demos contain long
     near-stationary segments; those contribute noise to any angular coupling.
  4  What does the repo's per-dimension normalizer do to the group action, on real data?

Reads only the ``action`` array out of the zarr, so it costs seconds and a few hundred MB,
not the 3.4 GB the full dataset would.

Usage:
    python scripts/libero_heading_audit.py
    python scripts/libero_heading_audit.py --zarr data/libero/libero10_N500.zarr --horizon 16
    python scripts/libero_heading_audit.py -o output/audit/libero10.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from oat.common.replay_buffer import ReplayBuffer  # noqa: E402
from oat.symmetry.coupling import MatchedHeadingPrior, SO2OrbitCoupling  # noqa: E402
from oat.symmetry.metrics import transport_stats  # noqa: E402
from oat.symmetry.normalizer import (  # noqa: E402
    equivariance_error,
    fit_so2_action_scale_offset,
)
from oat.symmetry.so2_chunk import (  # noqa: E402
    SO2ChunkSpec,
    chunk_heading,
    circular_moments,
    vector_energy_fraction,
)


def build_chunks(actions: np.ndarray, episode_ends: np.ndarray, horizon: int,
                 stride: int, max_chunks: int, seed: int = 0) -> torch.Tensor:
    """Non-overlapping-ish chunks that never cross an episode boundary."""
    starts = []
    prev = 0
    for end in episode_ends:
        for s in range(prev, int(end) - horizon + 1, stride):
            starts.append(s)
        prev = int(end)
    rng = np.random.default_rng(seed)
    if len(starts) > max_chunks:
        starts = rng.choice(np.asarray(starts), size=max_chunks, replace=False)
    idx = np.asarray(starts)[:, None] + np.arange(horizon)[None, :]
    return torch.from_numpy(actions[idx].astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/libero/libero10_N500.zarr")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-chunks", type=int, default=40000)
    ap.add_argument("--action-key", default="action")
    ap.add_argument("-o", "--output", default="output/audit/libero10_heading_audit.json")
    args = ap.parse_args()

    spec = SO2ChunkSpec.libero_osc_pose()
    print(f"loading actions from {args.zarr} ...")
    rb = ReplayBuffer.copy_from_path(args.zarr, keys=[args.action_key])
    actions = np.asarray(rb[args.action_key])
    episode_ends = np.asarray(rb.episode_ends)
    print(f"  {actions.shape[0]} steps, {len(episode_ends)} episodes, action_dim={actions.shape[1]}")
    if actions.shape[1] != spec.action_dim:
        raise SystemExit(
            f"action_dim={actions.shape[1]} but the SO(2) spec expects {spec.action_dim}. "
            f"Edit the spec before going further."
        )

    raw = build_chunks(actions, episode_ends, args.horizon, args.stride, args.max_chunks)
    print(f"  {raw.shape[0]} chunks of horizon {args.horizon}\n")

    report = {"zarr": args.zarr, "horizon": args.horizon, "n_chunks": int(raw.shape[0])}

    # ---- normalizers -----------------------------------------------------------------
    flat = raw.reshape(-1, spec.action_dim)
    lo, hi = flat.min(0).values, flat.max(0).values
    rng = (hi - lo).clamp_min(1e-4)
    mm_scale, mm_offset = 2.0 / rng, -1.0 - (2.0 / rng) * lo
    e_mm = equivariance_error(raw, mm_scale, mm_offset, spec)
    so2_scale, so2_offset = fit_so2_action_scale_offset(raw, spec, vector_mode="rms")
    e_so2 = equivariance_error(raw, so2_scale, so2_offset, spec)

    print("=" * 78)
    print("1  NORMALIZER")
    print("=" * 78)
    print(f"  per-dim min-max (repo default) : equivariance err {e_mm:.3e}")
    print(f"      scale(dx)={float(mm_scale[0]):.4f}  scale(dy)={float(mm_scale[1]):.4f}"
          f"  offset(dx)={float(mm_offset[0]):+.4f}")
    print(f"  SO(2) block (rms)              : equivariance err {e_so2:.3e}")
    print(f"      scale(dx)=scale(dy)={float(so2_scale[0]):.4f}  offset 0")
    report["normalizer"] = {
        "minmax_equivariance_error": e_mm,
        "so2_equivariance_error": e_so2,
        "so2_scale": [float(v) for v in so2_scale],
        "so2_offset": [float(v) for v in so2_offset],
    }
    x = raw * so2_scale + so2_offset      # everything below lives in the equivariant space

    # ---- energy ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2  PER-BLOCK ENERGY  (after SO(2) normalization)")
    print("=" * 78)
    energy = vector_energy_fraction(x, spec)
    for k, v in energy.items():
        print(f"  {k:<28}{v:8.4f}")
    report["energy"] = energy
    v1 = energy.get("vec1_energy_frac", 0.0)
    if v1 < 0.05:
        print(f"\n  !! (wx,wy) carries {v1:.1%} of chunk energy. The coupling is effectively")
        print(f"     translation-only. Use SO2ChunkSpec.translation_only() and say so.")

    # ---- headings --------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3  HEADINGS  -- the go / no-go gate")
    print("=" * 78)
    theta, conf = chunk_heading(x, spec)
    moments = circular_moments(theta, orders=(1, 2, 4, 8))
    w1 = MatchedHeadingPrior(spec).uniformity_gap(x)
    for k, v in moments.items():
        print(f"  heading {k:<22}{v:8.4f}")
    print(f"  W1(heading, uniform)          {w1:8.4f} rad")
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    cq = torch.quantile(conf, qs)
    print(f"  heading confidence quantiles  "
          + "  ".join(f"p{int(q*100)}={float(c):.3f}" for q, c in zip(qs, cq)))
    for thr in (0.02, 0.05, 0.10):
        print(f"    fraction with confidence < {thr:<5} : {float((conf < thr).float().mean()):.3f}")
    report["heading"] = {
        **moments,
        "W1_to_uniform": w1,
        "confidence_quantiles": {f"p{int(q*100)}": float(c) for q, c in zip(qs, cq)},
        "low_confidence_frac": {str(t): float((conf < t).float().mean()) for t in (0.02, 0.05, 0.10)},
    }

    print("\n  VERDICT")
    if w1 < 0.10:
        print("    W1 is small: headings are close to uniform, the S5 obstruction does not")
        print("    bite on this dataset, and mode='rot' is safe with the standard prior.")
        report["verdict"] = "rot_safe"
    elif w1 < 0.40:
        print("    W1 is moderate: rot will shift the prior noticeably. Run P4 and P5 and")
        print("    compare -- the matched prior should recover a measurable amount.")
        report["verdict"] = "matched_prior_recommended"
    else:
        print("    W1 is large: rot with the standard N(0,I) inference prior is a real")
        print("    train/test mismatch. Treat P5 (matched prior) as the primary rot variant")
        print("    and P2 (perm, angle) as the marginal-exact comparison.")
        report["verdict"] = "matched_prior_required"

    # ---- what each coupling would do -------------------------------------------------
    print("\n" + "=" * 78)
    print("4  COUPLING PREVIEW  (batch 256, real chunks, no training)")
    print("=" * 78)
    print(f"  angle gap cannot go below W1 = {w1:.4f} for any marginal-preserving coupling\n")
    print(f"  {'coupling':<22}{'path len':>10}{'angle gap':>11}{'cond var':>11}{'src R1':>9}")
    variants = {
        "A/iid": dict(mode="iid"),
        "P2/perm-angle": dict(mode="perm", cost="angle"),
        "P3/perm-euclid": dict(mode="perm", cost="euclidean"),
        "P4/rot-heading": dict(mode="rot", align="heading"),
        "P6/perm+rot-group": dict(mode="perm_rot", cost="group_ot"),
    }
    g = torch.Generator().manual_seed(0)
    n_batches = min(24, x.shape[0] // 256)
    report["coupling_preview"] = {}
    for name, cfg in variants.items():
        c = SO2OrbitCoupling(spec, **cfg)
        rows, r1s = [], []
        for b in range(n_batches):
            x1 = x[b * 256 : (b + 1) * 256]
            z0 = torch.randn(x1.shape, generator=g)
            zc, d = c(z0, x1)
            rows.append(transport_stats(zc, x1, spec))
            r1s.append(d["coupling/src_heading_R1"])
        agg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        r1 = sum(r1s) / len(r1s)
        print(f"  {name:<22}{agg['path_len']:>10.4f}{agg['angle_gap']:>11.4f}"
              f"{agg['cond_vel_var']:>11.4f}{r1:>9.4f}")
        report["coupling_preview"][name] = {**agg, "src_heading_R1": r1}

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
