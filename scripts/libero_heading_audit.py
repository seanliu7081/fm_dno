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

GATE F0c (PLAN_fewstep_coupling.md S2.3)
----------------------------------------
Section 4's coupling preview was originally measured at ``scalar_weight=0.0`` and B=256 --
i.e. with the assignment cost blind to the invariant channels, which on LIBERO-10 carry
84.5% of the raw chunk energy, and at a batch size no training arm uses.  ``--scalar-weight``,
``--batch-size`` and ``--block-weights`` re-measure it under the settings the arms actually
train with, and the preview now prints the path-length reduction against the iid row, which
is the number F0c gates on:

    B=128 improves the reduction by < 2 points over B=32   -> drop the P3b batch-size arm
    euclidean @ scalar_weight=1.0 still reads < 10%        -> P3's prior drops accordingly

**Every default is unchanged on purpose** (``scalar_weight=0.0``, ``batch_size=256``,
``block_weights=None``), so an invocation written before these flags existed still audits
the same configuration.  Two things about section 4's numbers did change and are not
settings: the source draws are now reseeded per variant, so every coupling is scored on the
*same* noise and a row-to-row path-length difference is the assignment rather than the draw;
and the chunk count is held fixed as ``--batch-size`` varies, so a B=32-vs-B=128 comparison
changes the assignment problem and nothing else.

Usage:
    python scripts/libero_heading_audit.py
    python scripts/libero_heading_audit.py --zarr data/libero/libero10_N500.zarr --horizon 16
    python scripts/libero_heading_audit.py -o output/audit/libero10.json

    # F0c: the settings every coupling arm actually trains with
    python scripts/libero_heading_audit.py --scalar-weight 1.0 --block-weights 1,0 \
        --batch-size 32  -o output/audit/f0c_sw1_b32.json
    python scripts/libero_heading_audit.py --scalar-weight 1.0 --block-weights 1,0 \
        --batch-size 128 -o output/audit/f0c_sw1_b128.json
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
    # ---- F0c knobs.  Defaults reproduce the original audit exactly. ------------------
    ap.add_argument("--scalar-weight", type=float, default=0.0,
                    help="weight of the SO(2)-invariant channels in the assignment cost. "
                         "The arms train at 1.0; the original audit measured 0.0.")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="assignment batch for section 4. Minibatch OT strengthens with B "
                         "and its conditional bias grows with it -- measure, do not assume.")
    ap.add_argument("--block-weights", default=None,
                    help="comma-separated per-vector-block weights, e.g. '1,0' for the "
                         "translation-only coupling every arm uses (G4'). Default: uniform.")
    args = ap.parse_args()

    block_weights = None
    if args.block_weights:
        block_weights = tuple(float(v) for v in args.block_weights.split(","))
    spec = SO2ChunkSpec(
        action_dim=7, vector_blocks=((0, 1), (3, 4)), scalar_dims=(2, 5, 6),
        block_weights=block_weights,
    )
    print(f"spec: block_weights={spec.weights}   coupling cost: "
          f"scalar_weight={args.scalar_weight}  batch={args.batch_size}")
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

    report = {
        "zarr": args.zarr, "horizon": args.horizon, "n_chunks": int(raw.shape[0]),
        "scalar_weight": float(args.scalar_weight),
        "coupling_batch_size": int(args.batch_size),
        "block_weights": list(spec.weights),
    }

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
    # GATE G4 must be read on the RAW actions, not the normalized ones.
    #
    # With vector_mode='rms' the SO(2) normalizer sets every vector block to unit
    # per-coordinate RMS, so the normalized per-block energies are equal BY CONSTRUCTION
    # and the < 0.05 threshold is unreachable no matter what the data looks like. Reading
    # the gate off the normalized numbers silently guarantees a pass. Both are printed
    # below; the gate fires on the raw share.
    print("\n" + "=" * 78)
    print("2  PER-BLOCK ENERGY  -- raw vs normalized")
    print("=" * 78)
    energy_raw = vector_energy_fraction(raw, spec)
    energy_norm = vector_energy_fraction(x, spec)
    print(f"  {'block':<24}{'RAW':>10}{'normalized':>14}   (gate reads RAW)")
    for k in energy_raw:
        print(f"  {k:<24}{energy_raw[k]:10.4f}{energy_norm.get(k, float('nan')):14.4f}")

    flat_raw = raw.reshape(-1, spec.action_dim)
    rms = [float(flat_raw[:, d].pow(2).mean().sqrt()) for d in range(spec.action_dim)]
    print("\n  per-dim raw RMS: " + "  ".join(
        f"{n}={v:.4f}" for n, v in zip(
            ("dx", "dy", "dz", "wx", "wy", "wz", "grip")[: spec.action_dim], rms)))

    report["energy"] = energy_norm            # kept under the original key for compatibility
    report["energy_raw"] = energy_raw
    report["energy_normalized"] = energy_norm
    report["raw_per_dim_rms"] = rms

    print("\n" + "-" * 78)
    print("  GATE G4  (threshold: raw vec1_energy_frac < 0.05)")
    v1_raw = energy_raw.get("vec1_energy_frac", 0.0)
    v1_norm = energy_norm.get("vec1_energy_frac", 0.0)
    report["G4_raw_vec1_energy_frac"] = v1_raw
    report["G4_fires"] = bool(v1_raw < 0.05)
    if v1_raw < 0.05:
        print(f"  !! FIRES: (wx,wy) carries {v1_raw:.2%} of RAW chunk energy "
              f"(normalized reads {v1_norm:.4f}).")
        print("     The rotation block is near-inert in physical units, and the rms")
        print("     normalizer then amplifies it to parity. Pick ONE policy and apply it")
        print("     to EVERY arm including the iid control; record it in RUNLOG.md:")
        print("       (b) policy.action_spec.block_weights=[1.0,0.0]   <- recommended")
        print("           translation-only heading/assignment; normalizer untouched, so an")
        print("           existing iid control stays valid.")
        print("       (a) policy.normalizer_vector_mode=global_rms")
        print("           keeps blocks in raw proportion, but CHANGES the normalizer and so")
        print("           needs its own iid control re-run.")
        print("       (c) keep the default and record the caveat in the writeup.")
    else:
        print(f"  does not fire: raw vec1_energy_frac = {v1_raw:.4f} >= 0.05")

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
    B = int(args.batch_size)
    print("\n" + "=" * 78)
    print(f"4  COUPLING PREVIEW  (batch {B}, scalar_weight {args.scalar_weight}, "
          f"real chunks, no training)")
    print("=" * 78)
    print(f"  angle gap cannot go below W1 = {w1:.4f} for any marginal-preserving coupling")
    print("  'path red.' is the reduction in mean path length against the iid row -- the")
    print("  quantity Gate F0c reads.\n")
    print(f"  {'coupling':<22}{'path len':>10}{'path red.':>11}{'angle gap':>11}"
          f"{'cond var':>11}{'src R1':>9}")
    variants = {
        "A/iid": dict(mode="iid"),
        "P2/perm-angle": dict(mode="perm", cost="angle"),
        "P3/perm-euclid": dict(mode="perm", cost="euclidean"),
        "P4/rot-heading": dict(mode="rot", align="heading"),
        "P6/perm+rot-group": dict(mode="perm_rot", cost="group_ot"),
    }
    # Same total number of chunks regardless of B, so the batch-size comparison changes the
    # assignment problem and nothing else.
    n_batches = max(1, min(24 * 256 // B, x.shape[0] // B))
    report["n_coupling_batches"] = int(n_batches)
    report["coupling_preview"] = {}
    baseline_path = None
    for name, cfg in variants.items():
        # reseed per variant: every coupling sees the SAME source draws, so a path-length
        # difference between rows is the assignment and not the noise.
        g = torch.Generator().manual_seed(0)
        c = SO2OrbitCoupling(spec, scalar_weight=args.scalar_weight, **cfg)
        rows, r1s = [], []
        for b in range(n_batches):
            x1 = x[b * B : (b + 1) * B]
            z0 = torch.randn(x1.shape, generator=g)
            zc, d = c(z0, x1)
            rows.append(transport_stats(zc, x1, spec))
            r1s.append(d["coupling/src_heading_R1"])
        agg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        r1 = sum(r1s) / len(r1s)
        if baseline_path is None:
            baseline_path = agg["path_len"]
        red = 1.0 - agg["path_len"] / max(baseline_path, 1e-12)
        print(f"  {name:<22}{agg['path_len']:>10.4f}{red * 100:>10.2f}%"
              f"{agg['angle_gap']:>11.4f}{agg['cond_vel_var']:>11.4f}{r1:>9.4f}")
        report["coupling_preview"][name] = {
            **agg, "src_heading_R1": r1, "path_len_reduction_vs_iid": float(red),
        }

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
