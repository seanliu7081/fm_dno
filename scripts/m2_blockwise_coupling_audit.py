#!/usr/bin/env python3
"""
Gate M2 of COUPLING_MECHANISM_NOTES.md -- is the blockwise assignment worth one GPU?

Construction B replaces the single 112-D minibatch matching with one independent matching per
SO(2) irrep block.  Minibatch OT's dilution is a dimension phenomenon, so lowering the
dimension without lowering the coupling's reach is the one legitimate lever left on the
data-side family.  This script measures whether it actually buys anything, on real chunks,
with no training.

WHAT IS MEASURED, AND WHY EACH ONE
----------------------------------
* **per-block path length reduction vs iid.**  The joint ``perm`` reads 4.9% at the arms'
  settings (RUNLOG, F0c).  If blockwise does not beat that -- and specifically if the
  **gripper** block does not beat it by a lot -- B is not worth a GPU.  The gripper is the
  notes' whole thesis for B: 73.5% of raw action energy, near-binary, one switch time, so
  effective dimension ~2-3 where the joint problem sees 112.
* **small-t conditional target variance.**  Path length is a proxy; the quantity a coupling
  must actually reduce is ``Var[x1 - z | x_t, t]``, the ambiguity the network is asked to
  average over.  It is largest at small ``t`` (where ``x_t`` is nearly the source and says
  little about the target), which is exactly where a coupling can inform it.  Estimated by
  k-NN in the **full** ``x_t`` space -- the network sees the whole chunk -- with the variance
  then read off per block.
* **within-task restriction on/off.**  Cross-task matching reorganizes structure the
  observation conditioning already explains: pure bias, no useful gain (notes S2).
  Restricting to same-``task_uid`` batch members should improve the gain/bias ratio for free.
* **B in {32, 128}.**  F0c showed the joint assignment gains only +1.65 points from 32 to
  128.  If blockwise scales better, that is a fact about the dimension argument.

Reads the same window-aligned chunks the policy trains on (via the M1 extractor), so the
numbers here and in M1 are directly comparable.

Usage:
    python scripts/m2_blockwise_coupling_audit.py
    python scripts/m2_blockwise_coupling_audit.py --batch-sizes 32,128 --n-chunks 6144
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
sys.path.append(str(pathlib.Path(__file__).parent))

from m1_residual_frame_audit import window_rows  # noqa: E402
from oat.common.replay_buffer import ReplayBuffer  # noqa: E402
from oat.common.seq_sampler import SequenceSampler, get_val_mask  # noqa: E402
from oat.symmetry.coupling import SO2OrbitCoupling  # noqa: E402
from oat.symmetry.coupling_blockwise import BlockwiseSO2Coupling, coupling_blocks  # noqa: E402
from oat.symmetry.normalizer import so2_scale_offset_from_stats  # noqa: E402
from oat.symmetry.so2_chunk import SO2ChunkSpec  # noqa: E402

BLOCK_DIMS = {"vec0 (dx,dy)": [0, 1], "vec1 (wx,wy)": [3, 4],
              "dz": [2], "wz": [5], "grip": [6], "ALL": list(range(7))}
T_GRID = (0.05, 0.10, 0.25, 0.50)


@torch.no_grad()
def cond_target_var(z0, x1, t, k=16, n_probe=2048, seed=0):
    """k-NN estimate of ``Var[x1 - z0 | x_t, t]``, per block.

    Neighbours are found in the **full** ``x_t`` space because that is what the network is
    conditioned on; the variance is then decomposed over blocks.  A weak estimator in
    absolute terms at 112 dimensions -- used only to compare couplings on identical data.
    """
    B = z0.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(B, generator=g)[: min(n_probe, B)].to(z0.device)
    u = (x1 - z0)
    xt = ((1 - t) * z0 + t * x1).reshape(B, -1)
    d = torch.cdist(xt[idx], xt)
    nb = torch.topk(d, k=k, largest=False).indices                 # (n, k)
    out = {}
    for name, dims in BLOCK_DIMS.items():
        v = u[:, :, dims][nb]                                      # (n, k, H, |dims|)
        out[name] = float(v.reshape(len(idx), k, -1).var(dim=1, unbiased=False).sum(-1).mean())
    return out


@torch.no_grad()
def run_variant(name, coupling, x, groups, batch, seed=0, device="cpu"):
    """Couple the pool batch by batch, then measure on the accumulated pairs."""
    n = (x.shape[0] // batch) * batch
    g = torch.Generator(device="cpu").manual_seed(seed)
    z_all, drift_block, drift_joint = [], 0.0, 0.0
    for b in range(0, n, batch):
        x1 = x[b: b + batch]
        z0 = torch.randn(x1.shape, generator=g).to(device)
        if coupling is None:
            zc, diag = z0, {}
        elif isinstance(coupling, BlockwiseSO2Coupling):
            gi = groups[b: b + batch] if coupling.group_restrict else None
            zc, diag = coupling(z0, x1, group_ids=gi)
        else:
            zc, diag = coupling(z0, x1)
        drift_block = max(drift_block, diag.get("coupling/blockwise_marginal_drift", 0.0))
        drift_joint = max(drift_joint, diag.get("coupling/src_norm_drift", 0.0))
        z_all.append(zc)
    z = torch.cat(z_all, 0)
    xx = x[:n]

    res = {"n_pairs": int(n), "batch": int(batch),
           "blockwise_marginal_drift": drift_block, "joint_norm_drift": drift_joint,
           "path_len": {}, "cond_var": {}}
    for bname, dims in BLOCK_DIMS.items():
        p = (xx[:, :, dims] - z[:, :, dims]).reshape(n, -1)
        res["path_len"][bname] = float(p.norm(dim=-1).mean())
    for t in T_GRID:
        res["cond_var"][f"t={t}"] = cond_target_var(z, xx, t, seed=seed)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/libero/libero10_N500.zarr")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n-obs-steps", type=int, default=2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-weights", default="1,0")
    ap.add_argument("--scalar-weight", type=float, default=1.0,
                    help="joint-perm reference cost weight, matching the trained arms")
    ap.add_argument("--batch-sizes", default="32,128")
    ap.add_argument("--n-chunks", type=int, default=6144)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-o", "--output", default="output/audit/m2_blockwise.json")
    args = ap.parse_args()

    bw = tuple(float(v) for v in args.block_weights.split(",")) if args.block_weights else None
    spec = SO2ChunkSpec(action_dim=7, vector_blocks=((0, 1), (3, 4)), scalar_dims=(2, 5, 6),
                        block_weights=bw)
    To, H = args.n_obs_steps, args.horizon
    dev = torch.device(args.device)

    # ---- the same window-aligned chunks the policy trains on --------------------------
    print(f"loading {args.zarr} (action + task_uid) ...")
    rb = ReplayBuffer.copy_from_path(args.zarr, keys=["action", "task_uid"])
    act = np.asarray(rb["action"])
    uid = np.asarray(rb["task_uid"]).reshape(-1)
    n_steps = act.shape[0]
    val_mask = get_val_mask(rb.n_episodes, args.val_ratio, args.seed)
    seq_len = To - 1 + 1 + (H - 1)
    sampler = SequenceSampler(rb, seq_len, To - 1, H - 1, episode_mask=~val_mask)
    idx = sampler.indices
    print(f"  {len(sampler)} train windows")

    a_idx = window_rows(idx, np.arange(To - 1, To - 1 + H), n_steps)
    anchor = np.clip(idx[:, 0] + (To - 1 - idx[:, 2]), 0, n_steps - 1)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(idx), size=min(args.n_chunks, len(idx)), replace=False)
    pick.sort()

    a = torch.from_numpy(act.astype(np.float32))
    stats = {"min": a.min(0).values, "max": a.max(0).values, "mean": a.mean(0), "std": a.std(0)}
    scale, offset = so2_scale_offset_from_stats(stats, spec, vector_mode="rms")
    x = torch.from_numpy(act[a_idx[pick]].astype(np.float32)) * scale + offset
    groups = torch.from_numpy(uid[anchor[pick]].astype(np.int64))
    x = x.to(dev)
    groups = groups.to(dev)
    print(f"  {x.shape[0]} chunks, {len(torch.unique(groups))} tasks, device {dev}")
    print(f"  blocks: {[b[0] for b in coupling_blocks(spec)]}")

    variants = {
        "A/iid": None,
        "P3/perm joint (euclid)": SO2OrbitCoupling(
            spec, mode="perm", cost="euclidean", scalar_weight=args.scalar_weight),
        "B/perm_block (group_ot)": BlockwiseSO2Coupling(spec, vector_cost="group_ot"),
        "B/perm_block (euclid)": BlockwiseSO2Coupling(spec, vector_cost="euclidean"),
        "B/perm_block within-task": BlockwiseSO2Coupling(
            spec, vector_cost="group_ot", group_restrict=True),
        "B/gripper block only": BlockwiseSO2Coupling(spec, blocks=["scalar6"]),
    }

    report = {"zarr": args.zarr, "horizon": H, "n_chunks": int(x.shape[0]),
              "block_weights": list(spec.weights), "scalar_weight": args.scalar_weight,
              "t_grid": list(T_GRID), "results": {}}

    for batch in [int(v) for v in args.batch_sizes.split(",")]:
        print("\n" + "=" * 92)
        print(f"BATCH {batch}")
        print("=" * 92)
        base = None
        rows = {}
        for name, c in variants.items():
            r = run_variant(name, c, x, groups, batch, seed=args.seed, device=dev)
            if base is None:
                base = r
            r["path_len_reduction"] = {
                k: 1.0 - r["path_len"][k] / max(base["path_len"][k], 1e-12)
                for k in BLOCK_DIMS
            }
            r["cond_var_reduction"] = {
                tk: {bk: 1.0 - r["cond_var"][tk][bk] / max(base["cond_var"][tk][bk], 1e-12)
                     for bk in BLOCK_DIMS}
                for tk in r["cond_var"]
            }
            rows[name] = r
            report["results"][f"B{batch}/{name}"] = r

        print("\n  path-length reduction vs iid, per block")
        print(f"  {'coupling':<26}" + "".join(f"{k:>15}" for k in BLOCK_DIMS))
        for name, r in rows.items():
            print(f"  {name:<26}" +
                  "".join(f"{r['path_len_reduction'][k] * 100:14.2f}%" for k in BLOCK_DIMS))

        print(f"\n  conditional target-variance reduction vs iid at small t "
              f"(ALL blocks, kNN in full x_t)")
        print(f"  {'coupling':<26}" + "".join(f"{'t=' + str(t):>12}" for t in T_GRID))
        for name, r in rows.items():
            print(f"  {name:<26}" +
                  "".join(f"{r['cond_var_reduction'][f't={t}']['ALL'] * 100:11.2f}%"
                          for t in T_GRID))

        print(f"\n  same, GRIPPER block only  (the notes' thesis for B)")
        print(f"  {'coupling':<26}" + "".join(f"{'t=' + str(t):>12}" for t in T_GRID))
        for name, r in rows.items():
            print(f"  {name:<26}" +
                  "".join(f"{r['cond_var_reduction'][f't={t}']['grip'] * 100:11.2f}%"
                          for t in T_GRID))

        print(f"\n  marginal audit   {'coupling':<24}{'per-block drift':>17}{'joint drift':>13}")
        for name, r in rows.items():
            print(f"  {'':<17}{name:<24}{r['blockwise_marginal_drift']:>17.2e}"
                  f"{r['joint_norm_drift']:>13.4f}")

    # ---- the gate ---------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("GATE M2")
    print("=" * 92)
    # The gate is read on the quantity a coupling must actually reduce -- the small-t
    # conditional target variance -- and on the BEST blockwise variant, not a nominated one.
    # (The first draft of this gate keyed on group_ot and would have mis-scored B: see the
    # vector_cost finding below.)
    b0 = int(args.batch_sizes.split(",")[0])
    tk = f"t={T_GRID[0]}"
    ref = report["results"][f"B{b0}/P3/perm joint (euclid)"]
    joint_path = ref["path_len_reduction"]["ALL"]
    joint_var = ref["cond_var_reduction"][tk]["ALL"]

    cands = {k: v for k, v in report["results"].items()
             if k.startswith(f"B{b0}/B/") and "within-task" not in k}
    best_name = max(cands, key=lambda k: cands[k]["cond_var_reduction"][tk]["ALL"])
    best = cands[best_name]
    print(f"  reference  joint perm       path {joint_path * 100:6.2f}%   "
          f"Var[u|x_t] @ {tk} {joint_var * 100:6.2f}%")
    print(f"  best blockwise variant     {best_name.split('/', 1)[1]}")
    print(f"                             path {best['path_len_reduction']['ALL'] * 100:6.2f}%   "
          f"Var[u|x_t] @ {tk} {best['cond_var_reduction'][tk]['ALL'] * 100:6.2f}%")
    print(f"  gripper block, best        path "
          f"{best['path_len_reduction']['grip'] * 100:6.2f}%   "
          f"Var[u|x_t] @ {tk} {best['cond_var_reduction'][tk]['grip'] * 100:6.2f}%")

    ratio = best["cond_var_reduction"][tk]["ALL"] / max(joint_var, 1e-9)
    verdict = ("WORTH A GPU -- blockwise roughly doubles the joint assignment's small-t "
               "conditional-variance reduction" if ratio >= 1.5 else
               "MARGINAL -- blockwise beats the joint assignment but not decisively"
               if ratio >= 1.15 else
               "NOT worth a GPU -- blockwise does not beat the joint assignment")
    print(f"\n  best/joint conditional-variance ratio = {ratio:.2f}x  ->  {verdict}")
    report["gate_m2"] = {
        "reference": "P3/perm joint (euclid)", "best_blockwise": best_name,
        "joint_path_reduction": joint_path, "joint_cond_var_reduction": joint_var,
        "best_path_reduction": best["path_len_reduction"]["ALL"],
        "best_cond_var_reduction": best["cond_var_reduction"][tk]["ALL"],
        "best_grip_cond_var_reduction": best["cond_var_reduction"][tk]["grip"],
        "ratio_vs_joint": ratio, "verdict": verdict,
    }

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
