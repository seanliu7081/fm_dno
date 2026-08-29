#!/usr/bin/env python3
"""
Gate M1 of COUPLING_MECHANISM_NOTES.md -- the half-hour that decides whether any coupling
has something left to work with.  No GPU, no training, no images.

TWO QUESTIONS, BOTH ANSWERABLE FROM THE ACTION AND STATE ARRAYS ALONE
---------------------------------------------------------------------

**M1a -- is Construction A alive?**  Construction A canonicalizes the source phase to a
reference heading computed *from the observation only*, identically at train and test time,
which is what makes it admissible where P4 was not (notes S1).  It is worth building iff that
reference actually predicts the chunk's heading.  The statistic is the circular concentration

    R1_resid = | E exp( i ( theta(x1) - theta_ref(o) ) ) |

against the absolute-heading concentration ``R1_abs = |E exp(i theta(x1))|`` as the do-nothing
baseline.  ``R1_resid >> R1_abs`` means the frame carries heading information and the source
can be handed to the network already pointing the right way.  ``R1_resid ~ R1_abs`` means it
carries none and A is dead.

The cheap non-learned candidate the notes propose is the recent end-effector motion
direction, ``theta_ref(o) = atan2(d_eef_y, d_eef_x)`` across the ``n_obs_steps`` the policy
actually receives.  Two more observation-only frames are measured beside it as controls: the
per-task circular mean (what ``task_uid`` alone buys) and, optionally, a small **state-only
MLP** trained to regress the heading.  The MLP is the upper bound on what *any* non-visual
``theta_ref`` head could do; it is evaluated on held-out episodes, and it does **not** bound a
vision-conditioned head, which could do better.

**M1b -- is there within-observation structure for ANY coupling to reduce?**  A coupling can
only help by lowering ``Var[x1 - z | x_t, t, o]``.  Conditioning on ``o`` already removes the
across-observation part; a condition-blind batch assignment mostly reorganizes exactly that
part (notes S2).  So what matters is how much chunk variance *survives* conditioning.  Two
progressively tighter conditioners, both observation-computable:

    (task_uid, progress decile)          coarse
    + k-NN in (eef_pos, eef_vel) within  tight -- the closest thing to conditioning on o
      that does not require running the vision encoder

Reported per SO(2) block, because the notes single out the gripper (73.5% of raw energy,
near-binary, one switch time) as the coordinate where a blockwise assignment might have room.

WHAT COUNTS AS A PASS
---------------------
    R1_resid >~ 0.5                 the frame is informative -> A is alive
    R1_resid ~ R1_abs (~0.15)       the frame knows nothing -> A dead unless the MLP probe
                                    rescues it; if that is weak too, A costs no GPU at all
    within-o variance fraction      if this is near zero the conditional is near-deterministic
                                    and NO coupling has anything to reduce (notes S2)

Usage:
    python scripts/m1_residual_frame_audit.py
    python scripts/m1_residual_frame_audit.py --no-probe          # skip the MLP
    python scripts/m1_residual_frame_audit.py -o output/audit/m1.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from oat.common.replay_buffer import ReplayBuffer  # noqa: E402
from oat.common.seq_sampler import SequenceSampler, get_val_mask  # noqa: E402
from oat.symmetry.normalizer import so2_scale_offset_from_stats  # noqa: E402
from oat.symmetry.so2_chunk import (  # noqa: E402
    SO2ChunkSpec,
    chunk_heading,
    circular_moments,
    wrap_angle,
)

STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "task_uid")


# --------------------------------------------------------------------------------------
# window extraction -- vectorised, and checked against SequenceSampler
# --------------------------------------------------------------------------------------


def window_rows(indices: np.ndarray, rows: np.ndarray, n_steps: int) -> np.ndarray:
    """Buffer index of padded-sequence row ``r`` for every window, as SequenceSampler pads.

    ``data[sample_start:sample_end] = buffer[buffer_start:buffer_end]`` and everything outside
    that span repeats the nearest edge, so row ``r`` reads buffer index
    ``clip(bs + (r - ss), bs, be - 1)``.
    """
    bs, be, ss = indices[:, 0:1], indices[:, 1:2], indices[:, 2:3]
    idx = bs + (rows[None, :] - ss)
    idx = np.clip(idx, bs, be - 1)
    return np.clip(idx, 0, n_steps - 1)


def check_vectorisation(sampler, arrays, indices, horizon, n_obs_steps, n_check=64):
    """Prove the vectorised extraction equals sample_sequence on a random subsample."""
    rng = np.random.default_rng(0)
    picks = rng.choice(len(sampler), size=min(n_check, len(sampler)), replace=False)
    a_rows = np.arange(n_obs_steps - 1, n_obs_steps - 1 + horizon)
    for i in picks:
        got = sampler.sample_sequence(int(i))
        idx = window_rows(indices[i: i + 1], a_rows, arrays["action"].shape[0])[0]
        if not np.allclose(arrays["action"][idx], got["action"][n_obs_steps - 1:
                                                              n_obs_steps - 1 + horizon]):
            raise SystemExit(f"vectorised window extraction disagrees at index {i}")
        o_idx = window_rows(indices[i: i + 1], np.arange(n_obs_steps),
                            arrays["action"].shape[0])[0]
        if not np.allclose(arrays["robot0_eef_pos"][o_idx],
                           got["robot0_eef_pos"][:n_obs_steps]):
            raise SystemExit(f"vectorised obs extraction disagrees at index {i}")
    return len(picks)


# --------------------------------------------------------------------------------------
# circular helpers
# --------------------------------------------------------------------------------------


def R1(theta: np.ndarray, weights: np.ndarray | None = None) -> float:
    if theta.size == 0:
        return float("nan")
    z = np.exp(1j * theta.astype(np.float64))
    if weights is None:
        return float(np.abs(z.mean()))
    w = weights.astype(np.float64)
    return float(np.abs((z * w).sum() / max(w.sum(), 1e-12)))


def circ_mae(theta: np.ndarray) -> float:
    if theta.size == 0:
        return float("nan")
    return float(np.abs((theta + np.pi) % (2 * np.pi) - np.pi).mean())


def circ_mean(theta: np.ndarray) -> float:
    return float(np.angle(np.exp(1j * theta.astype(np.float64)).mean()))


# --------------------------------------------------------------------------------------
# the state-only heading probe (upper bound on non-visual theta_ref)
# --------------------------------------------------------------------------------------


def run_probe(feat_tr, th_tr, w_tr, feat_va, th_va, w_va, epochs=40, seed=0):
    """Small circular regression: features -> unit vector.  Returns residual R1 on val.

    Trained by maximising ``E w * cos(theta_hat - theta)``, which is the circular analogue of
    least squares and the right objective for a phase.  Held-out episodes, so this is a
    generalisation bound and not a fit.
    """
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = feat_tr.mean(0, keepdims=True), feat_tr.std(0, keepdims=True) + 1e-6
    Xtr = torch.from_numpy(((feat_tr - mu) / sd).astype(np.float32)).to(dev)
    Xva = torch.from_numpy(((feat_va - mu) / sd).astype(np.float32)).to(dev)
    Ttr = torch.from_numpy(th_tr.astype(np.float32)).to(dev)
    Wtr = torch.from_numpy(w_tr.astype(np.float32)).to(dev)

    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 2),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    n, bs = Xtr.shape[0], 4096
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i: i + bs]
            out = net(Xtr[j])
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cos = out[:, 0] * torch.cos(Ttr[j]) + out[:, 1] * torch.sin(Ttr[j])
            loss = -(Wtr[j] * cos).sum() / Wtr[j].sum().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        out = net(Xva)
        pred = torch.atan2(out[:, 1], out[:, 0]).cpu().numpy()
    return pred, float(loss.item())


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/libero/libero10_N500.zarr")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n-obs-steps", type=int, default=2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-weights", default="1,0",
                    help="vector-block weights for the heading, matching every trained arm")
    ap.add_argument("--min-chunk-conf", type=float, default=0.05,
                    help="chunks whose planar deltas cancel carry no heading; gate them")
    ap.add_argument("--min-ref-speed", type=float, default=1e-4,
                    help="metres of planar eef motion over the obs window below which "
                         "theta_ref is meaningless")
    ap.add_argument("--knn-k", type=int, default=16)
    ap.add_argument("--knn-probe", type=int, default=20000)
    ap.add_argument("--probe", dest="probe", action="store_true", default=True)
    ap.add_argument("--no-probe", dest="probe", action="store_false")
    ap.add_argument("-o", "--output", default="output/audit/m1_residual_frame.json")
    args = ap.parse_args()

    bw = tuple(float(v) for v in args.block_weights.split(",")) if args.block_weights else None
    spec = SO2ChunkSpec(action_dim=7, vector_blocks=((0, 1), (3, 4)), scalar_dims=(2, 5, 6),
                        block_weights=bw)
    To, H = args.n_obs_steps, args.horizon
    report = {"zarr": args.zarr, "horizon": H, "n_obs_steps": To,
              "block_weights": list(spec.weights), "seed": args.seed,
              "val_ratio": args.val_ratio}

    # ---- windows, exactly as ZarrDataset builds them ---------------------------------
    print(f"loading {args.zarr} (action + state only) ...")
    rb = ReplayBuffer.copy_from_path(args.zarr, keys=["action", *STATE_KEYS])
    arrays = {k: np.asarray(rb[k]) for k in ["action", *STATE_KEYS]}
    n_steps = arrays["action"].shape[0]
    ep_ends = np.asarray(rb.episode_ends)
    val_mask = get_val_mask(rb.n_episodes, args.val_ratio, args.seed)
    seq_len = To - 1 + 1 + (H - 1)

    samplers = {
        "train": SequenceSampler(rb, seq_len, To - 1, H - 1, episode_mask=~val_mask),
        "val": SequenceSampler(rb, seq_len, To - 1, H - 1, episode_mask=val_mask),
    }
    print(f"  {n_steps} steps, {rb.n_episodes} episodes "
          f"-> {len(samplers['train'])} train / {len(samplers['val'])} val windows")
    n_checked = check_vectorisation(samplers["train"], arrays, samplers["train"].indices,
                                    H, To)
    print(f"  vectorised extraction verified against sample_sequence on {n_checked} windows")

    # ---- SO(2) normalizer, fitted the way the policy does -----------------------------
    a = torch.from_numpy(arrays["action"].astype(np.float32))
    stats = {"min": a.min(0).values, "max": a.max(0).values,
             "mean": a.mean(0), "std": a.std(0)}
    scale, offset = so2_scale_offset_from_stats(stats, spec, vector_mode="rms")
    report["normalizer_scale"] = [float(v) for v in scale]

    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    # task_uid is not 0-based on this dataset (LIBERO-10 uses 30..39); densify it once so
    # the one-hot and the per-task tables are indexable.
    uid_values = np.unique(arrays["task_uid"].reshape(-1))
    uid_to_dense = {int(v): i for i, v in enumerate(uid_values)}
    n_tasks = len(uid_values)
    report["task_uids"] = [int(v) for v in uid_values]
    print(f"  {n_tasks} tasks, uids {uid_values.min()}..{uid_values.max()} -> dense 0..{n_tasks - 1}")

    def build(split):
        s = samplers[split]
        idx = s.indices
        a_rows = np.arange(To - 1, To - 1 + H)
        o_rows = np.arange(To)
        a_idx = window_rows(idx, a_rows, n_steps)
        o_idx = window_rows(idx, o_rows, n_steps)

        chunks = torch.from_numpy(arrays["action"][a_idx].astype(np.float32))
        chunks = chunks * scale + offset                       # normalized space
        theta, conf = chunk_heading(chunks, spec)

        eef = arrays["robot0_eef_pos"][o_idx].astype(np.float64)      # (W, To, 3)
        d = eef[:, -1] - eef[:, 0]
        theta_ref = np.arctan2(d[:, 1], d[:, 0])
        ref_speed = np.linalg.norm(d[:, :2], axis=-1)

        anchor = idx[:, 0] + (To - 1 - idx[:, 2])
        anchor = np.clip(anchor, 0, n_steps - 1)
        ep_id = np.searchsorted(ep_ends, anchor, side="right")
        progress = ((anchor - ep_starts[ep_id])
                    / np.maximum(ep_ends[ep_id] - ep_starts[ep_id], 1))
        raw_uid = arrays["task_uid"][anchor].reshape(-1).astype(np.int64)
        task = np.array([uid_to_dense[int(v)] for v in raw_uid], dtype=np.int64)

        feat = np.concatenate([
            eef.reshape(len(idx), -1),                                  # To * 3
            d, ref_speed[:, None],                                      # 3 + 1
            arrays["robot0_eef_quat"][o_idx].reshape(len(idx), -1),     # To * 4
            arrays["robot0_gripper_qpos"][o_idx].reshape(len(idx), -1),  # To * 2
            np.eye(n_tasks)[task],                                      # task one-hot
        ], axis=1)

        return dict(theta=theta.numpy().astype(np.float64),
                    conf=conf.numpy().astype(np.float64),
                    chunks=chunks.numpy(), theta_ref=theta_ref, ref_speed=ref_speed,
                    task=task, progress=progress, eef=eef, feat=feat)

    D = {s: build(s) for s in ("train", "val")}
    tr = D["train"]

    # ---------------------------------------------------------------------------------
    # M1a -- the residual frame
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("M1a  RESIDUAL FRAME  -- does an observation-only heading predict the chunk's?")
    print("=" * 78)

    keep = (tr["conf"] > args.min_chunk_conf) & (tr["ref_speed"] > args.min_ref_speed)
    print(f"  windows                      {len(tr['theta'])}")
    print(f"  usable (conf>{args.min_chunk_conf}, speed>{args.min_ref_speed:g})"
          f"        {int(keep.sum())}  ({keep.mean():.1%})")

    r1_abs = R1(tr["theta"])
    r1_abs_keep = R1(tr["theta"][keep])
    resid_eef = wrap_angle(torch.from_numpy(tr["theta"] - tr["theta_ref"])).numpy()

    # per-task circular mean: the control for "what does task_uid alone buy"
    task_mean = np.zeros(n_tasks)
    for t in range(n_tasks):
        m = tr["task"] == t
        task_mean[t] = circ_mean(tr["theta"][m]) if m.any() else 0.0
    resid_task = wrap_angle(torch.from_numpy(tr["theta"] - task_mean[tr["task"]])).numpy()

    frames = {
        "absolute (no frame)": tr["theta"],
        "per-task circular mean": resid_task,
        "eef-motion direction": resid_eef,
    }
    print(f"\n  {'frame':<28}{'R1':>8}{'R1 (gated)':>13}{'circ MAE':>11}"
          f"{'MAE gated':>11}")
    report["m1a"] = {"n_windows": int(len(tr["theta"])),
                     "usable_frac": float(keep.mean()),
                     "R1_absolute": r1_abs, "frames": {}}
    for name, res in frames.items():
        row = {"R1": R1(res), "R1_gated": R1(res[keep]),
               "circ_MAE": circ_mae(res), "circ_MAE_gated": circ_mae(res[keep]),
               "R1_conf_weighted": R1(res, tr["conf"])}
        print(f"  {name:<28}{row['R1']:8.4f}{row['R1_gated']:13.4f}"
              f"{row['circ_MAE']:11.4f}{row['circ_MAE_gated']:11.4f}")
        report["m1a"]["frames"][name] = row

    # per-task residual for the eef frame -- a frame can work on some tasks and not others
    print(f"\n  per-task (eef-motion frame, gated):  {'task':<6}{'n':>7}{'R1_abs':>9}{'R1_res':>9}")
    per_task = {}
    for t in range(n_tasks):
        m = keep & (tr["task"] == t)
        if m.sum() < 50:
            continue
        per_task[int(t)] = {"n": int(m.sum()), "R1_abs": R1(tr["theta"][m]),
                            "R1_resid": R1(resid_eef[m])}
        print(f"  {'':<38}{t:<6}{int(m.sum()):>7}{per_task[int(t)]['R1_abs']:>9.4f}"
              f"{per_task[int(t)]['R1_resid']:>9.4f}")
    report["m1a"]["per_task_eef_frame"] = per_task

    # ---- the learned upper bound ------------------------------------------------------
    if args.probe:
        print("\n  state-only MLP probe (held-out episodes) -- upper bound on any NON-VISUAL")
        print("  theta_ref head. Does not bound a vision-conditioned one.")
        va = D["val"]
        keep_va = (va["conf"] > args.min_chunk_conf)
        pred, last_loss = run_probe(
            tr["feat"][keep], tr["theta"][keep], tr["conf"][keep],
            va["feat"], va["theta"], va["conf"],
        )
        resid_probe = wrap_angle(torch.from_numpy(va["theta"] - pred)).numpy()
        r1_probe = R1(resid_probe[keep_va])
        r1_abs_va = R1(va["theta"][keep_va])
        print(f"    val R1 absolute              {r1_abs_va:.4f}")
        print(f"    val R1 after MLP frame       {r1_probe:.4f}   "
              f"(circ MAE {circ_mae(resid_probe[keep_va]):.4f})")
        report["m1a"]["mlp_probe"] = {
            "val_R1_absolute": r1_abs_va, "val_R1_residual": r1_probe,
            "val_circ_MAE_residual": circ_mae(resid_probe[keep_va]),
            "train_loss": last_loss, "n_val": int(keep_va.sum()),
            "features": "eef_pos(To) + d_eef + |d| + eef_quat(To) + gripper(To) + task onehot",
        }

    best = max(report["m1a"]["frames"]["eef-motion direction"]["R1_gated"],
               report["m1a"].get("mlp_probe", {}).get("val_R1_residual", 0.0))
    verdict = ("ALIVE -- the frame is informative" if best >= 0.5 else
               "MARGINAL -- informative but weak" if best >= 0.30 else
               "DEAD -- no observation-only frame found that predicts chunk heading")
    print(f"\n  GATE M1a: best residual R1 = {best:.4f} vs absolute {r1_abs_keep:.4f}"
          f"  ->  {verdict}")
    report["m1a"]["best_residual_R1"] = best
    report["m1a"]["verdict"] = verdict

    # ---------------------------------------------------------------------------------
    # M1b -- within-observation structure
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("M1b  WITHIN-o STRUCTURE  -- how much chunk variance survives conditioning?")
    print("=" * 78)
    print("  A coupling can only reduce variance the conditioning has NOT already removed.")
    print("  Fractions are of the unconditional chunk variance, per SO(2) block.\n")

    X = tr["chunks"].reshape(len(tr["chunks"]), H, 7)
    blocks = {"vec0 (dx,dy)": [0, 1], "vec1 (wx,wy)": [3, 4],
              "dz": [2], "wz": [5], "grip": [6], "ALL": list(range(7))}

    def var_of(sel, mask=None):
        v = X[:, :, sel] if mask is None else X[mask][:, :, sel]
        return float(v.reshape(len(v), -1).var(axis=0, ddof=0).sum())

    total = {k: var_of(v) for k, v in blocks.items()}

    # conditioner 1: (task_uid, progress decile)
    dec = np.clip((tr["progress"] * 10).astype(int), 0, 9)
    gid = tr["task"] * 10 + dec
    within_tp = {k: 0.0 for k in blocks}
    wsum = 0.0
    for g in np.unique(gid):
        m = gid == g
        if m.sum() < 8:
            continue
        w = float(m.sum())
        wsum += w
        for k, sel in blocks.items():
            within_tp[k] += w * var_of(sel, m)
    within_tp = {k: v / max(wsum, 1e-9) for k, v in within_tp.items()}

    # conditioner 2: k-NN in (eef_pos, d_eef) within task -- the tightest observation-only
    # conditioner available without running the vision encoder
    rng = np.random.default_rng(0)
    probe_idx = rng.choice(len(X), size=min(args.knn_probe, len(X)), replace=False)
    feat_knn = np.concatenate([tr["eef"][:, -1], tr["eef"][:, -1] - tr["eef"][:, 0],
                               tr["progress"][:, None]], axis=1)
    feat_knn = (feat_knn - feat_knn.mean(0)) / (feat_knn.std(0) + 1e-9)
    within_knn = {k: 0.0 for k in blocks}
    n_used = 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for t in range(n_tasks):
        pool = np.where(tr["task"] == t)[0]
        probe = np.intersect1d(probe_idx, pool)
        if len(pool) < args.knn_k + 1 or len(probe) == 0:
            continue
        F = torch.from_numpy(feat_knn[pool].astype(np.float32)).to(dev)
        Q = torch.from_numpy(feat_knn[probe].astype(np.float32)).to(dev)
        nb = []
        for i in range(0, len(Q), 2048):
            d = torch.cdist(Q[i: i + 2048], F)
            nb.append(torch.topk(d, k=args.knn_k, largest=False).indices.cpu().numpy())
        nb = pool[np.concatenate(nb, axis=0)]                  # (n_probe, k)
        neigh = X[nb]                                          # (n_probe, k, H, 7)
        for k_, sel in blocks.items():
            v = neigh[:, :, :, sel].reshape(len(nb), args.knn_k, -1)
            within_knn[k_] += float(v.var(axis=1, ddof=0).sum(-1).sum())
        n_used += len(nb)
    within_knn = {k: v / max(n_used, 1) for k, v in within_knn.items()}

    print(f"  {'block':<16}{'total var':>12}{'| task,progress':>17}{'| kNN(eef)':>13}")
    report["m1b"] = {"blocks": {}, "knn_k": args.knn_k, "n_knn_probe": int(n_used)}
    for k in blocks:
        f_tp = within_tp[k] / max(total[k], 1e-12)
        f_kn = within_knn[k] / max(total[k], 1e-12)
        print(f"  {k:<16}{total[k]:12.4f}{f_tp:16.1%}{f_kn:13.1%}")
        report["m1b"]["blocks"][k] = {
            "total_var": total[k], "within_task_progress_frac": f_tp,
            "within_knn_frac": f_kn,
        }

    all_kn = report["m1b"]["blocks"]["ALL"]["within_knn_frac"]
    print(f"\n  GATE M1b: {all_kn:.1%} of chunk variance survives the tightest "
          f"observation-only conditioner.")
    if all_kn < 0.15:
        print("    Near-deterministic given o: there is very little for ANY data-side")
        print("    coupling to reduce. This is the notes' S2 worry, confirmed.")
    else:
        print("    Substantial within-o spread remains -- a coupling has something to act on;")
        print("    whether it acts on the USEFUL part is what M2/M3 test.")
    report["m1b"]["verdict_within_o_frac_all"] = all_kn

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
