#!/usr/bin/env python3
"""
Standalone validation of the SO(2) coupling machinery -- run this before LIBERO.

Nothing here needs the robot stack: it builds a synthetic action-chunk distribution with
the same irrep structure as a LIBERO 7-D delta-EEF chunk and checks the five claims the
design rests on.

  1  MARGINAL AUDIT.  Does each coupling leave the source marginal N(0, I)?
     Run twice: with uniform target headings (the toy problem's situation) and with
     strongly non-uniform ones (a robot dataset's situation).  ``rot`` should pass the
     first and fail the second; ``perm`` should pass both.  This is the single most
     important check, because it is the specific way a naive toy-to-robot port breaks.
  2  TRANSPORT.  Path length, angle gap, conditional velocity variance per coupling.
  3  NORMALIZER.  Per-dim min-max vs the SO(2) block normalizer, measured as
     ||N(rho a) - rho N(a)|| / ||rho N(a)||.
  4  OPTIMIZER.  IrrepAdam vs plain Adam: do trajectories started from z and rho(phi) z
     stay related by rho(phi)?
  5  END-TO-END.  Train a small conditional flow-matching model per coupling and compare
     sample quality, few-step error, straightness, and orbit steering gain.

Usage:
    python scripts/validate_so2_coupling.py            # ~5 min on CPU
    python scripts/validate_so2_coupling.py --quick    # ~1 min
    python scripts/validate_so2_coupling.py --tests 1,3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oat.dno.irrep_adam import IrrepAdam  # noqa: E402
from oat.symmetry.coupling import MatchedHeadingPrior, SO2OrbitCoupling  # noqa: E402
from oat.symmetry.metrics import (  # noqa: E402
    conditional_velocity_variance,
    few_step_gap,
    orbit_steering_gain,
    source_orbit_equivariance,
    straightness,
    transport_stats,
)
from oat.symmetry.normalizer import (  # noqa: E402
    equivariance_error,
    fit_so2_action_scale_offset,
)
from oat.symmetry.so2_chunk import (  # noqa: E402
    SO2ChunkSpec,
    chunk_heading,
    circular_moments,
    rotate_chunk,
    wrap_angle,
)

SPEC = SO2ChunkSpec.libero_osc_pose()
H = 8


# ======================================================================================
# Synthetic chunk distribution with LIBERO's irrep structure
# ======================================================================================


def make_chunks(n: int, heading: str = "uniform", g: torch.Generator = None) -> Dict[str, torch.Tensor]:
    """(n, H, 7) chunks: 4 motion primitives, rotated to a heading drawn from ``heading``.

    Canonical-frame primitives (before rotation) differ in speed and curvature, so the
    distribution is multimodal in the invariant coordinates -- the chunk analogue of the
    double ring's two radii -- while the heading supplies the SO(2) orbit direction.

    The condition vector reveals the heading only noisily.  That is the realistic
    situation: a LIBERO agentview image does tell you roughly where the object is, it just
    does not transform under a world rotation.  The residual heading ambiguity is exactly
    what the coupling is supposed to remove from the regression target.
    """
    dev = torch.device("cpu")
    mode = torch.randint(0, 4, (n,), generator=g)
    speed = torch.tensor([0.35, 0.9, 0.55, 0.7])[mode]
    curve = torch.tensor([0.0, 0.0, 0.45, -0.45])[mode]

    if heading == "uniform":
        theta = torch.rand(n, generator=g) * 2 * math.pi
    elif heading == "vonmises":
        # bimodal, concentrated: what a real demo set looks like in world frame
        base = torch.where(torch.rand(n, generator=g) < 0.65, 0.6, 2.7)
        theta = base + 0.35 * torch.randn(n, generator=g)
    else:
        raise ValueError(heading)

    h = torch.arange(H, dtype=torch.float32)
    prog = (h / max(H - 1, 1))[None, :]
    ang = curve[:, None] * prog
    vx = speed[:, None] * torch.cos(ang)
    vy = speed[:, None] * torch.sin(ang)
    wz_local = 0.25 * curve[:, None] * torch.ones_like(prog)
    wx = 0.20 * torch.cos(3.0 * prog + mode[:, None].float())
    wy = 0.20 * torch.sin(3.0 * prog + mode[:, None].float())
    dz = (-0.30 * prog + 0.1).expand(n, H)
    grip = torch.where(mode[:, None] % 2 == 0, -1.0, 1.0).expand(n, H).clone()

    x = torch.stack([vx, vy, dz, wx, wy, wz_local, grip], dim=-1)
    x = x + 0.03 * torch.randn(x.shape, generator=g)
    x = rotate_chunk(x, theta, SPEC)

    theta_obs = theta + 0.55 * torch.randn(n, generator=g)
    cond = torch.cat(
        [
            torch.nn.functional.one_hot(mode, 4).float(),
            torch.cos(theta_obs)[:, None],
            torch.sin(theta_obs)[:, None],
            speed[:, None],
            curve[:, None],
        ],
        dim=-1,
    )
    return {"x": x.to(dev), "cond": cond.to(dev), "theta": theta, "mode": mode}


# name -> (coupling kwargs, use a heading-matched inference prior)
COUPLINGS: Dict[str, tuple] = {
    "A/iid":             (dict(mode="iid"), False),
    "P1/perm-group":     (dict(mode="perm", cost="group_ot"), False),
    "P2/perm-angle":     (dict(mode="perm", cost="angle"), False),
    "P3/perm-euclid":    (dict(mode="perm", cost="euclidean"), False),
    "C1/rot-kabsch":     (dict(mode="rot", align="kabsch"), False),
    "C2/rot-heading":    (dict(mode="rot", align="heading"), False),
    "C3/rot-head+prior": (dict(mode="rot", align="heading"), True),
    "C4/rot-head-k2":    (dict(mode="rot", align="heading", kappa=2.0), True),
    "F/perm+rot-group":  (dict(mode="perm_rot", cost="group_ot"), False),
}


def fmt(v, w=9, p=4):
    return f"{v:{w}.{p}f}" if isinstance(v, float) else f"{v:>{w}}"


# ======================================================================================
# Test 1 -- marginal audit
# ======================================================================================


def test_marginal(n: int = 4096, batch: int = 256) -> None:
    print("\n" + "=" * 100)
    print("TEST 1  MARGINAL AUDIT -- does the coupling deform the source?")
    print("=" * 100)
    print("  R1 of the coupled source heading should be ~0 (uniform on the circle).")
    print("  norm-drift must be exactly 0 for permutation modes (no source is modified).\n")

    for heading in ("uniform", "vonmises"):
        g = torch.Generator().manual_seed(0)
        data = make_chunks(n, heading, g)
        tgt_R1 = circular_moments(chunk_heading(data["x"], SPEC)[0])["R1"]
        print(f"  target headings: {heading:<9}  (target R1 = {tgt_R1:.3f})")
        print(f"    {'coupling':<20}{'src R1':>9}{'src R2':>9}{'coord mean':>12}"
              f"{'coord std':>11}{'norm drift':>12}")
        for name, (cfg, _) in COUPLINGS.items():
            c = SO2OrbitCoupling(SPEC, **cfg)
            R1s, R2s, means, stds, drifts = [], [], [], [], []
            for s in range(0, n, batch):
                x1 = data["x"][s : s + batch]
                if x1.shape[0] < batch:
                    break
                z0 = torch.randn(x1.shape, generator=g)
                zc, d = c(z0, x1)
                R1s.append(d["coupling/src_heading_R1"])
                R2s.append(d["coupling/src_heading_R2"])
                means.append(d["coupling/src_mean_absmax"])
                stds.append(d["coupling/src_std_max"])
                drifts.append(d["coupling/src_norm_drift"])
            agg = lambda v: sum(v) / len(v)
            flag = "  <-- DEFORMED" if agg(R1s) > 0.10 else ""
            print(f"    {name:<20}{fmt(agg(R1s))}{fmt(agg(R2s))}{fmt(agg(means), 12)}"
                  f"{fmt(agg(stds), 11)}{fmt(agg(drifts), 12)}{flag}")
        print()
    print("  NOTE: a 'DEFORMED' source is only a problem if inference still draws from")
    print("  N(0, I).  MatchedHeadingPrior removes the mismatch by sampling the induced")
    print("  law instead -- see method C3 in test 5.\n")


# ======================================================================================
# Test 2 -- transport diagnostics
# ======================================================================================


def test_transport(n: int = 1024, batch: int = 256) -> None:
    print("=" * 100)
    print("TEST 2  TRANSPORT -- does the coupling actually simplify the regression?")
    print("=" * 100)
    g = torch.Generator().manual_seed(1)
    data = make_chunks(n, "vonmises", g)
    lb = MatchedHeadingPrior(SPEC).uniformity_gap(data["x"])
    print(f"  circular W1(target heading, uniform) = {lb:.4f} rad")
    print(f"  --> no marginal-preserving coupling can push 'angle gap' below this.\n")
    print(f"  {'coupling':<20}{'path len':>10}{'path len^2':>12}{'angle gap':>11}{'cond var':>11}")
    for name, (cfg, _) in COUPLINGS.items():
        c = SO2OrbitCoupling(SPEC, **cfg)
        rows: List[Dict[str, float]] = []
        for s in range(0, n, batch):
            x1 = data["x"][s : s + batch]
            if x1.shape[0] < batch:
                break
            z0 = torch.randn(x1.shape, generator=g)
            zc, _ = c(z0, x1)
            rows.append(transport_stats(zc, x1, SPEC))
        a = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        print(f"  {name:<20}{fmt(a['path_len'], 10)}{fmt(a['path_len_sq'], 12)}"
              f"{fmt(a['angle_gap'], 11)}{fmt(a['cond_vel_var'], 11)}")
    print("\n  reading: 'perm' on a rotation-invariant cost (P1) cannot reduce Euclidean")
    print("  transport, because the cost it matches on deliberately ignores the angle it")
    print("  is not allowed to change. Match on the angle (P2) or the full metric (P3).\n")


# ======================================================================================
# Test 3 -- normalizer
# ======================================================================================


def test_normalizer(n: int = 8192) -> None:
    print("=" * 100)
    print("TEST 3  NORMALIZER -- does the action normalizer commute with a rotation?")
    print("=" * 100)
    g = torch.Generator().manual_seed(2)
    a = make_chunks(n, "vonmises", g)["x"]
    flat = a.reshape(-1, 7)

    lo, hi = flat.min(0).values, flat.max(0).values
    rng = (hi - lo).clamp_min(1e-4)
    mm_scale = 2.0 / rng
    mm_offset = -1.0 - mm_scale * lo
    e_mm = equivariance_error(a, mm_scale, mm_offset, SPEC)

    print(f"  {'normalizer':<34}{'equivariance err':>18}{'scale(dx)':>11}{'scale(dy)':>11}")
    print(f"  {'per-dim min-max (repo default)':<34}{e_mm:>18.3e}"
          f"{float(mm_scale[0]):>11.4f}{float(mm_scale[1]):>11.4f}")
    for vm in ("rms", "quantile"):
        s, o = fit_so2_action_scale_offset(a, SPEC, vector_mode=vm)
        e = equivariance_error(a, s, o, SPEC)
        print(f"  {'SO(2) block (' + vm + ')':<34}{e:>18.3e}"
              f"{float(s[0]):>11.4f}{float(s[1]):>11.4f}")
    print(f"\n  interpretation: min-max gives scale(dx) != scale(dy) and a nonzero offset,")
    print(f"  so a rotation in normalized space is not a rotation in raw space.\n")


# ======================================================================================
# Test 4 -- optimizer equivariance
# ======================================================================================


def test_optimizer(n: int = 64, steps: int = 25) -> None:
    print("=" * 100)
    print("TEST 4  OPTIMIZER -- is the noise-space update rotation-compatible?")
    print("=" * 100)
    g = torch.Generator().manual_seed(3)
    z0 = torch.randn(n, H, 7, generator=g)
    target = torch.randn(n, H, 7, generator=g)
    phi = torch.full((n,), 0.7)

    def rotinv_loss(z, tgt):
        """Invariant objective: only norms and relative geometry, no fixed direction."""
        zc = torch.complex(z[..., 0], z[..., 1])
        tc = torch.complex(tgt[..., 0], tgt[..., 1])
        return (torch.abs((zc.conj() * tc).sum(-1)) * -1.0).sum() + 0.1 * (z ** 2).sum()

    print(f"  {'optimizer':<22}{'||z_k(rho z0) - rho z_k(z0)||':>32}")
    for name, equi in (("IrrepAdam", True), ("plain Adam", False)):
        za, zb = z0.clone(), rotate_chunk(z0, phi, SPEC)
        ta, tb = target, rotate_chunk(target, phi, SPEC)
        oa = IrrepAdam(SPEC, lr=0.05, equivariant=equi)
        ob = IrrepAdam(SPEC, lr=0.05, equivariant=equi)
        for _ in range(steps):
            for z, t, o in ((za, ta, oa), (zb, tb, ob)):
                z.requires_grad_(True)
                (grad,) = torch.autograd.grad(rotinv_loss(z, t), z)
                new = o.step(z.detach(), grad)
                if z is za:
                    za = new
                else:
                    zb = new
        err = float((zb - rotate_chunk(za, phi, SPEC)).norm() / zb.norm())
        print(f"  {name:<22}{err:>32.3e}")
    print()


# ======================================================================================
# Test 5 -- end-to-end conditional flow matching
# ======================================================================================


class TinyFM(nn.Module):
    def __init__(self, d_x: int, d_c: int, hidden: int = 256, n_freq: int = 32):
        super().__init__()
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(d_x + d_c + 2 * n_freq, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_x),
        )

    def forward(self, x, t, c):
        B = x.shape[0]
        xf = x.reshape(B, -1)
        f = torch.exp(torch.linspace(0, 4, self.n_freq, device=x.device))
        te = torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], -1)
        return self.net(torch.cat([xf, te, c], -1)).reshape_as(x)


def sliced_wasserstein(a: torch.Tensor, b: torch.Tensor, n_proj: int = 128, g=None) -> float:
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    d = a.shape[-1]
    p = torch.randn(d, n_proj, generator=g)
    p = p / p.norm(dim=0, keepdim=True)
    pa = (a @ p).sort(dim=0).values
    pb = (b @ p).sort(dim=0).values
    m = min(pa.shape[0], pb.shape[0])
    return float((pa[:m] - pb[:m]).abs().mean())


def _train_model(cfg: dict, use_prior: bool, seed: int, steps: int, batch: int = 256):
    """Shared training routine for tests 5 and 6."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    train = make_chunks(20000, "vonmises", g)

    # SO(2) block normalizer, so that the whole experiment lives in an equivariant space
    scale, offset = fit_so2_action_scale_offset(train["x"], SPEC, vector_mode="rms")
    xn = train["x"] * scale + offset

    model = TinyFM(H * 7, train["cond"].shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    coupling = SO2OrbitCoupling(SPEC, **cfg)

    n = xn.shape[0]
    seen_sources = []
    for it in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        x1, c = xn[idx], train["cond"][idx]
        z0, _ = coupling(torch.randn(x1.shape, generator=g), x1)
        if use_prior and it >= steps - 40:
            seen_sources.append(z0)
        t = torch.rand(batch, generator=g)
        xt = (1 - t[:, None, None]) * z0 + t[:, None, None] * x1
        loss = ((model(xt, t, c) - (x1 - z0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # The prior is fitted on the sources the model was ACTUALLY trained on, so it is
    # exact for every coupling variant rather than only for align='heading', kappa=inf.
    prior = MatchedHeadingPrior(SPEC).fit(torch.cat(seen_sources)) if use_prior else None
    model.eval()
    return model, scale, offset, prior


def train_one(cfg: dict, use_prior: bool, seed: int, steps: int, batch: int = 256) -> Dict[str, float]:
    model, scale, offset, prior = _train_model(cfg, use_prior, seed, steps, batch)

    # ---- evaluation -----------------------------------------------------------------
    ge = torch.Generator().manual_seed(7)
    test = make_chunks(2048, "vonmises", ge)
    xt_n = test["x"] * scale + offset
    cond = test["cond"]

    def draw(shape):
        z = torch.randn(shape, generator=ge)
        return prior.sample(z) if prior is not None else z

    def sample(z, n_steps=20, c=cond):
        x = z
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt)
            x = x + dt * model(x, t, c)
        return x

    with torch.no_grad():
        z = draw(xt_n.shape)
        gen = sample(z, 20)
        out = {"SWD": sliced_wasserstein(gen, xt_n, g=ge)}

        th_gen, _ = chunk_heading(gen, SPEC)
        out["heading_MAE"] = float(torch.abs(wrap_angle(th_gen - test["theta"])).mean())

        out.update(
            few_step_gap(lambda zz, nn_: sample(zz, nn_, c=cond[:512]), z[:512], steps=(1, 2, 4))
        )
        out.update(straightness(lambda xx, tt: model(xx, tt, cond[:512]), z[:512], n_steps=20))

        sub = slice(0, 256)
        fn = lambda zz: sample(zz, 20, c=cond[sub])
        out.update(source_orbit_equivariance(fn, z[sub], SPEC, n_angles=5))
        out.update(orbit_steering_gain(fn, z[sub], SPEC, n_angles=7))
    return out


def test_end_to_end(seeds: int, steps: int) -> None:
    print("=" * 100)
    print("TEST 5  END-TO-END -- train a small conditional FM per coupling")
    print("=" * 100)
    print(f"  {seeds} seed(s) x {steps} steps, non-uniform target headings, "
          f"SO(2) block normalizer.")
    print(f"  Inference prior is iid N(0,I) except for the '+prior' rows, which sample")
    print(f"  the coupling's own induced source law.\n")
    cols = ["SWD", "heading_MAE", "few_step_gap_N1", "few_step_gap_N4",
            "straightness", "src_orbit_eq_rel", "orbit_steering_gain"]
    short = {"SWD": "SWD", "heading_MAE": "headMAE", "few_step_gap_N1": "gapN1",
             "few_step_gap_N4": "gapN4", "straightness": "straight",
             "src_orbit_eq_rel": "srcEqErr", "orbit_steering_gain": "steerGain"}
    head = f"  {'coupling':<20}" + "".join(f"{short[c]:>12}" for c in cols)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, (cfg, use_prior) in COUPLINGS.items():
        t0 = time.perf_counter()
        runs = [train_one(cfg, use_prior, s, steps) for s in range(seeds)]
        agg = {k: sum(r[k] for r in runs) / len(runs) for k in cols}
        row = f"  {name:<20}" + "".join(f"{agg[c]:>12.4f}" for c in cols)
        print(row + f"  [{time.perf_counter() - t0:.0f}s]")
    print("\n  lower is better for SWD / headMAE / gapN* / straight / srcEqErr;")
    print("  steerGain should approach 1.0 for a coupling that makes the source phase")
    print("  control the output heading -- that is what stage 1 of DNO consumes.\n")


# ======================================================================================
# Test 6 -- the 2x2 that the whole DNO argument rests on
# ======================================================================================


DNO_METHODS = ["A/iid", "P3/perm-euclid", "C3/rot-head+prior"]


def test_orbit_dno(seed: int = 0, steps: int = 2500, K: int = 16, n_test: int = 512) -> None:
    """{iid, orbit-coupled} policy  x  {stage-1 orbit search on, off}.

    Test 5 shows the orbit-coupled model has a steering gain near 1 -- it has learned to
    read the chunk's heading off the source phase.  The flip side is that at inference the
    source phase is drawn independently of the observation, so the heading is chosen by
    chance: conditional heading error goes UP even as the marginal stays plausible.

    That is not a defect to be regularised away, it is the contract.  A policy that
    delegates the heading to the noise must be *given* the heading -- which is exactly what
    a one-dimensional orbit search over a task objective does, for the cost of one batched
    forward pass.  This test checks that the trade actually closes.

    The task objective here is deliberately weak and noisy: "head toward the direction the
    observation suggests", using the same noisy heading estimate the condition vector
    already carries.  That is the shape of a real geometric objective -- informative, not
    an oracle.
    """
    print("=" * 100)
    print("TEST 6  ORBIT DNO -- does a 1-D orbit search close the conditional gap?")
    print("=" * 100)
    print(f"  K={K} orbit angles, one batched forward pass, no gradients.\n")

    ge = torch.Generator().manual_seed(11)
    test = make_chunks(n_test, "vonmises", ge)
    cond = test["cond"]
    theta_obs = torch.atan2(cond[:, 5], cond[:, 4])

    def task_loss(gen):                       # (B,H,7) -> (B,) ; lower is better
        th, _ = chunk_heading(gen, SPEC)
        return -torch.cos(th - theta_obs)

    print(f"  {'method':<20}{'steerGain':>11}{'headMAE':>10}{'headMAE':>10}"
          f"{'task L':>10}{'task L':>10}")
    print(f"  {'':<20}{'':>11}{'(no DNO)':>10}{'(+orbit)':>10}{'(no DNO)':>10}{'(+orbit)':>10}")
    print("  " + "-" * 71)

    for name in DNO_METHODS:
        cfg, use_prior = COUPLINGS[name]
        model, scale, offset, prior = _train_model(cfg, use_prior, seed, steps)

        def sample(z, c, n_steps=20):
            x = z
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = torch.full((x.shape[0],), i * dt)
                x = x + dt * model(x, t, c)
            return x

        with torch.no_grad():
            z = torch.randn((n_test, H, 7), generator=ge)
            if prior is not None:
                z = prior.sample(z)

            base = sample(z, cond)
            th0, _ = chunk_heading(base, SPEC)
            mae0 = float(torch.abs(wrap_angle(th0 - test["theta"])).mean())
            tl0 = float(task_loss(base).mean())

            gain = orbit_steering_gain(lambda zz: sample(zz, cond), z, SPEC, 7)["orbit_steering_gain"]

            # ---- stage 1: exhaustive search over the orbit --------------------------
            angles = torch.arange(K, dtype=torch.float32) * (2 * math.pi / K)
            losses, gens = [], []
            for k in range(K):
                zk = rotate_chunk(z, angles[k].expand(n_test), SPEC)
                gk = sample(zk, cond)
                gens.append(gk)
                losses.append(task_loss(gk))
            L = torch.stack(losses, 1)                         # (B, K)
            best = L.argmin(1)
            G = torch.stack(gens, 1)                           # (B, K, H, 7)
            sel = G[torch.arange(n_test), best]

            th1, _ = chunk_heading(sel, SPEC)
            mae1 = float(torch.abs(wrap_angle(th1 - test["theta"])).mean())
            tl1 = float(L.min(1).values.mean())

        print(f"  {name:<20}{gain:>11.3f}{mae0:>10.3f}{mae1:>10.3f}{tl0:>10.3f}{tl1:>10.3f}")

    print("\n  Reading: with steering gain 0 the K rotations of an iid policy's source are K")
    print("  UNRELATED samples, so the search degenerates into best-of-K resampling -- which")
    print("  still helps a noisy objective.  The claim is the sharper one: at identical")
    print("  compute the coupling turns that blind draw into a structured 1-D search over the")
    print("  coordinate that matters, and the coupled policy ends up MORE conditionally")
    print("  accurate than the iid one despite starting less so.\n")


# ======================================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tests", default="1,2,3,4,5,6")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.steps = 1, 700
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    which = {int(t) for t in args.tests.split(",") if t.strip()}

    print(f"\nSO(2) coupling validation   spec: vector_blocks={SPEC.vector_blocks} "
          f"scalar_dims={SPEC.scalar_dims}  H={H}")
    if 1 in which:
        test_marginal(n=2048 if args.quick else 4096)
    if 2 in which:
        test_transport()
    if 3 in which:
        test_normalizer()
    if 4 in which:
        test_optimizer()
    if 5 in which:
        test_end_to_end(args.seeds, args.steps)
    if 6 in which:
        test_orbit_dno(steps=args.steps, K=8 if args.quick else 16,
                       n_test=256 if args.quick else 512)


if __name__ == "__main__":
    main()
