"""
Diagnostics for symmetry-aware flow policies.

WHAT "EQUIVARIANCE" CAN AND CANNOT MEAN HERE
--------------------------------------------
The textbook property is

    F(rho_Z(g) z | rho_C(g) o) = rho_X(g) F(z | o).                              (1)

With LIBERO's fixed ``agentview`` camera you cannot evaluate (1), and not because of a
measurement difficulty: a global rotation of the world rotates the camera and the objects
together, so the rendered image is *invariant* rather than equivariant.  Hu et al.,
"3D Equivariant Visuomotor Policy Learning via Spherical Projection" (NeurIPS 2025
Spotlight, arXiv:2505.16969) states this explicitly and is the right citation for why the
RGB setting is the honest one to study a *relaxed* notion in.

So this module measures three genuinely different things, and the design doc is careful
never to call the first one by the name of the third:

1. ``source_orbit_equivariance`` -- hold ``o`` fixed, rotate only the source noise:

       E || F(rho(phi) z | o) - rho(phi) F(z | o) ||  /  E || F(z | o) ||.

   This is measurable in the real RGB setting, it is exactly the property the orbit
   stage of DNO relies on, and it is the property the coupling most directly targets.
   Call it *noise-space* or *source-orbit* equivariance -- never just "equivariance".

2. ``orbit_steering_gain`` -- the same probe read as a control gain: how many radians the
   generated chunk's heading turns per radian of source rotation.  1.0 is ideal, 0.0
   means the noise phase is a nuisance variable the network has learned to ignore.

3. ``policy_equivariance`` -- the real property (1).  Requires an observation transform,
   so it is only available in a state-only or point-cloud variant of the task.  Provided
   so the state-obs ablation can report the honest number.

Plus the transport diagnostics that carried the toy result: path length, conditional
velocity variance, and straightness.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch

from oat.symmetry.so2_chunk import (
    SO2ChunkSpec,
    chunk_heading,
    rotate_chunk,
    wrap_angle,
)

SampleFn = Callable[[torch.Tensor], torch.Tensor]
"""z (B, H, A) -> generated chunk (B, H, A), with the observation already bound."""


# --------------------------------------------------------------------------------------
# 1 + 2: noise-space probes (available with fixed RGB observations)
# --------------------------------------------------------------------------------------


@torch.no_grad()
def source_orbit_equivariance(
    sample_fn: SampleFn,
    z: torch.Tensor,
    spec: SO2ChunkSpec,
    n_angles: int = 8,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Relative error of ``F(rho(phi) z) == rho(phi) F(z)`` at fixed observation."""
    base = sample_fn(z)
    errs, rels = [], []
    for k in range(1, n_angles + 1):
        phi = torch.full((z.shape[0],), 2 * math.pi * k / (n_angles + 1), device=z.device)
        lhs = sample_fn(rotate_chunk(z, phi, spec))
        rhs = rotate_chunk(base, phi, spec)
        num = (lhs - rhs).reshape(z.shape[0], -1).norm(dim=-1)
        den = rhs.reshape(z.shape[0], -1).norm(dim=-1) + eps
        errs.append(num.mean())
        rels.append((num / den).mean())
    return {
        "src_orbit_eq_abs": float(torch.stack(errs).mean()),
        "src_orbit_eq_rel": float(torch.stack(rels).mean()),
    }


@torch.no_grad()
def orbit_steering_gain(
    sample_fn: SampleFn,
    z: torch.Tensor,
    spec: SO2ChunkSpec,
    n_angles: int = 16,
) -> Dict[str, float]:
    """d(output heading) / d(source rotation).  1.0 = the noise phase steers the chunk.

    Estimated with a circular least-squares fit: the complex mean of
    ``exp(i (dtheta_out - phi))`` has modulus 1 and phase 0 exactly when the gain is 1.
    ``gain`` itself is the slope of a wrapped linear fit through the origin.
    """
    base = sample_fn(z)
    th0, _ = chunk_heading(base, spec)

    phis, dth = [], []
    for k in range(1, n_angles + 1):
        phi_val = 2 * math.pi * k / (n_angles + 1)
        phi = torch.full((z.shape[0],), phi_val, device=z.device)
        out = sample_fn(rotate_chunk(z, phi, spec))
        th, _ = chunk_heading(out, spec)
        phis.append(phi)
        dth.append(wrap_angle(th - th0))

    phi_all = torch.cat(phis)
    dth_all = torch.cat(dth)

    # slope through the origin, on the circle: minimise sum |wrap(dth - g*phi)|^2
    grid = torch.linspace(-0.5, 2.0, 251, device=z.device)
    resid = wrap_angle(dth_all[None, :] - grid[:, None] * phi_all[None, :]).pow(2).mean(-1)
    gain = float(grid[int(torch.argmin(resid))])

    consistency = float(torch.abs(torch.exp(1j * (dth_all - phi_all).to(torch.float32)).mean()))
    return {"orbit_steering_gain": gain, "orbit_phase_consistency": consistency}


# --------------------------------------------------------------------------------------
# 3: the real thing, for the state-observation ablation
# --------------------------------------------------------------------------------------


@torch.no_grad()
def policy_equivariance(
    sample_with_obs: Callable[[torch.Tensor, dict], torch.Tensor],
    z: torch.Tensor,
    obs: dict,
    obs_transform: Callable[[dict, torch.Tensor], dict],
    spec: SO2ChunkSpec,
    n_angles: int = 8,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Full F(rho z | rho o) vs rho F(z | o).  Needs an observation that transforms."""
    base = sample_with_obs(z, obs)
    rels = []
    for k in range(1, n_angles + 1):
        phi = torch.full((z.shape[0],), 2 * math.pi * k / (n_angles + 1), device=z.device)
        lhs = sample_with_obs(rotate_chunk(z, phi, spec), obs_transform(obs, phi))
        rhs = rotate_chunk(base, phi, spec)
        num = (lhs - rhs).reshape(z.shape[0], -1).norm(dim=-1)
        den = rhs.reshape(z.shape[0], -1).norm(dim=-1) + eps
        rels.append((num / den).mean())
    return {"policy_eq_rel": float(torch.stack(rels).mean())}


@torch.no_grad()
def field_equivariance(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    spec: SO2ChunkSpec,
    n_angles: int = 8,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Local velocity-field version, at fixed observation.  The analogue of the toy
    ``E_field``, evaluated on whatever spatial distribution ``x`` is drawn from.

    Evaluate it on *several* distributions -- common iid interpolation points, each
    method's own training paths, each method's own inference trajectories -- exactly as
    the toy Round-2 plan calls for.  A single spatial distribution hides the difference
    between global relaxed equivariance and trajectory-local equivariance.
    """
    base = velocity_fn(x, t)
    rels = []
    for k in range(1, n_angles + 1):
        phi = torch.full((x.shape[0],), 2 * math.pi * k / (n_angles + 1), device=x.device)
        lhs = velocity_fn(rotate_chunk(x, phi, spec), t)
        rhs = rotate_chunk(base, phi, spec)
        num = (lhs - rhs).reshape(x.shape[0], -1).norm(dim=-1)
        den = rhs.reshape(x.shape[0], -1).norm(dim=-1) + eps
        rels.append((num / den).mean())
    return {"field_eq_rel": float(torch.stack(rels).mean())}


# --------------------------------------------------------------------------------------
# Transport diagnostics
# --------------------------------------------------------------------------------------


@torch.no_grad()
def conditional_velocity_variance(
    z0: torch.Tensor,
    x1: torch.Tensor,
    n_probe: int = 512,
    k: int = 16,
    n_time: int = 8,
) -> float:
    """KNN estimate of ``E_t Var[u | x_t, t]`` for a coupling, with ``u = x1 - z0``.

    The quantity the toy result identified as the mechanism: a coupling helps by making
    the regression target less ambiguous at a given point of the interpolation.
    High-dimensional KNN is a weak estimator in absolute terms; it is used here only to
    compare couplings on identical data.
    """
    B = z0.shape[0]
    n_probe = min(n_probe, B)
    idx = torch.randperm(B, device=z0.device)[:n_probe]
    z0, x1 = z0[idx], x1[idx]
    u = (x1 - z0).reshape(n_probe, -1)

    total = 0.0
    for i in range(n_time):
        t = torch.full((n_probe, 1, 1), (i + 0.5) / n_time, device=z0.device, dtype=z0.dtype)
        xt = ((1 - t) * z0 + t * x1).reshape(n_probe, -1)
        d = torch.cdist(xt, xt)
        nn_idx = torch.topk(d, k=min(k, n_probe), largest=False).indices
        neigh = u[nn_idx]                                   # (n, k, D)
        total += float(neigh.var(dim=1, unbiased=False).sum(-1).mean())
    return total / n_time


@torch.no_grad()
def transport_stats(z0: torch.Tensor, x1: torch.Tensor, spec: SO2ChunkSpec) -> Dict[str, float]:
    path = (x1 - z0).reshape(x1.shape[0], -1)
    tz, _ = chunk_heading(z0, spec)
    tx, _ = chunk_heading(x1, spec)
    return {
        "path_len": float(path.norm(dim=-1).mean()),
        "path_len_sq": float((path ** 2).sum(-1).mean()),
        "angle_gap": float(torch.abs(wrap_angle(tz - tx)).mean()),
        "cond_vel_var": conditional_velocity_variance(z0, x1),
    }


@torch.no_grad()
def straightness(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    n_steps: int = 50,
) -> Dict[str, float]:
    """Rectified-flow straightness of the *learned* field, plus the few-step gap.

    ``straightness = E int_0^1 || (x_1 - x_0) - v(x_t, t) ||^2 dt`` -- 0 for a perfectly
    straight flow, which is exactly when one Euler step suffices.  This is the practical
    payoff to argue for in a robotics venue: shorter transport buys inference steps.
    """
    x = z.clone()
    xs, vs = [], []
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((z.shape[0],), i * dt, device=z.device, dtype=z.dtype)
        v = velocity_fn(x, t)
        xs.append(x.clone())
        vs.append(v)
        x = x + dt * v
    disp = (x - z).reshape(z.shape[0], 1, -1)
    V = torch.stack(vs, dim=1).reshape(z.shape[0], n_steps, -1)
    return {
        "straightness": float(((disp - V) ** 2).sum(-1).mean()),
        "path_arclength": float((V.norm(dim=-1).mean(1) * 1.0).mean()),
        "displacement": float(disp.reshape(z.shape[0], -1).norm(dim=-1).mean()),
    }


@torch.no_grad()
def few_step_gap(
    sample_fn_n: Callable[[torch.Tensor, int], torch.Tensor],
    z: torch.Tensor,
    steps=(1, 2, 4, 10),
    ref_steps: int = 100,
) -> Dict[str, float]:
    """|| F_N(z) - F_ref(z) || / || F_ref(z) || for a few small step budgets."""
    ref = sample_fn_n(z, ref_steps)
    den = ref.reshape(z.shape[0], -1).norm(dim=-1).clamp_min(1e-8)
    out = {}
    for n in steps:
        cur = sample_fn_n(z, n)
        num = (cur - ref).reshape(z.shape[0], -1).norm(dim=-1)
        out[f"few_step_gap_N{n}"] = float((num / den).mean())
    return out
