"""
Per-point orbit-steering probe: the scatter behind ``orbit_steering_gain``.

WHY THIS EXISTS
---------------
``metrics.orbit_steering_gain`` runs the right experiment and then throws the evidence away.
It rotates the source by a grid of angles, measures how far the generated chunk's heading
turns, fits a slope, and returns two scalars. Two scalars cannot separate

    "the output heading tracks the source rotation one-for-one"

from

    "the residuals happen to average to a slope of 1".

Only the scatter can. This module runs the identical loop and keeps every
``(phi, delta_theta)`` pair, then derives the gain and the phase consistency **from exactly
the points it returns** — so a figure drawn from these points and the scalar quoted beside it
cannot disagree.

Added alongside ``metrics.orbit_steering_gain`` rather than replacing it:
``report_coupling_diagnostics.py`` depends on that function's current behaviour, and
EXPERIMENT_PLAN §0.1 forbids modifying files other code relies on.

ON THE READOUT SPEC
-------------------
``chunk_heading`` weights the vector blocks by ``spec.block_weights``, and each checkpoint
carries its own spec in its config. An arm trained before the G4' decision (full two-block
spec) and one trained after it (``block_weights=[1.0, 0.0]``) therefore report *different
quantities* under their own specs, and overlaying them would be meaningless.

So the caller passes ONE readout spec and it is applied to every arm. This is safe rather
than a fudge: ``rotate_chunk`` acts on ``spec.vector_blocks`` and rotates every block
regardless of the weights, so forcing the readout changes what is **measured**, never what is
**done** to the noise.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch

from oat.symmetry.so2_chunk import SO2ChunkSpec, chunk_heading, rotate_chunk, wrap_angle

SampleFn = Callable[[torch.Tensor], torch.Tensor]


def _fit_gain(dtheta: torch.Tensor, phi: torch.Tensor) -> float:
    """Slope through the origin on the circle — identical grid to ``orbit_steering_gain``."""
    grid = torch.linspace(-0.5, 2.0, 251, device=dtheta.device)
    resid = wrap_angle(dtheta[None, :] - grid[:, None] * phi[None, :]).pow(2).mean(-1)
    return float(grid[int(torch.argmin(resid))])


def _phase_consistency(dtheta: torch.Tensor, phi: torch.Tensor) -> float:
    """|E exp(i(dtheta - phi))| — 1.0 iff gain is exactly 1 with no scatter."""
    return float(torch.abs(torch.exp(1j * (dtheta - phi).to(torch.float32)).mean()))


@torch.no_grad()
def orbit_steering_points(
    sample_fn: SampleFn,
    z: torch.Tensor,
    spec: SO2ChunkSpec,
    n_angles: int = 24,
    conf_threshold: float = 0.05,
) -> Dict:
    """Rotate the source over a grid and keep every ``(phi, delta_theta)`` pair.

    Args:
        sample_fn: ``z -> generated chunk``, observation already bound.
        z: (B, H, A) source noise, fixed before any rotation.
        spec: the **readout** spec, forced identically across arms.
        n_angles: grid size. Angles are ``2*pi*k/(n_angles+1)`` for ``k = 1..n_angles``,
            matching ``orbit_steering_gain`` so gains stay comparable.
        conf_threshold: ``chunk_heading`` confidence below which a point is flagged. Points
            are flagged, never dropped — how many there are is part of the result.

    Returns:
        dict with flat ``phi`` / ``dtheta`` / ``confidence`` lists (length B*n_angles),
        the derived ``gain`` and ``phase_consistency``, and the low-confidence fraction.
    """
    base = sample_fn(z)
    th0, conf0 = chunk_heading(base, spec)

    phis, dths, confs = [], [], []
    for k in range(1, n_angles + 1):
        phi_val = 2.0 * math.pi * k / (n_angles + 1)
        phi = torch.full((z.shape[0],), phi_val, device=z.device, dtype=th0.dtype)
        out = sample_fn(rotate_chunk(z, phi, spec))
        th, conf = chunk_heading(out, spec)
        phis.append(phi)
        dths.append(wrap_angle(th - th0))
        # a delta is only as trustworthy as the weaker of the two headings it differences
        confs.append(torch.minimum(conf0, conf))

    phi_all = torch.cat(phis)
    dth_all = torch.cat(dths)
    conf_all = torch.cat(confs)

    low = conf_all < conf_threshold
    return {
        "phi": [float(v) for v in phi_all.cpu()],
        "dtheta": [float(v) for v in dth_all.cpu()],
        "confidence": [float(v) for v in conf_all.cpu()],
        "gain": _fit_gain(dth_all, phi_all),
        "phase_consistency": _phase_consistency(dth_all, phi_all),
        "n_points": int(phi_all.numel()),
        "n_obs": int(z.shape[0]),
        "n_angles": int(n_angles),
        "conf_threshold": float(conf_threshold),
        "low_confidence_frac": float(low.float().mean()),
        # the same fit restricted to trustworthy points, so the caption can say whether the
        # low-confidence tail is doing any of the work
        "gain_high_conf": (_fit_gain(dth_all[~low], phi_all[~low])
                           if int((~low).sum()) > 8 else float("nan")),
        "phase_consistency_high_conf": (_phase_consistency(dth_all[~low], phi_all[~low])
                                        if int((~low).sum()) > 8 else float("nan")),
    }


def readout_spec_signature(spec: SO2ChunkSpec) -> str:
    """Canonical string for a spec, so the renderer can refuse mismatched arms."""
    return (f"action_dim={spec.action_dim};vector_blocks={spec.vector_blocks};"
            f"scalar_dims={spec.scalar_dims};block_weights={spec.weights}")
