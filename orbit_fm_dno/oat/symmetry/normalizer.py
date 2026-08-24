"""
An SO(2)-compatible action normalizer.

WHY THIS FILE EXISTS
--------------------
``LinearNormalizer.fit(mode='limits')`` -- the repo default, inherited from Diffusion
Policy -- maps every action dimension independently to [-1, 1]:

    a_norm[d] = scale[d] * a[d] + offset[d],   scale[d] = 2 / (max_d - min_d)

Independently.  So ``scale[0] != scale[1]`` and ``offset != 0`` in general, which means
the normalized action space is an *anisotropic sheared* copy of the raw one.  A rotation
about z is then no longer a rotation:

    N(rho(phi) a)  !=  rho(phi) N(a).

Every geometric statement in this project -- the coupling, the equivariance metrics, the
orbit stage of DNO -- lives in the normalized space the network actually sees.  If the
normalizer breaks the group action, none of them mean anything.  So the normalizer has
to be fixed first, and the fix has to be validated (see
``scripts/validate_so2_coupling.py``, test 3) rather than assumed.

THE FIX
-------
Constrain the affine map to commute with rho(phi):

  * frequency-1 (vector) blocks: one shared scale per block, offset exactly 0.
    A scalar multiple of the identity commutes with a rotation; an offset never does.
  * frequency-0 (scalar) blocks: unconstrained affine, since rotations act trivially
    on them.  ``dz``, ``dwz`` and the gripper channel keep the usual min-max treatment.

Two ways to pick the shared scale:

  ``rms``       scale = 1 / sqrt(E[(a_i^2 + a_j^2) / 2]) -- unit per-coordinate RMS.
                Matches the isotropic N(0, I) source in second moment, which is what
                the flow-matching interpolation actually cares about.
  ``quantile``  scale = 1 / quantile_q(||(a_i, a_j)||) -- the bounded analogue of
                ``limits``: a q-fraction of blocks land inside the unit disc.  Use this
                if you need the bounded range that ``limits`` was giving you.

I could not find prior work analysing this interaction, so treat it as both a hazard and
a contribution: report P0 (per-dim min-max) vs P1 (this normalizer) with an otherwise
identical iid policy, so that the normalizer change is never confounded with the
coupling change.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from oat.symmetry.so2_chunk import SO2ChunkSpec, rotate_chunk

_VECTOR_MODES = ("rms", "quantile")
_SCALAR_MODES = ("limits", "gaussian")


def fit_so2_action_scale_offset(
    actions: torch.Tensor,
    spec: SO2ChunkSpec,
    vector_mode: str = "rms",
    scalar_mode: str = "limits",
    quantile: float = 0.995,
    output_scale: float = 1.0,
    range_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit a rotation-commuting affine action normalizer.

    Args:
        actions: (N, A) or (N, H, A) raw actions from the training set.
        spec: irrep layout.
        vector_mode: ``rms`` or ``quantile``.
        scalar_mode: ``limits`` (min-max to +/-output_scale) or ``gaussian``.
        quantile: used when ``vector_mode='quantile'``.
        output_scale: target magnitude, matching the repo's [-1, 1] convention.

    Returns:
        ``(scale, offset)``, both shape (A,), such that ``a_norm = a * scale + offset``.
    """
    if vector_mode not in _VECTOR_MODES:
        raise ValueError(f"vector_mode must be one of {_VECTOR_MODES}, got {vector_mode}")
    if scalar_mode not in _SCALAR_MODES:
        raise ValueError(f"scalar_mode must be one of {_SCALAR_MODES}, got {scalar_mode}")

    a = torch.as_tensor(actions, dtype=torch.float32).reshape(-1, spec.action_dim)
    scale = torch.ones(spec.action_dim, dtype=torch.float32)
    offset = torch.zeros(spec.action_dim, dtype=torch.float32)

    for i, j in spec.vector_blocks:
        if vector_mode == "rms":
            sigma = torch.sqrt(((a[:, i] ** 2 + a[:, j] ** 2) / 2.0).mean())
        else:
            norms = torch.sqrt(a[:, i] ** 2 + a[:, j] ** 2)
            sigma = torch.quantile(norms, quantile)
        sigma = torch.clamp(sigma, min=range_eps)
        s = output_scale / sigma
        scale[i] = s
        scale[j] = s
        # offsets stay exactly zero: a translation never commutes with a rotation.

    for d in spec.scalar_dims:
        col = a[:, d]
        if scalar_mode == "limits":
            lo, hi = col.min(), col.max()
            rng = hi - lo
            if rng < range_eps:
                scale[d] = 1.0
                offset[d] = -col.mean()
            else:
                scale[d] = 2.0 * output_scale / rng
                offset[d] = -output_scale - scale[d] * lo
        else:
            std = col.std()
            scale[d] = 1.0 / std if std > range_eps else 1.0
            offset[d] = -col.mean() * scale[d]

    return scale, offset


def so2_scale_offset_from_stats(
    stats: Dict[str, torch.Tensor],
    spec: SO2ChunkSpec,
    vector_mode: str = "rms",
    scalar_mode: str = "limits",
    output_scale: float = 1.0,
    range_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same fit, but from the summary statistics a fitted ``LinearNormalizer`` already holds.

    This is what makes the whole thing a zero-touch drop-in: the policy can rebuild its own
    action normalizer inside ``set_normalizer`` from ``params_dict['action']['input_stats']``,
    so ``TrainPolicyWorkspace`` never has to be modified and checkpoints round-trip normally.

    ``rms`` needs only second moments, which are recoverable exactly:
    ``E[a^2] = mean^2 + std^2``.  ``absmax`` is the bounded analogue of ``limits`` --
    ``sigma_b = sqrt(m_i^2 + m_j^2)`` with ``m = max(|min|, |max|)`` -- which guarantees every
    observed block lands inside the unit disc, at the cost of being driven by outliers.  Use
    ``fit_so2_action_scale_offset`` on the raw actions if you want a true quantile.
    """
    if vector_mode not in ("rms", "absmax"):
        raise ValueError(f"vector_mode must be 'rms' or 'absmax', got {vector_mode}")
    mean = torch.as_tensor(stats["mean"]).float()
    std = torch.as_tensor(stats["std"]).float()
    lo = torch.as_tensor(stats["min"]).float()
    hi = torch.as_tensor(stats["max"]).float()

    scale = torch.ones(spec.action_dim)
    offset = torch.zeros(spec.action_dim)

    for i, j in spec.vector_blocks:
        if vector_mode == "rms":
            second = 0.5 * ((mean[i] ** 2 + std[i] ** 2) + (mean[j] ** 2 + std[j] ** 2))
            sigma = torch.sqrt(second.clamp_min(range_eps ** 2))
        else:
            m_i = torch.maximum(lo[i].abs(), hi[i].abs())
            m_j = torch.maximum(lo[j].abs(), hi[j].abs())
            sigma = torch.sqrt((m_i ** 2 + m_j ** 2).clamp_min(range_eps ** 2))
        s = output_scale / sigma.clamp_min(range_eps)
        scale[i] = s
        scale[j] = s

    for d in spec.scalar_dims:
        if scalar_mode == "limits":
            rng = hi[d] - lo[d]
            if rng < range_eps:
                scale[d] = 1.0
                offset[d] = -mean[d]
            else:
                scale[d] = 2.0 * output_scale / rng
                offset[d] = -output_scale - scale[d] * lo[d]
        else:
            scale[d] = 1.0 / std[d] if std[d] > range_eps else 1.0
            offset[d] = -mean[d] * scale[d]

    return scale, offset


def equivariance_error(
    actions: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    spec: SO2ChunkSpec,
    n_angles: int = 8,
) -> float:
    """Relative ||N(rho a) - rho N(a)|| for a candidate (scale, offset).

    ~1e-7 for a correct SO(2) normalizer; O(0.1-1) for per-dimension min-max.
    """
    a = torch.as_tensor(actions, dtype=torch.float32)
    if a.ndim == 2:
        a = a[:, None, :]
    errs = []
    for k in range(n_angles):
        phi = torch.full((a.shape[0],), 2 * torch.pi * k / n_angles)
        lhs = rotate_chunk(a, phi, spec) * scale + offset
        rhs = rotate_chunk(a * scale + offset, phi, spec)
        errs.append((lhs - rhs).norm() / rhs.norm().clamp_min(1e-12))
    return float(torch.stack(errs).mean())


# --------------------------------------------------------------------------------------
# Integration with the repo's LinearNormalizer
# --------------------------------------------------------------------------------------


def patch_action_normalizer(
    normalizer,
    actions: torch.Tensor,
    spec: SO2ChunkSpec,
    key: str = "action",
    **fit_kwargs,
):
    """Replace ``normalizer[key]`` in a fitted ``LinearNormalizer`` in place.

    Drop this into ``TrainPolicyWorkspace.__init__`` right after
    ``normalizer = dataset.get_normalizer()``::

        from oat.symmetry.so2_chunk import SO2ChunkSpec
        from oat.symmetry.normalizer import patch_action_normalizer
        spec = SO2ChunkSpec.libero_osc_pose()
        patch_action_normalizer(normalizer, dataset.replay_buffer['action'][:], spec)

    Observation normalizers are left alone: with fixed RGB cameras the observation does
    not carry a group action anyway (see the design doc, section on what "equivariance"
    can and cannot mean here).
    """
    import torch.nn as nn

    scale, offset = fit_so2_action_scale_offset(actions, spec, **fit_kwargs)
    a = torch.as_tensor(actions, dtype=torch.float32).reshape(-1, spec.action_dim)

    params = nn.ParameterDict(
        {
            "scale": nn.Parameter(scale, requires_grad=False),
            "offset": nn.Parameter(offset, requires_grad=False),
            "input_stats": nn.ParameterDict(
                {
                    "min": nn.Parameter(a.min(0).values, requires_grad=False),
                    "max": nn.Parameter(a.max(0).values, requires_grad=False),
                    "mean": nn.Parameter(a.mean(0), requires_grad=False),
                    "std": nn.Parameter(a.std(0), requires_grad=False),
                }
            ),
        }
    )
    for p in params.parameters():
        p.requires_grad_(False)
    normalizer.params_dict[key] = params
    return normalizer
