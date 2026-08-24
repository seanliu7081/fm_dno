"""
Symmetry-aware source-target couplings for flow-matching action-chunk policies.

Design space (this is the 2x2 that generalises the toy problem's A / C / D):

                       |  do NOT apply phi*      |  APPLY phi*
    -------------------+-------------------------+---------------------------------
    identity pairing   |  A  ``iid``             |  C  ``rot``     (toy Method C)
    batch assignment   |  P  ``perm``            |  F  ``perm_rot`` (Klein-style)

and an orthogonal choice of assignment cost:

    ``group_ot``   maximise |S_ij|      -- phase-invariant, symmetry-aware
    ``euclidean``  maximise Re(S_ij)    -- symmetry-blind control (OT-CFM / Immiscible)
    ``angle``      minimise circular distance between chunk headings (cheap, O(B log B))

WHY THE PERMUTATION VARIANT IS THE DEFAULT FOR ROBOT DATA
--------------------------------------------------------
In the double-ring toy problem the target angles are uniform, so *rotating* the source
so its phase matches the target's leaves the source marginal exactly N(0, I): the map
z -> rho(theta_1 - theta_0(z)) z only replaces z's orbit coordinate with an independent
uniform draw.

Robot demonstrations are not like that.  Planar headings of LIBERO action chunks are
strongly non-uniform and task-dependent (the arm mostly reaches toward where the objects
happen to be), so ``rot`` makes the *training* source inherit that non-uniform angular
density while inference still draws from an isotropic N(0, I).  That prior shift is a
train/test mismatch, and it is the single most likely way a naive port of the toy result
fails.

A pairing that only *permutes* sources within the minibatch cannot have this problem: no
source sample is ever modified, so the batch's source marginal is exactly the empirical
N(0, I) it started as.  This is the same argument Immiscible Diffusion (Li et al.,
NeurIPS 2024, arXiv:2406.12303) makes for assignment-based noise-data pairing, and the
marginal-preservation guarantee for minibatch couplings is Multisample Flow Matching
(Pooladian et al., ICML 2023, arXiv:2304.14772).  ``diagnose()`` measures the residual
angular non-uniformity so the claim is checked rather than assumed.

Both variants are implemented, because ``rot`` is the literal transfer of the toy result
and therefore the control you need in order to say anything about the toy-to-robot gap.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from oat.symmetry.so2_chunk import (
    SO2ChunkSpec,
    align_phase,
    as_complex,
    chunk_heading,
    circular_moments,
    pairwise_alignment,
    pairwise_euclidean_gain,
    rotate_chunk,
    scalar_part,
    wrap_angle,
)

_VALID_MODES = ("iid", "rot", "perm", "perm_rot")
_VALID_COSTS = ("group_ot", "euclidean", "angle")
_VALID_ASSIGNMENTS = ("hungarian", "circular_sort")


class SO2OrbitCoupling(nn.Module):
    """Couples a Gaussian source batch to a target action-chunk batch.

    Args:
        spec: irrep layout of the action vector.
        mode: one of ``iid`` / ``rot`` / ``perm`` / ``perm_rot``.
        cost: assignment cost, one of ``group_ot`` / ``euclidean`` / ``angle``.
            Ignored when ``mode`` does not use an assignment.
        scalar_weight: weight of the invariant channels in the assignment cost.  0.0
            couples on geometry only; >0 also matches gripper / dz / dwz content.
        kappa: von Mises concentration of the dither added to the applied rotation.
            ``inf`` is exact alignment (toy Round 1); finite values interpolate toward
            an independent coupling and are the chunk analogue of the planned soft
            radial sweep.  Only used when a rotation is applied.
        coupling_prob: fraction of the batch that receives the coupling.  The remainder
            keeps its independent pairing.  A cheap, exactly-marginal-preserving way to
            weaken the coupling, and a useful regulariser.
        min_heading_confidence: if > 0, chunks whose planar deltas cancel over the
            horizon (a hold or a reversal, where the heading is meaningless) are excluded
            from the coupling and left iid.
        assignment: solver for the assignment; ``circular_sort`` is only valid with
            ``cost='angle'`` and avoids the scipy dependency.
        align: which equivariant phase coordinate the rotation acts on.
            ``kabsch`` uses ``arg(sum_k conj(z_k) x_k)`` -- the transport-optimal choice,
            and the 2-D analogue of Klein et al.'s alignment step.
            ``heading`` sets ``theta(z) := theta(x)`` using the aggregate heading.  It is
            slightly worse transport but has an exactly characterisable induced source
            law (heading replaced by the target's, everything else untouched), which is
            what ``MatchedHeadingPrior`` needs in order to remove the prior shift
            exactly rather than approximately.  See that class.
    """

    def __init__(
        self,
        spec: SO2ChunkSpec,
        mode: str = "perm_rot",
        cost: str = "group_ot",
        scalar_weight: float = 0.0,
        kappa: float = float("inf"),
        coupling_prob: float = 1.0,
        min_heading_confidence: float = 0.0,
        assignment: str = "hungarian",
        align: str = "kabsch",
        collect_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode}")
        if cost not in _VALID_COSTS:
            raise ValueError(f"cost must be one of {_VALID_COSTS}, got {cost}")
        if align not in ("kabsch", "heading"):
            raise ValueError(f"align must be 'kabsch' or 'heading', got {align}")
        if assignment not in _VALID_ASSIGNMENTS:
            raise ValueError(f"assignment must be one of {_VALID_ASSIGNMENTS}, got {assignment}")
        if assignment == "circular_sort" and cost != "angle":
            raise ValueError("assignment='circular_sort' requires cost='angle'")
        if not 0.0 <= coupling_prob <= 1.0:
            raise ValueError(f"coupling_prob must be in [0, 1], got {coupling_prob}")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive (use mode='iid' for no coupling)")

        self.spec = spec
        self.mode = mode
        self.cost = cost
        self.scalar_weight = float(scalar_weight)
        self.kappa = float(kappa)
        self.coupling_prob = float(coupling_prob)
        self.min_heading_confidence = float(min_heading_confidence)
        self.assignment = assignment
        self.align = align
        self.collect_diagnostics = bool(collect_diagnostics)

    # -- public ------------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self, z0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return the coupled source and a dict of diagnostics.

        Args:
            z0: (B, H, A) independent source sample, already at the training noise scale.
            x1: (B, H, A) normalised target action chunk.
        """
        if z0.shape != x1.shape:
            raise ValueError(f"shape mismatch: z0 {tuple(z0.shape)} vs x1 {tuple(x1.shape)}")
        z_in = z0
        B = z0.shape[0]

        if self.mode == "iid" or B < 2:
            return z0, self._diagnose(z_in, z0, x1, active=torch.zeros(B, dtype=torch.bool))

        active = self._active_mask(z0, x1)

        z = z0
        if self.mode in ("perm", "perm_rot"):
            z = self._apply_assignment(z, x1, active)
        if self.mode in ("rot", "perm_rot"):
            z = self._apply_rotation(z, x1, active)

        return z, self._diagnose(z_in, z, x1, active)

    # -- internals ---------------------------------------------------------------------

    def _active_mask(self, z0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        B = z0.shape[0]
        active = torch.ones(B, dtype=torch.bool, device=z0.device)
        if self.coupling_prob < 1.0:
            active &= torch.rand(B, device=z0.device) < self.coupling_prob
        if self.min_heading_confidence > 0.0:
            _, conf = chunk_heading(x1, self.spec)
            active &= conf > self.min_heading_confidence
        return active

    def _assignment_gain(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(B, B) gain matrix to be MAXIMISED by the assignment."""
        if self.cost == "group_ot":
            gain, _ = pairwise_alignment(z, x, self.spec)
        elif self.cost == "euclidean":
            gain = pairwise_euclidean_gain(z, x, self.spec)
        else:  # angle
            tz, _ = chunk_heading(z, self.spec)
            tx, _ = chunk_heading(x, self.spec)
            d = torch.abs(wrap_angle(tz[:, None] - tx[None, :]))
            gain = -d
        if self.scalar_weight > 0.0 and self.spec.n_scalar_dims > 0:
            sz = scalar_part(z, self.spec)
            sx = scalar_part(x, self.spec)
            gain = gain + self.scalar_weight * (sz @ sx.transpose(0, 1))
        return gain

    def _apply_assignment(
        self, z: torch.Tensor, x: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        """Permute sources among the active rows.  Never modifies a source sample."""
        idx = torch.nonzero(active, as_tuple=False).flatten()
        if idx.numel() < 2:
            return z
        zs, xs = z[idx], x[idx]

        if self.assignment == "circular_sort":
            perm = _circular_sort_assignment(zs, xs, self.spec)
        else:
            gain = self._assignment_gain(zs, xs)
            perm = _hungarian_assignment(gain)

        out = z.clone()
        out[idx] = zs[perm]
        return out

    def _apply_rotation(
        self, z: torch.Tensor, x: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        """Rotate each active source onto its target's orbit position."""
        if self.align == "kabsch":
            phi = align_phase(z, x, self.spec)
        else:
            tz, _ = chunk_heading(z, self.spec)
            tx, _ = chunk_heading(x, self.spec)
            phi = wrap_angle(tx - tz)
        if math.isfinite(self.kappa):
            dither = torch.distributions.VonMises(
                loc=torch.zeros_like(phi),
                concentration=torch.full_like(phi, self.kappa),
            ).sample()
            phi = phi + dither
        phi = torch.where(active, phi, torch.zeros_like(phi))
        return rotate_chunk(z, phi, self.spec)

    def _diagnose(
        self,
        z_in: torch.Tensor,
        z_out: torch.Tensor,
        x1: torch.Tensor,
        active: torch.Tensor,
    ) -> Dict[str, float]:
        if not self.collect_diagnostics:
            return {}
        spec = self.spec
        d: Dict[str, float] = {}
        d["coupling/active_frac"] = float(active.float().mean())

        path = (x1 - z_out).reshape(x1.shape[0], -1)
        d["coupling/path_len"] = float(path.norm(dim=-1).mean())
        d["coupling/path_len_sq"] = float((path ** 2).sum(-1).mean())

        tz, _ = chunk_heading(z_out, spec)
        tx, cx = chunk_heading(x1, spec)
        d["coupling/angle_gap"] = float(torch.abs(wrap_angle(tz - tx)).mean())
        d["coupling/target_heading_conf"] = float(cx.mean())

        # radius / magnitude correlation on the vector part
        rz = as_complex(z_out, spec).abs().pow(2).sum(-1).sqrt()
        rx = as_complex(x1, spec).abs().pow(2).sum(-1).sqrt()
        if rz.numel() > 1:
            rzc, rxc = rz - rz.mean(), rx - rx.mean()
            denom = (rzc.norm() * rxc.norm()).clamp_min(1e-12)
            d["coupling/radius_corr"] = float((rzc * rxc).sum() / denom)

        # --- marginal audit: has the coupling deformed the source? --------------------
        for k, v in circular_moments(tz).items():
            d[f"coupling/src_heading_{k}"] = v
        for k, v in circular_moments(tx).items():
            d[f"coupling/tgt_heading_{k}"] = v
        flat = z_out.reshape(-1, spec.action_dim)
        d["coupling/src_mean_absmax"] = float(flat.mean(0).abs().max())
        d["coupling/src_std_min"] = float(flat.std(0).min())
        d["coupling/src_std_max"] = float(flat.std(0).max())
        # the permutation modes must leave the multiset of sources untouched
        d["coupling/src_norm_drift"] = float(
            (z_out.reshape(z_out.shape[0], -1).norm(dim=-1).sort().values
             - z_in.reshape(z_in.shape[0], -1).norm(dim=-1).sort().values).abs().max()
        )
        return d

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, cost={self.cost}, align={self.align}, kappa={self.kappa}, "
            f"scalar_weight={self.scalar_weight}, coupling_prob={self.coupling_prob}"
        )


# --------------------------------------------------------------------------------------
# Matched inference prior
# --------------------------------------------------------------------------------------


class MatchedHeadingPrior:
    """An inference prior that matches what a ``rot`` coupling actually trained on.

    THE OBSTRUCTION THIS RESOLVES
    -----------------------------
    Measured on synthetic chunks with realistically non-uniform target headings
    (``scripts/validate_so2_coupling.py``, tests 1-2), the picture is:

        coupling            angular gap    source heading R1 (0 = undeformed)
        iid                     1.60              0.05
        perm  (any cost)        1.05-1.57         0.06         <- marginal exact
        rot / perm+rot          0.83-0.85         0.27         <- prior deformed

    That gap is not an implementation defect, it is a constraint.  If the source's
    angular law must stay uniform and the target's is not, then the expected angular
    transport is bounded below by the circular Wasserstein distance
    ``W_1(Uniform, p_theta) > 0``.  Optimal circular OT (``mode='perm', cost='angle'``)
    attains that bound, and **nothing marginal-preserving can beat it**.  Buying more
    transport than that necessarily costs prior deformation.  In the double-ring toy
    problem ``p_theta`` is uniform, the bound is 0, and the tension is invisible; on
    robot demonstrations it is the whole story.

    THE RESOLUTION
    --------------
    Stop insisting the inference prior be isotropic.  Flow matching needs the source you
    *sample* from to equal the source you *trained* on -- not for either to be N(0, I).
    So: train with ``mode='rot', align='heading'``, then sample from the induced law.

    With ``align='heading'`` the induced law is exactly characterisable.  Decompose
    ``z = (theta, w)`` into its heading and the orbit-invariant remainder; for an
    isotropic Gaussian those are independent with ``theta ~ U(0, 2pi)``.  The coupling
    replaces ``theta`` by ``theta(x_1)`` and touches nothing else, so the induced source
    is exactly "N(0, I) with the heading redrawn from ``p_theta``".  Fitting a circular
    histogram of the training headings and re-imposing it at inference reproduces that
    law exactly -- no deformation, and the full angular transport benefit kept.

    (With ``align='kabsch'`` the phase coordinate that gets replaced is target-dependent,
    so the induced law is only approximately of this form -- test 1 shows a residual
    coordinate-mean shift.  That is the price of the transport-optimal alignment, and the
    reason both alignments are implemented.)

    Usage::

        prior = MatchedHeadingPrior(spec)
        prior.fit(train_action_chunks_normalized)     # once, after fitting the normalizer
        z = prior.sample(policy.sample_prior(B, device=dev))   # at inference

    Report ``W_1(source heading, uniform)`` alongside every result: it is the honest
    statement of how far the prior moved, and it is the second axis of the Pareto plot
    whose first axis is transport cost.
    """

    def __init__(self, spec: SO2ChunkSpec, n_bins: int = 128) -> None:
        self.spec = spec
        self.n_bins = int(n_bins)
        self._cdf: Optional[torch.Tensor] = None
        self._edges: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit(self, x1: torch.Tensor) -> "MatchedHeadingPrior":
        """Fit the circular heading histogram of a set of (normalised) target chunks."""
        theta, _ = chunk_heading(x1, self.spec)
        theta = torch.remainder(theta, 2 * math.pi)
        edges = torch.linspace(0, 2 * math.pi, self.n_bins + 1, device=theta.device)
        counts = torch.histc(theta.float(), bins=self.n_bins, min=0.0, max=2 * math.pi)
        counts = counts + 1e-3 * counts.mean().clamp_min(1e-6)      # never a zero bin
        cdf = torch.cumsum(counts, 0)
        self._cdf = cdf / cdf[-1]
        self._edges = edges
        return self

    @torch.no_grad()
    def sample(self, z: torch.Tensor) -> torch.Tensor:
        """Re-impose the fitted heading law on an isotropic Gaussian sample."""
        if self._cdf is None:
            raise RuntimeError("MatchedHeadingPrior.fit() must be called first")
        cdf = self._cdf.to(z.device)
        edges = self._edges.to(z.device)
        u = torch.rand(z.shape[0], device=z.device)
        idx = torch.searchsorted(cdf, u.clamp(0, 1 - 1e-6)).clamp(0, self.n_bins - 1)
        width = edges[1] - edges[0]
        theta_star = edges[idx] + width * torch.rand(z.shape[0], device=z.device)
        theta_z, _ = chunk_heading(z, self.spec)
        return rotate_chunk(z, wrap_angle(theta_star - theta_z), self.spec)

    @torch.no_grad()
    def uniformity_gap(self, x1: torch.Tensor) -> float:
        """Circular W1 between the target heading law and uniform.

        The lower bound on angular transport for *any* marginal-preserving coupling, and
        therefore the number that says whether ``perm`` can help at all on your dataset.
        """
        theta, _ = chunk_heading(x1, self.spec)
        theta = torch.remainder(theta, 2 * math.pi).sort().values
        n = theta.numel()
        unif = torch.linspace(0, 2 * math.pi, n + 1, device=theta.device)[:n]
        # minimise over the cut point, as circular OT requires
        best = float("inf")
        for s in range(0, n, max(1, n // 64)):
            shifted = torch.roll(theta, s)
            d = torch.abs(wrap_angle(shifted - unif)).mean()
            best = min(best, float(d))
        return best


# --------------------------------------------------------------------------------------
# Assignment solvers
# --------------------------------------------------------------------------------------


def _hungarian_assignment(gain: torch.Tensor) -> torch.Tensor:
    """argmax_perm sum_i gain[perm[i], i].  Returns ``perm`` of shape (B,).

    ``perm[i]`` is the index of the source assigned to target ``i``.
    Falls back to a greedy solve if scipy is unavailable.
    """
    B = gain.shape[0]
    cost = (-gain).detach().float().cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        perm = torch.empty(B, dtype=torch.long)
        perm[torch.as_tensor(cols)] = torch.as_tensor(rows)
    except ImportError:  # pragma: no cover - greedy fallback
        perm = _greedy_assignment(torch.as_tensor(cost))
    return perm.to(gain.device)


def _greedy_assignment(cost: torch.Tensor) -> torch.Tensor:
    B = cost.shape[0]
    cost = cost.clone()
    perm = torch.empty(B, dtype=torch.long)
    for _ in range(B):
        flat = torch.argmin(cost)
        r, c = int(flat // B), int(flat % B)
        perm[c] = r
        cost[r, :] = float("inf")
        cost[:, c] = float("inf")
    return perm


def _circular_sort_assignment(
    z: torch.Tensor, x: torch.Tensor, spec: SO2ChunkSpec
) -> torch.Tensor:
    """O(B log B + B^2) optimal transport on the circle for the angle-only cost.

    For equal-weight empirical measures on S^1 with a convex cost of geodesic distance,
    an optimal plan is cyclically monotone, so it suffices to sort both angle sets and
    search the B cyclic shifts of the rank matching.
    """
    B = z.shape[0]
    tz, _ = chunk_heading(z, spec)
    tx, _ = chunk_heading(x, spec)
    oz = torch.argsort(tz)
    ox = torch.argsort(tx)
    sz, sx = tz[oz], tx[ox]

    d = torch.abs(wrap_angle(sz[None, :] - sx[:, None]))          # (rank_x, rank_z)
    ar = torch.arange(B, device=z.device)
    shift_idx = (ar[None, :] + ar[:, None]) % B                    # (shift, rank_x)
    costs = torch.stack([d.gather(1, shift_idx[s].unsqueeze(1)).sum() for s in range(B)])
    s_best = int(torch.argmin(costs))

    perm = torch.empty(B, dtype=torch.long, device=z.device)
    perm[ox] = oz[(ar + s_best) % B]
    return perm


# --------------------------------------------------------------------------------------
# Config helper
# --------------------------------------------------------------------------------------


def build_coupling(cfg: Optional[dict], spec: SO2ChunkSpec) -> SO2OrbitCoupling:
    """Build a coupling from a plain dict (hydra-friendly).  ``None`` -> iid."""
    cfg = dict(cfg or {})
    cfg.pop("_target_", None)
    kappa = cfg.pop("kappa", float("inf"))
    if isinstance(kappa, str):  # yaml ".inf" / "inf"
        kappa = float(kappa)
    if kappa is None:
        kappa = float("inf")
    return SO2OrbitCoupling(spec=spec, kappa=kappa, **cfg)
