"""
Construction B: irrep-blockwise independent batch assignment.

THE FACT THAT OPENS THE ROOM
----------------------------
The source is isotropic Gaussian, so its coordinate blocks are independent.  Applying a
*different* within-batch permutation to each block,

    z'_i = [ z_{pi_0(i)}^{(dx,dy)}, z_{pi_1(i)}^{(wx,wy)}, z_{pi_2(i)}^{(dz)},
             z_{pi_3(i)}^{(wz)},    z_{pi_4(i)}^{(grip)} ]

leaves every block's batch marginal exactly as it was -- each block value is used exactly
once -- and leaves the product structure of ``N(0, I)`` intact, because the blocks were
independent to begin with.  The data side is never touched.  So the marginal-exactness that
made ``mode='perm'`` safe survives, while the assignment problem drops from one 112-D
matching to five low-dimensional ones.  Minibatch OT's dilution is a *dimension* phenomenon;
this is the one legitimate way to lower the dimension without lowering the coupling's reach.

WHY THE SYMMETRY EARNS ITS KEEP HERE, AND ONLY HERE
---------------------------------------------------
The valid factorizations are exactly the block structures under which the source law is a
product -- and the SO(2) irrep decomposition *is* the canonical such structure for this
action space.  Frequency-1 blocks get a rotation-aware or angular cost; invariant channels
get a Euclidean one.  "Group-aware coupling" becomes a claim about *which product structure
to exploit*, not about applying rotations, which the admissibility criterion forbids in the
target-aligned form that P4 used (SR 0.008).

WHERE THE MONEY IS SUPPOSED TO BE
---------------------------------
The gripper channel is 73.5% of raw action energy, essentially binary with one switch time --
a 16-D coordinate of effective dimension ~2-3.  A per-block assignment can match source
sign-patterns to target open/close profiles in a way no 112-D joint assignment could, and
that is exactly the coordinate where mismatched pairs make the regression target bimodal at
small ``t``.

HONEST STATUS -- READ THIS BEFORE QUOTING ANY NUMBER FROM IT
------------------------------------------------------------
This is a *better-conditioned* ``perm``, not a categorically different object.  Two caveats
that do not go away:

  1. Like every batch assignment, its gain and its train/test source bias are **the same
     number**.  The tilt toward the targets *is* the transport reduction, and it is also
     exactly the amount by which the training source law departs from the ``N(0, I)`` that
     inference samples.  Strengthening the coupling strengthens the inconsistency in
     proportion.  It stays in class (3) of the notes, never class (2) or (i).
  2. Blockwise permutation preserves each block's empirical batch marginal **exactly** (a
     stronger statement than joint ``perm`` can make per block) but does *not* preserve the
     empirical joint: it manufactures block combinations that were not in the batch.  Under
     the true product law ``N(0, I)`` those combinations are equally typical, so the
     population marginal is untouched; at finite batch size it is one more finite-sample
     tilt.  ``diagnose`` reports both drifts so the claim is checked rather than asserted:
     per-block drift must be ~0, joint drift will not be.

``group_ids`` restricts assignments to batch members sharing an observation label (in
practice ``task_uid``).  Cross-task matching reorganizes structure the conditioning already
explains -- pure bias with no useful gain -- so restricting it should improve the
gain/bias ratio for free.  It needs B >= 128, or task-grouped sampling, to leave enough
same-task members per batch.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from oat.symmetry.coupling import SO2OrbitCoupling, _hungarian_assignment
from oat.symmetry.so2_chunk import (
    SO2ChunkSpec,
    chunk_heading,
    circular_moments,
    wrap_angle,
)

_VECTOR_COSTS = ("group_ot", "euclidean", "angle")
_SCALAR_COSTS = ("euclidean",)


def coupling_blocks(spec: SO2ChunkSpec,
                    skip_zero_weight: bool = False) -> List[Tuple[str, Tuple[int, ...], bool]]:
    """``(name, dims, is_vector)`` for every independently-permutable block.

    ``block_weights`` is deliberately ignored by default.  Those weights exist to arbitrate a
    *shared* assignment -- ``[1.0, 0.0]`` means "do not let the near-inert (wx,wy) block vote
    on the joint heading".  Blockwise there is no joint vote to protect, so every block gets
    its own assignment on its own merits.  ``skip_zero_weight=True`` restores the old
    exclusion if a matched ablation is wanted.
    """
    out: List[Tuple[str, Tuple[int, ...], bool]] = []
    for b, (i, j) in enumerate(spec.vector_blocks):
        if skip_zero_weight and spec.weights[b] == 0.0:
            continue
        out.append((f"vec{b}", (i, j), True))
    for d in spec.scalar_dims:
        out.append((f"scalar{d}", (d,), False))
    return out


class BlockwiseSO2Coupling(SO2OrbitCoupling):
    """``mode='perm_block'``: one independent within-batch permutation per irrep block.

    Args:
        spec: irrep layout.
        vector_cost: assignment cost on frequency-1 blocks -- ``group_ot`` (phase-invariant
            modulus, symmetry-aware), ``euclidean`` (real part, symmetry-blind control) or
            ``angle`` (circular distance between block headings).
        blocks: subset of block names to permute, e.g. ``['scalar6']`` for a gripper-only
            ablation.  ``None`` permutes all of them.
        skip_zero_weight: see ``coupling_blocks``.
        group_restrict: if True, ``forward`` requires ``group_ids`` and solves one assignment
            per group.
    """

    def __init__(
        self,
        spec: SO2ChunkSpec,
        vector_cost: str = "group_ot",
        blocks: Optional[Sequence[str]] = None,
        skip_zero_weight: bool = False,
        group_restrict: bool = False,
        coupling_prob: float = 1.0,
        min_heading_confidence: float = 0.0,
        collect_diagnostics: bool = True,
        **unused,
    ) -> None:
        super().__init__(
            spec=spec, mode="perm", cost="euclidean",
            coupling_prob=coupling_prob,
            min_heading_confidence=min_heading_confidence,
            collect_diagnostics=collect_diagnostics,
        )
        if vector_cost not in _VECTOR_COSTS:
            raise ValueError(f"vector_cost must be one of {_VECTOR_COSTS}, got {vector_cost}")
        self.mode = "perm_block"
        self.vector_cost = vector_cost
        self.group_restrict = bool(group_restrict)
        self.skip_zero_weight = bool(skip_zero_weight)
        all_blocks = coupling_blocks(spec, skip_zero_weight)
        names = [b[0] for b in all_blocks]
        if blocks is not None:
            wanted = list(blocks)
            bad = [b for b in wanted if b not in names]
            if bad:
                raise ValueError(f"unknown block(s) {bad}; available: {names}")
            all_blocks = [b for b in all_blocks if b[0] in wanted]
        self.blocks = all_blocks
        self.block_names = [b[0] for b in self.blocks]

    def extra_repr(self) -> str:
        return (f"mode=perm_block, vector_cost={self.vector_cost}, "
                f"blocks={self.block_names}, group_restrict={self.group_restrict}, "
                f"coupling_prob={self.coupling_prob}")

    # -- costs -------------------------------------------------------------------------

    def _block_gain(self, zb: torch.Tensor, xb: torch.Tensor, is_vector: bool) -> torch.Tensor:
        """(B, B) gain to MAXIMISE, computed on this block's coordinates alone.

        ``gain[i, j]`` scores source ``i`` against target ``j``.
        """
        if is_vector:
            Z = torch.complex(zb[..., 0], zb[..., 1])            # (B, H)
            X = torch.complex(xb[..., 0], xb[..., 1])
            if self.vector_cost == "group_ot":
                return torch.abs(Z.conj() @ X.transpose(0, 1))
            if self.vector_cost == "euclidean":
                return torch.real(Z.conj() @ X.transpose(0, 1))
            tz = torch.angle(Z.sum(-1))
            tx = torch.angle(X.sum(-1))
            return -torch.abs(wrap_angle(tz[:, None] - tx[None, :]))
        S = zb.reshape(zb.shape[0], -1)                          # (B, H)
        T = xb.reshape(xb.shape[0], -1)
        return S @ T.transpose(0, 1)

    def _solve(self, zb: torch.Tensor, xb: torch.Tensor, is_vector: bool,
               rows: torch.Tensor) -> torch.Tensor:
        """Permutation of ``rows`` (indices into the batch) for one block."""
        if rows.numel() < 2:
            return rows
        gain = self._block_gain(zb[rows], xb[rows], is_vector)
        perm = _hungarian_assignment(gain)
        return rows[perm]

    # -- forward -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        z0: torch.Tensor,
        x1: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if z0.shape != x1.shape:
            raise ValueError(f"shape mismatch: z0 {tuple(z0.shape)} vs x1 {tuple(x1.shape)}")
        B = z0.shape[0]
        z_in = z0
        if B < 2 or not self.blocks:
            return z0, self._diagnose_block(z_in, z0, x1,
                                            torch.zeros(B, dtype=torch.bool, device=z0.device))

        active = self._active_mask(z0, x1)
        if self.group_restrict and group_ids is None:
            raise ValueError(
                "group_restrict=True but no group_ids were supplied. The policy must pass an "
                "observation label (task_uid) through to the coupling."
            )

        idx = torch.nonzero(active, as_tuple=False).flatten()
        if idx.numel() < 2:
            return z0, self._diagnose_block(z_in, z0, x1, active)

        if self.group_restrict:
            g = group_ids.reshape(-1)[idx]
            groups = [idx[g == v] for v in torch.unique(g)]
        else:
            groups = [idx]

        out = z0.clone()
        sizes = []
        for name, dims, is_vector in self.blocks:
            dl = list(dims)
            zb, xb = z0[..., dl], x1[..., dl]
            for rows in groups:
                if rows.numel() < 2:
                    continue
                src = self._solve(zb, xb, is_vector, rows)
                sub = out[rows]                       # (n, H, A) -- advanced indexing copies
                sub[:, :, dl] = z0[src][:, :, dl]
                out[rows] = sub
                sizes.append(int(rows.numel()))

        diag = self._diagnose_block(z_in, out, x1, active)
        diag["coupling/n_groups"] = float(len(groups))
        diag["coupling/mean_group_size"] = float(sum(sizes) / max(len(sizes), 1))
        return out, diag

    # -- diagnostics -------------------------------------------------------------------

    def _diagnose_block(self, z_in, z_out, x1, active) -> Dict[str, float]:
        d = super()._diagnose(z_in, z_out, x1, active)
        if not self.collect_diagnostics:
            return d
        # Per-block drift must be exactly 0 -- that is the marginal-exactness claim. The
        # joint drift will NOT be 0, because blockwise permutation manufactures block
        # combinations that were not in the batch; see the module docstring.
        worst = 0.0
        for name, dims, _ in self.blocks:
            dl = list(dims)
            a = z_in[..., dl].reshape(z_in.shape[0], -1).norm(dim=-1).sort().values
            b = z_out[..., dl].reshape(z_out.shape[0], -1).norm(dim=-1).sort().values
            worst = max(worst, float((a - b).abs().max()))
        d["coupling/blockwise_marginal_drift"] = worst
        return d
