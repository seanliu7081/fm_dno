"""
SO(2)-about-gravity geometry for robot action chunks.

The whole design rests on one observation: a LIBERO / robosuite ``OSC_POSE`` action

    a = [dx, dy, dz, wx, wy, wz, grip] in R^7

decomposes under a rotation R_phi about the world z axis into

    (dx, dy)     -> frequency-1 vector  (planar translation delta)
    (wx, wy)     -> frequency-1 vector  (planar part of the axis-angle rotation delta)
    dz, wz, grip -> frequency-0 scalars (invariant)

This is exactly the [s, v_x, v_y] structure of the double-ring toy problem, with two
vector blocks and three scalar blocks instead of one and one.  A whole action chunk
x in R^{H x 7} carries the group action

    (rho(phi) x)_{h} = [R_phi (dx,dy)_h, dz_h, R_phi (wx,wy)_h, wz_h, grip_h]

with a *single, shared* phi across the horizon -- the group acts on the chunk, not on
individual timesteps.  (The repo's ``SO3ActionChunkAug`` already follows that
convention: one random rotation per chunk.)

Everything below is written in a "complex view": each 2-D vector block at each
timestep becomes one complex number, so that rho(phi) is simply multiplication by
e^{i phi}.  That makes the two operations we need one-liners:

    heading of a chunk      theta(x)  = arg( sum_k x_k )
    optimal alignment angle phi*(z,x) = arg( sum_k conj(z_k) x_k )

The second one is the 2-D closed form of the Kabsch step in Klein et al.,
"Equivariant flow matching" (NeurIPS 2023, arXiv:2306.15030).

CAVEAT ON THE CONTROLLER FRAME
------------------------------
This decomposition assumes the policy's action deltas are expressed in the world /
robot-base frame (robosuite ``OSC_POSE`` default).  If the controller consumes
end-effector-frame deltas, a world rotation acts trivially on the action and the
symmetry statement changes completely.  Check ``controller_configs`` before using.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


# --------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SO2ChunkSpec:
    """Irrep layout of one action vector under SO(2) about gravity.

    Args:
        action_dim: width of the action vector.
        vector_blocks: index pairs ``(i, j)`` that form a frequency-1 (2-D) vector.
        scalar_dims: indices that are invariant (frequency 0).
        block_weights: relative weight of each vector block inside the coupling cost.
            ``None`` means uniform.  After the SO(2) block normalizer every vector block
            has roughly unit per-coordinate RMS, so uniform is the right default; use
            ``(1.0, 0.0)`` to build a translation-only coupling.
    """

    action_dim: int = 7
    vector_blocks: Tuple[Tuple[int, int], ...] = ((0, 1), (3, 4))
    scalar_dims: Tuple[int, ...] = (2, 5, 6)
    block_weights: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        # hydra / omegaconf hands us ListConfig; coerce so the dataclass stays hashable
        object.__setattr__(self, "action_dim", int(self.action_dim))
        object.__setattr__(
            self, "vector_blocks", tuple(tuple(int(i) for i in b) for b in self.vector_blocks)
        )
        object.__setattr__(self, "scalar_dims", tuple(int(d) for d in self.scalar_dims))
        if self.block_weights is not None:
            object.__setattr__(
                self, "block_weights", tuple(float(w) for w in self.block_weights)
            )

        seen = set()
        for blk in self.vector_blocks:
            if len(blk) != 2:
                raise ValueError(f"vector block must be a pair, got {blk}")
            for idx in blk:
                if not 0 <= idx < self.action_dim:
                    raise ValueError(f"index {idx} out of range for action_dim={self.action_dim}")
                if idx in seen:
                    raise ValueError(f"index {idx} appears in more than one block")
                seen.add(idx)
        for idx in self.scalar_dims:
            if not 0 <= idx < self.action_dim:
                raise ValueError(f"scalar index {idx} out of range")
            if idx in seen:
                raise ValueError(f"index {idx} appears in more than one block")
            seen.add(idx)
        if len(seen) != self.action_dim:
            missing = sorted(set(range(self.action_dim)) - seen)
            raise ValueError(
                f"every action dimension must be assigned to a block; missing {missing}"
            )
        if self.block_weights is not None:
            if len(self.block_weights) != len(self.vector_blocks):
                raise ValueError("block_weights must have one entry per vector block")
            if any(w < 0 for w in self.block_weights):
                raise ValueError("block_weights must be non-negative")
            if sum(self.block_weights) <= 0:
                raise ValueError("block_weights must not be all zero")

    # -- convenience -----------------------------------------------------------------

    @property
    def n_vector_blocks(self) -> int:
        return len(self.vector_blocks)

    @property
    def n_scalar_dims(self) -> int:
        return len(self.scalar_dims)

    @property
    def weights(self) -> Tuple[float, ...]:
        if self.block_weights is None:
            return tuple(1.0 for _ in self.vector_blocks)
        return self.block_weights

    @property
    def vector_index(self) -> Tuple[int, ...]:
        """Flat list of every dimension belonging to a vector block."""
        out: list = []
        for i, j in self.vector_blocks:
            out.extend((i, j))
        return tuple(out)

    @staticmethod
    def libero_osc_pose() -> "SO2ChunkSpec":
        """The 7-D delta-EEF action used by LIBERO / MimicGen / robomimic."""
        return SO2ChunkSpec(
            action_dim=7, vector_blocks=((0, 1), (3, 4)), scalar_dims=(2, 5, 6)
        )

    @staticmethod
    def translation_only() -> "SO2ChunkSpec":
        """Ablation: couple on the planar translation block alone."""
        return SO2ChunkSpec(
            action_dim=7,
            vector_blocks=((0, 1), (3, 4)),
            scalar_dims=(2, 5, 6),
            block_weights=(1.0, 0.0),
        )


# --------------------------------------------------------------------------------------
# Complex view
# --------------------------------------------------------------------------------------


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def as_complex(x: torch.Tensor, spec: SO2ChunkSpec) -> torch.Tensor:
    """(B, H, A) real -> (B, H * n_vector_blocks) complex, weighted by sqrt(block_weight).

    ``rho(phi)`` acts on the result as multiplication by ``exp(i phi)``, and the squared
    modulus equals the block-weighted squared norm of the vector part.
    """
    if x.ndim != 3:
        raise ValueError(f"expected (B, H, A), got {tuple(x.shape)}")
    if x.shape[-1] != spec.action_dim:
        raise ValueError(f"expected action_dim={spec.action_dim}, got {x.shape[-1]}")
    xw = x.to(_work_dtype(x.dtype))
    parts = []
    for (i, j), w in zip(spec.vector_blocks, spec.weights):
        if w == 0.0:
            continue
        scale = math.sqrt(w)
        parts.append(torch.complex(xw[..., i] * scale, xw[..., j] * scale))
    if not parts:
        raise ValueError("spec has no active vector block (all block_weights are zero)")
    z = torch.stack(parts, dim=-1)                      # (B, H, V)
    return z.reshape(z.shape[0], -1)                    # (B, H*V)


def scalar_part(x: torch.Tensor, spec: SO2ChunkSpec) -> torch.Tensor:
    """(B, H, A) -> (B, H * n_scalar_dims), the SO(2)-invariant channels."""
    if spec.n_scalar_dims == 0:
        return x.new_zeros(x.shape[0], 0)
    s = x[..., list(spec.scalar_dims)]
    return s.reshape(s.shape[0], -1)


def rotate_chunk(x: torch.Tensor, phi: torch.Tensor, spec: SO2ChunkSpec) -> torch.Tensor:
    """Apply rho(phi) to an action chunk.  Differentiable, out-of-place.

    Args:
        x: (B, H, A)
        phi: (B,) or scalar tensor -- one angle per chunk.
    """
    if x.ndim != 3:
        raise ValueError(f"expected (B, H, A), got {tuple(x.shape)}")
    phi = torch.as_tensor(phi, device=x.device, dtype=_work_dtype(x.dtype))
    if phi.ndim == 0:
        phi = phi.expand(x.shape[0])
    if phi.shape != (x.shape[0],):
        raise ValueError(f"phi must be (B,) = ({x.shape[0]},), got {tuple(phi.shape)}")
    c = torch.cos(phi)[:, None].to(x.dtype)
    s = torch.sin(phi)[:, None].to(x.dtype)

    cols = list(x.unbind(-1))
    for i, j in spec.vector_blocks:
        xi, xj = cols[i], cols[j]
        cols[i] = c * xi - s * xj
        cols[j] = s * xi + c * xj
    return torch.stack(cols, dim=-1)


# --------------------------------------------------------------------------------------
# Headings and alignment
# --------------------------------------------------------------------------------------


def chunk_heading(x: torch.Tensor, spec: SO2ChunkSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equivariant heading of a chunk: theta(rho(phi) x) = theta(x) + phi.

    Returns ``(theta, confidence)`` where confidence is ``|sum_k x_k| / sqrt(m)``.
    Confidence collapses to ~0 for chunks whose planar deltas cancel over the horizon
    (a hold / reversal), in which case the heading carries no information.  Used for
    diagnostics and for gating, never inside the coupling cost -- the cross-correlation
    below is the numerically better-behaved object.
    """
    z = as_complex(x, spec)
    c = z.sum(dim=-1)
    m = z.shape[-1]
    return torch.angle(c), torch.abs(c) / math.sqrt(m)


def align_phase(z: torch.Tensor, x: torch.Tensor, spec: SO2ChunkSpec) -> torch.Tensor:
    """Angle phi* minimising ||rho(phi) z - x||^2 for each paired (z_i, x_i).

    Closed form (2-D Kabsch):  phi* = arg( sum_k conj(z_k) x_k ).
    For a single vector block this reduces exactly to phi* = theta(x) - theta(z),
    i.e. the toy problem's orbit alignment.
    """
    Z = as_complex(z, spec)
    X = as_complex(x, spec)
    return torch.angle((Z.conj() * X).sum(dim=-1))


def pairwise_alignment(
    z: torch.Tensor, x: torch.Tensor, spec: SO2ChunkSpec
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-pairs alignment between a source batch and a target batch.

    Returns ``(gain, phi)`` where

        gain[i, j] = | sum_k conj(z_i,k) x_j,k |   (group-aware, phase-invariant)
        phi[i, j]  = arg( sum_k conj(z_i,k) x_j,k )

    Note ``min_phi ||rho(phi) z_i - x_j||^2 = ||z_i||^2 + ||x_j||^2 - 2 gain[i, j]``.
    The two norm terms are constant along a row / column respectively, so they do not
    change the optimal assignment: **the group-aware assignment depends on the moduli
    |S_ij| alone**, while a symmetry-blind Euclidean assignment uses Re(S_ij).  That
    single difference -- modulus versus real part -- is precisely what "symmetry-aware
    coupling" means here.
    """
    Z = as_complex(z, spec)                     # (Bz, m)
    X = as_complex(x, spec)                     # (Bx, m)
    S = Z.conj() @ X.transpose(0, 1)            # (Bz, Bx)
    return torch.abs(S), torch.angle(S)


def pairwise_euclidean_gain(
    z: torch.Tensor, x: torch.Tensor, spec: SO2ChunkSpec
) -> torch.Tensor:
    """Symmetry-blind gain Re(S_ij) on the vector part (control condition)."""
    Z = as_complex(z, spec)
    X = as_complex(x, spec)
    return torch.real(Z.conj() @ X.transpose(0, 1))


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap to (-pi, pi]."""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def circular_moments(theta: torch.Tensor, orders: Sequence[int] = (1, 2, 4, 8)) -> dict:
    """|E[exp(i k theta)]| for each k.  All ~0 iff theta is uniform on the circle.

    This is the test that tells you whether a coupling has silently deformed the
    angular marginal of the source.
    """
    out = {}
    for k in orders:
        c = torch.exp(1j * k * theta.to(_work_dtype(theta.dtype))).mean()
        out[f"R{k}"] = float(torch.abs(c))
    return out


def vector_energy_fraction(x: torch.Tensor, spec: SO2ChunkSpec) -> dict:
    """Per-block share of the total chunk energy -- a sanity check before coupling.

    If, say, the (wx, wy) block holds 1% of the energy after normalization, the coupling
    is effectively translation-only and you should say so rather than pretend otherwise.
    """
    total = (x ** 2).sum().clamp_min(1e-12)
    out = {}
    for b, (i, j) in enumerate(spec.vector_blocks):
        e = (x[..., i] ** 2 + x[..., j] ** 2).sum()
        out[f"vec{b}_energy_frac"] = float(e / total)
    for d in spec.scalar_dims:
        out[f"scalar{d}_energy_frac"] = float((x[..., d] ** 2).sum() / total)
    return out
