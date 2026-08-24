"""
A rotation-compatible Adam for noise-space optimization.

The DNO literature review flags this exactly right: "ordinary coordinate-wise Adam can
break continuous rotation equivariance."  The reason is the preconditioner.  Adam's step
is ``lr * m / (sqrt(v) + eps)`` with ``v`` accumulated per coordinate, i.e. a *diagonal*
matrix ``D``.  A rotation ``R`` acting inside a 2-D vector block commutes with ``D`` only
when ``D`` is a multiple of the identity on that block.  With independent ``v`` for
``z_x`` and ``z_y`` it is not, so two runs started from ``z`` and from ``rho(phi) z`` --
which see gradients related by the same rotation -- drift apart.

The fix is one line: share the second moment across the two coordinates of every
frequency-1 block.  Then ``D`` is a scalar on each block, the update commutes with
``rho(phi)``, and the whole optimization is equivariant:

    z_k(rho(phi) z_0)  =  rho(phi) z_k(z_0)   for every iterate k.

The first moment needs no treatment -- it is a linear function of the gradients, so it
rotates covariantly on its own.  Invariant (frequency-0) channels keep ordinary
per-coordinate adaptivity, which is exactly right: nothing constrains them.

Functional by design (state lives in the object, parameters are passed in and returned),
because DNO re-initialises the optimizer every control cycle and threading an
``nn.Parameter`` through that is more trouble than it is worth.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from oat.symmetry.so2_chunk import SO2ChunkSpec


class IrrepAdam:
    """Adam whose preconditioner is constant on each SO(2) vector block.

    Args:
        spec: irrep layout of the action vector.
        lr: step size.
        betas: Adam moment decay rates.  The default ``(0.9, 0.99)`` is tuned for the
            few-dozen-step budget of test-time optimization rather than training.
        eps: denominator floor.
        equivariant: set ``False`` to recover ordinary per-coordinate Adam.  Keep the
            switch -- "plain Adam vs IrrepAdam" is the ablation that shows the
            preconditioner matters, and the doc claims it does.
    """

    def __init__(
        self,
        spec: SO2ChunkSpec,
        lr: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        equivariant: bool = True,
    ) -> None:
        self.spec = spec
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.equivariant = bool(equivariant)
        self.reset()

    def reset(self) -> None:
        self._m: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None
        self._t = 0

    @torch.no_grad()
    def step(self, z: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """One update.  ``z`` and ``grad`` are (B, H, A); returns the new ``z``."""
        if self._m is None:
            self._m = torch.zeros_like(z)
            self._v = torch.zeros_like(z)
        self._t += 1

        self._m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        self._v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

        v = self._v
        if self.equivariant:
            v = v.clone()
            for i, j in self.spec.vector_blocks:
                shared = 0.5 * (v[..., i] + v[..., j])
                v[..., i] = shared
                v[..., j] = shared

        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = v / (1 - self.beta2 ** self._t)
        return z - self.lr * m_hat / (v_hat.sqrt() + self.eps)


class ScalarSGD:
    """Plain scalar-step-size SGD with momentum -- trivially equivariant.

    The DNO review's other recommendation.  Slower than ``IrrepAdam`` but has no
    preconditioner to get wrong, so it is the reference point when debugging an
    equivariance failure in the optimizer.
    """

    def __init__(self, lr: float = 0.05, momentum: float = 0.9) -> None:
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.reset()

    def reset(self) -> None:
        self._buf: Optional[torch.Tensor] = None

    @torch.no_grad()
    def step(self, z: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if self._buf is None:
            self._buf = torch.zeros_like(z)
        self._buf.mul_(self.momentum).add_(grad)
        return z - self.lr * self._buf
