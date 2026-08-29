"""
Construction B: flow policy trained with an irrep-blockwise independent batch assignment.

One independent within-batch permutation per SO(2) block instead of a single 112-D matching.
The mechanism, the marginal argument and the honest caveats all live in
``oat.symmetry.coupling_blockwise``; this module is only the wiring.

WHAT GATE M2 MEASURED, AND WHAT IT CHANGED ABOUT THE DESIGN
-----------------------------------------------------------
On real LIBERO-10 chunks at B=32, ``block_weights=[1,0]``, against an iid source:

    coupling                       path len    Var[u | x_t] @ t=0.05
    joint perm (euclidean)            5.65%                   14.04%
    blockwise, vector_cost=group_ot   3.56%                   13.23%
    blockwise, vector_cost=euclidean 11.76%                   29.49%

So the viable form of B is **euclidean per-block cost**, and it roughly doubles the joint
assignment on both quantities.  ``group_ot`` -- the phase-invariant modulus, the
"symmetry-aware" cost -- is the *wrong* choice here and the reason is exact: it scores the
transport achievable **after an optimal rotation**, but a permutation applies no rotation, so
it optimizes a bound it cannot realize.  On the vector blocks it buys 0.1-0.8% of path
length, i.e. nothing.  That is a clean mechanistic negative and it belongs in the write-up:
the symmetry earns its keep in B by choosing *which product structure to factor over*, not by
supplying the cost.

``group_restrict`` (match only within ``task_uid``) was predicted to improve the gain/bias
ratio.  Measured, it *lowers* the gain at both batch sizes (11.8% -> 3.0% at B=32) because it
shrinks the matching pool to ~3 same-task members per batch.  Off by default; it needs
task-grouped sampling to be testable at all.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from oat.policy.flow_policy_orbit import OrbitFlowPolicy
from oat.symmetry.coupling_blockwise import BlockwiseSO2Coupling


class BlockwiseCouplingFlowPolicy(OrbitFlowPolicy):
    """``OrbitFlowPolicy`` with ``mode='perm_block'``.

    Args:
        blockwise: dict forwarded to ``BlockwiseSO2Coupling`` (``vector_cost``, ``blocks``,
            ``group_restrict``, ``skip_zero_weight``, ``coupling_prob``,
            ``min_heading_confidence``).
        group_key: observation key used as the assignment group label when
            ``blockwise.group_restrict`` is on.  Must be an integer-valued state port.
    """

    def __init__(
        self,
        *args,
        blockwise: Optional[Dict] = None,
        group_key: str = "task_uid",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.coupling.mode != "iid":
            raise ValueError(
                f"BlockwiseCouplingFlowPolicy replaces the coupling entirely; set "
                f"policy.coupling.mode=iid so the two cannot both fire. Got "
                f"'{self.coupling.mode}'."
            )
        cfg = dict(blockwise or {})
        cfg.pop("_target_", None)
        self.blockwise = BlockwiseSO2Coupling(spec=self.action_spec, **cfg)
        self.group_key = str(group_key)
        print(f"  blockwise : {self.blockwise.extra_repr()}\n")

    def get_policy_name(self) -> str:
        return "blockwise_" + super().get_policy_name()

    def _group_ids(self, batch) -> Optional[torch.Tensor]:
        if not self.blockwise.group_restrict:
            return None
        obs = batch["obs"]
        if self.group_key not in obs:
            raise KeyError(
                f"blockwise.group_restrict=True needs obs['{self.group_key}']; the batch has "
                f"{sorted(obs)}."
            )
        g = obs[self.group_key]
        # (B, To, 1) state port -> one label per sample, read at the current step
        return g[:, -1].reshape(g.shape[0], -1)[:, 0].long()

    def forward(self, batch) -> torch.Tensor:
        """The parent's loss with the blockwise assignment in place of the joint one."""
        x1 = self.normalizer["action"].normalize(batch["action"])        # (B, H, A)
        B = x1.shape[0]
        device = x1.device

        cond = self.obs_encoder(batch["obs"])                             # (B, To, d)

        z0 = self.prior_noise_scale * torch.randn_like(x1)
        z0, diag = self.blockwise(z0, x1, group_ids=self._group_ids(batch))
        self.last_coupling_diagnostics = diag

        t = torch.rand(B, device=device, dtype=x1.dtype)
        t_b = t[:, None, None]
        xt = (1.0 - t_b) * z0 + t_b * x1
        v_target = x1 - z0

        v_pred = self.model(xt, self._scale_t(t), cond)
        return F.mse_loss(v_pred, v_target)

    # predict_action is inherited verbatim: inference draws z ~ N(0, I), exactly as the
    # baseline does. That is the point of a permutation coupling -- it never modifies a
    # source sample, so the population source law is untouched and the sampler need not
    # change. The residual finite-batch tilt is the honest caveat, documented in
    # oat/symmetry/coupling_blockwise.py.
