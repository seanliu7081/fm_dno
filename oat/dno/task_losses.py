"""
Differentiable task objectives for test-time steering of an action-chunk policy.

All losses take **unnormalized** actions ``a`` of shape (B, H, 7) in the policy's native
delta-EEF space and return a per-sample loss of shape (B,).  Unnormalizing before the
loss matters: metres and radians are the units the constraints are actually written in,
and the SO(2) block normalizer's scales are not the same for the translation and rotation
blocks.

CONTROLLER GAINS
----------------
Turning delta actions into a predicted end-effector path needs the controller's output
scaling.  robosuite's ``OSC_POSE`` default maps the [-1, 1] action box to
+/-0.05 m and +/-0.5 rad per environment step; LIBERO inherits that.  ``trans_gain`` and
``rot_gain`` below default to those numbers, but **check the ``controller_configs`` of
the env you are actually running** -- a wrong gain silently rescales every geometric
constraint and is a very quiet way to make an ablation meaningless.

The forward rollout used here is deliberately naive (``p_{h+1} = p_h + gain * a_h``): it
ignores controller lag, contact, and the fact that OSC only tracks the target
approximately.  That is fine for smoothness, workspace, and clearance shaping, and it is
not fine as a substitute for a dynamics model -- see the design doc's risk section.

TIERS
-----
  Tier 0  no extra learning, no privileged state: smoothness, seam continuity, per-step
          limits, workspace box, table clearance.  This is the tier to run first, because
          it isolates *steerability* -- whether the noise space can be optimized at all
          -- from the question of whether the objective is a good proxy for success.
  Tier 1  privileged: obstacle SDFs from simulator object poses.  Analysis only; report
          it as an oracle.
  Tier 2  learned: a success classifier or Q function on (obs, chunk).  This is the DSRL
          comparison (Wagenmaker et al., CoRL 2025, arXiv:2506.15799) and the point at
          which reward hacking becomes the dominant risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch

TRANS_GAIN_DEFAULT = 0.05   # metres per unit action, robosuite OSC_POSE default
ROT_GAIN_DEFAULT = 0.5      # radians per unit action


# --------------------------------------------------------------------------------------
# Kinematic rollout
# --------------------------------------------------------------------------------------


def rollout_positions(
    a: torch.Tensor,
    eef_pos: torch.Tensor,
    trans_gain: float = TRANS_GAIN_DEFAULT,
) -> torch.Tensor:
    """(B, H, 7), (B, 3) -> (B, H, 3) predicted end-effector positions."""
    return eef_pos[:, None, :] + trans_gain * torch.cumsum(a[..., :3], dim=1)


# --------------------------------------------------------------------------------------
# Tier 0
# --------------------------------------------------------------------------------------


def smoothness(a: torch.Tensor, dims: int = 6) -> torch.Tensor:
    """Second-difference energy of the chunk -- penalises jerk within the chunk."""
    x = a[..., :dims]
    if x.shape[1] < 3:
        return x.new_zeros(x.shape[0])
    d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return (d2 ** 2).sum(dim=(-1, -2))


def seam_continuity(a: torch.Tensor, prev_action: Optional[torch.Tensor], dims: int = 6) -> torch.Tensor:
    """Match the first action of the chunk to the last one actually executed.

    Receding-horizon replanning re-generates a chunk every ``n_action_steps``; without
    this term the seam is where a diffusion or flow policy visibly stutters.
    """
    if prev_action is None:
        return a.new_zeros(a.shape[0])
    return ((a[:, 0, :dims] - prev_action[..., :dims]) ** 2).sum(-1)


def step_limit(
    a: torch.Tensor,
    max_trans: float = 0.9,
    max_rot: float = 0.9,
) -> torch.Tensor:
    """Keep per-step commands inside the controller's linear range (action box is +/-1)."""
    tn = a[..., :3].norm(dim=-1)
    rn = a[..., 3:6].norm(dim=-1)
    return (torch.relu(tn - max_trans) ** 2).sum(-1) + (torch.relu(rn - max_rot) ** 2).sum(-1)


def workspace_box(
    a: torch.Tensor,
    eef_pos: torch.Tensor,
    lo: Sequence[float],
    hi: Sequence[float],
    trans_gain: float = TRANS_GAIN_DEFAULT,
) -> torch.Tensor:
    """Penalise the predicted path leaving an axis-aligned workspace box."""
    p = rollout_positions(a, eef_pos, trans_gain)
    lo_t = torch.as_tensor(lo, device=a.device, dtype=a.dtype)
    hi_t = torch.as_tensor(hi, device=a.device, dtype=a.dtype)
    below = torch.relu(lo_t - p)
    above = torch.relu(p - hi_t)
    return (below ** 2 + above ** 2).sum(dim=(-1, -2))


def table_clearance(
    a: torch.Tensor,
    eef_pos: torch.Tensor,
    z_min: float,
    trans_gain: float = TRANS_GAIN_DEFAULT,
) -> torch.Tensor:
    """Keep the predicted path above a table plane."""
    p = rollout_positions(a, eef_pos, trans_gain)
    return (torch.relu(z_min - p[..., 2]) ** 2).sum(-1)


def descent_limit(
    a: torch.Tensor,
    eef_pos: torch.Tensor,
    max_descent: float = 0.15,
    trans_gain: float = TRANS_GAIN_DEFAULT,
) -> torch.Tensor:
    """Penalise the chunk driving the end-effector more than ``max_descent`` below its
    CURRENT height. A *relative* clearance floor.

    WHY THIS REPLACES ``table_clearance`` ON LIBERO-10
    --------------------------------------------------
    ``table_clearance`` needs one absolute plane ``z_min``. Measured on
    ``libero10_N500.zarr`` the demonstrated end-effector height is **bimodal**, with an
    empty gap at 0.815-0.889 and the ten tasks splitting five/five:

        low  group  eef z in [0.446, 0.781]      (all five LIVING_ROOM tasks)
        high group  eef z in [0.910, 1.332]      (four KITCHEN + STUDY)

    So no scalar is both safe and active. The plan's placeholder 0.82 lies *above* the
    maximum height of every low-group task, i.e. it would penalise **every demonstrated
    step** of half the benchmark. Pushing it below the global 1st percentile (0.448) makes
    it safe but identically zero for the entire high group -- an inert term carrying the
    largest weight in the Tier-0 objective.

    Anchoring to the current height sidesteps the choice: the penalty is by construction
    identical at eef z = 0.50, 0.95 and 1.20, so one setting is simultaneously safe and
    active on both scene families. It encodes "do not dive" rather than "stay above a
    plane", which is the constraint actually wanted.

    Note it is SO(2)-invariant (it reads only the z channel), unlike ``workspace_box``.
    """
    p = rollout_positions(a, eef_pos, trans_gain)                 # (B, H, 3)
    floor = eef_pos[:, 2].unsqueeze(-1) - float(max_descent)      # (B, 1)
    return (torch.relu(floor - p[..., 2]) ** 2).sum(-1)


def gripper_decisiveness(a: torch.Tensor, dim: int = 6) -> torch.Tensor:
    """Push the gripper channel toward a committed +/-1 rather than an ambiguous middle.

    Off by default: it is a genuine help when a policy dithers at a grasp, and a genuine
    way to force a premature close when it does not.  Turn it on knowingly.
    """
    g = a[..., dim].clamp(-1.5, 1.5)
    return ((1.0 - g ** 2) ** 2).sum(-1)


# --------------------------------------------------------------------------------------
# Tier 1 (privileged)
# --------------------------------------------------------------------------------------


def sphere_obstacles(
    a: torch.Tensor,
    eef_pos: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    margin: float = 0.02,
    trans_gain: float = TRANS_GAIN_DEFAULT,
) -> torch.Tensor:
    """Hinge SDF penalty against sphere-approximated obstacles.  (B,3+) centers, (K,) radii.

    Uses simulator object poses, so it is privileged information: report any number that
    depends on it as an oracle ablation, never as a deployable result.
    """
    p = rollout_positions(a, eef_pos, trans_gain)                     # (B, H, 3)
    d = (p[:, :, None, :] - centers[None, None, :, :]).norm(dim=-1)   # (B, H, K)
    return (torch.relu(radii[None, None, :] + margin - d) ** 2).sum(dim=(-1, -2))


def goal_reaching(
    a: torch.Tensor,
    eef_pos: torch.Tensor,
    goal: torch.Tensor,
    trans_gain: float = TRANS_GAIN_DEFAULT,
    terminal_only: bool = True,
) -> torch.Tensor:
    """Distance of the predicted path (or its endpoint) to a target position."""
    p = rollout_positions(a, eef_pos, trans_gain)
    if terminal_only:
        return ((p[:, -1] - goal) ** 2).sum(-1)
    return ((p - goal[:, None, :]) ** 2).sum(-1).min(dim=1).values


# --------------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------------


@dataclass
class CompositeTaskLoss:
    """Weighted sum of the terms above.

    ``__call__(a, ctx) -> (B,)`` where ``ctx`` carries whatever the enabled terms need
    (``eef_pos``, ``prev_action``, ``centers``, ``radii``, ``goal``).  Per-term values are
    stashed in ``last_terms`` so a rollout can log which constraint is actually binding
    -- without that, a DNO ablation is uninterpretable.

    A note on invariance: for the whole optimization to be SO(2)-equivariant, every
    enabled term must transform correctly.  ``smoothness``, ``step_limit`` and
    ``gripper_decisiveness`` are invariant.  ``workspace_box`` is **not** -- an
    axis-aligned box has only C4 symmetry -- and ``table_clearance``/``sphere_obstacles``
    are invariant only if the scene rotates with the action.  That is fine and expected:
    a real task objective usually breaks the symmetry, which is precisely why the
    *policy* is asked for relaxed rather than exact equivariance.  Just do not then claim
    the DNO stage is equivariant.
    """

    weights: Dict[str, float] = field(default_factory=dict)
    trans_gain: float = TRANS_GAIN_DEFAULT
    workspace_lo: Optional[Sequence[float]] = None
    workspace_hi: Optional[Sequence[float]] = None
    table_z: Optional[float] = None
    max_descent: float = 0.15
    obstacle_margin: float = 0.02
    last_terms: Dict[str, float] = field(default_factory=dict)

    def __call__(self, a: torch.Tensor, ctx: Optional[Dict] = None) -> torch.Tensor:
        ctx = ctx or {}
        total = a.new_zeros(a.shape[0])
        terms: Dict[str, torch.Tensor] = {}

        w = self.weights
        if w.get("smoothness"):
            terms["smoothness"] = smoothness(a)
        if w.get("seam"):
            terms["seam"] = seam_continuity(a, ctx.get("prev_action"))
        if w.get("step_limit"):
            terms["step_limit"] = step_limit(a)
        if w.get("gripper"):
            terms["gripper"] = gripper_decisiveness(a)
        if w.get("workspace") and self.workspace_lo is not None:
            terms["workspace"] = workspace_box(
                a, ctx["eef_pos"], self.workspace_lo, self.workspace_hi, self.trans_gain
            )
        if w.get("table") and self.table_z is not None:
            terms["table"] = table_clearance(a, ctx["eef_pos"], self.table_z, self.trans_gain)
        if w.get("descent") and ctx.get("eef_pos") is not None:
            terms["descent"] = descent_limit(
                a, ctx["eef_pos"], self.max_descent, self.trans_gain
            )
        if w.get("obstacles") and ctx.get("centers") is not None:
            terms["obstacles"] = sphere_obstacles(
                a, ctx["eef_pos"], ctx["centers"], ctx["radii"],
                self.obstacle_margin, self.trans_gain,
            )
        if w.get("goal") and ctx.get("goal") is not None:
            terms["goal"] = goal_reaching(a, ctx["eef_pos"], ctx["goal"], self.trans_gain)

        for name, value in terms.items():
            total = total + w[name] * value
        self.last_terms = {k: float(v.mean()) for k, v in terms.items()}
        return total
