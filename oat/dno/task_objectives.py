"""Task-directed costs for optimizing a frozen action generator's input noise.

The position rollout below is a local OSC command approximation, not a world
model: it predicts neither contact nor object motion and is not task reward.
Actual episode success must still come from the environment.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn


class TaskGeometryObjective(nn.Module):
    """Lower-is-better position/gripper cost on the executed action prefix.

    ``actions`` must be physical policy outputs [B, H, 7], after action
    unnormalization but before robosuite's OSC input scaling. ``goal_pos`` is
    the current stage's end-effector target, not necessarily the final object
    placement position. The observation and stage remain fixed during a DNO
    inner optimization; advance them only after actual environment execution.

    Required context: eef_pos [B,3], goal_pos [B,3], translation_gain scalar,
    [B], or [B,3]. Optional: active scalar/[B], desired_gripper scalar/[B],
    translation_input_min/max with the same shapes as translation_gain.
    LIBERO uses -1 for opening and +1 for closing its Panda gripper.
    """

    def __init__(
        self,
        n_action_steps: int = 8,
        position_weight: float = 1.0,
        gripper_weight: float = 0.001,
        path_weight: float = 0.1,
    ):
        super().__init__()
        if not isinstance(n_action_steps, int) or n_action_steps < 1:
            raise ValueError("n_action_steps must be a positive integer")
        for name, value in (
            ("position_weight", position_weight),
            ("gripper_weight", gripper_weight),
            ("path_weight", path_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.n_action_steps = n_action_steps
        self.position_weight = position_weight
        self.gripper_weight = gripper_weight
        self.path_weight = path_weight

    @staticmethod
    def _scalar(value, actions: Tensor, name: str) -> Tensor:
        value = torch.as_tensor(value, device=actions.device, dtype=actions.dtype)
        batch = actions.shape[0]
        if value.ndim == 0:
            return value.expand(batch)
        if value.shape == (batch, 1):
            return value[:, 0]
        if value.shape != (batch,):
            raise ValueError(f"{name} must be scalar or [B], got {tuple(value.shape)}")
        return value

    @classmethod
    def _translation(cls, value, actions: Tensor, name: str) -> Tensor:
        value = torch.as_tensor(value, device=actions.device, dtype=actions.dtype)
        if value.shape == (actions.shape[0], 3):
            return value[:, None, :]
        return cls._scalar(value, actions, name)[:, None, None]

    def forward(self, actions: Tensor, context: Mapping[str, Tensor]) -> Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError("actions must have shape [B,H,7]")
        if actions.shape[1] < self.n_action_steps:
            raise ValueError("action horizon is shorter than n_action_steps")
        if not actions.is_floating_point():
            raise ValueError("actions must be floating point, unnormalized OSC commands")
        batch = actions.shape[0]
        eef = torch.as_tensor(context["eef_pos"], device=actions.device, dtype=actions.dtype)
        goal = torch.as_tensor(context["goal_pos"], device=actions.device, dtype=actions.dtype)
        if eef.shape != (batch, 3) or goal.shape != (batch, 3):
            raise ValueError("eef_pos and goal_pos must have shape [B,3]")
        gain = self._translation(context["translation_gain"], actions, "translation_gain")
        lower = self._translation(context.get("translation_input_min", -1.0), actions,
                                  "translation_input_min")
        upper = self._translation(context.get("translation_input_max", 1.0), actions,
                                  "translation_input_max")
        prefix = actions[:, :self.n_action_steps]
        command = torch.maximum(torch.minimum(prefix[..., :3], upper), lower)
        positions = eef[:, None, :] + torch.cumsum(command * gain, dim=1)
        squared_distance = (positions - goal[:, None, :]).square().sum(dim=-1)
        cost = self.position_weight * (
            squared_distance[:, -1] + self.path_weight * squared_distance.mean(dim=1)
        )
        if "desired_gripper" in context and self.gripper_weight:
            desired = self._scalar(context["desired_gripper"], actions, "desired_gripper")
            gripper_error = (prefix[..., 6].clamp(-1, 1) - desired[:, None]).square()
            cost = cost + self.gripper_weight * gripper_error.mean(dim=1)
        active = self._scalar(context.get("active", 1.0), actions, "active").bool()
        return torch.where(active, cost, torch.zeros_like(cost))
