"""A small, identity-initialized noise selector distilled from DNO searches.

This module does not implement reinforcement learning. Its output is a bounded
correction to a *particular* source noise, conditioned on the frozen policy's
features and the current geometric subgoal. Checkpoint identity checks prevent
accidentally reusing a selector with a different E/F action generator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn


CONTEXT_DIM = 9
CONTEXT_SCHEMA = "goal_minus_eef_xyz,desired_gripper,phase_onehot_5_v1"
PHASE_NAMES = ("reach", "grasp", "lift", "transport", "release")
FORMAT_VERSION = 2


def build_context_features(
    context: Mapping[str, Any],
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return [B, 9] task features; accept batched tensors or one numpy context.

    ``desired_gripper`` is -1 for open and +1 for closed. Missing gripper and
    phase use zero features (an unspecified phase is not encoded as reach).
    The caller must use the same physical frame/units for goal and end effector.
    """
    dtype = dtype or torch.float32

    def tensor(value):
        return torch.as_tensor(value, device=device, dtype=dtype)

    if "context_features" in context:
        result = tensor(context["context_features"])
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2 or result.shape[-1] != CONTEXT_DIM:
            raise ValueError("context_features must have shape [B, 9] or [9].")
        if batch_size is not None:
            if result.shape[0] == 1:
                result = result.expand(batch_size, -1)
            elif result.shape[0] != batch_size:
                raise ValueError("context_features batch does not match noise batch.")
        return result
    if "goal_pos" not in context or "eef_pos" not in context:
        raise ValueError("Task context requires goal_pos and eef_pos, or context_features.")
    goal, eef = tensor(context["goal_pos"]), tensor(context["eef_pos"])
    if goal.ndim == 1:
        goal = goal.unsqueeze(0)
    if eef.ndim == 1:
        eef = eef.unsqueeze(0)
    if goal.ndim != 2 or eef.ndim != 2 or goal.shape[1] != 3 or eef.shape[1] != 3:
        raise ValueError("goal_pos/eef_pos must have shape [B, 3] or [3].")
    batch_size = batch_size or max(goal.shape[0], eef.shape[0])
    if goal.shape[0] not in (1, batch_size) or eef.shape[0] not in (1, batch_size):
        raise ValueError("Position batch does not match noise batch.")
    displacement = goal.expand(batch_size, -1) - eef.expand(batch_size, -1)

    def scalar_batch(value, name):
        item = tensor(value).reshape(-1)
        if item.numel() == 1:
            item = item.expand(batch_size)
        if item.numel() != batch_size:
            raise ValueError(f"{name} must be scalar or have one entry per batch item.")
        return item

    gripper = scalar_batch(context.get("desired_gripper", 0.0), "desired_gripper")
    phase_features = displacement.new_zeros(batch_size, len(PHASE_NAMES))
    if "phase" in context:
        phase = scalar_batch(context["phase"], "phase")
        if not bool(((phase >= 0) & (phase < len(PHASE_NAMES)) & (phase == phase.floor())).all()):
            raise ValueError("phase must be an integer from 0 through 4.")
        phase_features.scatter_(1, phase.long().unsqueeze(-1), 1.0)
    return torch.cat((displacement, gripper.unsqueeze(-1), phase_features), dim=-1)


class NoiseInitializer(nn.Module):
    """Predict a bounded residual while preserving the random noise input.

    A zero-initialized final layer makes the initial result exactly base_noise.
    The correction's RMS is strictly below max_residual_rms for finite logits.
    This bound is a search constraint, not a guarantee of task improvement.
    """

    def __init__(
        self,
        cond_dim: int,
        horizon: int,
        action_dim: int,
        n_obs_steps: int = 2,
        context_dim: int = CONTEXT_DIM,
        hidden_dims: Sequence[int] = (128, 128),
        max_residual_rms: float = 0.5,
        cond_std_floor: float = 0.1,
        displacement_std_floor: float = 0.05,
        normalized_input_clip: float = 10.0,
    ):
        super().__init__()
        if any(int(value) <= 0 for value in (cond_dim, horizon, action_dim, n_obs_steps)):
            raise ValueError("All condition/noise dimensions must be positive.")
        if context_dim != CONTEXT_DIM:
            raise ValueError(f"This context schema requires context_dim={CONTEXT_DIM}.")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if not 0 < float(max_residual_rms) < float("inf"):
            raise ValueError("max_residual_rms must be finite and positive.")
        for name, value in (("cond_std_floor", cond_std_floor),
                            ("displacement_std_floor", displacement_std_floor),
                            ("normalized_input_clip", normalized_input_clip)):
            if not 0 < float(value) < float("inf"):
                raise ValueError(f"{name} must be finite and positive.")
        self.cond_dim = int(cond_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.n_obs_steps = int(n_obs_steps)
        self.context_dim = int(context_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.max_residual_rms = float(max_residual_rms)
        self.cond_std_floor = float(cond_std_floor)
        self.displacement_std_floor = float(displacement_std_floor)
        self.normalized_input_clip = float(normalized_input_clip)
        input_dim = self.n_obs_steps * self.cond_dim + self.horizon * self.action_dim + self.context_dim
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        layers: list[nn.Module] = []
        previous = input_dim
        for width in self.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        output = nn.Linear(previous, self.horizon * self.action_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def structure(self) -> dict[str, Any]:
        return {
            "cond_dim": self.cond_dim,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "n_obs_steps": self.n_obs_steps,
            "context_dim": self.context_dim,
            "hidden_dims": list(self.hidden_dims),
            "max_residual_rms": self.max_residual_rms,
            "cond_std_floor": self.cond_std_floor,
            "displacement_std_floor": self.displacement_std_floor,
            "normalized_input_clip": self.normalized_input_clip,
        }

    def input_features(self, cond, base_noise, context) -> torch.Tensor:
        batch = base_noise.shape[0]
        if base_noise.ndim != 3 or tuple(base_noise.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError(f"base_noise must have shape [B, {self.horizon}, {self.action_dim}].")
        if tuple(cond.shape) != (batch, self.n_obs_steps, self.cond_dim):
            raise ValueError(f"cond must have shape [B, {self.n_obs_steps}, {self.cond_dim}].")
        task_features = build_context_features(
            context, batch_size=batch, device=base_noise.device, dtype=base_noise.dtype,
        )
        return torch.cat((cond.flatten(1), base_noise.flatten(1), task_features), dim=-1)

    @torch.no_grad()
    def fit_input_normalizer(self, cond, base_noise, context) -> None:
        """Fit condition/displacement statistics only, using training episodes.

        Noise retains its E/F source coordinates. Gripper/phase are already
        bounded semantic features; fitting their variance on one phase could
        otherwise amplify an unseen phase by orders of magnitude.
        """
        features = self.input_features(cond, base_noise, context)
        if features.shape[0] == 0 or not bool(torch.isfinite(features).all()):
            raise ValueError("Normalizer requires nonempty, finite training features.")
        self.input_mean.zero_()
        self.input_std.fill_(1.0)
        cond_end = self.n_obs_steps * self.cond_dim
        context_start = cond_end + self.horizon * self.action_dim
        for columns, std_floor in (
            (slice(0, cond_end), self.cond_std_floor),
            (slice(context_start, context_start + 3), self.displacement_std_floor),
        ):
            self.input_mean[columns].copy_(features[:, columns].mean(0))
            self.input_std[columns].copy_(features[:, columns].std(0, unbiased=False).clamp_min(std_floor))

    def normalized_input_features(self, cond, base_noise, context) -> torch.Tensor:
        """Clip only the fitted condition/displacement blocks; keep noise raw."""
        features = self.input_features(cond, base_noise, context)
        normalized = (features - self.input_mean) / self.input_std
        cond_end = self.n_obs_steps * self.cond_dim
        context_start = cond_end + self.horizon * self.action_dim
        bound = self.normalized_input_clip
        return torch.cat((
            normalized[:, :cond_end].clamp(-bound, bound),
            features[:, cond_end:context_start],
            normalized[:, context_start:context_start + 3].clamp(-bound, bound),
            features[:, context_start + 3:],
        ), dim=-1)

    def forward(self, cond, base_noise, context) -> torch.Tensor:
        features = self.normalized_input_features(cond, base_noise, context)
        raw = self.net(features).reshape_as(base_noise)
        # Smooth radial saturation, with a nonzero denominator at initialization.
        scale = self.max_residual_rms / (
            self.max_residual_rms ** 2 + raw.square().mean(dim=(1, 2), keepdim=True)
        ).sqrt()
        return base_noise + raw * scale


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The base checkpoint includes the frozen reference encoder and its stats."""
    result = dict(identity)
    if result.get("source_policy") not in ("E", "F"):
        raise ValueError("identity.source_policy must be E or F.")
    digest = result.get("base_checkpoint_sha256", "")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("identity.base_checkpoint_sha256 must be a lowercase SHA-256 hex digest.")
    return result


def save_noise_initializer(
    path: str | Path,
    initializer: NoiseInitializer,
    *,
    identity: Mapping[str, Any],
    training: Mapping[str, Any] | None = None,
) -> None:
    """Save architecture, normalizer, identity and optional training metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "method": "dno_teacher_distillation",
        "context_schema": CONTEXT_SCHEMA,
        "structure": initializer.structure(),
        "identity": validate_identity(identity),
        "state_dict": {key: value.detach().cpu() for key, value in initializer.state_dict().items()},
        "training": dict(training or {}),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_noise_initializer(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any],
    map_location: torch.device | str = "cpu",
) -> NoiseInitializer:
    """Load in eval mode, refusing incompatible policy/source identities."""
    expected = validate_identity(expected_identity)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("context_schema") != CONTEXT_SCHEMA:
        raise ValueError("Unsupported noise initializer checkpoint format/context schema.")
    actual = validate_identity(payload["identity"])
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Noise initializer identity mismatch for {key}.")
    initializer = NoiseInitializer(**payload["structure"])
    initializer.load_state_dict(payload["state_dict"], strict=True)
    return initializer.to(map_location).eval()
