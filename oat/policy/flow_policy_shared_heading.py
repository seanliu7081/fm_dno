"""Matched ordinary-flow ablations with a heading head on shared observations.

All modes retain the baseline action normalizer and IID Gaussian source. Future
actions supervise the heading head only; inference always uses observations.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.policy.flow_policy import FlowPolicy


HEADING_MODES = ("baseline", "auxiliary", "condition")


class SharedHeadingObservationEncoder(BaseObservationEncoder):
    """Encode once, predict one heading per observation window, pad all modes alike.

The baseline head is an online diagnostic: it receives detached features, so its
supervision cannot change the encoder. The auxiliary mode supervises the shared
encoder without feeding a heading to the flow network. The condition mode also
feeds the predicted direction and validity probability to every observation token.
"""

    def __init__(self, obs_encoder, n_obs_steps, hidden_dim=128, mode="baseline"):
        super().__init__()
        if n_obs_steps < 1 or hidden_dim < 1:
            raise ValueError("n_obs_steps and hidden_dim must be positive")
        self.policy_encoder = obs_encoder
        self.n_obs_steps = int(n_obs_steps)
        self.head = nn.Sequential(
            nn.Linear(n_obs_steps * obs_encoder.output_feature_dim(), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.set_mode(mode)

    def set_mode(self, mode):
        if mode not in HEADING_MODES:
            raise ValueError(f"heading_mode must be one of {HEADING_MODES}, got {mode!r}")
        self.mode = mode

    def modalities(self):
        return self.policy_encoder.modalities()

    def output_feature_dim(self):
        return self.policy_encoder.output_feature_dim() + 3

    def set_normalizer(self, normalizer):
        self.policy_encoder.set_normalizer(normalizer)
        for module in self.policy_encoder.modules():
            if isinstance(module, LinearNormalizer):
                module.requires_grad_(False)

    def encode_with_heading(self, obs_dict):
        features = self.policy_encoder(obs_dict)
        if features.ndim != 3 or features.shape[1] != self.n_obs_steps:
            raise ValueError(
                f"Expected (B, {self.n_obs_steps}, D) observation features, "
                f"got {tuple(features.shape)}"
            )
        head_features = features.detach() if self.mode == "baseline" else features
        output = self.head(head_features.flatten(start_dim=1))
        # Calculate direction/losses in float32 under mixed precision as well.
        output = output.float()
        raw_direction = output[:, :2]
        norm = raw_direction.norm(dim=-1, keepdim=True)
        unit = raw_direction / norm.clamp_min(1e-6)
        fallback = torch.zeros_like(unit)
        fallback[:, 0] = 1
        direction = torch.where(norm > 1e-6, unit, fallback)
        valid_logit = output[:, 2]
        prediction = {
            "direction": direction,
            "theta": torch.atan2(direction[:, 1], direction[:, 0]),
            "valid_logit": valid_logit,
            "confidence": valid_logit.sigmoid(),
        }
        if self.mode == "condition":
            extra = torch.cat((direction, prediction["confidence"][:, None]), dim=-1)
            extra = extra.to(features)
        else:
            extra = features.new_zeros(features.shape[0], 3)
        cond = torch.cat((features, extra[:, None].expand(-1, features.shape[1], -1)), -1)
        return cond, prediction

    def forward(self, obs_dict):
        return self.encode_with_heading(obs_dict)[0]


class SharedHeadingFlowPolicy(FlowPolicy):
    """Baseline, shared auxiliary-heading, and joint heading-condition ablations.

Construct once and deepcopy before switching ``heading_mode`` for matched initial
weights. All modes have identical parameter shapes. The added conditioning columns
start at zero, so the initial flow prediction is identical in all three modes.

``heading_xy_rms`` must be fitted on training episodes only and is independent of
the ordinary per-coordinate action normalizer. It sets the validity-label scale;
it does not alter either actions or flow noise.
"""

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        heading_mode: str = "baseline",
        heading_hidden_dim: int = 128,
        heading_horizon: Optional[int] = None,
        heading_xy_rms: float = 1.0,
        min_target_confidence: float = 0.05,
        heading_loss_weight: float = 0.1,
        validity_loss_weight: float = 0.1,
        **kwargs,
    ):
        heading_horizon = horizon if heading_horizon is None else int(heading_horizon)
        if not 1 <= heading_horizon <= horizon:
            raise ValueError("heading_horizon must be between 1 and policy horizon")
        if not math.isfinite(min_target_confidence) or min_target_confidence <= 0:
            raise ValueError("min_target_confidence must be finite and positive")
        if any(not math.isfinite(w) or w < 0 for w in (heading_loss_weight, validity_loss_weight)):
            raise ValueError("Heading loss weights must be finite and non-negative")
        encoder = SharedHeadingObservationEncoder(
            obs_encoder, n_obs_steps, heading_hidden_dim, heading_mode,
        )
        super().__init__(
            shape_meta=shape_meta, obs_encoder=encoder, horizon=horizon,
            n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, **kwargs,
        )
        if self.action_dim < 2:
            raise ValueError("XY heading requires at least two action dimensions")
        if not isinstance(getattr(self.model, "cond_obs_emb", None), nn.Linear):
            raise ValueError("SharedHeadingFlowPolicy currently requires the transformer backbone")
        with torch.no_grad():
            self.model.cond_obs_emb.weight[:, -3:].zero_()
        self.heading_horizon = heading_horizon
        self.min_target_confidence = float(min_target_confidence)
        self.heading_loss_weight = float(heading_loss_weight)
        self.validity_loss_weight = float(validity_loss_weight)
        self.register_buffer("heading_xy_rms", torch.tensor(1.0))
        self.set_heading_xy_rms(heading_xy_rms)

    @property
    def heading_mode(self):
        return self.obs_encoder.mode

    def set_heading_mode(self, mode):
        self.obs_encoder.set_mode(mode)

    @torch.no_grad()
    def set_heading_xy_rms(self, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("heading_xy_rms must be finite and positive")
        self.heading_xy_rms.fill_(value)

    @torch.no_grad()
    def heading_targets(self, actions):
        """Raw XY resultant direction; holds/reversals have no valid direction."""
        if actions.ndim != 3 or actions.shape[1] < self.heading_horizon or actions.shape[2] < 2:
            raise ValueError("actions must contain (B, heading_horizon, at least 2) values")
        resultant = actions[:, :self.heading_horizon, :2].float().sum(dim=1)
        norm = resultant.norm(dim=-1)
        theta = torch.atan2(resultant[:, 1], resultant[:, 0])
        confidence = norm / (self.heading_xy_rms * math.sqrt(self.heading_horizon))
        return {
            "theta": theta,
            "direction": torch.stack((theta.cos(), theta.sin()), dim=-1),
            "confidence": confidence,
            "valid": confidence >= self.min_target_confidence,
        }

    def loss_components(self, batch, noise=None, t=None):
        """One encoder forward; supplied IID noise/time enable paired comparisons.

Baseline ``loss`` includes diagnostic-head supervision, whose features are
detached. Use ``flow_loss`` for comparing action objectives between modes.
        """
        x1 = self.normalizer["action"].normalize(batch["action"])
        cond, prediction = self.obs_encoder.encode_with_heading(batch["obs"])
        if noise is None:
            noise = torch.randn_like(x1)
        elif noise.shape != x1.shape:
            raise ValueError("noise must have the same shape as the action chunk")
        if t is None:
            t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        elif t.shape != (x1.shape[0],):
            raise ValueError("t must have shape (B,)")
        x0 = self.prior_noise_scale * noise
        t_b = t[:, None, None]
        xt = (1.0 - t_b) * x0 + t_b * x1
        velocity = self.model(xt, self._scale_t(t), cond)
        flow_mse_per_sample = (velocity - (x1 - x0)).square().mean(dim=(1, 2))
        flow_loss = flow_mse_per_sample.mean()

        target = self.heading_targets(batch["action"])
        valid = target["valid"].to(prediction["direction"].dtype)
        count = valid.sum().clamp_min(1)
        cosine = (prediction["direction"] * target["direction"]).sum(-1).clamp(-1, 1)
        heading_loss = ((1 - cosine) * valid).sum() / count
        validity_loss = F.binary_cross_entropy_with_logits(prediction["valid_logit"], valid)
        weighted_heading_loss = (
            self.heading_loss_weight * heading_loss + self.validity_loss_weight * validity_loss
        )
        error = prediction["theta"] - target["theta"]
        error = torch.remainder(error + math.pi, 2 * math.pi) - math.pi
        return {
            "loss": flow_loss + weighted_heading_loss,
            "flow_loss": flow_loss,
            "flow_mse_per_sample": flow_mse_per_sample,
            "heading_loss": heading_loss,
            "validity_loss": validity_loss,
            "weighted_heading_loss": weighted_heading_loss,
            "valid_fraction": valid.mean().detach(),
            "heading_cosine": ((cosine * valid).sum() / count).detach(),
            "heading_error_deg": ((error.abs() * valid).sum() / count * (180 / math.pi)).detach(),
            "prediction": prediction,
            "target": target,
        }

    def forward(self, batch):
        return self.loss_components(batch)["loss"]
