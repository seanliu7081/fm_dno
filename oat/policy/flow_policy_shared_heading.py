"""Matched ordinary-flow ablations with a heading head on shared observations.

All modes retain the baseline action normalizer. The optional heading source
rotates realized noise in raw action coordinates before normalizing it again.
Future actions supervise the heading head only; inference uses observations.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.policy.flow_policy import FlowPolicy


HEADING_MODES = ("baseline", "auxiliary", "condition")
SOURCE_MODES = ("iid", "heading")


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
it does not alter either actions or flow noise. ``source_mode="heading"`` adds
a separately selected source transform, with its heading input detached. Its
local angular RNG does not consume global NumPy or PyTorch random draws; pass
explicit ``angular_jitter`` to reproduce a particular draw after checkpoint load.
``source_jitter="zero"`` disables angular dither in both training and inference;
an explicit ``angular_jitter`` still overrides it for controlled diagnostics.
Source settings are constructor metadata, not new state-dict tensors, so old IID
checkpoints still load strictly.
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
        source_mode: str = "iid",
        source_kappa: float = 4.0,
        source_jitter: str = "vonmises",
        min_source_confidence: float = 0.5,
        source_seed: int = 0,
        source_vector_blocks: Optional[Sequence[Sequence[int]]] = None,
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
        zero_columns = getattr(self.model, "zero_condition_columns", None)
        if callable(zero_columns):
            zero_columns(range(self.obs_feature_dim - 3, self.obs_feature_dim))
        elif isinstance(getattr(self.model, "cond_obs_emb", None), nn.Linear):
            # Preserve existing transformer initialization and checkpoint keys.
            with torch.no_grad():
                self.model.cond_obs_emb.weight[:, -3:].zero_()
        elif self.backbone_type == "starvla_dit":
            # StarVLA feeds raw observation features to cross-attention K/V
            # projections rather than using a shared observation embedding.
            with torch.no_grad():
                for block in self.model.blocks:
                    if not block.cross_attention:
                        continue
                    attention = block.attention
                    if attention.in_proj_weight is not None:
                        attention.in_proj_weight[attention.embed_dim:, -3:].zero_()
                    else:
                        attention.k_proj_weight[:, -3:].zero_()
                        attention.v_proj_weight[:, -3:].zero_()
        else:
            raise ValueError("Heading flow backbone must support observation condition initialization")
        self.heading_horizon = heading_horizon
        self.min_target_confidence = float(min_target_confidence)
        self.heading_loss_weight = float(heading_loss_weight)
        self.validity_loss_weight = float(validity_loss_weight)
        self.register_buffer("heading_xy_rms", torch.tensor(1.0))
        self.set_heading_xy_rms(heading_xy_rms)
        if not math.isfinite(source_kappa) or source_kappa < 0:
            raise ValueError("source_kappa must be finite and non-negative")
        if not math.isfinite(min_source_confidence) or not 0 <= min_source_confidence <= 1:
            raise ValueError("min_source_confidence must be in [0, 1]")
        self.source_kappa = float(source_kappa)
        self.set_source_jitter(source_jitter)
        self.min_source_confidence = float(min_source_confidence)
        if source_vector_blocks is None:
            source_vector_blocks = ((0, 1), (3, 4)) if self.action_dim >= 5 else ((0, 1),)
        self.set_source_vector_blocks(source_vector_blocks)
        self.set_source_mode(source_mode)
        self.set_source_seed(source_seed)

    @property
    def heading_mode(self):
        return self.obs_encoder.mode

    def set_heading_mode(self, mode):
        self.obs_encoder.set_mode(mode)

    def set_source_mode(self, mode):
        if mode not in SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {SOURCE_MODES}, got {mode!r}")
        self.source_mode = mode

    def set_source_vector_blocks(self, blocks):
        """Select nonoverlapping raw 2D blocks; translation XY must be included."""
        blocks = tuple(tuple(block) for block in blocks)
        if not blocks or any(len(block) != 2 for block in blocks):
            raise ValueError("source_vector_blocks must contain pairs of action indices")
        indices = [index for block in blocks for index in block]
        if any(not isinstance(index, (int, np.integer)) or not 0 <= index < self.action_dim
               for index in indices):
            raise ValueError("source_vector_blocks indices must be integers within action dimensions")
        if len(indices) != len(set(indices)):
            raise ValueError("source_vector_blocks must not overlap or repeat indices")
        if (0, 1) not in blocks:
            raise ValueError("source_vector_blocks must include translation XY (0, 1)")
        self.source_vector_blocks = tuple(tuple(int(index) for index in block) for block in blocks)

    def set_source_jitter(self, mode):
        if mode not in ("vonmises", "zero"):
            raise ValueError("source_jitter must be 'vonmises' or 'zero'")
        self.source_jitter = mode

    def set_source_seed(self, seed):
        """Reset the private CPU angular RNG without affecting any global RNG."""
        self._source_rng = np.random.default_rng(seed)

    def _transform_source(self, z, prediction, angular_jitter=None):
        if self.source_mode == "iid":
            return z, torch.zeros(z.shape[0], device=z.device, dtype=torch.bool)
        if z.ndim != 3 or z.shape[-1] != self.action_dim or z.shape[1] != self.horizon:
            raise ValueError("source must have shape (B, horizon, action_dim)")
        if prediction["theta"].shape != (z.shape[0],) or prediction["confidence"].shape != (z.shape[0],):
            raise ValueError("source heading and confidence must have shape (B,)")
        if angular_jitter is None:
            if self.source_jitter == "zero":
                angular_jitter = torch.zeros(z.shape[0], device=z.device, dtype=torch.float32)
            else:
                angular_jitter = torch.as_tensor(
                    self._source_rng.vonmises(0., self.source_kappa, size=z.shape[0]),
                    device=z.device, dtype=torch.float32,
                )
        else:
            angular_jitter = torch.as_tensor(angular_jitter, device=z.device, dtype=torch.float32)
            if angular_jitter.shape != (z.shape[0],):
                raise ValueError("angular_jitter must have shape (B,)")
            if not bool(torch.isfinite(angular_jitter).all()):
                raise ValueError("angular_jitter must be finite")
        # The source is a target distribution for this first experiment: gradients
        # still reach the head through the condition and auxiliary loss, but not
        # by moving the flow interpolation endpoints via a learned source angle.
        theta = prediction["theta"].detach().to(device=z.device, dtype=torch.float32)
        confidence = prediction["confidence"].detach().to(device=z.device)
        raw = self.normalizer["action"].unnormalize(z.float())
        resultant = raw[:, :self.heading_horizon, :2].sum(dim=1)
        active = torch.isfinite(theta) & torch.isfinite(confidence)
        active = active & (confidence >= self.min_source_confidence)
        # Treat cancellation at affine-roundtrip precision as an undefined
        # heading instead of amplifying tiny residuals with atan2.
        roundoff = (8 * torch.finfo(raw.dtype).eps
                    * raw[:, :self.heading_horizon, :2].abs().sum(dim=(1, 2)).clamp_min(1))
        active = active & torch.isfinite(resultant).all(-1) & (resultant.norm(dim=-1) > roundoff)
        # A safe inactive resultant avoids undefined atan2 gradients at zero.
        safe_resultant = torch.where(
            active[:, None], resultant,
            torch.tensor([1., 0.], device=z.device, dtype=resultant.dtype),
        )
        theta_z = torch.atan2(safe_resultant[:, 1], safe_resultant[:, 0])
        phi = torch.where(active, theta + angular_jitter - theta_z, torch.zeros_like(theta_z))
        cosine, sine = phi.cos()[:, None], phi.sin()[:, None]
        raw_columns = list(raw.unbind(-1))
        for i, j in self.source_vector_blocks:
            raw_columns[i] = cosine * raw[..., i] - sine * raw[..., j]
            raw_columns[j] = sine * raw[..., i] + cosine * raw[..., j]
        rotated = self.normalizer["action"].normalize(torch.stack(raw_columns, dim=-1)).to(z.dtype)
        # Preserve all scalar channels and gated-off samples bit for bit, avoiding
        # rounding from an otherwise algebraically identical affine roundtrip.
        vector_indices = {i for block in self.source_vector_blocks for i in block}
        out_columns = [rotated[..., i] if i in vector_indices else z[..., i]
                       for i in range(self.action_dim)]
        output = torch.stack(out_columns, dim=-1)
        return torch.where(active[:, None, None], output, z), active

    def transform_source(self, z, prediction, angular_jitter=None):
        """Map an already scaled normalized source as N(R_phi(N^-1(z))).

The rotation aligns the realized raw XY resultant to predicted heading plus
angular jitter, and rotates the selected planar vector blocks together. Raw block norms
are preserved; anisotropic Min-Max scaling need not preserve normalized norms.
        """
        return self._transform_source(z, prediction, angular_jitter)[0]

    def predict_action(self, obs_dict, noise=None, angular_jitter=None, return_source=False):
        """Euler inference with the same optional source map as flow training.

``noise`` is an unscaled standard Gaussian draw. Source diagnostics are opt-in so
ordinary IID inference retains the original output keys and random draw order.
        """
        cond, prediction = self.obs_encoder.encode_with_heading(obs_dict)
        batch_size = cond.shape[0]
        if noise is None:
            noise = torch.randn(
                batch_size, self.horizon, self.action_dim,
                device=self.device, dtype=cond.dtype,
            )
        elif noise.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError("noise must have shape (B, horizon, action_dim)")
        source, active = self._transform_source(
            self.prior_noise_scale * noise, prediction, angular_jitter,
        )
        x = source
        dt = 1.0 / self.num_inference_steps
        for i in range(self.num_inference_steps):
            t = torch.full((batch_size,), i * dt, device=cond.device, dtype=cond.dtype)
            x = x + dt * self.model(x, self._scale_t(t), cond)
        action_pred = self.normalizer["action"].unnormalize(x)
        result = {"action": action_pred[:, :self.n_action_steps], "action_pred": action_pred}
        if return_source:
            result.update(source=source, source_active=active, prediction=prediction)
        return result

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

    def loss_components(self, batch, noise=None, t=None, angular_jitter=None):
        """One encoder forward; supplied IID noise/time enable paired comparisons.

Baseline ``loss`` includes diagnostic-head supervision, whose features are
detached. Compare ``flow_loss`` only between arms using the same source/path;
a changed source changes the target velocity and loss scale. For different source
laws, compare generated actions or closed-loop outcomes instead.
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
        x0, source_active = self._transform_source(
            self.prior_noise_scale * noise, prediction, angular_jitter,
        )
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
            "source": x0,
            "source_active": source_active,
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
