"""Flow matching with an observation-predicted, full-rank heading prior.

Unlike exact source-angle alignment, this conditional Gaussian retains support
in every action direction. Heading is still supplied explicitly to the velocity
field and supervised only from training actions. No task geometry is consulted.
"""

from __future__ import annotations

import math

import torch

from oat.policy.flow_policy_heading_zero import HeadingZeroFlowPolicy


class HeadingGaussianFlowPolicy(HeadingZeroFlowPolicy):
    def __init__(self, *args, prior_heading_mean=0.5,
                 prior_parallel_std=1.0, prior_perpendicular_std=0.5, **kwargs):
        for name, value in (("prior_heading_mean", prior_heading_mean),
                            ("prior_parallel_std", prior_parallel_std),
                            ("prior_perpendicular_std", prior_perpendicular_std)):
            if not math.isfinite(value) or (value < 0 if name == "prior_heading_mean" else value <= 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'prior_heading_mean' else 'positive'}")
        super().__init__(*args, **kwargs)
        if not math.isfinite(self.prior_noise_scale) or self.prior_noise_scale <= 0:
            raise ValueError("HeadingGaussianFlowPolicy requires positive finite prior_noise_scale")
        self.prior_heading_mean = float(prior_heading_mean)
        self.prior_parallel_std = float(prior_parallel_std)
        self.prior_perpendicular_std = float(prior_perpendicular_std)

    def get_policy_name(self):
        return "flowpolicy_heading_gaussian"

    def prepare_for_training(self):
        # Frozen pretrained perception initializes after checkpoint restoration.
        # Its weights and readiness buffer are subsequently embedded in this policy.
        for module in self.obs_encoder.modules():
            prepare = getattr(module, "prepare_for_training", None)
            if callable(prepare):
                prepare()

    def _transform_source(self, z, prediction, angular_jitter=None):
        if self.source_mode == "iid":
            return z, torch.zeros(z.shape[0], device=z.device, dtype=torch.bool)
        if z.ndim != 3 or z.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("source must have shape (B, horizon, action_dim)")
        theta = prediction["theta"].detach().to(device=z.device, dtype=torch.float32)
        confidence = prediction["confidence"].detach().to(device=z.device)
        if theta.shape != (z.shape[0],) or confidence.shape != (z.shape[0],):
            raise ValueError("source heading and confidence must have shape (B,)")
        active = torch.isfinite(theta) & torch.isfinite(confidence)
        active &= confidence >= self.min_source_confidence
        theta = torch.where(active, theta, torch.zeros_like(theta))
        if angular_jitter is not None:
            jitter = torch.as_tensor(angular_jitter, device=z.device, dtype=torch.float32)
            if jitter.shape != theta.shape or not torch.isfinite(jitter).all():
                raise ValueError("angular_jitter must be finite and have shape (B,)")
            theta = theta + jitter
        direction = torch.stack((theta.cos(), theta.sin()), -1)[:, None]
        perpendicular = torch.stack((-theta.sin(), theta.cos()), -1)[:, None]
        # Define physical directions BEFORE the potentially anisotropic affine
        # action normalization. A scalar raw scale preserves those directions.
        params = self.normalizer["action"].params_dict
        scale = params["scale"][:2].detach().float().to(z.device)
        raw_scale = scale.reciprocal().square().mean().sqrt()
        parallel = self.prior_parallel_std * z[..., :1].float()
        parallel = parallel + self.prior_heading_mean * self.prior_noise_scale
        across = self.prior_perpendicular_std * z[..., 1:2].float()
        raw_xy = raw_scale * (parallel * direction + across * perpendicular)
        offset = params["offset"][:2].detach().float().to(z.device)
        xy = (raw_xy * scale + offset).to(z.dtype)
        transformed = torch.cat((xy, z[..., 2:]), -1)
        return torch.where(active[:, None, None], transformed, z), active
