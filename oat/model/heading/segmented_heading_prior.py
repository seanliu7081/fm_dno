"""Masked motion labels, fixed-denominator losses, and segmented XY sources.

Directions describe sums of raw translation commands, not robot yaw. Geometry
is constructed before action normalization, whose XY scales may be unequal.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def segment_targets(
    raw_actions: Tensor,
    action_mask: Tensor,
    heading_xy_rms: Tensor | float,
    num_segments: int = 4,
    threshold: float = 0.05,
) -> dict[str, Tensor]:
    """Summarize real raw XY commands in equally sized future segments.

    ``heading_xy_rms`` is a positive scalar fitted on each training frame once.
    A nonempty stationary/cancelling segment has a negative validity label;
    an empty segment contributes to neither auxiliary loss.
    """
    if raw_actions.ndim != 3 or raw_actions.shape[-1] < 2:
        raise ValueError("raw_actions must have shape (B, horizon, action_dim>=2)")
    batch_size, horizon = raw_actions.shape[:2]
    if num_segments < 1 or horizon < 1 or horizon % num_segments:
        raise ValueError("horizon must be positive and divisible by num_segments")
    if action_mask.shape != (batch_size, horizon):
        raise ValueError("action_mask must have shape (B, horizon)")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")

    segment_length = horizon // num_segments
    raw_xy = raw_actions[..., :2].float().reshape(batch_size, num_segments, segment_length, 2)
    mask = action_mask.to(device=raw_actions.device, dtype=torch.bool).reshape(
        batch_size, num_segments, segment_length
    )
    count = mask.sum(dim=-1)
    resultant = torch.where(mask[..., None], raw_xy, 0.0).sum(dim=-2)
    magnitude = resultant.norm(dim=-1)
    rms = torch.as_tensor(heading_xy_rms, dtype=torch.float32, device=raw_actions.device)
    if rms.numel() != 1:
        raise ValueError("heading_xy_rms must be a scalar")
    score = magnitude / (rms.reshape(()) * count.clamp_min(1).float().sqrt())
    has_data = count > 0
    valid = has_data & (score >= threshold)
    direction = resultant / magnitude[..., None].clamp_min(1e-6)
    return dict(direction=direction, valid=valid, has_data=has_data, count=count, score=score)


def heading_losses(
    prediction: Mapping[str, Tensor], target: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Return float32 losses and detached metric numerator/count pairs.

    Both loss denominators include every B*K slot, even masked slots. Metrics
    use their appropriate population: valid headings for angular error and
    nonempty segments for validity accuracy and label prevalence.
    """
    direction = prediction["direction"].float()
    valid_logit = prediction["valid_logit"].float()
    target_direction = target["direction"].float()
    valid = target["valid"].bool()
    has_data = target["has_data"].bool()
    if direction.ndim != 3 or direction.shape[-1] != 2 or direction.shape != target_direction.shape:
        raise ValueError("predicted and target directions must have shape (B, K, 2)")
    if any(value.shape != direction.shape[:2] for value in (valid_logit, valid, has_data)):
        raise ValueError("validity logits and target masks must have shape (B, K)")
    cosine = (direction * target_direction).sum(dim=-1).clamp(-1.0, 1.0)
    direction_loss = ((1.0 - cosine) * valid.float()).mean()
    validity_loss = (
        F.binary_cross_entropy_with_logits(valid_logit, valid.float(), reduction="none")
        * has_data.float()
    ).mean()

    with torch.no_grad():
        valid_count = valid.float().sum()
        data_count = has_data.float().sum()
        total_count = valid_count.new_tensor(valid.numel())
        correct = (valid_logit >= 0) == valid
        metrics = dict(
            angle_error_sum=(cosine.acos().rad2deg() * valid.float()).sum(),
            angle_error_count=valid_count,
            validity_correct_sum=(correct & has_data).float().sum(),
            validity_correct_count=data_count,
            valid_target_sum=(valid & has_data).float().sum(),
            valid_target_count=data_count,
            valid_segment_sum=valid_count,
            valid_segment_count=total_count,
        )
    return dict(direction_loss=direction_loss, validity_loss=validity_loss, **metrics)


def segmented_gaussian_source(
    noise: Tensor,
    prediction: Mapping[str, Tensor],
    action_scale: Tensor,
    action_offset: Tensor,
    noise_scale: float = 1.0,
    mean: float = 0.5,
    parallel_std: float = 1.0,
    perpendicular_std: float = 0.5,
    confidence_threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Transform unscaled IID noise using detached predicted segment headings.

    Returns the source in the input noise dtype and an active mask of shape
    ``[B,K]``. Invalid predictions retain the normalized-space IID source, as
    do all non-XY channels. Noise gradients remain available; predictions and
    normalizer parameters receive no gradient through this construction.
    """
    for name, value in (("noise_scale", noise_scale), ("parallel_std", parallel_std),
                        ("perpendicular_std", perpendicular_std)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not math.isfinite(mean) or mean < 0:
        raise ValueError("mean must be finite and nonnegative")
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0,1]")
    if noise.ndim != 3 or noise.shape[-1] < 2:
        raise ValueError("noise must have shape (B, horizon, action_dim>=2)")

    direction = prediction["direction"].detach().to(device=noise.device, dtype=torch.float32)
    confidence = prediction["confidence"].detach().to(device=noise.device, dtype=torch.float32)
    if direction.ndim != 3 or direction.shape[0] != noise.shape[0] or direction.shape[-1] != 2:
        raise ValueError("predicted direction must have shape (B, K, 2)")
    if confidence.shape != direction.shape[:2]:
        raise ValueError("predicted confidence must have shape (B, K)")
    num_segments = direction.shape[1]
    if num_segments < 1 or noise.shape[1] < 1 or noise.shape[1] % num_segments:
        raise ValueError("source horizon must be positive and divisible by the segment count")
    norm = direction.norm(dim=-1, keepdim=True)
    active = torch.isfinite(direction).all(dim=-1) & torch.isfinite(confidence)
    active = active & (norm.squeeze(-1) > 1e-6) & (confidence >= confidence_threshold)
    # Sanitize inactive directions before any arithmetic, including NaN values.
    direction = torch.where(active[..., None], direction, direction.new_tensor([1.0, 0.0]))
    direction = F.normalize(direction, dim=-1, eps=1e-6)
    direction = direction.repeat_interleave(noise.shape[1] // num_segments, dim=1)
    perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)

    scale = action_scale.detach().to(device=noise.device, dtype=torch.float32)
    offset = action_offset.detach().to(device=noise.device, dtype=torch.float32)
    if scale.ndim != 1 or scale.numel() < 2 or offset.shape != scale.shape:
        raise ValueError("action_scale and action_offset must be matching per-channel vectors")
    scale, offset = scale[:2], offset[:2]
    raw_scale = scale.reciprocal().square().mean().sqrt()
    z = noise.float() * noise_scale
    parallel = mean * noise_scale + parallel_std * z[..., :1]
    across = perpendicular_std * z[..., 1:2]
    raw_xy = raw_scale * (parallel * direction + across * perpendicular)
    source_xy = raw_xy * scale + offset
    timestep_active = active.repeat_interleave(noise.shape[1] // num_segments, dim=1)
    source_xy = torch.where(timestep_active[..., None], source_xy, z[..., :2])
    source = torch.cat((source_xy, z[..., 2:]), dim=-1)
    return source.to(noise.dtype), active
