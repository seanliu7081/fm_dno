"""Heading Gaussian mathematics in raw action coordinates.

This module is independent of the VLM and simulator. It preserves fm_dno's
heading-Gaussian source, including detached source construction and its fallback.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ActionStatistics(nn.Module):
    def __init__(self, statistics: dict):
        super().__init__()
        for name in ("action", "state"):
            for key in ("scale", "offset"):
                value = torch.from_numpy(np.asarray(statistics[name][key], dtype=np.float32).copy())
                if not torch.isfinite(value).all() or (key == "scale" and (value <= 0).any()):
                    raise ValueError(f"Invalid {name} {key}")
                self.register_buffer(f"{name}_{key}", value)
        rms = float(statistics["heading_xy_rms"])
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError("heading_xy_rms must be finite and positive")
        self.register_buffer("heading_xy_rms", torch.from_numpy(np.asarray(rms, dtype=np.float32).copy()))

    def _apply(self, fn, recurse=True):
        # DeepSpeed converts the model to BF16; statistics must retain FP32
        # precision (casting them back after BF16 rounding would be too late).
        originals = dict(self.named_buffers(recurse=False))
        super()._apply(fn, recurse=recurse)
        for name, original in originals.items():
            setattr(self, name, original.to(device=getattr(self, name).device, dtype=torch.float32))
        return self

    def normalize(self, value, kind="action"):
        return value.float() * getattr(self, f"{kind}_scale") + getattr(self, f"{kind}_offset")

    def unnormalize(self, value, kind="action"):
        return (value.float() - getattr(self, f"{kind}_offset")) / getattr(self, f"{kind}_scale")


def heading_prediction(output: torch.Tensor) -> dict:
    output = output.float()
    norm = output[:, :2].norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(output[:, :2])
    fallback[:, 0] = 1
    direction = torch.where(norm > 1e-6, output[:, :2] / norm.clamp_min(1e-6), fallback)
    return {"direction": direction, "valid_logit": output[:, 2], "confidence": output[:, 2].sigmoid()}


def gaussian_source(noise, prediction, statistics, *, mean=.5, parallel_std=1., perpendicular_std=.5,
                    noise_scale=1., confidence_threshold=.5):
    if not all(math.isfinite(x) for x in (mean, parallel_std, perpendicular_std, noise_scale)):
        raise ValueError("Source parameters must be finite")
    if mean < 0 or min(parallel_std, perpendicular_std, noise_scale) <= 0:
        raise ValueError("Gaussian source requires nonnegative mean and positive standard deviations")
    z = noise.float() * noise_scale
    direction = prediction["direction"].detach().float()
    confidence = prediction["confidence"].detach().float()
    active = torch.isfinite(direction).all(-1) & torch.isfinite(confidence) & (confidence >= confidence_threshold)
    direction = torch.where(active[:, None], direction, torch.zeros_like(direction))[:, None]
    perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), -1)
    raw_scale = statistics.action_scale[:2].reciprocal().square().mean().sqrt()
    raw_xy = raw_scale * ((mean * noise_scale + parallel_std * z[..., :1]) * direction
                          + perpendicular_std * z[..., 1:2] * perpendicular)
    xy = raw_xy * statistics.action_scale[:2] + statistics.action_offset[:2]
    transformed = torch.cat((xy, z[..., 2:]), -1)
    return torch.where(active[:, None, None], transformed, z), active


def heading_supervision(raw_actions, action_mask, prediction, statistics, threshold=.05):
    mask = action_mask.bool()
    count = mask.sum(-1)
    resultant = (raw_actions.float()[..., :2] * mask[..., None]).sum(1)
    norm = resultant.norm(dim=-1)
    score = norm / (statistics.heading_xy_rms * count.clamp_min(1).float().sqrt())
    valid = (score >= threshold) & (count > 0)
    target = resultant / norm[:, None].clamp_min(1e-8)
    cosine = (prediction["direction"] * target).sum(-1).clamp(-1, 1)
    # Reduce per example so gradient accumulation and microbatch regrouping
    # preserve the same objective, including invalid or empty headings.
    direction_loss = ((1 - cosine) * valid).mean()
    present = count > 0
    bce = F.binary_cross_entropy_with_logits(prediction["valid_logit"], valid.float(), reduction="none")
    validity_loss = (bce * present).mean()
    error = (torch.acos(cosine) * (180 / math.pi) * valid).sum() / valid.sum().clamp_min(1)
    return direction_loss, validity_loss, error.detach(), valid.float().mean()
