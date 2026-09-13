"""Observation-conditioned future-segment queries for coarse XY motion plans."""

from __future__ import annotations

import math

import torch
from torch import nn


class _MotionDecoderBlock(nn.Module):
    def __init__(self, token_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(token_dim)
        self.cross_norm = nn.LayerNorm(token_dim)
        self.ff_norm = nn.LayerNorm(token_dim)
        self.self_attention = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, feedforward_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward_dim, token_dim), nn.Dropout(dropout),
        )
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, queries, memory):
        query = self.self_norm(queries)
        attended, _ = self.self_attention(query, query, query, need_weights=False)
        queries = queries + self.attention_dropout(attended)
        attended, _ = self.cross_attention(self.cross_norm(queries), memory, memory, need_weights=False)
        queries = queries + self.attention_dropout(attended)
        return queries + self.ff(self.ff_norm(queries))


class MotionQueryDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        num_segments: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(token_dim, num_segments, num_layers, num_heads, feedforward_dim) < 1:
            raise ValueError("Motion decoder dimensions and layer counts must be positive")
        if token_dim % 2 or token_dim % num_heads:
            raise ValueError("token_dim must be even and divisible by num_heads")
        self.token_dim = token_dim
        self.num_segments = num_segments
        self.queries = nn.Parameter(torch.empty(1, num_segments, token_dim))
        self.motion_type_embedding = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.motion_type_embedding, std=0.02)
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(token_dim // 2).float() / (token_dim // 2))
        phases = torch.arange(num_segments).float()[:, None] * frequencies[None]
        self.register_buffer("segment_position", torch.cat((phases.sin(), phases.cos()), dim=-1)[None])
        self.blocks = nn.ModuleList(
            _MotionDecoderBlock(token_dim, num_heads, feedforward_dim, dropout) for _ in range(num_layers)
        )
        self.readout = nn.Linear(token_dim, 3)
        self.geometry_projection = nn.Linear(3, token_dim)
        self.motion_norm = nn.LayerNorm(token_dim)

    def forward(self, obs_tokens: torch.Tensor) -> dict:
        if obs_tokens.ndim != 3 or obs_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"obs_tokens must have shape (B,L,{self.token_dim})")
        queries = (self.queries + self.segment_position.to(self.queries.dtype)).expand(obs_tokens.shape[0], -1, -1)
        queries = queries.to(obs_tokens.dtype)
        for block in self.blocks:
            queries = block(queries, obs_tokens)

        # The explicit geometric readout remains float32 even during BF16
        # training; gradients stay connected through the conditioning route.
        with torch.autocast(device_type=obs_tokens.device.type, enabled=False):
            raw = torch.nn.functional.linear(queries.float(), self.readout.weight.float(), self.readout.bias.float())
            vector = raw[..., :2]
            norm = vector.norm(dim=-1, keepdim=True)
            direction = vector / norm.clamp_min(1e-6)
            direction = torch.where(norm > 1e-6, direction, direction.new_tensor([1.0, 0.0]))
            valid_logit = raw[..., 2]
            confidence = valid_logit.sigmoid()
            geometry = torch.cat((direction, confidence[..., None]), dim=-1)
        geometry_features = self.geometry_projection(geometry.to(self.geometry_projection.weight.dtype))
        motion_tokens = self.motion_norm(
            queries + geometry_features.to(queries.dtype) + self.motion_type_embedding.to(queries.dtype)
        )
        return {
            "direction": direction,
            "valid_logit": valid_logit,
            "confidence": confidence,
            "motion_tokens": motion_tokens,
        }
