"""Ordered action-command history with explicit first/second differences."""
from __future__ import annotations

import torch
from torch import nn


class PastActionTokenEncoder(nn.Module):
    """Encode normalized commands; missing slots have learned content, not actions."""

    def __init__(self, action_dim=7, token_dim=256, past_n=7):
        super().__init__()
        if min(action_dim, token_dim) < 1 or past_n < 3:
            raise ValueError('Positive dimensions and at least three history steps are required')
        self.action_dim, self.token_dim, self.past_n = action_dim, token_dim, past_n
        def projection():
            return nn.Sequential(nn.Linear(action_dim, token_dim), nn.GELU(),
                                 nn.Linear(token_dim, token_dim))
        self.raw_proj = projection()
        self.delta_proj = projection()
        self.delta2_proj = projection()
        self.lag_embedding = nn.Embedding(past_n, token_dim)
        self.type_embedding = nn.Embedding(3, token_dim)
        self.validity_embedding = nn.Embedding(2, token_dim)
        self.missing_embedding = nn.Parameter(torch.empty(1, 1, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        for embedding in (self.lag_embedding, self.type_embedding, self.validity_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.normal_(self.missing_embedding, std=0.02)

    def forward(self, normalized_past, past_mask):
        if normalized_past.ndim != 3 or normalized_past.shape[1:] != (self.past_n, self.action_dim):
            raise ValueError('History must have shape (B,past_n,action_dim)')
        if past_mask.shape != normalized_past.shape[:2] or past_mask.dtype != torch.bool:
            raise ValueError('past_mask must be boolean with shape (B,past_n)')
        past = torch.where(past_mask[..., None], normalized_past, 0.0)
        valid = torch.cat((past_mask, past_mask[:, -2:].all(-1, keepdim=True),
                           past_mask[:, -3:].all(-1, keepdim=True)), dim=1)
        delta = past[:, -1] - past[:, -2]
        delta2 = past[:, -1] - 2 * past[:, -2] + past[:, -3]
        content = torch.cat((self.raw_proj(past), self.delta_proj(delta)[:, None],
                             self.delta2_proj(delta2)[:, None]), dim=1)
        content = torch.where(valid[..., None], content, self.missing_embedding.to(content.dtype))
        positions = torch.cat((self.lag_embedding.weight, self.lag_embedding.weight[-1:].expand(2, -1)))
        types = torch.cat((self.type_embedding.weight[:1].expand(self.past_n, -1),
                           self.type_embedding.weight[1:]))
        return self.norm(content + positions.to(content.dtype)[None]
                         + types.to(content.dtype)[None]
                         + self.validity_embedding(valid.long()).to(content.dtype))
