"""Flow transformer with time modulation and observation cross-attention.

Each action step is one token. Time supplies adaLN-Zero scale, shift and
residual gates, while ordered observation tokens remain cross-attention memory.
The forward interface matches ``TransformerForDiffusion`` for flow policies.
"""

import math

import torch
from torch import nn


def _modulate(tokens, shift, scale):
    return tokens * (1 + scale[:, None, :]) + shift[:, None, :]


class _TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        half = width // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width))

    def forward(self, timestep):
        # Keep the phase calculation in float32 even for low-precision policies.
        phase = timestep.float()[:, None] * self.frequencies.float()[None, :]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class _MixedAdaLNZeroBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm_self = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_cross = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))

    def forward(self, tokens, memory, time_condition):
        (shift_self, scale_self, gate_self,
         shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(time_condition).chunk(9, dim=-1)
        query = _modulate(self.norm_self(tokens), shift_self, scale_self)
        residual = self.self_attn(query, query, query, need_weights=False)[0]
        tokens = tokens + gate_self[:, None, :] * self.residual_dropout(residual)
        query = _modulate(self.norm_cross(tokens), shift_cross, scale_cross)
        residual = self.cross_attn(query, memory, memory, need_weights=False)[0]
        tokens = tokens + gate_cross[:, None, :] * self.residual_dropout(residual)
        residual = self.mlp(_modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp))
        return tokens + gate_mlp[:, None, :] * self.residual_dropout(residual)


class _FinalLayer(nn.Module):
    def __init__(self, width, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.linear = nn.Linear(width, output_dim)

    def forward(self, tokens, time_condition):
        shift, scale = self.adaLN_modulation(time_condition).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(tokens), shift, scale))


class MixedAdaLNZeroTransformer(nn.Module):
    """Bidirectional action transformer with separate time and observation paths.

    Args follow the flow policy's existing transformer constructor. ``timestep``
    is already scaled into the policy's 0--1000 time range; it can be a scalar,
    a one-element tensor, or one value per batch item. ``cond`` contains up to
    ``n_obs_steps`` ordered observation tokens, without a time token.

    Residual gates and the velocity output start at zero. Therefore each block
    initially acts as identity and the entire network predicts zero velocity.
    """

    def __init__(
        self, input_dim, output_dim, horizon, n_obs_steps, cond_dim,
        n_layer=4, n_head=4, n_emb=256, p_drop_emb=0.1, p_drop_attn=0.1,
    ):
        super().__init__()
        dimensions = {
            "input_dim": input_dim, "output_dim": output_dim, "horizon": horizon,
            "n_obs_steps": n_obs_steps, "cond_dim": cond_dim,
            "n_layer": n_layer, "n_head": n_head, "n_emb": n_emb,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if n_emb % 2 or n_emb % n_head:
            raise ValueError("n_emb must be even and divisible by n_head")
        if not 0 <= p_drop_emb <= 1 or not 0 <= p_drop_attn <= 1:
            raise ValueError("dropout probabilities must be between 0 and 1")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.cond_dim = cond_dim
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.empty(1, horizon, n_emb))
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.empty(1, n_obs_steps, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.cond_encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb),
        )
        self.time_emb = _TimeEmbedding(n_emb)
        self.blocks = nn.ModuleList(
            _MixedAdaLNZeroBlock(n_emb, n_head, p_drop_attn) for _ in range(n_layer)
        )
        self.final_layer = _FinalLayer(n_emb, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cond_pos_emb, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, sample, timestep, cond):
        if sample.ndim != 3 or sample.shape[-1] != self.input_dim:
            raise ValueError(f"sample must have shape [B, T, {self.input_dim}]")
        batch_size, length, _ = sample.shape
        if batch_size < 1 or not 1 <= length <= self.horizon:
            raise ValueError(f"sample requires a nonempty batch and 1 <= T <= {self.horizon}")
        if cond.ndim != 3 or cond.shape[0] != batch_size or cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond must have shape [B, To, {self.cond_dim}] matching sample's batch")
        if not 1 <= cond.shape[1] <= self.n_obs_steps:
            raise ValueError(f"cond requires 1 <= To <= {self.n_obs_steps}")
        if cond.device != sample.device:
            raise ValueError("sample and cond must be on the same device")
        timestep = torch.as_tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim == 1 and timestep.numel() in (1, batch_size):
            timestep = timestep.expand(batch_size)
        else:
            raise ValueError("timestep must be a scalar, [1], or [B]")
        tokens = self.drop(self.input_emb(sample) + self.pos_emb[:, :length])
        memory = self.cond_encoder(self.drop(
            self.cond_obs_emb(cond) + self.cond_pos_emb[:, :cond.shape[1]]
        ))
        time_condition = self.time_emb(timestep)
        for block in self.blocks:
            tokens = block(tokens, memory, time_condition)
        return self.final_layer(tokens, time_condition)
