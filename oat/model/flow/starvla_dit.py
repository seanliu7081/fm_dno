# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for fm_dno from StarVLA's GR00T_ActionHeader.py and NVIDIA's
# flow_matching_head/{action_encoder,cross_attention_dit}.py. This adaptation
# uses only PyTorch, accepts existing observation features, retains continuous
# timesteps, and scales the latent/output dimensions to the configured width.
# Observation-frame positions replace the position information that the source
# receives from its VLM, keeping fm_dno's separately encoded frames ordered.

"""GR00T-style flow Transformer with time AdaLN and alternating attention.

This is ordinary AdaLN, without adaLN-Zero's residual gates or zero-initialized
modulation. It is an action-head adaptation, with no VLM or layerwise features.
"""

import math
from typing import Union

import torch
from torch import nn


def _sinusoidal_embedding(
    timesteps: torch.Tensor,
    dimension: int,
    *,
    flip_sin_to_cos: bool = False,
    frequency_shift: int = 0,
) -> torch.Tensor:
    """Encode continuous time using the source action/DiT frequency conventions."""
    half_dim = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / (half_dim - frequency_shift)
    )
    phases = timesteps.float().unsqueeze(-1) * frequencies
    sine, cosine = phases.sin(), phases.cos()
    return torch.cat((cosine, sine) if flip_sin_to_cos else (sine, cosine), dim=-1)


class _TimeAdaLayerNorm(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.norm = nn.LayerNorm(dimension, eps=1e-5, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 2 * dimension))

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(time_embedding).chunk(2, dim=-1)
        return self.norm(hidden) * (1 + scale[:, None]) + shift[:, None]


class _AlternatingAttentionBlock(nn.Module):
    def __init__(
        self, dimension: int, n_head: int, cond_dim: int, cross_attention: bool, dropout: float
    ):
        super().__init__()
        self.cross_attention = cross_attention
        self.norm_attention = _TimeAdaLayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            n_head,
            kdim=cond_dim if cross_attention else dimension,
            vdim=cond_dim if cross_attention else dimension,
            # The source diffusers Attention drops projected outputs, not
            # attention probabilities; its final_dropout adds a second drop.
            dropout=0.0,
            batch_first=True,
        )
        self.attention_dropout = nn.Sequential(nn.Dropout(dropout), nn.Dropout(dropout))
        self.norm_ff = nn.LayerNorm(dimension, eps=1e-5, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(
        self, hidden: torch.Tensor, time_embedding: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        query = self.norm_attention(hidden, time_embedding)
        context = cond if self.cross_attention else query
        attended, _ = self.attention(query, context, context, need_weights=False)
        hidden = hidden + self.attention_dropout(attended)
        return hidden + self.ff(self.norm_ff(hidden))


class StarVLAFlowTransformer(nn.Module):
    """Predict action velocity from noisy actions, flow time, and observations.

    ``n_layer`` counts individual attention/FFN blocks: cross-attention first,
    then self-attention, repeated. At least two blocks allow action positions
    and learned register tokens to interact. Self-attention is bidirectional.

    ``timestep`` is already in the policy's 0..1000 scale. It is neither scaled
    again nor rounded to an integer, keeping gradients through flow time.
    Observations (including optional heading features) are cross-attention
    keys/values with their existing ``cond_dim``; they are not pooled. Learned
    observation-frame positions supply temporal order absent from the per-frame
    observation encoder (the source VLM already supplied position information).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int,
        cond_dim: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        num_register_tokens: int = 32,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("output_dim", output_dim),
            ("horizon", horizon),
            ("n_obs_steps", n_obs_steps),
            ("cond_dim", cond_dim),
            ("n_head", n_head),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(n_layer, int) or n_layer < 2:
            raise ValueError("n_layer must be at least 2 to include action self-attention")
        if not isinstance(n_emb, int) or n_emb < 2 or n_emb % 2 or n_emb % n_head:
            raise ValueError("n_emb must be positive, even, and divisible by n_head")
        if not isinstance(num_register_tokens, int) or num_register_tokens < 0:
            raise ValueError("num_register_tokens must be a nonnegative integer")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.cond_dim = cond_dim
        self.n_emb = n_emb

        # Source action encoder: project actions, concatenate sinusoidal time,
        # and mix them with a SiLU MLP before adding learned action positions.
        self.action_projection = nn.Linear(input_dim, n_emb)
        self.action_time_mlp = nn.Sequential(
            nn.Linear(2 * n_emb, n_emb), nn.SiLU(), nn.Linear(n_emb, n_emb)
        )
        self.action_pos_emb = nn.Parameter(torch.empty(1, horizon, n_emb))
        self.obs_pos_emb = nn.Parameter(torch.empty(1, n_obs_steps, cond_dim))
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, n_emb))
        nn.init.normal_(self.action_pos_emb, std=0.02)
        nn.init.normal_(self.obs_pos_emb, std=0.02)
        nn.init.normal_(self.register_tokens, std=0.02)
        self.embedding_dropout = nn.Dropout(p_drop_emb)

        # The DiT timestep encoder uses 256 cosine/sine channels, distinct from
        # the action encoder's n_emb sine/cosine channels.
        self.time_encoder = nn.Sequential(nn.Linear(256, n_emb), nn.SiLU(), nn.Linear(n_emb, n_emb))
        self.blocks = nn.ModuleList(
            _AlternatingAttentionBlock(n_emb, n_head, cond_dim, index % 2 == 0, p_drop_attn)
            for index in range(n_layer)
        )
        self.norm_out = nn.LayerNorm(n_emb, eps=1e-6, elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(n_emb, 2 * n_emb))
        self.output_projection = nn.Linear(n_emb, n_emb)
        self.action_decoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.ReLU(), nn.Linear(4 * n_emb, output_dim)
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if sample.ndim != 3 or sample.shape[-1] != self.input_dim:
            raise ValueError(f"sample must have shape (B, H, {self.input_dim})")
        batch_size, horizon, _ = sample.shape
        if not 0 < horizon <= self.horizon:
            raise ValueError(f"sample horizon must be between 1 and {self.horizon}")
        if cond.shape != (batch_size, self.n_obs_steps, self.cond_dim):
            raise ValueError(
                f"cond must have shape (B, {self.n_obs_steps}, {self.cond_dim})"
            )
        timesteps = torch.as_tensor(timestep, device=sample.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.ndim == 1 and timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.shape != (batch_size,):
            raise ValueError("timestep must be a scalar or have shape (B,)")

        action_features = self.action_projection(sample)
        action_time = _sinusoidal_embedding(timesteps, self.n_emb).to(action_features.dtype)
        action_time = action_time[:, None].expand(-1, horizon, -1)
        action_features = self.action_time_mlp(torch.cat((action_features, action_time), dim=-1))
        action_features = action_features + self.action_pos_emb[:, :horizon].to(action_features.dtype)
        registers = self.register_tokens.expand(batch_size, -1, -1).to(action_features.dtype)
        hidden = self.embedding_dropout(torch.cat((registers, action_features), dim=1))

        time_features = _sinusoidal_embedding(
            timesteps, 256, flip_sin_to_cos=True, frequency_shift=1
        ).to(self.time_encoder[0].weight.dtype)
        time_embedding = self.time_encoder(time_features)
        cond = cond + self.obs_pos_emb.to(cond.dtype)
        for block in self.blocks:
            hidden = block(hidden, time_embedding, cond)

        # The GR00T output path modulates LayerNorm with time before its output
        # projection and a separate action decoder MLP. Register outputs are
        # unnecessary, so decode only the trailing action tokens.
        hidden = hidden[:, -horizon:]
        shift, scale = self.output_modulation(time_embedding).chunk(2, dim=-1)
        hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        return self.action_decoder(self.output_projection(hidden))
