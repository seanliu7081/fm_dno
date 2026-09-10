"""Temporal convolutional velocity field with observation-conditioned FiLM.

The adapter keeps the flow policy's ``(sample, timestep, cond)`` API.
Observation tokens are flattened in temporal order and supplied to every U-Net
residual block. Timesteps are already scaled by the policy; no diffusion noise
schedule or diffusion objective is introduced here.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from oat.model.diffusion.conditional_unet1d import (
    ConditionalResidualBlock1D, ConditionalUnet1D,
)


class FlowUnet1D(nn.Module):
    """Adapt the repository's conditional U-Net to a flow velocity backbone.

    ``down_dims`` controls U-Net capacity. The shared transformer arguments
    ``n_layer``, ``n_head``, ``n_emb`` and dropout probabilities are accepted for
    policy API compatibility and do not configure this convolutional backbone.
    Short or odd action horizons are edge-padded for downsampling and cropped
    back to their original length after prediction.
    """

    def __init__(
        self, input_dim, output_dim, horizon, n_obs_steps, cond_dim,
        n_layer=4, n_head=4, n_emb=256, p_drop_emb=0.1, p_drop_attn=0.1,
        down_dims: Sequence[int] = (128, 256, 512),
        diffusion_step_embed_dim: int = 128, kernel_size: int = 3,
        n_groups: int = 8, cond_predict_scale: bool = True,
    ):
        super().__init__()
        for name, value in {
            "input_dim": input_dim, "output_dim": output_dim, "horizon": horizon,
            "n_obs_steps": n_obs_steps, "cond_dim": cond_dim,
            "diffusion_step_embed_dim": diffusion_step_embed_dim,
            "kernel_size": kernel_size, "n_groups": n_groups,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if output_dim != input_dim:
            raise ValueError("FlowUnet1D requires matching input_dim and output_dim")
        if diffusion_step_embed_dim < 4 or diffusion_step_embed_dim % 2:
            raise ValueError("diffusion_step_embed_dim must be even and at least 4")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve action length")
        down_dims = tuple(down_dims)
        if not down_dims or any(
            isinstance(width, bool) or not isinstance(width, int)
            or width < 1 or width % n_groups for width in down_dims
        ):
            raise ValueError("down_dims must contain positive multiples of n_groups")
        # ConditionalUnet1D's final Conv1dBlock always uses eight groups.
        if down_dims[0] % 8:
            raise ValueError("down_dims[0] must be divisible by 8")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.cond_dim = cond_dim
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.temporal_multiple = 2 ** (len(down_dims) - 1)
        self.unet = ConditionalUnet1D(
            input_dim=input_dim, global_cond_dim=n_obs_steps * cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims, kernel_size=kernel_size,
            n_groups=n_groups, cond_predict_scale=cond_predict_scale,
        )

    @torch.no_grad()
    def zero_condition_columns(self, columns):
        """Initially ignore selected features in every observation token.

        Shared-heading ablations use this to neutralize their three added
        condition features without changing the existing U-Net parameter layout.
        The columns remain trainable after initialization.
        """
        columns = tuple(columns)
        if any(not isinstance(index, int) or not 0 <= index < self.cond_dim
               for index in columns):
            raise ValueError("condition columns must index observation features")
        indices = [self.diffusion_step_embed_dim + step * self.cond_dim + index
                   for step in range(self.n_obs_steps) for index in columns]
        for module in self.unet.modules():
            if isinstance(module, ConditionalResidualBlock1D):
                module.cond_encoder[1].weight[:, indices] = 0

    def forward(self, sample, timestep, cond):
        if sample.ndim != 3 or sample.shape[1:] != (self.horizon, self.input_dim):
            raise ValueError("sample must have shape (B, horizon, input_dim)")
        batch_size = sample.shape[0]
        if cond.shape != (batch_size, self.n_obs_steps, self.cond_dim):
            raise ValueError("cond must have shape (B, n_obs_steps, cond_dim)")
        # Avoid the underlying diffusion module's integer cast for Python
        # scalar timesteps, preserving continuous time for flow matching.
        timestep = torch.as_tensor(timestep, device=sample.device, dtype=torch.float32)
        if timestep.ndim > 1 or timestep.numel() not in (1, batch_size):
            raise ValueError("timestep must be scalar or have shape (B,)")
        timestep = timestep.reshape(-1).expand(batch_size)
        padding = (-self.horizon) % self.temporal_multiple
        if padding:
            sample = F.pad(sample.transpose(1, 2), (0, padding), mode="replicate").transpose(1, 2)
        velocity = self.unet(sample, timestep, global_cond=cond.flatten(start_dim=1))
        return velocity[:, :self.horizon]
