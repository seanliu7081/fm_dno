"""Spatial RGB, proprioception, and categorical task memory for motion decoding."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torchvision.models import resnet18

from oat.model.common.normalizer import LinearNormalizer, _normalize
from oat.perception.base_obs_encoder import BaseObservationEncoder


class SpatialTokenObservationEncoder(BaseObservationEncoder):
    """Keep image-grid tokens and use one crop offset per camera/history window.

    Token order is camera, observation time, grid row, grid column, followed
    by chronological state tokens and a single raw-ID task token. All visual
    trunks are independent, randomly initialized ResNet18s with GroupNorm.
    """

    def __init__(
        self,
        shape_meta: Dict,
        n_obs_steps: int = 2,
        crop_shape: Tuple[int, int] = (116, 116),
        token_dim: int = 256,
        task_ids: Sequence[int] = tuple(range(30, 40)),
        state_keys: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if not isinstance(n_obs_steps, int) or n_obs_steps < 1:
            raise ValueError("n_obs_steps must be a positive integer")
        if not isinstance(token_dim, int) or token_dim < 4 or token_dim % 4:
            raise ValueError("token_dim must be a positive multiple of four")
        task_ids = tuple(int(task_id) for task_id in task_ids)
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must contain distinct raw categorical IDs")
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.token_dim = token_dim
        self.crop_shape = tuple(int(size) for size in crop_shape)
        if len(self.crop_shape) != 2 or min(self.crop_shape) < 1:
            raise ValueError("crop_shape must contain two positive sizes")
        self.rgb_ports = [name for name, meta in shape_meta["obs"].items()
                          if meta.get("type") == "rgb"]
        if not self.rgb_ports:
            raise ValueError("Spatial observation encoding requires RGB observations")
        if state_keys is None:
            # Preserve the original LIBERO schema and checkpoint dimensions.
            self.state_ports = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
            for port, dimension in zip(self.state_ports, (3, 4, 2)):
                if port not in shape_meta["obs"] or tuple(shape_meta["obs"][port]["shape"]) != (dimension,):
                    raise ValueError(f"{port} must have shape ({dimension},)")
        else:
            if isinstance(state_keys, (str, bytes)):
                raise ValueError("state_keys must be a nonempty sequence of distinct observation names")
            self.state_ports = list(state_keys)
            if (not self.state_ports
                    or any(not isinstance(port, str) or not port for port in self.state_ports)
                    or len(set(self.state_ports)) != len(self.state_ports)):
                raise ValueError("state_keys must be a nonempty sequence of distinct observation names")
            for port in self.state_ports:
                if port == "task_uid" or port in self.rgb_ports:
                    raise ValueError(f"state_keys cannot include RGB or task_uid observations: {port}")
                if port not in shape_meta["obs"]:
                    raise ValueError(f"Unknown state observation: {port}")
                shape = tuple(shape_meta["obs"][port].get("shape", ()))
                if (len(shape) != 1 or not isinstance(shape[0], Integral)
                        or isinstance(shape[0], bool) or shape[0] < 1):
                    raise ValueError(f"State observation {port} must have a positive one-dimensional shape")
        state_dim = sum(int(shape_meta["obs"][port]["shape"][0]) for port in self.state_ports)
        if "task_uid" not in shape_meta["obs"]:
            raise ValueError("A raw task_uid observation is required")
        self.image_shapes = {}
        for port in self.rgb_ports:
            shape = tuple(shape_meta["obs"][port]["shape"])
            if len(shape) != 3 or shape[2] != 3:
                raise ValueError(f"{port} must have channels-last RGB shape (H,W,3)")
            if any(crop > size for crop, size in zip(self.crop_shape, shape[:2])):
                raise ValueError(f"crop_shape exceeds the image dimensions for {port}")
            self.image_shapes[port] = shape

        # Each standard ResNet18 spatial downsampling stage rounds upward.
        self.grid_shape = tuple((size + 31) // 32 for size in self.crop_shape)
        self.visual_trunks = nn.ModuleDict()
        self.visual_projections = nn.ModuleDict()
        for port in self.rgb_ports:
            trunk = resnet18(
                weights=None,
                norm_layer=lambda channels: nn.GroupNorm(channels // 16, channels),
            )
            self.visual_trunks[port] = nn.Sequential(*list(trunk.children())[:-2])
            self.visual_projections[port] = nn.Sequential(nn.Linear(512, token_dim), nn.LayerNorm(token_dim))
        self.state_projection = nn.Sequential(nn.Linear(state_dim, token_dim), nn.LayerNorm(token_dim))
        self.camera_embedding = nn.Embedding(len(self.rgb_ports), token_dim)
        self.time_embedding = nn.Embedding(n_obs_steps, token_dim)
        self.type_embedding = nn.Embedding(3, token_dim)  # visual / state / task
        self.task_embedding = nn.Embedding(len(task_ids), token_dim)
        self.register_buffer("task_ids", torch.tensor(task_ids, dtype=torch.long))
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(token_dim // 4).float() / (token_dim // 4))
        self.register_buffer("spatial_frequencies", frequencies, persistent=False)
        for embedding in (self.camera_embedding, self.time_embedding, self.type_embedding, self.task_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        self.normalizer = LinearNormalizer()

    def modalities(self):
        return ["rgb", "state"]

    def output_feature_dim(self) -> int:
        return self.token_dim

    def output_token_count(self) -> int:
        spatial_count = len(self.rgb_ports) * self.n_obs_steps * math.prod(self.grid_shape)
        return spatial_count + self.n_obs_steps + 1

    def set_normalizer(self, normalizer: LinearNormalizer):
        for port in self.rgb_ports + self.state_ports:
            if port not in normalizer.params_dict:
                raise ValueError(f"Missing training normalizer for observation {port}")
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.requires_grad_(False)

    def _crop_history(self, images: torch.Tensor):
        """Crop [B,T,H,W,C] with offsets shared across each example's history."""
        batch, time, height, width, channels = images.shape
        crop_height, crop_width = self.crop_shape
        if self.training:
            top = torch.randint(height - crop_height + 1, (batch,), device=images.device)
            left = torch.randint(width - crop_width + 1, (batch,), device=images.device)
        else:
            top = torch.full((batch,), (height - crop_height) // 2, device=images.device, dtype=torch.long)
            left = torch.full((batch,), (width - crop_width) // 2, device=images.device, dtype=torch.long)
        rows = top[:, None] + torch.arange(crop_height, device=images.device)[None]
        columns = left[:, None] + torch.arange(crop_width, device=images.device)[None]
        row_index = rows[:, None, :, None, None].expand(batch, time, crop_height, width, channels)
        crops = images.gather(2, row_index)
        column_index = columns[:, None, None, :, None].expand(batch, time, crop_height, crop_width, channels)
        return crops.gather(3, column_index), top, left

    def _spatial_encoding(self, top, left, image_shape):
        height, width = image_shape[:2]
        grid_height, grid_width = self.grid_shape
        crop_height, crop_width = self.crop_shape
        rows = (torch.arange(grid_height, device=top.device).float() + 0.5) * crop_height / grid_height
        columns = (torch.arange(grid_width, device=top.device).float() + 0.5) * crop_width / grid_width
        y = (top[:, None].float() + rows[None]) / height
        x = (left[:, None].float() + columns[None]) / width
        y = y[:, :, None].expand(-1, grid_height, grid_width).flatten(1)
        x = x[:, None, :].expand(-1, grid_height, grid_width).flatten(1)
        x_phase = x[..., None] * (2 * math.pi) * self.spatial_frequencies.float()
        y_phase = y[..., None] * (2 * math.pi) * self.spatial_frequencies.float()
        return torch.cat((x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1)

    def _task_indices(self, raw_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        if raw_ids.shape == (batch_size, self.n_obs_steps, 1):
            raw_ids = raw_ids.squeeze(-1)
        if raw_ids.shape != (batch_size, self.n_obs_steps):
            raise ValueError("task_uid must have shape (B,n_obs_steps,1) or (B,n_obs_steps)")
        if not torch.all(raw_ids == raw_ids[:, :1]):
            raise ValueError("All observation-history frames must carry the same task_uid")
        matches = raw_ids[:, :1] == self.task_ids[None]
        if not torch.all(matches.any(dim=-1)):
            raise ValueError(f"Unknown raw task_uid; expected one of {self.task_ids.tolist()}")
        return matches.long().argmax(dim=-1)

    def forward(self, obs_dict: Dict) -> torch.Tensor:
        sample = obs_dict[self.rgb_ports[0]]
        batch = sample.shape[0]
        task_indices = self._task_indices(obs_dict["task_uid"], batch)
        tokens = []
        for camera_index, port in enumerate(self.rgb_ports):
            expected_shape = (batch, self.n_obs_steps, *self.image_shapes[port])
            if tuple(obs_dict[port].shape) != expected_shape:
                raise ValueError(f"{port} must have shape {expected_shape}")
            images = _normalize(obs_dict[port], self.normalizer.params_dict[port], forward=True)
            crops, top, left = self._crop_history(images)
            crops = crops.permute(0, 1, 4, 2, 3).reshape(batch * self.n_obs_steps, 3, *self.crop_shape)
            features = self.visual_trunks[port](crops)
            if tuple(features.shape[-2:]) != self.grid_shape:
                raise RuntimeError("ResNet output grid does not match the configured token geometry")
            visual = self.visual_projections[port](features.flatten(2).transpose(1, 2))
            visual = visual.reshape(batch, self.n_obs_steps, math.prod(self.grid_shape), self.token_dim)
            position = self._spatial_encoding(top, left, self.image_shapes[port])
            visual = visual + position[:, None].to(visual.dtype)
            visual = visual + self.camera_embedding.weight[camera_index].to(visual.dtype)
            visual = visual + self.time_embedding.weight[None, :, None].to(visual.dtype)
            visual = visual + self.type_embedding.weight[0].to(visual.dtype)
            tokens.append(visual.flatten(1, 2))

        states = []
        for port in self.state_ports:
            expected_shape = (batch, self.n_obs_steps, *self.shape_meta["obs"][port]["shape"])
            if tuple(obs_dict[port].shape) != expected_shape:
                raise ValueError(f"{port} must have shape {expected_shape}")
            states.append(_normalize(obs_dict[port], self.normalizer.params_dict[port], forward=True))
        state_tokens = self.state_projection(torch.cat(states, dim=-1))
        state_tokens = state_tokens + self.time_embedding.weight[None].to(state_tokens.dtype)
        state_tokens = state_tokens + self.type_embedding.weight[1].to(state_tokens.dtype)
        tokens.append(state_tokens)
        task_token = self.task_embedding(task_indices) + self.type_embedding.weight[2]
        tokens.append(task_token[:, None].to(state_tokens.dtype))
        return torch.cat(tokens, dim=1)
