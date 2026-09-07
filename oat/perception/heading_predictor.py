"""Observation-only reference directions, supervised by demonstration action chunks.

The predictor owns its observation encoder and normalizers so the entire reference
frame can be frozen before training a flow policy. Future actions are used only by
``targets`` / ``loss`` during offline supervision; ``forward`` accepts observations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.symmetry.normalizer import so2_scale_offset_from_stats
from oat.symmetry.so2_chunk import SO2ChunkSpec, chunk_heading, wrap_angle


class ObservationHeadingPredictor(nn.Module):
    """Predict a planar unit direction and the probability its target is well defined.

    ``confidence`` predicts whether the demonstration chunk has a non-negligible
    resultant planar vector. It is a validity probability, not a calibrated estimate
    of angular accuracy. Both image+state and state-only encoders use this interface.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        obs_encoder: BaseObservationEncoder,
        n_obs_steps: int = 2,
        hidden_dim: int = 128,
        min_target_confidence: float = 0.05,
    ) -> None:
        super().__init__()
        if n_obs_steps < 1 or hidden_dim < 1:
            raise ValueError("n_obs_steps and hidden_dim must be positive")
        if not math.isfinite(min_target_confidence) or min_target_confidence <= 0:
            raise ValueError("min_target_confidence must be finite and positive")
        self.obs_encoder = obs_encoder
        self.n_obs_steps = int(n_obs_steps)
        self.hidden_dim = int(hidden_dim)
        self.min_target_confidence = float(min_target_confidence)
        self.head = nn.Sequential(
            nn.Linear(self.n_obs_steps * obs_encoder.output_feature_dim(), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.normalizer = LinearNormalizer()
        self.normalizer_vector_mode = "rms"
        self._frozen = False

    def forward(self, obs_dict: Dict) -> Dict[str, torch.Tensor]:
        features = self.obs_encoder(obs_dict)
        if features.ndim != 3 or features.shape[1] != self.n_obs_steps:
            raise ValueError(
                "Heading encoder must return (B, n_obs_steps, D); "
                f"expected {self.n_obs_steps} observation steps, got {tuple(features.shape)}"
            )
        output = self.head(features.flatten(start_dim=1))
        raw_direction = output[:, :2]
        norm = raw_direction.norm(dim=-1, keepdim=True)
        unit = raw_direction / norm.clamp_min(1e-6)
        fallback = torch.zeros_like(unit)
        fallback[:, 0] = 1
        direction = torch.where(norm > 1e-6, unit, fallback)
        valid_logit = output[:, 2]
        return {
            "direction": direction,
            "theta": torch.atan2(direction[:, 1], direction[:, 0]),
            "valid_logit": valid_logit,
            "confidence": valid_logit.sigmoid(),
        }

    def freeze(self) -> "ObservationHeadingPredictor":
        """Freeze encoder, head and normalization; parent.train() cannot change mode."""
        self._frozen = True
        self.requires_grad_(False)
        self.train(False)
        return self

    def train(self, mode: bool = True) -> "ObservationHeadingPredictor":
        super().train(False if self._frozen else mode)
        return self

    def set_normalizer(
        self,
        normalizer: LinearNormalizer,
        action_spec: Union[SO2ChunkSpec, Mapping],
        vector_mode: str = "rms",
    ) -> None:
        """Copy training statistics and use the same SO(2) action map as the policy."""
        spec = self._spec(action_spec)
        reference = next(self.head.parameters())
        self.normalizer.load_state_dict(normalizer.state_dict())
        if "action" not in self.normalizer.params_dict:
            raise ValueError("Heading supervision requires an action normalizer")
        params = self.normalizer.params_dict["action"]
        stats = {key: value.detach().cpu() for key, value in params["input_stats"].items()}
        scale, offset = so2_scale_offset_from_stats(stats, spec, vector_mode=vector_mode)
        with torch.no_grad():
            params["scale"].copy_(scale.to(params["scale"]))
            params["offset"].copy_(offset.to(params["offset"]))
        self.normalizer_vector_mode = vector_mode
        self.obs_encoder.set_normalizer(self.normalizer)
        # LinearNormalizer constructs tensors while loading, including on CPU.
        self.to(device=reference.device, dtype=reference.dtype)
        if self._frozen:
            self.freeze()

    @torch.no_grad()
    def targets(
        self, actions: torch.Tensor, action_spec: Union[SO2ChunkSpec, Mapping]
    ) -> Dict[str, torch.Tensor]:
        """Labels from raw demonstration actions, with holds/reversals masked out."""
        if "action" not in self.normalizer.params_dict:
            raise RuntimeError("Call set_normalizer or load_checkpoint before creating targets")
        normalized = self.normalizer["action"].normalize(actions)
        theta, confidence = chunk_heading(normalized, self._spec(action_spec))
        valid = confidence >= self.min_target_confidence
        return {
            "theta": theta,
            "direction": torch.stack((theta.cos(), theta.sin()), dim=-1),
            "confidence": confidence,
            "valid": valid,
        }

    def loss(
        self,
        batch: Dict,
        action_spec: Union[SO2ChunkSpec, Mapping],
        prediction: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if prediction is None:
            prediction = self(batch["obs"])
        target = self.targets(batch["action"], action_spec)
        valid = target["valid"].to(prediction["direction"].dtype)
        count = valid.sum().clamp_min(1)
        cosine = (prediction["direction"] * target["direction"]).sum(-1).clamp(-1, 1)
        heading_loss = ((1 - cosine) * valid).sum() / count
        validity_loss = F.binary_cross_entropy_with_logits(prediction["valid_logit"], valid)
        error = wrap_angle(prediction["theta"] - target["theta"]).abs()
        return {
            "loss": heading_loss + validity_loss,
            "heading_loss": heading_loss,
            "validity_loss": validity_loss,
            "valid_fraction": valid.mean().detach(),
            "heading_cosine": ((cosine * valid).sum() / count).detach(),
            "heading_error_deg": ((error * valid).sum() / count * (180 / math.pi)).detach(),
        }

    @staticmethod
    def _spec(action_spec: Union[SO2ChunkSpec, Mapping]) -> SO2ChunkSpec:
        return action_spec if isinstance(action_spec, SO2ChunkSpec) else SO2ChunkSpec(**action_spec)

    def _encoder_signature(self) -> str:
        """Fingerprint input order and preprocessing, which tensor shapes cannot check."""
        from omegaconf import OmegaConf

        attributes = (
            "state_ports", "rgb_ports", "rgb_keys", "text_ports", "obs_shapes",
            "input_shape", "crop_height", "crop_width", "num_crops", "pos_enc", "p",
        )
        descriptors = []
        for name, module in self.obs_encoder.named_modules():
            # Fitted normalizers have extra ParameterDict children; their contents
            # are loaded separately and are not an architectural difference.
            if "normalizer" in name.split("."):
                continue
            descriptor = {
                "name": name,
                "class": type(module).__module__ + "." + type(module).__qualname__,
            }
            for attr in attributes:
                if hasattr(module, attr):
                    value = getattr(module, attr)
                    if OmegaConf.is_config(value):
                        value = OmegaConf.to_container(value, resolve=True)
                    if isinstance(value, Mapping):
                        value = list(value.items())  # preserve input-port order
                    descriptor[attr] = value
            if hasattr(module, "shape_meta"):
                shape_meta = module.shape_meta
                if OmegaConf.is_config(shape_meta):
                    shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
                descriptor["observation_shapes"] = list(shape_meta["obs"].items())
            descriptors.append(descriptor)
        encoded = json.dumps(
            descriptors, sort_keys=True,
            default=lambda value: value.tolist() if hasattr(value, "tolist") else list(value),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def save_checkpoint(
        self,
        path: Union[str, Path],
        *,
        action_spec: Union[SO2ChunkSpec, Mapping],
        horizon: int,
        model_config: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save tensors and plain metadata; no pickled model or executable config objects."""
        if "action" not in self.normalizer.params_dict:
            raise RuntimeError("Cannot save a heading predictor without fitted normalization")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        details = dict(metadata or {})
        details.update(
            n_obs_steps=self.n_obs_steps,
            hidden_dim=self.hidden_dim,
            feature_dim=self.obs_encoder.output_feature_dim(),
            encoder_signature=self._encoder_signature(),
            min_target_confidence=self.min_target_confidence,
            normalizer_vector_mode=self.normalizer_vector_mode,
            horizon=int(horizon),
            action_spec=asdict(self._spec(action_spec)),
            model_config=model_config,
        )
        # Plain JSON-compatible metadata is safe with torch.load(weights_only=True).
        details = json.loads(json.dumps(details))
        payload = {
            "version": self.CHECKPOINT_VERSION,
            "metadata": details,
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        *,
        action_spec: Optional[Union[SO2ChunkSpec, Mapping]] = None,
        horizon: Optional[int] = None,
        normalizer_vector_mode: Optional[str] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Dict:
        """Load a predictor into an explicitly instantiated architecture and validate labels."""
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or payload.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError("Unsupported observation heading checkpoint version")
        metadata = payload["metadata"]
        expected = {
            "n_obs_steps": self.n_obs_steps,
            "hidden_dim": self.hidden_dim,
            "feature_dim": self.obs_encoder.output_feature_dim(),
            "encoder_signature": self._encoder_signature(),
            "min_target_confidence": self.min_target_confidence,
        }
        if horizon is not None:
            expected["horizon"] = int(horizon)
        if normalizer_vector_mode is not None:
            expected["normalizer_vector_mode"] = normalizer_vector_mode
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Heading checkpoint {key}={metadata.get(key)!r} does not match {value!r}"
                )
        if action_spec is not None and self._spec(metadata["action_spec"]) != self._spec(action_spec):
            raise ValueError("Heading checkpoint action_spec does not match the policy action_spec")
        reference = next(self.head.parameters())
        device, dtype = reference.device, reference.dtype
        self.load_state_dict(payload["state_dict"], strict=True)
        if "action" not in self.normalizer.params_dict:
            raise ValueError("Heading checkpoint has no action normalization")
        self.normalizer_vector_mode = metadata["normalizer_vector_mode"]
        self.to(device=device, dtype=dtype)
        if self._frozen:
            self.freeze()
        return metadata


def build_heading_predictor(
    *,
    shape_meta: Mapping,
    reference_mode: str = "image_state",
    n_obs_steps: int = 2,
    hidden_dim: int = 128,
    min_target_confidence: float = 0.05,
    vision_encoder: Optional[Mapping] = None,
    state_encoder: Optional[Mapping] = None,
) -> ObservationHeadingPredictor:
    """Build the offline trainer's encoder architecture from a policy Hydra config.

    Use ``_recursive_: false`` on this factory so child encoder configs reach the
    fused encoder uninstantiated. The state ablation removes RGB ports before any
    image encoder is constructed. This function never reads a checkpoint.
    """
    from omegaconf import OmegaConf
    from oat.perception.fused_obs_encoder import FusedObservationEncoder

    if reference_mode not in ("image_state", "state"):
        raise ValueError("reference_mode must be 'image_state' or 'state'")

    def plain_config(value):
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return copy.deepcopy(value)

    shape_meta = plain_config(shape_meta)
    if reference_mode == "state":
        shape_meta["obs"] = {
            key: attr for key, attr in shape_meta["obs"].items() if attr["type"] == "state"
        }
        vision_encoder = None
    if not shape_meta["obs"]:
        raise ValueError("The selected reference_mode contains no observations")
    encoder = FusedObservationEncoder(
        shape_meta=shape_meta,
        vision_encoder=plain_config(vision_encoder),
        state_encoder=plain_config(state_encoder),
    )
    return ObservationHeadingPredictor(
        obs_encoder=encoder, n_obs_steps=n_obs_steps, hidden_dim=hidden_dim,
        min_target_confidence=min_target_confidence,
    )
