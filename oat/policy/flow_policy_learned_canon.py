"""Frozen image/state reference directions for canonical flow sources and ablations.

The reference predicts only from observations. Its complete encoder, head and
normalizers are embedded in policy checkpoints. Loading the standalone reference
is deferred until training initialization or first use, so policy checkpoints can
be restored without the original reference file.
"""

from __future__ import annotations

from typing import Optional

import torch

from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.heading_predictor import ObservationHeadingPredictor
from oat.policy.flow_policy_canon import CanonicalPhaseFlowPolicy
from oat.symmetry.so2_chunk import SO2ChunkSpec


class ReferenceObservationEncoder(BaseObservationEncoder):
    """Trainable policy features plus an independent, frozen heading predictor."""

    def __init__(
        self, obs_encoder, reference_model, reference_checkpoint, action_spec,
        horizon, normalizer_vector_mode, include_reference=False,
    ):
        super().__init__()
        self.policy_encoder = obs_encoder
        self.reference_model = reference_model.freeze()
        self.reference_checkpoint = reference_checkpoint
        self.action_spec = action_spec
        self.horizon = horizon
        self.normalizer_vector_mode = normalizer_vector_mode
        self.include_reference = bool(include_reference)
        self.register_buffer("reference_loaded", torch.tensor(False))

    def modalities(self):
        return self.policy_encoder.modalities()

    def output_feature_dim(self):
        return self.policy_encoder.output_feature_dim() + (3 if self.include_reference else 0)

    def ensure_reference(self):
        if bool(self.reference_loaded):
            return
        if not self.reference_checkpoint:
            raise RuntimeError(
                "A trained reference is required: set reference_checkpoint, or load a "
                "complete learned canonical policy checkpoint before using the policy."
            )
        self.reference_model.load_checkpoint(
            self.reference_checkpoint, action_spec=self.action_spec, horizon=self.horizon,
            normalizer_vector_mode=self.normalizer_vector_mode,
        )
        self.reference_model.freeze()
        self.reference_loaded.fill_(True)

    def set_normalizer(self, normalizer):
        # The reference must retain its own offline training statistics.
        self.policy_encoder.set_normalizer(normalizer)

    def predict_reference(self, obs_dict):
        self.ensure_reference()
        parameter = next(self.reference_model.head.parameters())
        # Parent training may use autocast whereas deployment does not. Use the
        # same reference precision and eval preprocessing in both paths.
        observations = {
            key: value.to(dtype=parameter.dtype) if torch.is_tensor(value)
            and value.is_floating_point() else value
            for key, value in obs_dict.items()
        }
        with torch.no_grad(), torch.autocast(device_type=parameter.device.type, enabled=False):
            return self.reference_model(observations)

    def encode_with_reference(self, obs_dict):
        prediction = self.predict_reference(obs_dict)
        cond = self.policy_encoder(obs_dict)
        if self.include_reference:
            extra = torch.cat((prediction["direction"], prediction["confidence"][:, None]), -1)
            extra = extra.to(device=cond.device, dtype=cond.dtype)
            cond = torch.cat((cond, extra[:, None, :].expand(-1, cond.shape[1], -1)), -1)
        return cond, prediction

    def forward(self, obs_dict):
        return self.encode_with_reference(obs_dict)[0]


class LearnedCanonicalPhaseFlowPolicy(CanonicalPhaseFlowPolicy):
    """Use a frozen observation-only reference in the source, condition, or both.

    ``reference_use='condition'`` keeps the original isotropic source. ``'source'``
    preserves the original conditioning width; ``'both'`` allows the controlled
    comparison where only source canonicalization changes relative to condition.
    ``confidence`` estimates target-heading validity, not angular accuracy.
    """

    def __init__(
        self,
        *args,
        obs_encoder: BaseObservationEncoder,
        reference_model: ObservationHeadingPredictor,
        reference_checkpoint: Optional[str] = None,
        reference_use: str = "source",
        min_reference_confidence: float = 0.5,
        kappa: float = 4.0,
        **kwargs,
    ):
        if reference_use not in ("source", "condition", "both"):
            raise ValueError("reference_use must be 'source', 'condition', or 'both'")
        if not 0.0 <= min_reference_confidence <= 1.0:
            raise ValueError("min_reference_confidence must be in [0, 1]")
        if not isinstance(reference_model, ObservationHeadingPredictor):
            raise TypeError("reference_model must be an ObservationHeadingPredictor")
        if reference_model.n_obs_steps != int(kwargs["n_obs_steps"]):
            raise ValueError("reference_model.n_obs_steps must match policy.n_obs_steps")
        if kwargs.get("inference_prior", "standard") != "standard":
            raise ValueError("Learned canonicalization requires inference_prior='standard'")
        spec_config = kwargs.get("action_spec")
        spec = SO2ChunkSpec(**spec_config) if spec_config else SO2ChunkSpec.libero_osc_pose()
        encoder = ReferenceObservationEncoder(
            obs_encoder, reference_model, reference_checkpoint, spec,
            int(kwargs["horizon"]), kwargs.get("normalizer_vector_mode", "rms"),
            include_reference=reference_use in ("condition", "both"),
        )
        super().__init__(*args, obs_encoder=encoder, kappa=kappa, **kwargs)
        self.reference_use = reference_use
        self.min_reference_confidence = float(min_reference_confidence)
        self.last_reference_diagnostics = {}

    def prepare_for_training(self):
        self.obs_encoder.ensure_reference()

    @property
    def reference_model(self):
        return self.obs_encoder.reference_model

    def _reference_description(self):
        return "learned(" + "+".join(self.reference_model.obs_encoder.modalities()) + ")"

    def get_policy_name(self):
        return "learned_" + super().get_policy_name()

    def _frame(self, prediction):
        theta = prediction["theta"]
        confidence = prediction["confidence"]
        valid = torch.isfinite(theta) & torch.isfinite(confidence)
        valid = valid & (confidence >= self.min_reference_confidence)
        self.last_reference_diagnostics = {
            "reference/valid_fraction": float(valid.float().mean()),
            "reference/mean_validity_probability": float(confidence.mean()),
        }
        # No EEF-speed gate: vision can supply a direction before the robot moves.
        return theta, valid

    def reference_heading(self, obs_dict):
        return self._frame(self.obs_encoder.predict_reference(obs_dict))

    def _prepare_observation(self, obs_dict):
        cond, prediction = self.obs_encoder.encode_with_reference(obs_dict)
        theta, valid = self._frame(prediction)
        return cond, theta, valid

    def _canonicalize_with_heading(self, z, theta_ref, valid, generator=None):
        if self.reference_use == "condition":
            self.last_canon_diagnostics = {
                "canon/valid_frac": 0.0, "canon/abs_phi_mean": 0.0, "canon/norm_drift": 0.0,
            }
            return z
        return super()._canonicalize_with_heading(z, theta_ref, valid, generator=generator)
