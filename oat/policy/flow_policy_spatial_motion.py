"""Spatial observation tokens and a coarse motion plan for DiT flow matching."""
from __future__ import annotations

import math

import torch

from oat.model.common.normalizer import LinearNormalizer
from oat.model.heading.motion_query_decoder import MotionQueryDecoder
from oat.model.heading.segmented_heading_prior import (
    heading_losses, segment_targets, segmented_gaussian_source,
)
from oat.policy.flow_policy import FlowPolicy


class SpatialMotionFlowPolicy(FlowPolicy):
    def __init__(self, shape_meta, obs_encoder, horizon, n_action_steps, n_obs_steps,
                 num_motion_segments=4, motion_decoder_layers=2,
                 motion_decoder_heads=4, motion_decoder_ff_dim=1024,
                 heading_loss_weight=0.1, validity_loss_weight=0.1,
                 min_target_confidence=0.05, min_source_confidence=0.5,
                 prior_heading_mean=0.5, prior_parallel_std=1.0,
                 prior_perpendicular_std=0.5, inference_bf16=True,
                 backbone_type="starvla_dit", backbone_kwargs=None, **kwargs):
        if backbone_type != "starvla_dit":
            raise ValueError("SpatialMotionFlowPolicy requires the starvla_dit backbone")
        if num_motion_segments < 1 or horizon % num_motion_segments:
            raise ValueError("The horizon must be divisible by num_motion_segments")
        if not 1 <= n_action_steps <= horizon:
            raise ValueError("n_action_steps must be within the prediction horizon")
        for name, value in (("heading_loss_weight", heading_loss_weight),
                            ("validity_loss_weight", validity_loss_weight),
                            ("prior_heading_mean", prior_heading_mean)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("min_target_confidence", min_target_confidence),
                            ("prior_parallel_std", prior_parallel_std),
                            ("prior_perpendicular_std", prior_perpendicular_std)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(min_source_confidence) or not 0 <= min_source_confidence <= 1:
            raise ValueError("min_source_confidence must be a probability")
        cond_len = obs_encoder.output_token_count() + num_motion_segments
        options = dict(backbone_kwargs or {})
        if options.get("cond_len", cond_len) != cond_len:
            raise ValueError("cond_len must match observation and motion token counts")
        if options.get("add_cond_pos_emb", False):
            raise ValueError("Spatial and motion tokens already carry position encodings")
        options.update(cond_len=cond_len, add_cond_pos_emb=False)
        super().__init__(shape_meta=shape_meta, obs_encoder=obs_encoder,
                         horizon=horizon, n_action_steps=n_action_steps,
                         n_obs_steps=n_obs_steps, backbone_type=backbone_type,
                         backbone_kwargs=options, **kwargs)
        if self.action_dim != 7:
            raise ValueError("This experiment requires seven-channel delta actions")
        if self.num_inference_steps < 1 or not math.isfinite(self.prior_noise_scale) or self.prior_noise_scale <= 0:
            raise ValueError("Flow sampling requires positive steps and noise scale")
        self.num_motion_segments = int(num_motion_segments)
        self.motion_decoder = MotionQueryDecoder(
            token_dim=obs_encoder.output_feature_dim(), num_segments=num_motion_segments,
            num_layers=motion_decoder_layers, num_heads=motion_decoder_heads,
            feedforward_dim=motion_decoder_ff_dim, dropout=0.0)
        self.heading_loss_weight = float(heading_loss_weight)
        self.validity_loss_weight = float(validity_loss_weight)
        self.min_target_confidence = float(min_target_confidence)
        self.min_source_confidence = float(min_source_confidence)
        self.prior_heading_mean = float(prior_heading_mean)
        self.prior_parallel_std = float(prior_parallel_std)
        self.prior_perpendicular_std = float(prior_perpendicular_std)
        self.inference_bf16 = bool(inference_bf16)
        self.register_buffer("heading_xy_rms", torch.tensor(1.0, dtype=torch.float32))
        self._training_metrics = {}

    def get_policy_name(self):
        return "flowpolicy_spatial_motion_dit"

    def set_normalizer(self, normalizer):
        super().set_normalizer(normalizer)
        for module in self.modules():
            if isinstance(module, LinearNormalizer):
                module.requires_grad_(False)

    def configure_from_dataset(self, dataset):
        rms = float(dataset.get_heading_xy_rms())
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError("Training XY RMS must be finite and positive")
        self.heading_xy_rms.fill_(rms)

    def encode_condition(self, obs_dict):
        observations = self.obs_encoder(obs_dict)
        prediction = self.motion_decoder(observations)
        condition = torch.cat((observations, prediction["motion_tokens"]), dim=1)
        return condition, prediction

    def make_source(self, noise, prediction):
        params = self.normalizer["action"].params_dict
        return segmented_gaussian_source(
            noise, prediction, params["scale"], params["offset"],
            noise_scale=self.prior_noise_scale, mean=self.prior_heading_mean,
            parallel_std=self.prior_parallel_std,
            perpendicular_std=self.prior_perpendicular_std,
            confidence_threshold=self.min_source_confidence)

    def loss_components(self, batch, noise=None, t=None):
        actions = batch["action"]
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("Actions must match the configured horizon and action dimension")
        cond, prediction = self.encode_condition(batch["obs"])
        x1 = self.normalizer["action"].normalize(actions).float()
        noise = torch.randn_like(x1) if noise is None else noise
        if noise.shape != x1.shape:
            raise ValueError("Source noise must match actions")
        x0, active = self.make_source(noise, prediction)
        t = torch.rand(len(x1), device=x1.device) if t is None else t
        if t.shape != (len(x1),):
            raise ValueError("Flow time must have shape (B,)")
        tau = t[:, None, None]
        xt = (1 - tau) * x0 + tau * x1
        velocity = self.model(xt, self._scale_t(t), cond)
        flow_loss = (velocity.float() - (x1 - x0)).square().mean()
        target = segment_targets(actions, batch["action_mask"], self.heading_xy_rms,
                                 num_segments=self.num_motion_segments,
                                 threshold=self.min_target_confidence)
        aux = heading_losses(prediction, target)
        total = (flow_loss + self.heading_loss_weight * aux["direction_loss"]
                 + self.validity_loss_weight * aux["validity_loss"])
        return {"loss": total, "flow_loss": flow_loss, "heading_loss": aux["direction_loss"],
                "validity_loss": aux["validity_loss"], "prediction": prediction,
                "target": target, "source": x0, "source_active": active}

    def forward(self, batch):
        result = self.loss_components(batch)
        if self.training:
            self._cache_metrics(result)
        return result["loss"]

    @torch.no_grad()
    def _cache_metrics(self, result):
        prediction, target = result["prediction"], result["target"]
        valid, present = target["valid"], target["has_data"]
        cosine = (prediction["direction"] * target["direction"]).sum(-1).clamp(-1, 1)
        angle = cosine.acos() * (180.0 / math.pi)
        slots = valid.new_tensor(valid.numel(), dtype=torch.float32)
        examples = slots.new_tensor(valid.shape[0])
        valid_count, present_count = valid.sum(), present.sum()
        metrics = {name: (result[name].detach() * examples, examples)
                   for name in ("flow_loss", "heading_loss", "validity_loss")}
        metrics.update({
            "heading_error_deg": ((angle * valid).sum(), valid_count),
            "target_nonempty_fraction": (present_count, slots),
            "target_valid_fraction": (valid_count, present_count),
            "valid_segment_fraction": (valid_count, slots),
            "validity_accuracy": (((prediction["confidence"] >= .5) == valid).logical_and(present).sum(), present_count),
            "confidence": (prediction["confidence"].sum(), slots),
            "source_active_fraction": (result["source_active"].sum(), slots),
        })
        for j in range(self.num_motion_segments):
            metrics[f"segment_{j}/heading_error_deg"] = ((angle[:, j] * valid[:, j]).sum(), valid[:, j].sum())
            metrics[f"segment_{j}/source_active_fraction"] = (result["source_active"][:, j].sum(), examples)
        self._training_metrics = {key: (num.detach().float(), count.detach().float())
                                  for key, (num, count) in metrics.items()}

    def pop_training_metrics(self):
        result, self._training_metrics = self._training_metrics, {}
        return result

    @torch.no_grad()
    def predict_action(self, obs_dict):
        enabled = self.inference_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=enabled):
            cond, prediction = self.encode_condition(obs_dict)
            noise = torch.randn(len(cond), self.horizon, self.action_dim,
                                device=cond.device, dtype=torch.float32)
            x, _ = self.make_source(noise, prediction)
            dt = 1.0 / self.num_inference_steps
            for i in range(self.num_inference_steps):
                t = torch.full((len(cond),), i * dt, device=cond.device)
                x = x + dt * self.model(x, self._scale_t(t), cond).float()
        action_pred = self.normalizer["action"].unnormalize(x)
        return {"action": action_pred[:, :self.n_action_steps], "action_pred": action_pred}

    def get_optimizer(self, policy_lr, obs_enc_lr, weight_decay, betas, heading_lr=1e-4):
        groups = []
        for label, module, rate in (("policy", self.model, policy_lr),
                                    ("encoder", self.obs_encoder, obs_enc_lr),
                                    ("motion", self.motion_decoder, heading_lr)):
            decay, no_decay = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                is_embedding = any(word in name for word in ("embedding", "pos_emb", "queries", "register_tokens"))
                (decay if param.ndim >= 2 and not is_embedding else no_decay).append(param)
            for suffix, params, decay_value in (("decay", decay, weight_decay), ("no_decay", no_decay, 0.0)):
                if params:
                    groups.append({"params": params, "lr": rate, "weight_decay": decay_value,
                                   "name": f"{label}_{suffix}"})
        covered = [id(p) for group in groups for p in group["params"]]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(covered) != len(set(covered)) or set(covered) != expected:
            raise RuntimeError("Optimizer must cover each trainable parameter exactly once")
        return torch.optim.AdamW(groups, betas=betas)

    def get_gradient_clip_groups(self):
        motion = [p for p in self.motion_decoder.parameters() if p.requires_grad]
        ids = {id(p) for p in motion}
        core = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        return {"core": core, "motion": motion}

    def create_dummy_observation(self, batch_size=1, device=None):
        device = self.device if device is None else device
        result = {}
        for key, shape in self.obs_key_shapes.items():
            size = (batch_size, self.n_obs_steps, *shape)
            if self.shape_meta_type(key) == "rgb":
                result[key] = torch.zeros(size, dtype=torch.uint8, device=device)
            elif key == "task_uid":
                task = int(self.obs_encoder.task_ids[0])
                result[key] = torch.full(size, task, dtype=torch.long, device=device)
            else:
                result[key] = torch.zeros(size, device=device)
                if key.endswith("quat"):
                    result[key][..., -1] = 1.0
        return result

    def shape_meta_type(self, key):
        return "rgb" if len(self.obs_key_shapes[key]) == 3 else "state"
