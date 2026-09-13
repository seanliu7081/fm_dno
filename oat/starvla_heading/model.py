"""StarVLA-compatible Qwen2.5-VL framework and native DiT Gaussian expert."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import subprocess

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLModel

from . import STARVLA_REVISION
from .prior import ActionStatistics, gaussian_source, heading_prediction, heading_supervision


def add_starvla_path(path=None):
    path = Path(path or os.environ.get("STARVLA_ROOT", "/workspace/deps/starVLA")).resolve()
    if not (path / "starVLA/model/modules/action_model/GR00T_ActionHeader.py").is_file():
        raise FileNotFoundError(f"Pinned StarVLA source missing: {path}; run scripts/setup_starvla_heading.sh")
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != STARVLA_REVISION:
        raise RuntimeError(f"Expected StarVLA {STARVLA_REVISION}, found {revision}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


add_starvla_path()
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY


class HeadingGaussianActionHead(FlowmatchingActionHead):
    """Native StarVLA expert with fm_dno sources, supervision, and continuous time.

VLM tokens remain attention memory; a masked pooled summary and both robot
states predict heading. Only the source route is detached. State tokens receive
time positions. Raw action labels enter this head so clipping cannot corrupt
heading supervision or source geometry.
"""
    def __init__(self, config: dict, statistics: dict, vl_hidden_dim: int):
        settings = dict(config)
        native = {
            "action_model_type": "DiT-B", "hidden_size": 1024, "state_dim": 9,
            "action_dim": 7, "action_horizon": 16, "add_pos_embed": True,
            "max_seq_len": 64, "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5, "noise_beta_beta": 1., "noise_s": .999,
            "num_timestep_buckets": 1000, "num_inference_timesteps": 10,
            "diffusion_model_cfg": {
                "cross_attention_dim": vl_hidden_dim + 3, "dropout": 0.,
                "final_dropout": True, "interleave_self_attention": True,
                "norm_type": "ada_norm", "num_layers": 16, "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
        native.update({k: v for k, v in settings.items() if k in native and k != "diffusion_model_cfg"})
        native["diffusion_model_cfg"].update(settings.get("diffusion_model_cfg", {}))
        native["diffusion_model_cfg"]["cross_attention_dim"] = vl_hidden_dim + 3
        super().__init__(OmegaConf.create({"framework": {"action_model": native}}))
        def initialize_embedding(module):
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0., std=.02)
        # Native initialization accesses Embedding.weight after child modules
        # are partitioned by ZeRO Init. Module.apply gathers before reinitializing.
        self.future_tokens.apply(initialize_embedding)
        self.position_embedding.apply(initialize_embedding)
        self.statistics = ActionStatistics(statistics)
        self.heading_settings = settings
        self.n_obs_steps = int(settings.get("n_obs_steps", 2))
        self.heading_head = nn.Sequential(
            nn.LayerNorm(vl_hidden_dim + self.n_obs_steps * native["state_dim"]),
            nn.Linear(vl_hidden_dim + self.n_obs_steps * native["state_dim"], 128),
            nn.SiLU(), nn.Linear(128, 3),
        )
        self.state_positions = nn.Parameter(torch.randn(1, self.n_obs_steps, self.input_embedding_dim) * .02)
        # Initial behavior ignores appended heading, as in fm_dno. The added
        # columns learn from flow loss; auxiliary losses train heading at step 0.
        def initialize_heading_columns(module):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight[:, -3:].zero_()
        for block in self.model.transformer_blocks:
            if block.cross_attention_dim is not None:
                # ZeRO Init intercepts Module.apply and gathers each Linear
                # before initialization; direct indexing a sharded weight fails.
                block.attn1.to_k.apply(initialize_heading_columns)
                block.attn1.to_v.apply(initialize_heading_columns)

    def condition(self, vl_embs, state, encoder_attention_mask):
        # ZeRO Init places parameters on CUDA but leaves from_numpy buffers on
        # CPU; its engine may then skip a whole-model .to(). Move only these
        # small FP32 statistics, never the partitioned model parameters.
        if self.statistics.action_scale.device != vl_embs.device:
            self.statistics.to(device=vl_embs.device)
        mask = encoder_attention_mask.bool() if encoder_attention_mask is not None else torch.ones(
            vl_embs.shape[:2], dtype=torch.bool, device=vl_embs.device)
        if not mask.any(-1).all():
            raise ValueError("Every sample must contain at least one unmasked VLM token")
        normalized_state = self.statistics.normalize(state, "state")
        if normalized_state.shape[1:] != (self.n_obs_steps, self.config.state_dim):
            raise ValueError("State must contain the configured observation history")
        pooled = (vl_embs.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        prediction = heading_prediction(self.heading_head(torch.cat((pooled, normalized_state.flatten(1)), -1).to(self.dtype)))
        heading = torch.cat((prediction["direction"], prediction["confidence"][:, None]), -1)
        memory = torch.cat((vl_embs, heading[:, None].expand(-1, vl_embs.shape[1], -1).to(vl_embs.dtype)), -1)
        return memory, normalized_state.to(self.dtype), prediction, mask

    def source(self, noise, prediction):
        settings = self.heading_settings
        return gaussian_source(noise, prediction, self.statistics,
            mean=settings.get("prior_heading_mean", .5),
            parallel_std=settings.get("prior_parallel_std", 1.),
            perpendicular_std=settings.get("prior_perpendicular_std", .5),
            noise_scale=settings.get("prior_noise_scale", 1.),
            confidence_threshold=settings.get("min_source_confidence", .5))

    def velocity(self, x, t, memory, state, mask):
        # Continuous Uniform(0,1) time, matching fm_dno, explicitly differs from
        # the upstream Beta/integer-bucket recipe.
        timesteps = t.float() * self.num_timestep_buckets
        action_features = self.action_encoder(x.to(self.dtype), timesteps)
        positions = torch.arange(x.shape[1], device=x.device)
        action_features = action_features + self.position_embedding(positions)[None]
        state_features = self.state_encoder(state) + self.state_positions
        registers = self.future_tokens(torch.arange(self.config.num_target_vision_tokens, device=x.device))[None].expand(x.shape[0], -1, -1)
        hidden = torch.cat((state_features, registers, action_features), 1)
        # Explicit additive mask avoids diffusers attention-mask bool/bias
        # ambiguities across versions. Self-attention receives no memory mask.
        attention_bias = torch.zeros(mask.shape, device=mask.device, dtype=memory.dtype)
        attention_bias.masked_fill_(~mask, torch.finfo(memory.dtype).min)
        output = self.model(hidden_states=hidden, encoder_hidden_states=memory,
                            encoder_attention_mask=attention_bias[:, None, :], timestep=timesteps)
        return self.action_decoder(output[:, -x.shape[1]:]).float()

    def forward(self, vl_embs, actions=None, state=None, encoder_attention_mask=None,
                action_mask=None, noise=None, t=None, return_metrics=False, inference=False):
        if inference:
            return self.predict_action(vl_embs, state, encoder_attention_mask, noise=noise)
        if actions.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("Actions must match configured prediction horizon and dimension")
        if action_mask is None:
            action_mask = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        memory, state, prediction, mask = self.condition(vl_embs, state, encoder_attention_mask)
        x1 = self.statistics.normalize(actions)
        noise = torch.randn_like(x1) if noise is None else noise
        x0, active = self.source(noise, prediction)
        t = torch.rand(len(x1), device=x1.device) if t is None else t
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        # Padded targets must not leak through bidirectional action attention.
        # Their slots carry source noise only and receive no supervised loss.
        xt = torch.where(action_mask[..., None].bool(), xt, x0)
        pred = self.velocity(xt, t, memory, state, mask)
        error = (pred - (x1 - x0)).square().mean(-1)
        loss_per_sample = (error * action_mask).sum(-1) / action_mask.sum(-1).clamp_min(1)
        flow_loss = loss_per_sample.mean()
        direction_loss, validity_loss, angle, valid_fraction = heading_supervision(
            actions, action_mask, prediction, self.statistics,
            threshold=self.heading_settings.get("min_target_confidence", .05))
        total = flow_loss + self.heading_settings.get("heading_loss_weight", .1) * direction_loss
        total = total + self.heading_settings.get("validity_loss_weight", .1) * validity_loss
        metrics = {"action_loss": total, "flow_loss": flow_loss.detach(),
                   "heading_loss": direction_loss.detach(), "validity_loss": validity_loss.detach(),
                   "heading_error_deg": angle, "source_active_fraction": active.float().mean(),
                   "target_valid_fraction": valid_fraction}
        return metrics if return_metrics else total

    @torch.no_grad()
    def predict_action(self, vl_embs, state=None, encoder_attention_mask=None, noise=None):
        memory, state, prediction, mask = self.condition(vl_embs, state, encoder_attention_mask)
        if noise is None:
            noise = torch.randn(len(vl_embs), self.action_horizon, self.action_dim, device=vl_embs.device)
        x, _ = self.source(noise, prediction)
        for step in range(self.num_inference_timesteps):
            t = torch.full((len(x),), step / self.num_inference_timesteps, device=x.device)
            x = x + self.velocity(x, t, memory, state, mask) / self.num_inference_timesteps
        return x


@FRAMEWORK_REGISTRY.register("QwenHeadingGaussian")
class QwenHeadingGaussian(nn.Module):
    """Native StarVLA forward/predict_action contract, without unused LM logits."""
    def __init__(self, config, statistics=None, *, pretrained=True):
        super().__init__()
        self.config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
        framework = self.config.get("framework", self.config)
        statistics = statistics or self.config.get("statistics")
        if statistics is None:
            raise ValueError("Training-only action/state statistics are required")
        self.framework_config = framework
        path = framework["qwenvl"]["base_vlm"]
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        self.processor.tokenizer.padding_side = "left"
        image_size = int(framework.get("image_size", 224))
        self.processor.image_processor.min_pixels = image_size * image_size
        self.processor.image_processor.max_pixels = image_size * image_size
        kwargs = {"attn_implementation": framework["qwenvl"].get("attn_implementation", "sdpa")}
        if pretrained:
            self.qwen = Qwen2_5_VLModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                       local_files_only=True, **kwargs)
        else:
            config = Qwen2_5_VLConfig.from_pretrained(path, local_files_only=True)
            self.qwen = Qwen2_5_VLModel._from_config(config, torch_dtype=torch.bfloat16, **kwargs)
        self.qwen.config.use_cache = False
        self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.action_model = HeadingGaussianActionHead(framework.get("action_model", {}), statistics,
                                                       self.qwen.config.text_config.hidden_size)
        # Full fine-tuning is explicit, including vision and visual merger.
        self.requires_grad_(True)

    def prepare_inputs(self, examples):
        texts, images = [], []
        for example in examples:
            if len(example["image"]) != 4:
                raise ValueError("Expected two frames of front and wrist images")
            content = [{"type": "text", "text": "Task: " + example["lang"]}]
            for i, img in enumerate(example["image"]):
                content.extend([{"type": "text", "text": f"{'Previous' if i < 2 else 'Current'} {'front' if i % 2 == 0 else 'wrist'} view:"},
                                {"type": "image", "image": img}])
                images.append(img)
            content.append({"type": "text", "text": "Predict the next robot actions for this task."})
            texts.append(self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                            tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        max_tokens = int(self.framework_config.get("max_input_tokens", 512))
        if inputs["input_ids"].shape[1] > max_tokens:
            raise ValueError(f"Input exceeds explicit {max_tokens}-token budget; adjust config after profiling")
        device = next(self.qwen.parameters()).device
        return inputs.to(device)

    def encode(self, examples):
        inputs = self.prepare_inputs(examples)
        outputs = self.qwen(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        state = torch.as_tensor(np.stack([x["state"] for x in examples]), device=outputs.last_hidden_state.device)
        return outputs.last_hidden_state, state, inputs["attention_mask"]

    def forward(self, examples, inference=False, **kwargs):
        tokens, state, mask = self.encode(examples)
        if inference:
            normalized = self.action_model(tokens, state=state, encoder_attention_mask=mask, inference=True, **kwargs)
            return {"normalized_actions": normalized}
        actions = torch.as_tensor(np.stack([x["action"] for x in examples]), device=tokens.device)
        action_mask = torch.as_tensor(np.stack([x.get("action_mask", np.ones(len(x["action"]), bool)) for x in examples]), device=tokens.device)
        return self.action_model(tokens, actions, state, encoder_attention_mask=mask,
                                 action_mask=action_mask, return_metrics=True, **kwargs)

    def compute_loss(self, tag, batch, loss_scale=None):
        if tag != "vla":
            raise ValueError("This experiment supports original-LIBERO VLA training only")
        return self.forward(batch)

    @torch.no_grad()
    def predict_action(self, examples, **kwargs):
        if isinstance(examples, dict):
            examples = [examples]
        tokens, state, mask = self.encode(examples)
        normalized = self.action_model(tokens, state=state, encoder_attention_mask=mask, inference=True, **kwargs)
        raw = self.action_model.statistics.unnormalize(normalized)
        return {"normalized_actions": normalized.cpu().numpy(), "actions": raw.cpu().numpy()}
