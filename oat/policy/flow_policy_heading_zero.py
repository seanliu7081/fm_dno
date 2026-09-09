"""Full-training policy for a shared heading condition and zero-jitter XY prior."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.policy.flow_policy_shared_heading import SharedHeadingFlowPolicy


class HeadingZeroFlowPolicy(SharedHeadingFlowPolicy):
    """The deployable zero-jitter variant, with the mini experiment's head optimizer.

    Observations supply both the explicit heading condition and the direction
    used to align translation-XY source noise. The shared head trains jointly;
    only its source-construction input is detached. Inherited explicit-jitter
    arguments remain available for diagnostics, while ordinary training and
    inference use exactly zero angular dither.

    The heading-label RMS is a checkpoint buffer. A training workspace calls
    ``configure_from_dataset`` to fit it from the dataset's training episodes;
    checkpoint inference restores the buffer without accessing a dataset.
    """

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        *,
        heading_mode: str = "condition",
        source_mode: str = "heading",
        source_jitter: str = "zero",
        source_vector_blocks: Sequence[Sequence[int]] = ((0, 1),),
        **kwargs,
    ):
        for key, actual, required in (
            ("heading_mode", heading_mode, "condition"),
            ("source_mode", source_mode, "heading"),
            ("source_jitter", source_jitter, "zero"),
        ):
            if actual != required:
                raise ValueError(f"HeadingZeroFlowPolicy requires {key}={required!r}; got {actual!r}")
        try:
            blocks = tuple(tuple(block) for block in source_vector_blocks)
        except TypeError as error:
            raise ValueError("HeadingZeroFlowPolicy requires source_vector_blocks=((0, 1),)") from error
        if blocks != ((0, 1),):
            raise ValueError("HeadingZeroFlowPolicy requires source_vector_blocks=((0, 1),)")
        super().__init__(
            shape_meta=shape_meta,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            heading_mode=heading_mode,
            source_mode=source_mode,
            source_jitter=source_jitter,
            source_vector_blocks=blocks,
            **kwargs,
        )

    def get_policy_name(self):
        return f"{super().get_policy_name()}_heading_zero"

    def configure_from_dataset(self, dataset):
        """Set the heading-label scale supplied by the training-only dataset fit."""
        self.set_heading_xy_rms(dataset.get_heading_xy_rms())

    def get_optimizer(
        self,
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
        heading_lr: float = 1e-3,
    ) -> torch.optim.Optimizer:
        optimizer = super().get_optimizer(policy_lr, obs_enc_lr, weight_decay, betas)
        heading = [parameter for parameter in self.obs_encoder.head.parameters() if parameter.requires_grad]
        heading_ids = {id(parameter) for parameter in heading}
        for group in optimizer.param_groups:
            group["params"] = [parameter for parameter in group["params"] if id(parameter) not in heading_ids]
        if heading:
            # The mini experiment uses one separately optimized, undecayed head.
            optimizer.add_param_group({"params": heading, "lr": heading_lr, "weight_decay": 0., "name": "heading"})
        optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if len(optimized) != len({id(parameter) for parameter in optimized}) or {id(parameter) for parameter in optimized} != expected:
            raise RuntimeError("HeadingZeroFlowPolicy optimizer must cover each trainable parameter exactly once")
        return optimizer

    def get_gradient_clip_groups(self):
        """Keep head clipping independent of the flow/observation gradient norm."""
        heading = [parameter for parameter in self.obs_encoder.head.parameters() if parameter.requires_grad]
        heading_ids = {id(parameter) for parameter in heading}
        core = [parameter for parameter in self.parameters()
                if parameter.requires_grad and id(parameter) not in heading_ids]
        return {"core": core, "heading": heading}
