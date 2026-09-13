"""Heading dataset with real-action masks for future segment supervision."""

from __future__ import annotations

import numpy as np
import torch

from oat.dataset.heading_zarr_dataset import HeadingZarrDataset


class MotionPlanZarrDataset(HeadingZarrDataset):
    """Keep endpoint-padded actions, identifying their real records separately.

    The mask is an auxiliary-label input only. In particular, it must not be
    supplied to the observation encoder or used to select the flow source.
    Validation copies inherit this method and retain training-only statistics.
    """

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        _, _, sample_start, sample_end = self.seq_sampler.indices[idx]
        # A two-observation / sixteen-action window has seventeen sampler slots;
        # the current observation and first action both occupy slot one.
        slots = np.arange(self.n_action_steps) + max(self.n_obs_steps - 1, 0)
        item["action_mask"] = torch.from_numpy(
            (slots >= sample_start) & (slots < sample_end)
        )
        return item
