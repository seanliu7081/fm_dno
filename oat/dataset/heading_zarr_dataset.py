"""Train-split normalized dataset with heading-specific action statistics."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from oat.dataset.train_split_zarr_dataset import TrainSplitZarrDataset


class HeadingZarrDataset(TrainSplitZarrDataset):
    def __init__(
        self,
        zarr_path: str,
        obs_keys: Sequence[str] = (),
        action_key: str = "action",
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
        rgb_keys: Sequence[str] = (),
    ):
        super().__init__(
            zarr_path=zarr_path,
            obs_keys=obs_keys,
            action_key=action_key,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            rgb_keys=rgb_keys,
        )
        self._heading_xy_rms = None

    def get_heading_xy_rms(self) -> float:
        """Raw per-coordinate XY RMS, independent of policy normalization."""
        if self._heading_xy_rms is None:
            actions = self._training_values(self.action_key)
            if actions.ndim != 2 or actions.shape[-1] < 2:
                raise ValueError("Heading statistics require per-frame actions with XY coordinates")
            self._heading_xy_rms = float(np.sqrt(np.mean(actions[:, :2].astype(np.float64) ** 2)))
        return self._heading_xy_rms

    def get_training_metadata(self):
        metadata = super().get_training_metadata()
        metadata["heading_xy_rms"] = self.get_heading_xy_rms()
        return metadata
