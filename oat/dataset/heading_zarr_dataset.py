"""Full-window heading dataset with normalization fitted on training episodes.

Sampling is inherited from :class:`ZarrDataset`: observations end at the first
predicted action, with edge repetition at episode boundaries. Only statistics
and validation-split bookkeeping differ from that dataset.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np

from oat.common.seq_sampler import SequenceSampler, get_val_mask
from oat.dataset.zarr_dataset import ZarrDataset
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class HeadingZarrDataset(ZarrDataset):
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
        obs_keys, rgb_keys = list(obs_keys), tuple(rgb_keys)
        if len(set(rgb_keys)) != len(rgb_keys):
            raise ValueError("rgb_keys must not contain duplicates")
        if not set(rgb_keys).issubset(obs_keys):
            raise ValueError("Every rgb_key must also appear in obs_keys")
        if max_train_episodes is not None and max_train_episodes < 1:
            raise ValueError("max_train_episodes must be positive when specified")
        super().__init__(
            zarr_path=zarr_path,
            obs_keys=obs_keys,
            action_key=action_key,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )
        for key in rgb_keys:
            array = self.replay_buffer[key]
            if key not in self.numeric_obs_keys or array.shape[-1] != 3:
                raise ValueError(f"RGB observation {key!r} must have three numeric channels last")

        self.rgb_keys = rgb_keys
        self.split_seed = int(seed)
        self.val_ratio = float(val_ratio)
        self._normalization_episode_mask = np.array(self.train_mask, copy=True)
        # Keep the original holdout: unused training episodes are not validation.
        self._validation_episode_mask = get_val_mask(
            self.replay_buffer.n_episodes, val_ratio, seed)
        self._normalization_episode_mask.setflags(write=False)
        self._validation_episode_mask.setflags(write=False)
        lengths = np.diff(np.r_[0, self.replay_buffer.episode_ends[:]])
        self._normalization_step_mask = np.repeat(self._normalization_episode_mask, lengths)
        self._normalization_step_mask.setflags(write=False)
        self._heading_xy_rms = None

    def get_validation_dataset(self):
        validation = copy.copy(self)
        validation.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self._validation_episode_mask,
        )
        # Preserve the base dataset convention: this mask identifies sampled
        # episodes. The independent immutable mask above always fits TRAIN stats.
        validation.train_mask = self._validation_episode_mask.copy()
        return validation

    def _training_values(self, key):
        values = np.asarray(self.replay_buffer[key][:])
        return values[self._normalization_step_mask]

    def get_normalizer(self, mode="limits", **kwargs):
        numeric_keys = [key for key in self.numeric_obs_keys if key not in self.rgb_keys]
        values = {"action": self._training_values(self.action_key)}
        values.update({key: self._training_values(key) for key in numeric_keys})
        normalizer = LinearNormalizer()
        normalizer.fit(data=values, last_n_dims=1, mode=mode, **kwargs)
        for key in self.rgb_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                np.array([[0., 0., 0.], [255., 255., 255.]], dtype=np.float32),
                mode="limits",
            )
        return normalizer

    def get_heading_xy_rms(self) -> float:
        """Raw per-coordinate XY RMS, independent of policy normalization."""
        if self._heading_xy_rms is None:
            actions = self._training_values(self.action_key)
            if actions.ndim != 2 or actions.shape[-1] < 2:
                raise ValueError("Heading statistics require per-frame actions with XY coordinates")
            self._heading_xy_rms = float(np.sqrt(np.mean(actions[:, :2].astype(np.float64) ** 2)))
        return self._heading_xy_rms

    def get_training_metadata(self):
        """JSON-compatible provenance; validation copies report the same fit."""
        ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        lengths = np.diff(np.r_[0, ends])
        train = self._normalization_episode_mask
        validation = self._validation_episode_mask
        return {
            "split_seed": self.split_seed,
            "val_ratio": self.val_ratio,
            "train_episodes": np.flatnonzero(train).tolist(),
            "validation_episodes": np.flatnonzero(validation).tolist(),
            "unused_episodes": np.flatnonzero(~(train | validation)).tolist(),
            "train_episode_count": int(train.sum()),
            "validation_episode_count": int(validation.sum()),
            "train_window_count": int(lengths[train].sum()),
            "validation_window_count": int(lengths[validation].sum()),
            "episode_ends": ends.tolist(),
            "normalizer_fit": "all frames from selected training episodes only; RGB fixed [0,255]",
            "rgb_keys": list(self.rgb_keys),
            "heading_xy_rms": self.get_heading_xy_rms(),
        }
