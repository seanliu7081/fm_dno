"""Train-split SMQ windows with causal action history and one previous window.

The current observation/action anchor is exactly MotionPlanZarrDataset's anchor.
Only actual actions preceding that anchor may enter history. Episode-start
history is zero-filled and explicitly masked, never endpoint-repeated.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from oat.dataset.motion_plan_zarr_dataset import MotionPlanZarrDataset


class SpatialMotionSelfPastDataset(MotionPlanZarrDataset):
    """Add self-past inputs without loading images across the action horizon.

    For current frame t, history covers [t-P, t), previous observations end at
    t-S, and previous history covers [t-S-P, t-S). S is the execution stride,
    not the prediction horizon. A previous window exists only when t >= S.

    ``obs['_p2n_context']`` supplies the same explicit context to the unchanged
    workspace's observation-only reconstruction call. It contains no current
    target actions or target validity mask, and previous observations never
    contain another context mapping. A policy must strip this mapping before
    observation normalization or encoding.
    """

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
        past_n: int = 7,
        n_exec_steps: int = 8,
    ):
        if n_obs_steps < 1 or n_action_steps < 1:
            raise ValueError("n_obs_steps and n_action_steps must be positive")
        if past_n < 1 or n_exec_steps < 1:
            raise ValueError("past_n and n_exec_steps must be positive")
        if past_n > n_exec_steps:
            raise ValueError("past_n must not exceed n_exec_steps")
        if n_exec_steps > n_action_steps:
            raise ValueError("n_exec_steps must not exceed the prediction horizon")
        if "_p2n_context" in obs_keys:
            raise ValueError("_p2n_context is reserved for causal history metadata")
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
        self.past_n = int(past_n)
        self.n_exec_steps = int(n_exec_steps)
        self._episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        self._episode_starts = np.r_[0, self._episode_ends[:-1]]

    def _sample_anchor(self, idx):
        buffer_start, _, sample_start, _ = self.seq_sampler.indices[idx]
        anchor = int(buffer_start - sample_start + self.n_obs_steps - 1)
        episode = int(np.searchsorted(self._episode_ends, anchor, side="right"))
        return anchor, int(self._episode_starts[episode]), int(self._episode_ends[episode])

    def _observations(self, anchor, episode_start, episode_end):
        frames = np.clip(
            np.arange(anchor - self.n_obs_steps + 1, anchor + 1),
            episode_start,
            episode_end - 1,
        )
        obs = {}
        for key in self.numeric_obs_keys:
            values = np.asarray(self.replay_buffer[key][frames])
            if values.dtype.kind == "f":
                values = values.astype(np.float32)
            obs[key] = torch.from_numpy(values)
        for key in self.text_obs_keys:
            value = self.replay_buffer[key][frames[0]]
            obs[key] = value.decode("utf-8") if isinstance(value, bytes) else value
        return obs

    def _history(self, anchor, episode_start):
        frames = np.arange(anchor - self.past_n, anchor)
        valid = frames >= episode_start
        actions = self.replay_buffer[self.action_key]
        values = np.zeros((self.past_n,) + actions.shape[1:], dtype=np.float32)
        if valid.any():
            values[valid] = actions[frames[valid]].astype(np.float32)
        return torch.from_numpy(values), torch.from_numpy(valid)

    def __getitem__(self, idx):
        anchor, episode_start, episode_end = self._sample_anchor(idx)
        obs = self._observations(anchor, episode_start, episode_end)
        target_frames = anchor + np.arange(self.n_action_steps)
        action_mask = torch.from_numpy(target_frames < episode_end)
        action = torch.from_numpy(
            self.replay_buffer[self.action_key][np.minimum(target_frames, episode_end - 1)]
            .astype(np.float32)
        )
        past_action, past_mask = self._history(anchor, episode_start)
        previous_anchor = anchor - self.n_exec_steps
        prev_obs = self._observations(previous_anchor, episode_start, episode_end)
        prev_past_action, prev_past_mask = self._history(previous_anchor, episode_start)
        context = {
            "past_action": past_action,
            "past_mask": past_mask,
            "prev_obs": prev_obs,
            "prev_past_action": prev_past_action,
            "prev_past_mask": prev_past_mask,
            "prev_window_valid": torch.tensor(previous_anchor >= episode_start, dtype=torch.bool),
        }
        obs["_p2n_context"] = context
        return {"obs": obs, "action": action, "action_mask": action_mask, **context}

    def get_training_metadata(self):
        metadata = super().get_training_metadata()
        metadata.update({
            "past_n": self.past_n,
            "n_exec_steps": self.n_exec_steps,
            "history_padding": "zeros with false mask; no cross-episode history",
            "previous_window_valid": "current episode frame >= execution stride",
        })
        return metadata
