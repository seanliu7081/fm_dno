"""Train-only baseline preprocessing without heading-dependent action assumptions."""

import json

import numpy as np
import pytest
import torch
import zarr

from oat.common.seq_sampler import downsample_mask, get_val_mask
from oat.dataset.train_split_zarr_dataset import TrainSplitZarrDataset


def write_dataset(path, action_dim, outlier_episodes=()):
    lengths = np.array([3, 5, 4, 2, 6, 3], dtype=np.int64)
    ends = np.cumsum(lengths)
    count = int(ends[-1])
    actions = np.arange(count * action_dim, dtype=np.float32).reshape(count, action_dim)
    states = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
    rgb = np.full((count, 2, 3, 3), 113, dtype=np.uint8)
    outliers = np.repeat(np.isin(np.arange(len(lengths)), outlier_episodes), lengths)
    actions[outliers], states[outliers], rgb[outliers] = 123456.0, -654321.0, 251
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    for key, values in {"control": actions, "state": states, "camera": rgb}.items():
        data.create_dataset(key, data=values)
    root.create_group("meta").create_dataset("episode_ends", data=ends)
    return actions, states, lengths


def make_dataset(path):
    return TrainSplitZarrDataset(
        str(path), obs_keys=["state", "camera"], rgb_keys=["camera"],
        action_key="control", n_obs_steps=2, n_action_steps=4,
        seed=42, val_ratio=1 / 3, max_train_episodes=1,
    )


@pytest.mark.parametrize("action_dim", [1, 3])
def test_neutral_dataset_has_train_only_stats_without_heading_metadata(tmp_path, action_dim):
    clean_path, outlier_path = tmp_path / "clean.zarr", tmp_path / "outliers.zarr"
    actions, states, lengths = write_dataset(clean_path, action_dim)
    validation_mask = get_val_mask(6, 1 / 3, 42)
    train_mask = downsample_mask(~validation_mask, 1, 42)
    write_dataset(outlier_path, action_dim, np.flatnonzero(~train_mask))
    clean, outliers = make_dataset(clean_path), make_dataset(outlier_path)
    normalizer = clean.get_normalizer()
    expected_state = normalizer.state_dict()
    validation = outliers.get_validation_dataset()
    for candidate in (outliers, validation):
        actual_state = candidate.get_normalizer().state_dict()
        assert actual_state.keys() == expected_state.keys()
        for key, expected in expected_state.items():
            torch.testing.assert_close(actual_state[key], expected, rtol=0, atol=0)
        assert not hasattr(candidate, "get_heading_xy_rms")
        assert not hasattr(candidate, "_heading_xy_rms")

    train_frames = np.repeat(train_mask, lengths)
    for key, values in {"action": actions, "state": states}.items():
        stats = normalizer[key].get_input_stats()
        np.testing.assert_array_equal(stats["min"].numpy(), values[train_frames].min(axis=0))
        np.testing.assert_array_equal(stats["max"].numpy(), values[train_frames].max(axis=0))
    assert "control" not in normalizer.params_dict
    rgb_stats = normalizer["camera"].get_input_stats()
    torch.testing.assert_close(rgb_stats["min"], torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(rgb_stats["max"], torch.full((3,), 255.), rtol=0, atol=0)
    assert clean[0]["action"].shape == (4, action_dim)
    assert clean[0]["obs"]["camera"].dtype == torch.uint8

    np.testing.assert_array_equal(validation.train_mask, validation_mask)
    metadata = clean.get_training_metadata()
    assert metadata == outliers.get_training_metadata() == validation.get_training_metadata()
    assert not any("heading" in key for key in metadata)
    assert metadata["train_episode_count"] == 1
    assert metadata["validation_episode_count"] == 2
    assert len(metadata["unused_episodes"]) == 3
    assert metadata["train_window_count"] == len(clean)
    assert metadata["validation_window_count"] == len(validation)
    assert json.loads(json.dumps(metadata)) == metadata
