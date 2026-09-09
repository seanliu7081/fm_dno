"""Episode leakage and full-window checks for the heading training dataset."""

import json

import numpy as np
import pytest
import torch
import zarr

from oat.common.seq_sampler import downsample_mask, get_val_mask
from oat.dataset.heading_zarr_dataset import HeadingZarrDataset


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_dataset(path, *, lengths=(3, 5, 4, 2, 6, 3), outlier_episodes=(), action_key="action"):
    lengths = np.asarray(lengths, dtype=np.int64)
    ends = np.cumsum(lengths)
    n = int(ends[-1])
    actions = (np.arange(n * 7, dtype=np.float32).reshape(n, 7) - 31) / 17
    states = np.column_stack((np.arange(n), -np.arange(n), np.ones(n))).astype(np.float32)
    task = np.repeat(np.arange(len(lengths)), lengths).astype(np.int64)[:, None]
    # Tiny HWC RGB arrays. Fixed bounds must not depend on their actual values.
    rgb = np.full((n, 2, 3, 3), 113, dtype=np.uint8)
    altered = np.repeat(np.isin(np.arange(len(lengths)), outlier_episodes), lengths)
    actions[altered] = 123456.0
    states[altered] = -654321.0
    rgb[altered] = 251
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    for key, value in {action_key: actions, "state": states, "task_uid": task, "camera": rgb}.items():
        data.create_dataset(key, data=value)
    root.create_group("meta").create_dataset("episode_ends", data=ends)
    return {"action": actions, "state": states, "camera": rgb, "episode_ends": ends}


def make_dataset(path, **kwargs):
    return HeadingZarrDataset(
        zarr_path=str(path), obs_keys=["state", "task_uid", "camera"],
        rgb_keys=["camera"], n_obs_steps=2, n_action_steps=4,
        seed=42, val_ratio=1 / 3, **kwargs,
    )


def assert_normalizers_equal(left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


def test_validation_outliers_do_not_change_training_statistics(tmp_path):
    clean_path, outlier_path = tmp_path / "clean.zarr", tmp_path / "outliers.zarr"
    original = write_dataset(clean_path)
    validation_mask = get_val_mask(6, 1 / 3, 42)
    write_dataset(outlier_path, outlier_episodes=np.flatnonzero(validation_mask))
    clean, outliers = make_dataset(clean_path), make_dataset(outlier_path)
    fitted = clean.get_normalizer()
    assert_normalizers_equal(fitted, outliers.get_normalizer())
    assert clean.get_heading_xy_rms() == outliers.get_heading_xy_rms()
    frame_mask = np.repeat(~validation_mask, np.diff(np.r_[0, original["episode_ends"]]))
    train_actions = original["action"][frame_mask]
    expected_rms = np.sqrt(np.mean(train_actions[:, :2].astype(np.float64) ** 2))
    assert clean.get_heading_xy_rms() == expected_rms
    for key in ("action", "state"):
        stats = fitted[key].get_input_stats()
        expected = original[key][frame_mask]
        np.testing.assert_array_equal(stats["min"].numpy(), expected.min(axis=0))
        np.testing.assert_array_equal(stats["max"].numpy(), expected.max(axis=0))


def test_rgb_uses_fixed_three_channel_bounds_without_fitting_image_values(tmp_path, monkeypatch):
    path = tmp_path / "rgb.zarr"
    write_dataset(path)
    dataset = make_dataset(path)
    numeric_values = dataset._training_values

    def prohibit_image_statistics(key):
        assert key != "camera", "Image pixels must never be read for normalization statistics"
        return numeric_values(key)

    monkeypatch.setattr(dataset, "_training_values", prohibit_image_statistics)
    normalizer = dataset.get_normalizer()
    rgb = normalizer["camera"]
    stats = rgb.get_input_stats()
    torch.testing.assert_close(stats["min"], torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(stats["max"], torch.full((3,), 255.), rtol=0, atol=0)
    torch.testing.assert_close(rgb.normalize(torch.tensor([[0., 127.5, 255.]])),
                               torch.tensor([[-1., 0., 1.]]), rtol=0, atol=1e-7)
    assert dataset[0]["obs"]["camera"].dtype == torch.uint8
    assert dataset[0]["obs"]["camera"].shape == (2, 2, 3, 3)


def test_all_episode_frames_have_correct_observation_and_action_alignment(tmp_path):
    path = tmp_path / "windows.zarr"
    source = write_dataset(path)
    dataset = make_dataset(path)
    ends = source["episode_ends"]
    starts = np.r_[0, ends[:-1]]
    for split in (dataset, dataset.get_validation_dataset()):
        index = 0
        for episode in np.flatnonzero(split.train_mask):
            for frame in range(starts[episode], ends[episode]):
                sample = split[index]
                obs_frames = np.maximum([frame - 1, frame], starts[episode])
                action_frames = np.minimum(frame + np.arange(4), ends[episode] - 1)
                np.testing.assert_array_equal(sample["obs"]["state"], source["state"][obs_frames])
                np.testing.assert_array_equal(sample["action"], source["action"][action_frames])
                index += 1
        assert len(split) == index
    assert len(dataset) + len(dataset.get_validation_dataset()) == ends[-1]


def test_validation_copy_keeps_training_stats_and_original_holdout_after_subsampling(tmp_path):
    path = tmp_path / "subsampled.zarr"
    val_mask = get_val_mask(6, 1 / 3, 42)
    train_mask = downsample_mask(~val_mask, 1, 42)
    source = write_dataset(path, outlier_episodes=np.flatnonzero(~train_mask))
    dataset = make_dataset(path, max_train_episodes=1)
    validation = dataset.get_validation_dataset()
    np.testing.assert_array_equal(dataset.train_mask, train_mask)
    np.testing.assert_array_equal(validation.train_mask, val_mask)
    assert not (dataset.train_mask & validation.train_mask).any()
    assert not ((~(train_mask | val_mask)) & validation.train_mask).any()
    assert_normalizers_equal(dataset.get_normalizer(), validation.get_normalizer())
    assert dataset.get_heading_xy_rms() == validation.get_heading_xy_rms()
    assert dataset.get_training_metadata() == validation.get_training_metadata()
    np.testing.assert_array_equal(validation.get_validation_dataset().train_mask, val_mask)
    metadata = dataset.get_training_metadata()
    assert metadata["train_episode_count"] == 1
    assert metadata["validation_episode_count"] == 2
    assert len(metadata["unused_episodes"]) == 3
    assert metadata["train_window_count"] == len(dataset)
    assert metadata["validation_window_count"] == len(validation)
    assert json.loads(json.dumps(metadata)) == metadata
    with pytest.raises(ValueError, match="read-only"):
        validation._normalization_episode_mask[0] = False
    # Mutating the public sampler mask cannot change the recorded fit population.
    before = dataset.get_normalizer()
    dataset.train_mask[:] = False
    assert_normalizers_equal(before, dataset.get_normalizer())
    assert len(source["action"]) == source["episode_ends"][-1]


def test_default_split_has_450_training_and_50_validation_episodes(tmp_path):
    path = tmp_path / "500_episodes.zarr"
    write_dataset(path, lengths=np.full(500, 2))
    dataset = HeadingZarrDataset(str(path), obs_keys=["state", "camera"], rgb_keys=["camera"])
    metadata = dataset.get_training_metadata()
    assert metadata["split_seed"] == 42
    assert metadata["train_episode_count"] == 450
    assert metadata["validation_episode_count"] == 50
    assert len(dataset) == 900
    assert len(dataset.get_validation_dataset()) == 100
    assert metadata["unused_episodes"] == []


def test_custom_action_key_fits_canonical_action_normalizer(tmp_path):
    path = tmp_path / "controls.zarr"
    source = write_dataset(path, action_key="control")
    dataset = make_dataset(path, action_key="control")
    normalizer = dataset.get_normalizer()
    assert "action" in normalizer.params_dict
    assert "control" not in normalizer.params_dict
    episode = np.flatnonzero(dataset.train_mask)[0]
    frame = np.r_[0, source["episode_ends"][:-1]][episode]
    np.testing.assert_array_equal(dataset[0]["action"][0], source["action"][frame])


@pytest.mark.parametrize("kwargs,match", [
    ({"obs_keys": ["state"], "rgb_keys": ["camera"]}, "Every rgb_key"),
    ({"obs_keys": ["camera"], "rgb_keys": ["camera", "camera"]}, "duplicates"),
    ({"max_train_episodes": 0}, "must be positive"),
])
def test_invalid_dataset_configuration_fails_before_loading(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HeadingZarrDataset("unused.zarr", **kwargs)
