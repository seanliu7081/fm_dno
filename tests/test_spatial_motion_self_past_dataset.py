"""Causal history alignment, unchanged SMQ targets, and train-only statistics."""

import json

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from oat.common.seq_sampler import downsample_mask, get_val_mask
from oat.dataset.motion_plan_zarr_dataset import MotionPlanZarrDataset
from oat.dataset.spatial_motion_self_past_dataset import SpatialMotionSelfPastDataset


CONTEXT_KEYS = {
    "past_action", "past_mask", "prev_obs", "prev_past_action", "prev_past_mask",
    "prev_window_valid",
}


def write_dataset(path, *, lengths=(1, 9, 25, 4, 18, 21), outlier_episodes=(), action_key="action"):
    lengths = np.asarray(lengths, dtype=np.int64)
    ends = np.cumsum(lengths)
    count = int(ends[-1])
    frame = np.arange(count)
    actions = np.repeat((frame + 1)[:, None], 7, axis=1).astype(np.float32)
    state = np.column_stack((frame, frame + 0.5)).astype(np.float64)
    camera = np.broadcast_to(frame[:, None, None, None], (count, 2, 3, 3)).astype(np.uint8)
    episode = np.repeat(np.arange(len(lengths)), lengths).astype(np.int64)[:, None]
    altered = np.repeat(np.isin(np.arange(len(lengths)), outlier_episodes), lengths)
    actions[altered], state[altered], camera[altered] = 123456.0, -654321.0, 251
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    for key, values in {action_key: actions, "state": state, "camera": camera, "task_uid": episode}.items():
        data.create_dataset(key, data=values)
    root.create_group("meta").create_dataset("episode_ends", data=ends)
    return {"action": actions, "state": state.astype(np.float32), "camera": camera,
            "episode_ends": ends, "task_uid": episode}


def make_dataset(path, cls=SpatialMotionSelfPastDataset, **kwargs):
    return cls(str(path), obs_keys=["state", "camera", "task_uid"], rgb_keys=["camera"],
               n_obs_steps=2, n_action_steps=16, seed=42, val_ratio=1 / 3, **kwargs)


def assert_stats_equal(left, right):
    left_state, right_state = left.get_normalizer().state_dict(), right.get_normalizer().state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        torch.testing.assert_close(left_state[key], right_state[key], rtol=0, atol=0)
    assert left.get_heading_xy_rms() == right.get_heading_xy_rms()


def test_all_windows_match_baseline_and_history_never_crosses_episodes(tmp_path):
    path = tmp_path / "windows.zarr"
    source = write_dataset(path)
    dataset = make_dataset(path)
    baseline = make_dataset(path, cls=MotionPlanZarrDataset)
    starts = np.r_[0, source["episode_ends"][:-1]]
    for split, reference in ((dataset, baseline),
                             (dataset.get_validation_dataset(), baseline.get_validation_dataset())):
        index = 0
        for episode in np.flatnonzero(split.train_mask):
            start, end = starts[episode], source["episode_ends"][episode]
            for anchor in range(start, end):
                sample, expected = split[index], reference[index]
                for key in ("action", "action_mask"):
                    torch.testing.assert_close(sample[key], expected[key], rtol=0, atol=0)
                for key in ("state", "camera", "task_uid"):
                    torch.testing.assert_close(sample["obs"][key], expected["obs"][key], rtol=0, atol=0)
                for prefix, history_anchor in (("", anchor), ("prev_", anchor - 8)):
                    frames = np.arange(history_anchor - 7, history_anchor)
                    valid = frames >= start
                    history = np.zeros((7, 7), dtype=np.float32)
                    history[valid] = source["action"][frames[valid]]
                    np.testing.assert_array_equal(sample[prefix + "past_action"], history)
                    np.testing.assert_array_equal(sample[prefix + "past_mask"], valid)
                    assert sample[prefix + "past_mask"].dtype == torch.bool
                previous_frames = np.clip([anchor - 9, anchor - 8], start, end - 1)
                for key in ("state", "camera", "task_uid"):
                    np.testing.assert_array_equal(sample["prev_obs"][key], source[key][previous_frames])
                assert sample["prev_window_valid"].dtype == torch.bool
                assert sample["prev_window_valid"].item() == (anchor - start >= 8)
                context = sample["obs"]["_p2n_context"]
                assert set(context) == CONTEXT_KEYS
                assert "_p2n_context" not in context["prev_obs"]
                assert "action" not in context and "action_mask" not in context
                for key in CONTEXT_KEYS:
                    assert context[key] is sample[key]
                index += 1
        assert index == len(split) == len(reference)


def test_execution_slice_exactly_matches_current_history_for_expert_previous_prediction(tmp_path):
    path = tmp_path / "alignment.zarr"
    source = write_dataset(path)
    dataset = make_dataset(path)
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["prev_window_valid"]:
            anchor, _, end = dataset._sample_anchor(index)
            previous_prediction = source["action"][np.minimum(anchor - 8 + np.arange(16), end - 1)]
            np.testing.assert_array_equal(previous_prediction[1:8], sample["past_action"])
            assert sample["past_mask"].all()


def test_context_collates_without_recursion_or_targets(tmp_path):
    path = tmp_path / "batch.zarr"
    write_dataset(path)
    batch = next(iter(DataLoader(make_dataset(path), batch_size=4)))
    context = batch["obs"]["_p2n_context"]
    assert set(context) == CONTEXT_KEYS
    assert context["past_action"].shape == (4, 7, 7)
    assert context["past_mask"].shape == (4, 7)
    assert context["prev_obs"]["camera"].shape == (4, 2, 2, 3, 3)
    assert context["prev_window_valid"].shape == (4,)
    for key in ("past_action", "past_mask", "prev_past_action", "prev_past_mask", "prev_window_valid"):
        torch.testing.assert_close(context[key], batch[key])


def test_split_statistics_and_holdout_are_inherited_without_leakage(tmp_path):
    clean_path, outlier_path = tmp_path / "clean.zarr", tmp_path / "outliers.zarr"
    validation_mask = get_val_mask(6, 1 / 3, 42)
    train_mask = downsample_mask(~validation_mask, 1, 42)
    write_dataset(clean_path)
    write_dataset(outlier_path, outlier_episodes=np.flatnonzero(~train_mask))
    clean = make_dataset(clean_path, max_train_episodes=1)
    outliers = make_dataset(outlier_path, max_train_episodes=1)
    validation = outliers.get_validation_dataset()
    baseline = make_dataset(clean_path, cls=MotionPlanZarrDataset, max_train_episodes=1)
    assert_stats_equal(clean, baseline)
    assert_stats_equal(clean, outliers)
    assert_stats_equal(outliers, validation)
    np.testing.assert_array_equal(outliers.train_mask, train_mask)
    np.testing.assert_array_equal(validation.train_mask, validation_mask)
    np.testing.assert_array_equal(validation.get_validation_dataset().train_mask, validation_mask)
    metadata = outliers.get_training_metadata()
    assert metadata == validation.get_training_metadata()
    assert metadata["past_n"] == 7 and metadata["n_exec_steps"] == 8
    assert metadata["train_window_count"] == len(outliers)
    assert metadata["validation_window_count"] == len(validation)
    assert json.loads(json.dumps(metadata)) == metadata


def test_getitem_reads_only_current_and_previous_observation_frames(tmp_path, monkeypatch):
    path = tmp_path / "reads.zarr"
    write_dataset(path)
    dataset = make_dataset(path)
    # Sampling through the baseline full sequence would load images for all
    # future action slots, despite requiring only two observation frames.
    def reject_sequence_read(*args, **kwargs):
        pytest.fail("self-past must not read the full sampler observation horizon")
    monkeypatch.setattr(dataset.seq_sampler, "sample_sequence", reject_sequence_read)
    camera = dataset.replay_buffer["camera"]
    reads = []

    class ObservedArray:
        shape = camera.shape
        def __getitem__(self, index):
            reads.append(np.asarray(index))
            return camera[index]

    dataset.replay_buffer.root["data"]["camera"] = ObservedArray()
    sample = dataset[len(dataset) // 2]
    assert sample["obs"]["camera"].shape[0] == 2
    assert len(reads) == 2
    assert all(len(indices) == 2 for indices in reads)


def test_nondefault_history_stride_and_action_key(tmp_path):
    path = tmp_path / "custom.zarr"
    source = write_dataset(path, action_key="control")
    dataset = make_dataset(path, action_key="control", past_n=3, n_exec_steps=4)
    for index in range(len(dataset)):
        item = dataset[index]
        anchor, start, _ = dataset._sample_anchor(index)
        frames = np.arange(anchor - 3, anchor)
        valid = frames >= start
        assert item["past_action"].shape == (3, 7)
        np.testing.assert_array_equal(item["past_action"][valid], source["action"][frames[valid]])
        assert item["prev_window_valid"].item() == (anchor - start >= 4)
    assert "action" in dataset.get_normalizer().params_dict
    assert "control" not in dataset.get_normalizer().params_dict


@pytest.mark.parametrize("kwargs,match", [
    ({"past_n": 0}, "must be positive"),
    ({"n_exec_steps": 0}, "must be positive"),
    ({"n_obs_steps": 0}, "must be positive"),
    ({"past_n": 9, "n_exec_steps": 8}, "must not exceed n_exec_steps"),
    ({"n_exec_steps": 17}, "must not exceed the prediction horizon"),
    ({"obs_keys": ["_p2n_context"]}, "reserved"),
])
def test_invalid_configuration_fails_before_loading(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpatialMotionSelfPastDataset("unused.zarr", **kwargs)
