"""Held-out conversion preserves the approved split and delta-action targets."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from scripts.prepare_mimicgen_action_mse import (
    array_sha256, file_sha256, prepare_action_mse_dataset,
)
from scripts.prepare_mimicgen_dataset import prepare_dataset


def make_split(tmp_path):
    source_dir = tmp_path / "source"
    (source_dir / "core").mkdir(parents=True)
    for task_name in ("stack_three_d1", "square_d2"):
        with h5py.File(source_dir / "core" / f"{task_name}.hdf5", "w") as source:
            data = source.create_group("data")
            data.attrs["env_args"] = json.dumps({"env_kwargs": {"controller_configs": {
                "type": "OSC_POSE", "control_delta": True}}})
            for index in range(5):
                demo = data.create_group(f"demo_{index}")
                demo.attrs["model_file"] = "<mujoco/>"
                demo.create_dataset("states", data=np.array([[index, .1], [index, .2]]))
                demo.create_dataset("actions", data=np.full((2, 7), index / 10, dtype=np.float64))
                obs = demo.create_group("obs")
                images = np.arange(36, dtype=np.uint8).reshape(2, 3, 2, 3)
                for key in ("agentview_image", "robot0_eye_in_hand_image"):
                    obs.create_dataset(key, data=images)
                for key, dim in (("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_gripper_qpos", 2)):
                    obs.create_dataset(key, data=np.zeros((2, dim)))
    split_dir = tmp_path / "split"
    prepare_dataset(source_dir, split_dir, ["stack_three_d1", "square_d2"], 2, 2, image_size=None)
    return split_dir / "mimicgen2_N2.manifest.json"


def test_converts_exact_manifest_order_and_keeps_training_untouched(tmp_path):
    split_path = make_split(tmp_path)
    split = json.loads(split_path.read_text())
    # The original manifest is authoritative, even if held-out IDs are reordered.
    for task in split["tasks"]:
        task["eval_demo_names"].reverse()
        task["eval_initial_state_sha256"].reverse()
    split_path.write_text(json.dumps(split))
    original_split = split_path.read_bytes()
    train = zarr.open(str(split_path.parent / split["zarr_path"]), mode="r")
    original_train_actions = train["data/action"][:].copy()
    original_eval_sha256 = file_sha256(split_path.parent / split["eval_hdf5"])
    output = tmp_path / "heldout.zarr"
    result = prepare_action_mse_dataset(split_path, output)
    heldout = zarr.open(str(output), mode="r")
    assert result["total_episodes"] == 4
    assert result["total_steps"] == 8
    assert heldout.attrs["split"] == "held_out_action_mse"
    assert result["source_split_manifest_sha256"] == file_sha256(split_path)
    np.testing.assert_array_equal(heldout["meta/episode_ends"][:], [2, 4, 6, 8])
    assert result["episode_ends_sha256"] == array_sha256(heldout["meta/episode_ends"][:])
    for task, expected in zip(result["tasks"], split["tasks"]):
        assert task["eval_demo_names"] == expected["eval_demo_names"] == ["demo_3", "demo_2"]
        assert not set(task["eval_demo_names"]) & set(expected["train_demo_names"])
        assert task["source_hdf5_sha256"] == file_sha256(Path(task["source_hdf5"]))
        with h5py.File(task["source_hdf5"]) as source:
            for episode in task["episodes"]:
                start, end = episode["start"], episode["end"]
                targets = source[f"data/{episode['demo_name']}/actions"][:].astype(np.float32)
                np.testing.assert_array_equal(heldout["data/action"][start:end], targets)
                assert episode["arrays"]["action"]["sha256"] == array_sha256(targets)
                np.testing.assert_array_equal(heldout["data/task_uid"][start:end], np.full((2, 1), task["task_uid"]))
                np.testing.assert_array_equal(heldout["data/agentview_rgb"][start:end], source[f"data/{episode['demo_name']}/obs/agentview_image"][:])
    assert split_path.read_bytes() == original_split
    np.testing.assert_array_equal(train["data/action"][:], original_train_actions)
    assert file_sha256(split_path.parent / split["eval_hdf5"]) == original_eval_sha256
    assert output.with_suffix(".manifest.json").is_file()
    with pytest.raises(FileExistsError):
        prepare_action_mse_dataset(split_path, output)


@pytest.mark.parametrize("corruption,match", [
    ("overlap", "overlap training"),
    ("initial_state", "evaluation initial states differ"),
    ("source_hash", "source HDF5 SHA256 differs"),
])
def test_refuses_split_or_source_corruption_and_cleans_stage(tmp_path, corruption, match):
    split_path = make_split(tmp_path)
    split = json.loads(split_path.read_text())
    if corruption == "overlap":
        split["tasks"][0]["eval_demo_names"][0] = "demo_0"
    elif corruption == "initial_state":
        split["tasks"][0]["eval_initial_state_sha256"][0] = "not-the-approved-state"
    else:
        split["source_download"] = {"sources": [{"task_name": "stack_three_d1", "sha256": "0" * 64}]}
    split_path.write_text(json.dumps(split))
    output = tmp_path / "heldout.zarr"
    with pytest.raises(ValueError, match=match):
        prepare_action_mse_dataset(split_path, output)
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not list(tmp_path.glob("heldout.partial-*"))
