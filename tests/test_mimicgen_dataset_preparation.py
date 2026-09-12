"""Check action preservation and held-out state isolation using tiny HDF5 files."""
import json

import cv2
import h5py
import numpy as np
import pytest
import zarr

from scripts.prepare_mimicgen_dataset import convert_episode, prepare_dataset, select_demos, state_sha256


def make_source(root, task="stack_three_d1", count=5, absolute=False):
    source = root / "core" / f"{task}.hdf5"
    source.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "StackThree_D1", "env_kwargs": {
            "controller_configs": {"type": "OSC_POSE", "control_delta": not absolute}}})
        for index in range(count):
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("states", data=np.array([[index, 0.1, 0.2], [index, 0.3, 0.4]]))
            demo.attrs["model_file"] = f"<mujoco model='demo_{index}'/>"
            demo.create_dataset("actions", data=np.arange(14).reshape(2, 7) / 20 - 0.5 + index / 100)
            obs = demo.create_group("obs")
            images = np.arange(36, dtype=np.uint8).reshape(2, 3, 2, 3)
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                obs.create_dataset(key, data=images)
            for key, dim in (("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_gripper_qpos", 2)):
                obs.create_dataset(key, data=np.zeros((2, dim)))
    return source


def test_preserves_delta_actions_image_orientation_and_isolates_eval(tmp_path):
    source = make_source(tmp_path / "source")
    output = tmp_path / "out"
    output.mkdir()  # existing directory is allowed, existing output artifacts are not
    manifest = prepare_dataset(tmp_path / "source", output, ["stack_three_d1"], train_count=2, eval_count=2, image_size=None)
    root = zarr.open(str(output / "mimicgen1_N2.zarr"), mode="r")
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [2, 4])
    with h5py.File(source) as f:
        expected_actions = np.concatenate([f[f"data/demo_{i}/actions"][:] for i in range(2)]).astype(np.float32)
        np.testing.assert_array_equal(root["data/action"][:], expected_actions)
        np.testing.assert_array_equal(root["data/agentview_rgb"][:2], f["data/demo_0/obs/agentview_image"][:])
    np.testing.assert_array_equal(root["data/task_uid"][:], np.zeros((4, 1), dtype=np.int64))
    task = manifest["tasks"][0]
    assert task["train_demo_names"] == ["demo_0", "demo_1"]
    assert task["eval_demo_names"] == ["demo_2", "demo_3"]
    assert not set(task["train_initial_state_sha256"]) & set(task["eval_initial_state_sha256"])
    with h5py.File(output / manifest["eval_hdf5"]) as f:
        assert set(f["data/stack_three_d1"]) == {"demo_2", "demo_3"}
        for index, name in enumerate(task["eval_demo_names"]):
            demo = f[f"data/stack_three_d1/{name}"]
            assert state_sha256(demo["states"][:]) == task["eval_initial_state_sha256"][index]
            assert "<mujoco" in demo.attrs["model_file"]
    assert (output / "mimicgen1_N2.manifest.json").exists()
    with pytest.raises(FileExistsError):
        prepare_dataset(tmp_path / "source", output, ["stack_three_d1"], 2, 2)


def test_skips_repeated_initial_states_in_holdout(tmp_path):
    source = make_source(tmp_path)
    with h5py.File(source, "a") as f:
        f["data/demo_2/states"][0] = f["data/demo_0/states"][0]
        train, tests = select_demos(f["data"], train_count=2, eval_count=2)
    assert train == ["demo_0", "demo_1"]
    assert tests == ["demo_3", "demo_4"]


def test_refuses_absolute_control_and_cleans_partial_output(tmp_path):
    make_source(tmp_path / "source", absolute=True)
    with pytest.raises(ValueError, match="Absolute-control"):
        prepare_dataset(tmp_path / "source", tmp_path / "out", ["stack_three_d1"], 2, 2)
    assert not list(tmp_path.glob("out.partial-*"))
    assert not list((tmp_path / "out").iterdir())


def test_refuses_insufficient_demos(tmp_path):
    source = make_source(tmp_path, count=3)
    with h5py.File(source) as f, pytest.raises(ValueError, match="at least 4"):
        select_demos(f["data"], train_count=2, eval_count=2)


def test_matches_runner_linear_resize(tmp_path):
    source = make_source(tmp_path)
    with h5py.File(source) as f:
        demo = f["data/demo_0"]
        episode = convert_episode(demo, "stack_three_d1", 3, image_size=128)
        expected = cv2.resize(demo["obs/agentview_image"][0], (128, 128), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(episode["agentview_rgb"][0], expected)
        np.testing.assert_array_equal(episode["task_uid"], np.full((2, 1), 3, dtype=np.int64))
