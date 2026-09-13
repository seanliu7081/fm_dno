"""Leakage, temporal alignment, prompt fidelity and provenance for VLA data."""

import json
import os
from pathlib import Path
import pickle

import h5py
import numpy as np
import pytest

from oat.starvla_heading.data import (
    LiberoHDF5Dataset,
    ORIGINAL_SUITES,
    axisangle_to_quaternion,
    canonical_task_map,
    collate_libero_samples,
    denormalize_array,
    load_manifest,
    manifest_sha256,
    normalize_array,
    prepare_libero_manifest,
)


PROMPT = ("Pick up the yellow and white mug beside the large plate, move it behind "
          "the red bowl, open the upper cabinet drawer, and place the mug inside "
          "before closing the drawer completely.")


def write_task(root, task="fixture_task", *, suite="libero_10", lengths=(3, 5, 4, 2),
               outlier_episodes=(), prompt=PROMPT):
    path = Path(root) / suite / f"{task}_demo.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = []
    with h5py.File(path, "w") as source:
        data = source.create_group("data")
        data.attrs["bddl_file_name"] = f"/original/libero/{suite}/{task}.bddl"
        data.attrs["problem_info"] = json.dumps({"language_instruction": prompt})
        data.attrs["num_demos"] = len(lengths)
        for episode, length in enumerate(lengths):
            demo = data.create_group(f"demo_{episode}")
            obs = demo.create_group("obs")
            actions = (np.arange(length * 7).reshape(length, 7) + episode * 10).astype(np.float32)
            position = (np.arange(length * 3).reshape(length, 3) + episode).astype(np.float32)
            gripper = np.full((length, 2), 0.01 + episode * 0.001, dtype=np.float32)
            if episode in outlier_episodes:
                actions[:] = 1e6
                position[:] = -2e6
                gripper[:] = 1e4
            orientation = np.zeros((length, 3), dtype=np.float32)
            orientation[:, 2] = np.pi / 2
            obs.create_dataset("ee_pos", data=position)
            obs.create_dataset("ee_ori", data=orientation)
            obs.create_dataset("gripper_states", data=gripper)
            demo.create_dataset("actions", data=actions)
            # Pixel value encodes camera, frame, and row independently.
            cameras = {}
            for offset, key in ((0, "agentview_rgb"), (100, "eye_in_hand_rgb")):
                pixels = np.zeros((length, 3, 4, 3), dtype=np.uint8)
                for t in range(length):
                    for row in range(3):
                        pixels[t, row] = offset + t * 10 + row
                obs.create_dataset(key, data=pixels)
                cameras[key] = pixels
            quat = axisangle_to_quaternion(orientation)
            original.append({"action": actions, "state": np.concatenate((position, quat, gripper), axis=-1),
                             **cameras})
    return path, original


def prepare(root, output, **kwargs):
    return prepare_libero_manifest(root, output, suites=("libero_10",),
                                   task_map={"libero_10": ("fixture_task",)},
                                   expected_demos_per_task=4, val_ratio=0.25, **kwargs)


def test_validation_outliers_never_affect_normalizer_or_heading_rms(tmp_path):
    clean_root, altered_root = tmp_path / "clean", tmp_path / "altered"
    _, raw = write_task(clean_root)
    clean = prepare(clean_root, tmp_path / "clean.json")
    holdout = [int(item["demo"].split("_")[-1]) for item in clean["episodes"]
               if item["split"] == "val"]
    write_task(altered_root, outlier_episodes=holdout)
    altered = prepare(altered_root, tmp_path / "altered.json")
    assert clean["statistics"] == altered["statistics"]
    assert clean["training_numeric_sha256"] == altered["training_numeric_sha256"]
    assert clean["episodes"] == altered["episodes"]
    train = [raw[index] for index in range(len(raw)) if index not in holdout]
    for key in ("action", "state"):
        values = np.concatenate([item[key] for item in train])
        stats = clean["statistics"][key]
        np.testing.assert_array_equal(stats["min"], values.min(axis=0))
        np.testing.assert_array_equal(stats["max"], values.max(axis=0))
        np.testing.assert_allclose(denormalize_array(normalize_array(values, stats), stats),
                                   values, atol=1e-6)
        normalized = normalize_array(values, stats)
        assert np.abs(normalized).max() <= 1.00001
    actions = np.concatenate([item["action"] for item in train])
    assert clean["statistics"]["heading_xy_rms"] == np.sqrt(np.mean(actions[:, :2].astype(np.float64)**2))


def test_every_frame_preserves_history_action_alignment_cameras_and_full_prompt(tmp_path):
    _, raw = write_task(tmp_path / "source")
    manifest_path = tmp_path / "data.json"
    manifest = prepare(tmp_path / "source", manifest_path)
    total = 0
    for split in ("train", "val"):
        dataset = LiberoHDF5Dataset(manifest_path, split, horizon=4, image_size=None)
        assert dataset.statistics == manifest["statistics"]
        index = 0
        for episode in dataset.episodes:
            source = raw[int(episode["demo"].split("_")[-1])]
            for timestep in range(episode["length"]):
                sample = dataset[index]
                assert sample["lang"] == PROMPT and len(sample["lang"]) > 128
                assert sample["episode_id"] == episode["episode_id"]
                assert sample["timestep"] == timestep
                assert "task_uid" not in sample
                history = np.maximum([timestep - 1, timestep], 0)
                predicted = np.minimum(timestep + np.arange(4), episode["length"] - 1)
                np.testing.assert_array_equal(sample["action"], source["action"][predicted])
                np.testing.assert_array_equal(sample["state"], source["state"][history])
                np.testing.assert_array_equal(sample["action_mask"],
                                              timestep + np.arange(4) < episode["length"])
                assert len(sample["image"]) == 4
                for image_index, frame in enumerate(history):
                    for camera_index, camera in enumerate(("agentview_rgb", "eye_in_hand_rgb")):
                        image = sample["image"][image_index * 2 + camera_index]
                        assert image.mode == "RGB"
                        np.testing.assert_array_equal(np.asarray(image), source[camera][frame, ::-1])
                index += 1
        assert len(dataset) == index
        assert dataset[-1]["timestep"] == dataset[index - 1]["timestep"]
        with pytest.raises(IndexError):
            dataset[len(dataset)]
        dataset.close()
        total += index
    assert total == sum(len(item["action"]) for item in raw)


def test_default_split_is_per_task_45_train_5_validation_and_order_independent(tmp_path):
    root = tmp_path / "source"
    tasks = ["first_task", "second_task"]
    for task in reversed(tasks):
        write_task(root, task, lengths=[1] * 50)
    kwargs = {"suites": ("libero_10",), "task_map": {"libero_10": tasks}}
    first = prepare_libero_manifest(root, tmp_path / "first.json", **kwargs)
    second = prepare_libero_manifest(root, tmp_path / "second.json", **kwargs)
    assert first == second
    assert first["sha256"] == manifest_sha256(first)
    assert first["train_episode_count"] == 90
    assert first["val_episode_count"] == 10
    for task_index in range(2):
        task_episodes = [item for item in first["episodes"] if item["task_index"] == task_index]
        assert sum(item["split"] == "train" for item in task_episodes) == 45
        assert sum(item["split"] == "val" for item in task_episodes) == 5
    assert len({item["episode_id"] for item in first["episodes"]}) == 100


def test_manifest_is_immutable_hash_checked_and_source_changes_are_rejected(tmp_path):
    source_path, _ = write_task(tmp_path / "source")
    path = tmp_path / "data.json"
    manifest = prepare(tmp_path / "source", path)
    with pytest.raises(FileExistsError, match="immutable"):
        prepare(tmp_path / "source", path)
    assert load_manifest(path) == manifest
    edited = dict(manifest)
    edited["seed"] += 1
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="hash mismatch"):
        LiberoHDF5Dataset(tampered)
    identity = source_path.stat()
    os.utime(source_path, ns=(identity.st_atime_ns, identity.st_mtime_ns + 1_000_000))
    dataset = LiberoHDF5Dataset(path)
    with pytest.raises(RuntimeError, match="Source differs"):
        dataset[0]


def test_worker_pickling_discards_open_hdf5_handles_and_images_resize(tmp_path):
    write_task(tmp_path / "source")
    path = tmp_path / "data.json"
    prepare(tmp_path / "source", path)
    original = LiberoHDF5Dataset(path, image_size=224)
    sample = original[0]
    assert all(image.size == (224, 224) for image in sample["image"])
    assert original._handles
    restored = pickle.loads(pickle.dumps(original))
    assert not restored._handles
    restored_sample = restored[0]
    np.testing.assert_array_equal(restored_sample["state"], sample["state"])
    np.testing.assert_array_equal(restored_sample["action"], sample["action"])
    assert collate_libero_samples([sample, restored_sample]) == [sample, restored_sample]
    original.close()
    restored.close()


def test_axisangle_matches_xyzw_and_handles_zero_without_nan():
    angles = np.array([[0, 0, 0], [0, 0, np.pi], [np.pi / 2, 0, 0], [1e-20, 0, 0]])
    result = axisangle_to_quaternion(angles)
    np.testing.assert_allclose(result[0], [0, 0, 0, 1])
    np.testing.assert_allclose(result[1], [0, 0, 1, 0], atol=1e-7)
    np.testing.assert_allclose(result[2], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1, atol=1e-7)
    assert np.isfinite(result).all()


def test_production_task_map_has_all_40_original_tasks():
    mapping = canonical_task_map()
    assert set(mapping) == set(ORIGINAL_SUITES)
    assert all(len(tasks) == 10 for tasks in mapping.values())
    assert len({task for tasks in mapping.values() for task in tasks}) == 40


def test_missing_suites_plus_data_and_wrong_demo_counts_fail_early(tmp_path):
    root = tmp_path / "source"
    write_task(root)
    with pytest.raises(FileNotFoundError, match="Missing 40 canonical"):
        prepare_libero_manifest(root, tmp_path / "missing.json")
    with pytest.raises(ValueError, match="original LIBERO suites"):
        prepare_libero_manifest(root, tmp_path / "plus.json", suites=("libero_plus",))
    with pytest.raises(ValueError, match="expected 50 demonstrations"):
        prepare_libero_manifest(root, tmp_path / "counts.json", suites=("libero_10",),
                                task_map={"libero_10": ["fixture_task"]})
    plus_root = tmp_path / "LIBERO-plus"
    write_task(plus_root)
    with pytest.raises(ValueError, match="cannot be used"):
        prepare(plus_root, tmp_path / "plus_source.json")


@pytest.mark.parametrize("defect,match", [("prompt", "instruction"), ("actions", "action shape"),
                                        ("state", "length or width"), ("camera", "RGB observation")])
def test_malformed_hdf5_contract_is_rejected(tmp_path, defect, match):
    source_path, _ = write_task(tmp_path / "source")
    with h5py.File(source_path, "a") as source:
        data, demo = source["data"], source["data/demo_0"]
        if defect == "prompt":
            data.attrs["problem_info"] = json.dumps({"language_instruction": ""})
        elif defect == "actions":
            del demo["actions"]
            demo.create_dataset("actions", data=np.zeros((3, 8)))
        elif defect == "state":
            del demo["obs/ee_ori"]
            demo["obs"].create_dataset("ee_ori", data=np.zeros((2, 3)))
        else:
            del demo["obs/agentview_rgb"]
            demo["obs"].create_dataset("agentview_rgb", data=np.zeros((3, 2, 2, 3)))
    with pytest.raises(ValueError, match=match):
        prepare(tmp_path / "source", tmp_path / "data.json")
