"""Convert the existing MimicGen evaluation demonstrations for action MSE.

This does not alter training Zarr, the training split, or simulator start states.
The exact held-out IDs are read from the original split manifest. Conversion
reuses the training conversion function, preserving normalized OSC_POSE deltas.
No action or observation normalizer is fitted on these held-out demonstrations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import h5py
import numpy as np
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oat.common.replay_buffer import ReplayBuffer
from scripts.prepare_mimicgen_dataset import (
    _initial_state, _metadata, convert_episode, state_sha256,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare_action_mse_dataset(split_manifest_path: Path, zarr_path: Path) -> dict:
    """Publish a separate validation-only replay after checking split provenance."""
    split_manifest_path = split_manifest_path.resolve()
    split_bytes = split_manifest_path.read_bytes()
    split = json.loads(split_bytes)
    if split.get("action_mode") != "delta":
        raise ValueError("Action MSE requires the existing delta-action split")
    tasks = split["tasks"]
    if not tasks or len({task["task_uid"] for task in tasks}) != len(tasks):
        raise ValueError("Expected distinct task IDs in the original split")
    if len({task["task_name"] for task in tasks}) != len(tasks):
        raise ValueError("Expected distinct task names in the original split")
    zarr_path = zarr_path.resolve()
    manifest_path = zarr_path.with_suffix(".manifest.json")
    if zarr_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {zarr_path} or {manifest_path}")
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    stage = zarr_path.parent / (zarr_path.stem + ".partial-" + uuid.uuid4().hex[:8])
    stage.mkdir()
    source_records = {
        item["task_name"]: item for item in split.get("source_download", {}).get("sources", [])
    }
    manifest = {
        "schema_version": 1,
        "purpose": "held-out demonstration action MSE; never training or normalizer fitting",
        "selection": "exact eval_demo_names from original training/evaluation split manifest",
        "source_split_manifest": str(split_manifest_path),
        "source_split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "source_download": split.get("source_download"),
        "zarr_path": zarr_path.name,
        "action_mode": "delta",
        "action_description": split.get("action_description"),
        "image_orientation": split.get("image_orientation"),
        "image_size": split["image_size"],
        "image_resize_interpolation": "cv2.INTER_LINEAR",
        "quaternion_order": split.get("quaternion_order"),
        "task_uid_encoding": "scalar integer [T, 1]",
        "array_hash_algorithm": "sha256(np.ascontiguousarray(array).tobytes()); dtype and shape recorded",
        "state_hash_algorithm": "sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes())",
        "tasks": [],
    }
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    published_zarr = False
    try:
        replay = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(stage / zarr_path.name)))
        expected_keys = None
        for task in tasks:
            task_name, task_uid = task["task_name"], int(task["task_uid"])
            names = task["eval_demo_names"]
            train_names = task["train_demo_names"]
            if (not names or len(set(names)) != len(names)
                    or set(names) & set(train_names)):
                raise ValueError(f"{task_name}: evaluation IDs are empty, duplicated, or overlap training")
            if len(names) != split["eval_tests_per_task"]:
                raise ValueError(f"{task_name}: evaluation demo count changed")
            if len(train_names) != split["train_demos_per_task"]:
                raise ValueError(f"{task_name}: training demo count changed")
            source_path = Path(task["source_hdf5"])
            if not source_path.is_absolute():
                source_path = split_manifest_path.parent / source_path
            source_path = source_path.resolve()
            print(f"{task_name}: verifying source SHA256", flush=True)
            source_sha256 = file_sha256(source_path)
            expected_sha256 = source_records.get(task_name, {}).get("sha256")
            if expected_sha256 and expected_sha256 != source_sha256:
                raise ValueError(f"{task_name}: source HDF5 SHA256 differs from original download")
            task_record = {
                "task_name": task_name, "task_uid": task_uid,
                "source_hdf5": str(source_path), "source_hdf5_sha256": source_sha256,
                "source_hdf5_bytes": source_path.stat().st_size,
                "eval_demo_names": names,
                "eval_initial_state_sha256": [], "episode_indices": [],
                "episodes": [], "total_steps": 0,
            }
            with h5py.File(source_path, "r") as source:
                data = source["data"]
                _metadata(data)  # rejects absolute control
                train_hashes = [state_sha256(_initial_state(data[name])) for name in train_names]
                if train_hashes != task["train_initial_state_sha256"]:
                    raise ValueError(f"{task_name}: training initial states differ from the original split")
                eval_hashes = [state_sha256(_initial_state(data[name])) for name in names]
                if eval_hashes != task["eval_initial_state_sha256"]:
                    raise ValueError(f"{task_name}: evaluation initial states differ from the original split")
                if set(train_hashes) & set(eval_hashes) or len(set(eval_hashes)) != len(eval_hashes):
                    raise ValueError(f"{task_name}: evaluation initial states overlap or are duplicated")
                task_record["eval_initial_state_sha256"] = eval_hashes
                for i, name in enumerate(names):
                    episode = convert_episode(data[name], task_name, task_uid, image_size=split["image_size"])
                    if expected_keys is None:
                        expected_keys = set(episode)
                    if set(episode) != expected_keys:
                        raise ValueError(f"{task_name}/{name}: inconsistent observation keys")
                    shapes = {key: list(value.shape[1:]) for key, value in episode.items()}
                    if shapes != split["observation_shapes"]:
                        raise ValueError(f"{task_name}/{name}: observation shapes differ from training")
                    arrays = {key: {"sha256": array_sha256(value), "shape": list(value.shape), "dtype": str(value.dtype)}
                              for key, value in episode.items()}
                    episode_index = replay.n_episodes
                    start = int(replay.n_steps)
                    replay.add_episode(episode, compressors=compressor)
                    task_record["episode_indices"].append(episode_index)
                    task_record["total_steps"] += len(episode["action"])
                    task_record["episodes"].append({
                        "demo_name": name, "episode_index": episode_index,
                        "start": start, "end": int(replay.n_steps),
                        "length": len(episode["action"]),
                        "initial_state_sha256": eval_hashes[i], "arrays": arrays,
                    })
                    if (i + 1) % 10 == 0 or i + 1 == len(names):
                        print(f"{task_name}: converted {i + 1}/{len(names)} held-out demos", flush=True)
            manifest["tasks"].append(task_record)
        manifest["total_episodes"] = replay.n_episodes
        manifest["total_steps"] = int(replay.n_steps)
        manifest["observation_shapes"] = {key: list(value.shape[1:]) for key, value in replay.data.items()}
        manifest["episode_ends_sha256"] = array_sha256(replay.episode_ends[:])
        replay.root.attrs.update({
            "action_mode": "delta", "split": "held_out_action_mse",
            "manifest": f"../{manifest_path.name}",
            "source_split_manifest_sha256": manifest["source_split_manifest_sha256"],
            "task_names": [task["task_name"] for task in tasks],
            "num_demos_per_task": split["eval_tests_per_task"],
        })
        (stage / manifest_path.name).write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(stage / zarr_path.name, zarr_path)
        published_zarr = True
        os.replace(stage / manifest_path.name, manifest_path)  # completion marker
        stage.rmdir()
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        if published_zarr and not manifest_path.exists():
            shutil.rmtree(zarr_path)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=Path("data/mimicgen/mimicgen6_N100.manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/mimicgen/mimicgen6_action_mse.zarr"))
    args = parser.parse_args()
    manifest = prepare_action_mse_dataset(args.split_manifest, args.output)
    print(f"Prepared {manifest['total_episodes']} held-out demos / {manifest['total_steps']} steps at {args.output}", flush=True)


if __name__ == "__main__":
    main()
