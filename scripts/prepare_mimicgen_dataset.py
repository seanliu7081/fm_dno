"""Build a Libero-format multi-task replay buffer and disjoint MimicGen tests.

The public MimicGen HDF5 actions already contain normalized OSC_POSE deltas.
They are copied verbatim (apart from float32 storage), never differentiated or
converted to absolute poses. Public HDF5 images already use upright orientation.

Example:
    python scripts/prepare_mimicgen_dataset.py --download
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

import cv2
import h5py
import numpy as np
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oat.common.replay_buffer import ReplayBuffer


HF_REPO = "amandlek/mimicgen_datasets"
DEFAULT_TASKS = (
    "stack_three_d1", "square_d2", "threading_d1",
    "hammer_cleanup_d1", "mug_cleanup_d1", "coffee_d2",
)
TASK_PROMPTS = {
    "stack_three": "Stack the three blocks.",
    "square": "Place the square nut on the peg.",
    "threading": "Thread the needle through the ring.",
    "hammer_cleanup": "Put the hammer in the drawer and close the drawer.",
    "mug_cleanup": "Put the mug in the drawer and close the drawer.",
    "coffee": "Insert the coffee pod and close the coffee machine.",
}
IMAGE_KEYS = {
    "agentview_image": "agentview_rgb",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand_rgb",
}
PROPRIO_KEYS = {
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_qpos": 2,
}


def state_sha256(state: np.ndarray) -> str:
    """Canonical state hash shared with the held-out evaluation runner."""
    return hashlib.sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes()).hexdigest()


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _demo_names(data: h5py.Group) -> list[str]:
    return sorted((name for name in data if name.startswith("demo_")),
                  key=lambda name: int(name.rsplit("_", 1)[1]))


def _metadata(data: h5py.Group) -> dict:
    if "env_args" not in data.attrs:
        raise ValueError("Source HDF5 needs data.attrs['env_args'] for reproducible evaluation")
    metadata = json.loads(_text(data.attrs["env_args"]))
    controller = metadata.get("env_kwargs", {}).get("controller_configs", {})
    if controller.get("type") != "OSC_POSE":
        raise ValueError(f"Expected OSC_POSE delta control, found {controller}")
    if controller.get("control_delta", True) is not True:
        raise ValueError("Absolute-control source cannot be used as delta actions")
    return metadata


def _initial_state(demo: h5py.Group) -> np.ndarray:
    if "states" not in demo or len(demo["states"]) == 0:
        raise ValueError(f"{demo.name} has no simulator state for train/eval disjointness")
    state = np.asarray(demo["states"][0], dtype=np.float64)
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError(f"{demo.name} has an invalid initial simulator state")
    return state


def select_demos(data: h5py.Group, train_count: int, eval_count: int) -> tuple[list[str], list[str]]:
    """Select first N demos and then unique held-out initial states numerically."""
    if train_count <= 0 or eval_count <= 0:
        raise ValueError("Train and evaluation demo counts must be positive")
    names = _demo_names(data)
    if len(names) < train_count + eval_count:
        raise ValueError(f"Need at least {train_count + eval_count} demos; found {len(names)}")
    train_names = names[:train_count]
    seen = {state_sha256(_initial_state(data[name])) for name in train_names}
    eval_names = []
    for name in names[train_count:]:
        fingerprint = state_sha256(_initial_state(data[name]))
        if fingerprint in seen:
            continue
        eval_names.append(name)
        seen.add(fingerprint)
        if len(eval_names) == eval_count:
            break
    if len(eval_names) != eval_count:
        raise ValueError(f"Only {len(eval_names)} unique unseen initial states are available")
    return train_names, eval_names


def convert_episode(demo: h5py.Group, task_name: str, task_uid: int, image_size: int | None = 128) -> dict[str, np.ndarray]:
    actions = np.asarray(demo["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError(f"{demo.name}: expected finite, nonempty delta actions [T, 7]")
    length = len(actions)
    obs = demo["obs"]
    episode = {"action": actions}
    for source, target in IMAGE_KEYS.items():
        images = np.asarray(obs[source])
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3 or len(images) != length:
            raise ValueError(f"{demo.name}/{source}: expected uint8 [T, H, W, 3]")
        if image_size is not None and images.shape[1:3] != (image_size, image_size):
            images = np.stack([cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                               for frame in images])
        episode[target] = images
    for key, dim in PROPRIO_KEYS.items():
        value = np.asarray(obs[key], dtype=np.float32)
        if value.shape != (length, dim) or not np.isfinite(value).all():
            raise ValueError(f"{demo.name}/{key}: expected finite [T, {dim}]")
        episode[key] = value
    # The public MimicGen corpus does not always store joint positions. The
    # policy uses EEF pose + gripper; include joint positions only if available.
    if "robot0_joint_pos" in obs:
        joint_pos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
        if joint_pos.shape != (length, 7) or not np.isfinite(joint_pos).all():
            raise ValueError(f"{demo.name}: invalid robot0_joint_pos")
        episode["robot0_joint_pos"] = joint_pos
    base_task = task_name.rsplit("_d", 1)[0]
    episode["prompt"] = np.full(length, TASK_PROMPTS.get(base_task, base_task.replace("_", " ")), dtype="<U128")
    episode["task_uid"] = np.full((length, 1), task_uid, dtype=np.int64)
    return episode


def download_sources(tasks: list[str], hdf5_dir: Path) -> dict:
    """Resolve one immutable HF revision and download into the chosen directory."""
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(HF_REPO, files_metadata=True)
    revision = info.sha
    siblings = {item.rfilename: item for item in info.siblings}
    sources = []
    for task in tasks:
        filename = f"core/{task}.hdf5"
        if filename not in siblings:
            raise ValueError(f"Official MimicGen dataset has no {filename}")
        entry = siblings[filename]
        print(f"Downloading {filename}: {entry.size / 1e9:.3f} GB", flush=True)
        path = hf_hub_download(HF_REPO, filename=filename, repo_type="dataset",
                               revision=revision, local_dir=str(hdf5_dir))
        expected = entry.lfs.sha256 if entry.lfs else None
        if expected:
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"SHA256 mismatch for {path}; refusing conversion")
        sources.append({"task_name": task, "path": path, "bytes": entry.size, "sha256": expected})
    record = {"repo_id": HF_REPO, "revision": revision, "sources": sources}
    (hdf5_dir / "download_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def prepare_dataset(hdf5_dir: Path, output_dir: Path, tasks: list[str],
                    train_count: int = 100, eval_count: int = 50, image_size: int | None = 128) -> dict:
    """Write only training trajectories to Zarr, and only held-out states to HDF5."""
    hdf5_dir = hdf5_dir.resolve()
    output_dir = output_dir.resolve()
    if len(set(tasks)) != len(tasks) or not tasks:
        raise ValueError("Provide a nonempty list of distinct tasks")
    for task in tasks:
        if not (hdf5_dir / "core" / f"{task}.hdf5").is_file():
            raise FileNotFoundError(hdf5_dir / "core" / f"{task}.hdf5")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [f"mimicgen{len(tasks)}_N{train_count}.zarr",
             f"mimicgen{len(tasks)}_N{train_count}.manifest.json", "eval_initial_states.hdf5"]
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(f"Output already exists: {output_dir / name}; choose a new directory")
    stage = output_dir.with_name(output_dir.name + ".partial-" + uuid.uuid4().hex[:8])
    stage.mkdir()
    replay_name = f"mimicgen{len(tasks)}_N{train_count}.zarr"
    manifest = {
        "schema_version": 1, "action_mode": "delta", "action_description": "normalized OSC_POSE delta xyz / axis-angle / gripper",
        "image_orientation": "public HDF5 upright RGB, copied without flipping",
        "image_size": image_size, "image_resize_interpolation": "cv2.INTER_LINEAR",
        "quaternion_order": "xyzw", "task_uid_encoding": "scalar integer [T, 1]",
        "state_hash_algorithm": "sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes())",
        "selection": "first numerically sorted train demos; subsequent unique states for evaluation",
        "train_demos_per_task": train_count, "eval_tests_per_task": eval_count,
        "zarr_path": replay_name, "eval_hdf5": "eval_initial_states.hdf5", "tasks": [],
    }
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    try:
        replay = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(stage / replay_name)))
        expected_keys = None
        with h5py.File(stage / manifest["eval_hdf5"], "w") as eval_file:
            for task_uid, task in enumerate(tasks):
                source_path = hdf5_dir / "core" / f"{task}.hdf5"
                with h5py.File(source_path, "r") as source:
                    data = source["data"]
                    env_meta = _metadata(data)
                    train_names, eval_names = select_demos(data, train_count, eval_count)
                    train_hashes = [state_sha256(_initial_state(data[name])) for name in train_names]
                    task_manifest = {
                        "task_name": task, "task_uid": task_uid, "source_hdf5": str(source_path),
                        "env_meta": env_meta, "horizon": 500 if task.startswith(("mug_cleanup", "hammer_cleanup")) else 400,
                        "train_demo_names": train_names, "eval_demo_names": eval_names,
                        "train_initial_state_sha256": train_hashes, "eval_initial_state_sha256": [],
                        "train_episode_indices": [], "train_steps": 0,
                    }
                    for index, name in enumerate(train_names):
                        episode = convert_episode(data[name], task, task_uid, image_size=image_size)
                        if expected_keys is None:
                            expected_keys = set(episode)
                        if set(episode) != expected_keys:
                            raise ValueError(f"Observation keys differ across demos: {task}/{name}")
                        task_manifest["train_episode_indices"].append(replay.n_episodes)
                        replay.add_episode(episode, compressors=compressor)
                        task_manifest["train_steps"] += len(episode["action"])
                        if index % 10 == 0 or index + 1 == train_count:
                            print(f"{task}: converted {index + 1}/{train_count} train demos", flush=True)
                    for name in eval_names:
                        demo = data[name]
                        if "model_file" not in demo.attrs:
                            raise ValueError(f"{demo.name} has no model_file for exact evaluation reset")
                        state = _initial_state(demo)
                        fingerprint = state_sha256(state)
                        group = eval_file.create_group(f"data/{task}/{name}")
                        group.create_dataset("states", data=state)
                        group.attrs["model_file"] = _text(demo.attrs["model_file"])
                        group.attrs["state_sha256"] = fingerprint
                        task_manifest["eval_initial_state_sha256"].append(fingerprint)
                    manifest["tasks"].append(task_manifest)
                    print(f"{task}: saved {train_count} train demonstrations and {eval_count} unseen tests", flush=True)
        manifest["total_train_episodes"] = replay.n_episodes
        manifest["total_train_steps"] = int(replay.n_steps)
        manifest["observation_shapes"] = {key: list(value.shape[1:]) for key, value in replay.data.items()}
        replay.root.attrs.update({"action_mode": "delta", "manifest": f"../mimicgen{len(tasks)}_N{train_count}.manifest.json",
                                  "task_names": tasks, "num_demos_per_task": train_count})
        download_manifest = hdf5_dir / "download_manifest.json"
        if download_manifest.is_file():
            manifest["source_download"] = json.loads(download_manifest.read_text())
        (stage / f"mimicgen{len(tasks)}_N{train_count}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name in (names[0], names[2], names[1]):  # manifest is the completion marker
            os.replace(stage / name, output_dir / name)
        stage.rmdir()
    except Exception:
        shutil.rmtree(stage)
        raise
    print(f"Prepared {output_dir / replay_name}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", type=Path, default=Path("/workspace/shared_data/mimicgen/hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/shared_data/mimicgen"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--train-demos", type=int, default=100)
    parser.add_argument("--eval-tests", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.download or args.download_only:
        download_sources(args.tasks, args.hdf5_dir)
    if not args.download_only:
        prepare_dataset(args.hdf5_dir, args.output_dir, args.tasks, args.train_demos, args.eval_tests, args.image_size)


if __name__ == "__main__":
    main()
