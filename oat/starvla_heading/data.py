"""Original LIBERO demonstrations for QwenVL + Heading Gaussian.

This adapter reads the original HDF5 files instead of the merged zarr's fixed
width prompt field. Splits and affine statistics are frozen in a manifest before
training. LIBERO-Plus is deliberately not a supported training suite.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image


ORIGINAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAMERA_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
STATE_LAYOUT = ("eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_quat_x",
                "eef_quat_y", "eef_quat_z", "eef_quat_w", "gripper_qpos_0",
                "gripper_qpos_1")
MANIFEST_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash every setting, task, split, source identity, and normalization value."""
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("Unsupported LIBERO manifest version")
    if manifest.get("dataset_origin") != "original_libero":
        raise ValueError("Training requires original LIBERO, never LIBERO-Plus")
    if manifest.get("sha256") != manifest_sha256(manifest):
        raise ValueError("LIBERO manifest hash mismatch; do not edit a frozen manifest")
    return manifest


def canonical_task_map(path: str | Path | None = None) -> dict[str, list[str]]:
    """Read the vendored official task list without importing the simulator."""
    if path is None:
        path = (Path(__file__).resolve().parents[2] / "third_party" / "LIBERO" /
                "libero" / "libero" / "benchmark" / "libero_suite_task_map.py")
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            mapping = ast.literal_eval(node.value)
            return {suite: list(mapping[suite]) for suite in ORIGINAL_SUITES}
    raise ValueError(f"Cannot read canonical LIBERO tasks from {path}")


def axisangle_to_quaternion(axisangle: np.ndarray) -> np.ndarray:
    """Convert axis-angle radians to robosuite's xyzw quaternion convention."""
    value = np.asarray(axisangle, dtype=np.float64)
    if value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError("LIBERO ee_ori must contain finite three-dimensional axis angles")
    angle = np.linalg.norm(value, axis=-1, keepdims=True)
    # np.sinc(z) = sin(pi*z)/(pi*z), including the analytic limit at zero.
    xyz = value * (0.5 * np.sinc(angle / (2 * np.pi)))
    return np.concatenate((xyz, np.cos(angle / 2)), axis=-1).astype(np.float32)


def read_state(obs: h5py.Group, selection: slice) -> np.ndarray:
    position = np.asarray(obs["ee_pos"][selection], dtype=np.float32)
    orientation = axisangle_to_quaternion(obs["ee_ori"][selection])
    gripper = np.asarray(obs["gripper_states"][selection], dtype=np.float32)
    state = np.concatenate((position, orientation, gripper), axis=-1)
    if state.ndim != 2 or state.shape[-1] != 9 or not np.isfinite(state).all():
        raise ValueError("LIBERO proprioception must be finite position3 + quat4 + gripper2")
    return state


def normalize_array(value: np.ndarray, statistics: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(value, dtype=np.float32) *
            np.asarray(statistics["scale"], dtype=np.float32) +
            np.asarray(statistics["offset"], dtype=np.float32))


def denormalize_array(value: np.ndarray, statistics: Mapping[str, Any]) -> np.ndarray:
    return ((np.asarray(value, dtype=np.float32) -
             np.asarray(statistics["offset"], dtype=np.float32)) /
            np.asarray(statistics["scale"], dtype=np.float32))


def _affine_statistics(low: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    width = high - low
    constant = width < 1e-4  # Same constant-channel behavior as OAT LinearNormalizer.
    scale = 2.0 / np.where(constant, 2.0, width)
    offset = np.where(constant, -low, -1.0 - scale * low)
    return {"scale": scale.tolist(), "offset": offset.tolist(),
            "min": low.tolist(), "max": high.tolist(),
            "formula": "normalized = raw * scale + offset", "mode": "limits"}


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _episode_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("demo_")
    return (int(suffix) if suffix.isdigit() else 2**63, name)


def _split_episode_ids(episode_ids: Sequence[str], seed: int,
                       val_ratio: float) -> set[str]:
    count = 0 if val_ratio == 0 else min(
        len(episode_ids) - 1, max(1, round(len(episode_ids) * val_ratio)))
    ranked = sorted(episode_ids, key=lambda key: hashlib.sha256(
        f"{seed}\0{key}".encode("utf-8")).hexdigest())
    return set(ranked[:count])


def prepare_libero_manifest(
    root: str | Path,
    output: str | Path,
    *,
    suites: Sequence[str] = ORIGINAL_SUITES,
    seed: int = 42,
    val_ratio: float = 0.1,
    expected_demos_per_task: int | None = 50,
    require_complete: bool = True,
    task_map: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Audit original demonstrations and write an immutable train/val manifest.

    Production defaults require all ten canonical tasks and fifty episodes per
    task in each requested suite. Tests may pass an explicit small ``task_map``.
    No RGB pixels are read for normalization, and validation episodes never
    contribute to action/state statistics or the raw XY RMS.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Manifest is immutable: {output}; choose a new output path")
    suites = tuple(suites)
    if not suites or len(set(suites)) != len(suites) or not set(suites) <= set(ORIGINAL_SUITES):
        raise ValueError(f"Training suites must be unique original LIBERO suites: {ORIGINAL_SUITES}")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if expected_demos_per_task is not None and expected_demos_per_task < 1:
        raise ValueError("expected_demos_per_task must be positive")
    if not root.is_dir():
        raise FileNotFoundError(f"Original LIBERO HDF5 root does not exist: {root}")
    if any("libero-plus" in part.lower() or "libero_plus" in part.lower()
           for part in root.parts):
        raise ValueError("LIBERO-Plus files cannot be used to prepare training data")
    mapping = canonical_task_map() if task_map is None else task_map
    expected = {task: suite for suite in suites for task in mapping[suite]}
    if len(expected) != sum(len(mapping[suite]) for suite in suites):
        raise ValueError("Canonical task names must be unique across training suites")
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.hdf5")):
        task = path.stem.removesuffix("_demo")
        if task not in expected:
            continue
        if any("libero-plus" in part.lower() or "libero_plus" in part.lower()
               for part in path.relative_to(root).parts):
            raise ValueError(f"LIBERO-Plus training source rejected: {path}")
        if task in found:
            raise ValueError(f"Duplicate HDF5 source for {task}: {found[task]} and {path}")
        found[task] = path
    missing = sorted(set(expected) - set(found))
    if require_complete and missing:
        raise FileNotFoundError(f"Missing {len(missing)} canonical original LIBERO tasks: " +
                                ", ".join(missing))
    if not found:
        raise FileNotFoundError(f"No canonical original LIBERO HDF5 demonstrations in {root}")

    tasks, episodes = [], []
    bounds = {"action": [np.full(7, np.inf), np.full(7, -np.inf)],
              "state": [np.full(9, np.inf), np.full(9, -np.inf)]}
    xy_squares, train_frames, val_frames = 0.0, 0, 0
    numerical_digest = hashlib.sha256()
    for task in sorted(found, key=lambda name: (suites.index(expected[name]), name)):
        path, suite = found[task], expected[task]
        identity = path.stat()
        with h5py.File(path, "r") as source:
            data = source["data"]
            bddl_task = Path(_decode(data.attrs["bddl_file_name"])).stem
            if bddl_task != task:
                raise ValueError(f"Task name mismatch in {path}: {bddl_task} != {task}")
            info = json.loads(_decode(data.attrs["problem_info"]))
            prompt = info.get("language_instruction")
            if not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt:
                raise ValueError(f"Missing or invalid full language instruction in {path}")
            demo_names = sorted((name for name in data if name.startswith("demo_")),
                                key=_episode_sort_key)
            if not demo_names:
                raise ValueError(f"No demonstration episodes in {path}")
            if int(data.attrs.get("num_demos", len(demo_names))) != len(demo_names):
                raise ValueError(f"num_demos disagrees with the episode groups in {path}")
            if expected_demos_per_task is not None and len(demo_names) != expected_demos_per_task:
                raise ValueError(f"{task}: expected {expected_demos_per_task} demonstrations, "
                                 f"found {len(demo_names)}")
            if val_ratio and len(demo_names) < 2:
                raise ValueError(f"Need at least two demonstrations for per-task validation: {task}")
            task_id = len(tasks)
            relative = str(path.relative_to(root))
            tasks.append({"suite": suite, "task": task, "lang": prompt, "file": relative,
                          "file_size": identity.st_size, "file_mtime_ns": identity.st_mtime_ns,
                          "episode_count": len(demo_names), "prompt_source": "data.attrs.problem_info"})
            episode_ids = [f"{suite}/{task}/{demo}" for demo in demo_names]
            validation = _split_episode_ids(episode_ids, seed, val_ratio)
            for demo_name, episode_id in zip(demo_names, episode_ids):
                demo = data[demo_name]
                action_ds, obs = demo["actions"], demo["obs"]
                length = len(action_ds)
                if length < 1 or action_ds.shape != (length, 7):
                    raise ValueError(f"Invalid action shape for {episode_id}: {action_ds.shape}")
                for key, width in (("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)):
                    if obs[key].shape != (length, width):
                        raise ValueError(f"Observation/action length or width mismatch: {episode_id}/{key}")
                for key in CAMERA_KEYS:
                    camera = obs[key]
                    if (camera.ndim != 4 or camera.shape[0] != length or
                            camera.shape[-1] != 3 or camera.dtype != np.dtype("uint8")):
                        raise ValueError(f"Invalid uint8 RGB observation: {episode_id}/{key}")
                split = "val" if episode_id in validation else "train"
                episodes.append({"episode_id": episode_id, "task_index": task_id,
                                 "demo": demo_name, "length": length, "split": split})
                if split == "val":
                    val_frames += length
                    continue
                action = np.asarray(action_ds[:], dtype=np.float32)
                if not np.isfinite(action).all():
                    raise ValueError(f"Non-finite actions in {episode_id}")
                state = read_state(obs, slice(None))
                numerical_digest.update(episode_id.encode("utf-8"))
                for key, values in (("action", action), ("state", state)):
                    bounds[key][0] = np.minimum(bounds[key][0], values.min(axis=0))
                    bounds[key][1] = np.maximum(bounds[key][1], values.max(axis=0))
                    numerical_digest.update(values.astype("<f4", copy=False).tobytes())
                xy_squares += float(np.square(action[:, :2].astype(np.float64)).sum())
                train_frames += length
        if path.stat().st_mtime_ns != identity.st_mtime_ns or path.stat().st_size != identity.st_size:
            raise RuntimeError(f"Dataset source changed while reading {path}")
    statistics = {key: _affine_statistics(*value) for key, value in bounds.items()}
    statistics["heading_xy_rms"] = float(np.sqrt(xy_squares / (train_frames * 2)))
    statistics["fit"] = "all unpadded frames from training episodes only"
    manifest = {
        "version": MANIFEST_VERSION, "dataset_origin": "original_libero",
        "root": str(root), "suites": list(suites), "seed": int(seed),
        "val_ratio": float(val_ratio), "split_algorithm": "per_task_sha256_seed_episode_id_v1",
        "require_complete": require_complete, "expected_demos_per_task": expected_demos_per_task,
        "missing_tasks": missing, "tasks": tasks, "episodes": episodes,
        "statistics": statistics, "training_numeric_sha256": numerical_digest.hexdigest(),
        "train_episode_count": sum(item["split"] == "train" for item in episodes),
        "val_episode_count": sum(item["split"] == "val" for item in episodes),
        "train_frame_count": train_frames, "val_frame_count": val_frames,
        "preprocessing": {"camera_keys": list(CAMERA_KEYS), "image_order": "time_major",
                          "flip_vertical": True, "state_layout": list(STATE_LAYOUT),
                          "quaternion_order": "xyzw", "action_dimensions": 7,
                          "observation_alignment": "history ends at first predicted action",
                          "action_padding": "repeat final action; action_mask excludes padding"},
    }
    acquisition_path = root / "SOURCE_MANIFEST.json"
    if acquisition_path.is_file():
        acquisition_bytes = acquisition_path.read_bytes()
        manifest["upstream_source"] = {
            "metadata_file": acquisition_path.name,
            "metadata_sha256": hashlib.sha256(acquisition_bytes).hexdigest(),
            "declared_source": json.loads(acquisition_bytes),
        }
    manifest["sha256"] = manifest_sha256(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting another worker's frozen manifest.
    with output.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    return manifest


class LiberoHDF5Dataset:
    """Map-style dataset of raw action chunks and PIL images for StarVLA.

    ``image`` contains ``n_obs_steps * 2`` RGB PIL images ordered oldest to
    newest, front then wrist. ``action`` and ``state`` remain in raw units;
    the model applies the affine statistics stored in ``manifest``. Episode
    IDs and task metadata are bookkeeping only, never model conditioning.
    """

    def __init__(self, manifest_path: str | Path, split: str = "train", *,
                 n_obs_steps: int = 2, horizon: int = 16,
                 image_size: int | None = 224, flip_vertical: bool = True,
                 max_open_files: int = 4):
        if split == "validation":
            split = "val"
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        if n_obs_steps < 1 or horizon < 1 or max_open_files < 1:
            raise ValueError("n_obs_steps, horizon and max_open_files must be positive")
        if image_size is not None and image_size < 1:
            raise ValueError("image_size must be positive or None")
        self.manifest_path = str(Path(manifest_path).resolve())
        self.manifest = load_manifest(self.manifest_path)
        if bool(flip_vertical) != self.manifest["preprocessing"]["flip_vertical"]:
            raise ValueError("Image orientation must match the frozen dataset manifest")
        self.split, self.n_obs_steps, self.horizon = split, n_obs_steps, horizon
        self.image_size, self.flip_vertical = image_size, flip_vertical
        self.max_open_files = max_open_files
        self.episodes = [item for item in self.manifest["episodes"] if item["split"] == split]
        self._ends = np.cumsum([item["length"] for item in self.episodes], dtype=np.int64)
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._pid = os.getpid()

    @property
    def statistics(self) -> dict[str, Any]:
        return self.manifest["statistics"]

    def __len__(self) -> int:
        return int(self._ends[-1]) if len(self._ends) else 0

    def _file(self, task_index: int) -> h5py.File:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if task_index in self._handles:
            self._handles.move_to_end(task_index)
            return self._handles[task_index]
        task = self.manifest["tasks"][task_index]
        path = Path(self.manifest["root"]) / task["file"]
        identity = path.stat()
        if (identity.st_size != task["file_size"] or
                identity.st_mtime_ns != task["file_mtime_ns"]):
            raise RuntimeError(f"Source differs from the frozen manifest: {path}")
        while len(self._handles) >= self.max_open_files:
            _, old = self._handles.popitem(last=False)
            old.close()
        source = h5py.File(path, "r")
        self._handles[task_index] = source
        return source

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self._ends, index, side="right"))
        episode = self.episodes[episode_index]
        timestep = int(index - (self._ends[episode_index - 1] if episode_index else 0))
        task = self.manifest["tasks"][episode["task_index"]]
        demo = self._file(episode["task_index"])["data"][episode["demo"]]
        first = max(0, timestep - self.n_obs_steps + 1)
        selected = np.maximum(np.arange(timestep - self.n_obs_steps + 1, timestep + 1), 0) - first
        state = read_state(demo["obs"], slice(first, timestep + 1))[selected]
        images = []
        cameras = {key: demo["obs"][key][first:timestep + 1] for key in CAMERA_KEYS}
        for frame in selected:
            for key in CAMERA_KEYS:
                pixels = cameras[key][frame]
                if self.flip_vertical:
                    pixels = pixels[::-1]
                image = Image.fromarray(np.ascontiguousarray(pixels))
                if self.image_size is not None:
                    image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
                images.append(image)
        available = min(self.horizon, episode["length"] - timestep)
        actions = np.asarray(demo["actions"][timestep:timestep + available], dtype=np.float32)
        if available < self.horizon:
            actions = np.pad(actions, ((0, self.horizon - available), (0, 0)), mode="edge")
        return {"image": images, "lang": task["lang"], "action": actions,
                "state": state, "action_mask": np.arange(self.horizon) < available,
                "episode_id": episode["episode_id"], "timestep": timestep}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        value = self.__dict__.copy()
        value["_handles"] = OrderedDict()
        return value

    def __del__(self):
        # h5py's C-extension globals may already be cleared at interpreter exit.
        # Explicit close() still reports errors during normal application use.
        try:
            if hasattr(self, "_handles"):
                self.close()
        except Exception:
            pass


def collate_libero_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep PIL images and variable-length language for the Qwen processor."""
    return list(samples)
