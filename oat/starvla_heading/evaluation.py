"""LIBERO and strict zero-shot LIBERO-Plus evaluation over a local policy server.

This module intentionally has no model or simulator imports at module scope. Run
it in the simulator virtualenv; the shared Qwen policy lives in its own process.
The wire protocol is versioned, uses lossless PNG images, and exchanges raw
simulator-domain actions. Images are temporal-major, then camera-major.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import time
import traceback
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

PROTOCOL_VERSION = 1
PROMPT_PROTOCOL = "original_manifest_and_plus_language_bddl_v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = dict(zip(SUITES, (220, 280, 300, 520)))
CAMERAS = ("agentview", "robot0_eye_in_hand")
_WORKER_SETTINGS = None
_WORKER_CLIENT = None
_WORKER_SUITES = {}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + json.dumps(value.shape).encode() + value.tobytes()).hexdigest()


def stable_seed(seed, *parts):
    return int(json_hash([int(seed), *parts])[:8], 16)


def encode_image(value):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Policy images must be uint8 HxWx3 RGB")
    buffer = io.BytesIO()
    Image.fromarray(value).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(value):
    with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("Expected a lossless RGB PNG")
        return np.asarray(image).copy()


def extract_observation(raw, image_size=None):
    """Use the wrapper-returned observation, retaining all Plus corruptions.

    Local LIBERO HDF5 conversion flips the image's vertical axis only. Do not
    copy the 180-degree rotation used by upstream policies trained on RLDS.
    """
    images = []
    for camera in CAMERAS:
        image = np.asarray(raw[f"{camera}_image"])
        if image.dtype != np.uint8:
            raise ValueError(f"Unexpected image dtype {image.dtype} for {camera}")
        image = np.ascontiguousarray(image[::-1])
        if image_size is not None:
            image = np.asarray(Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BICUBIC)).copy()
        images.append(image)
    state = np.concatenate([np.asarray(raw[key], dtype=np.float32).reshape(-1) for key in
                            ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")])
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("Expected finite xyz + xyzw quaternion + 2 gripper positions")
    return {"images": images, "state": state.copy()}


def validate_metadata(metadata, benchmark="libero_plus", smoke=False):
    expected = {
        "protocol_version": PROTOCOL_VERSION, "n_obs_steps": 2,
        "camera_names": list(CAMERAS), "state_dim": 9, "action_dim": 7,
        "prediction_horizon": 16, "execution_horizon": 8,
        "image_orientation": "vertical_flip", "variant": "heading_gaussian",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Policy metadata {key!r} must be {value!r}, got {metadata.get(key)!r}")
    if not isinstance(metadata.get("image_size"), int) or metadata["image_size"] < 1:
        raise ValueError("Policy metadata requires positive image_size")
    if metadata.get("camera_render_size", 128) != 128 or metadata.get("image_resize", "bicubic") != "bicubic":
        raise ValueError("Policy preprocessing requires 128-pixel rendering followed by bicubic resizing")
    sha = metadata.get("checkpoint_sha256", "")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Policy metadata requires a checkpoint SHA-256")
    if benchmark == "libero_plus" and not smoke:
        for field in ("training_benchmark", "selection_benchmark"):
            if metadata.get(field) != "libero":
                raise ValueError(f"Strict zero-shot evaluation requires {field}='libero'")
    return metadata


class PolicyClient:
    def __init__(self, server_url, expected_metadata=None, timeout=300):
        parsed = urlparse(server_url)
        if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("The inference server must be a localhost HTTP endpoint")
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.metadata = self._request("/metadata")
        if expected_metadata is not None and self.metadata != expected_metadata:
            raise RuntimeError("Inference server metadata differs from the frozen evaluation manifest")

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.server_url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(8192).decode("utf-8", errors="replace")
            raise RuntimeError(f"Policy server HTTP {error.code}: {detail}") from error

    def predict(self, history, language, seed, request_id):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "checkpoint_sha256": self.metadata["checkpoint_sha256"],
            "request_id": request_id, "seed": int(seed), "language": language,
            "images": [encode_image(image) for frame in history for image in frame["images"]],
            "state": [frame["state"].tolist() for frame in history],
        }
        response = self._request("/predict", payload)
        if response.get("checkpoint_sha256") != self.metadata["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed during evaluation")
        return validate_actions(response["actions"], self.metadata["prediction_horizon"])


def validate_actions(value, horizon=16):
    actions = np.asarray(value, dtype=np.float32)
    if actions.shape != (horizon, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite ({horizon}, 7) raw actions, got {actions.shape}")
    return actions


def load_official_states(suite, task_index):
    # Retain the official resolver: Plus aliases language/view/texture states
    # and stores object-layout states in a different directory. Torch 2.6+
    # needs this switch for the trusted NumPy-containing official assets.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    states = np.asarray(suite.get_task_init_states(task_index))
    if states.ndim != 2 or states.shape[0] < 1 or not np.isfinite(states).all():
        raise ValueError("Official state resolver did not return finite (episodes, state) data")
    return states


def classification_index(classification, suite_name):
    entries = classification[suite_name]
    indexed = {entry["name"]: entry for entry in entries}
    if len(indexed) != len(entries):
        raise ValueError(f"Duplicate classification names in {suite_name}")
    for entry in entries:
        if not entry.get("category") or "difficulty_level" not in entry:
            raise ValueError("Classification requires category and difficulty_level")
    return indexed


def load_instruction_catalog(path, expected_sha256=None):
    """Use the frozen training instructions, never the Plus filename suffixes."""
    from .data import load_manifest
    manifest = load_manifest(path)
    if expected_sha256 is not None and manifest["sha256"] != expected_sha256:
        raise ValueError("Instruction manifest differs from the policy's training manifest")
    instructions = {suite: {} for suite in SUITES}
    for task in manifest["tasks"]:
        suite, name, language = task["suite"], task["task"], task["lang"]
        if suite not in instructions or name in instructions[suite]:
            raise ValueError("Instruction manifest has an unknown suite or duplicate task")
        if not isinstance(language, str) or not language.strip() or chr(0) in language:
            raise ValueError("Instruction manifest contains an invalid instruction")
        instructions[suite][name] = language
    if any(len(tasks) != 10 for tasks in instructions.values()):
        raise ValueError("Instruction manifest must cover all 40 original LIBERO tasks")
    return {"protocol": PROMPT_PROTOCOL, "training_manifest_sha256": manifest["sha256"],
            "instruction_catalog_sha256": json_hash(instructions), "instructions": instructions}


def language_variant_prompt(task, bddl_root):
    """Read the exact rewrite with the same BDDL resolver as official Plus.

    Only the filename's view/init-state suffix is removed. Natural-language
    text is never trimmed, lowercased, or inferred from the base BDDL.
    """
    from libero.libero.envs import bddl_utils
    name = task.name.split("_view_", 1)[0]
    path = Path(bddl_root) / task.problem_folder / (name + ".bddl")
    return {"language": bddl_utils.get_problem_info(str(path))["language_instruction"],
            "language_bddl_sha256": file_hash(path)}


def resolve_task_prompt(task, suite_name, benchmark, instruction_catalog, language_loader=None):
    if instruction_catalog is None or instruction_catalog.get("protocol") != PROMPT_PROTOCOL:
        raise ValueError("Evaluation requires the frozen original instruction catalog")
    instructions = instruction_catalog["instructions"]
    if json_hash(instructions) != instruction_catalog["instruction_catalog_sha256"]:
        raise ValueError("Frozen instruction catalog changed")
    candidates = [name for name in instructions[suite_name]
                  if task.name == name or task.name.startswith(name + "_")]
    if len(candidates) != 1:
        raise ValueError(f"Task {task.name} has no unique original LIBERO instruction")
    original = candidates[0]
    suffix = task.name[len(original):]
    if benchmark == "libero" and suffix:
        raise ValueError("Original LIBERO cannot contain perturbation variants")
    if suffix and not re.fullmatch(r"_(?:language|table|tb|view|add|light|moved|level[1-5])(?:_.*)?", suffix):
        raise ValueError(f"Unknown Plus perturbation suffix: {suffix}")
    source, bddl_sha = "original_training_manifest", None
    if suffix.startswith("_language_"):
        if language_loader is None:
            raise ValueError("Language variants require the official BDDL instruction loader")
        rewritten = language_loader(task)
        language, bddl_sha = rewritten["language"], rewritten["language_bddl_sha256"]
        source = "libero_plus_language_bddl"
        if language != task.language:
            raise ValueError("Official Plus language instruction differs from its variant BDDL")
    else:
        language = instructions[suite_name][original]
    if not isinstance(language, str) or not language.strip() or chr(0) in language:
        raise ValueError(f"Task {task.name} lacks a valid instruction")
    return {"language": language, "canonical_task_name": original, "prompt_source": source,
            "prompt_sha256": json_hash(language), "language_bddl_sha256": bddl_sha,
            "source_task_language_sha256": json_hash(task.language)}


def audit_prompt_mapping(plan, instruction_catalog):
    instructions = instruction_catalog["instructions"]
    if (instruction_catalog.get("protocol") != PROMPT_PROTOCOL or
            json_hash(instructions) != instruction_catalog["instruction_catalog_sha256"]):
        raise ValueError("Prompt audit has an invalid frozen instruction catalog")
    mapping = {}
    for episode in plan:
        if episode["prompt_sha256"] != json_hash(episode["language"]):
            raise ValueError("Prompt audit instruction hash mismatch")
        original = episode["canonical_task_name"]
        candidates = [name for name in instructions[episode["suite"]]
                      if episode["task_name"] == name or episode["task_name"].startswith(name + "_")]
        if candidates != [original]:
            raise ValueError("Prompt audit lacks a unique canonical task")
        rewritten = episode["task_name"].startswith(original + "_language_")
        if rewritten:
            sha = episode["language_bddl_sha256"]
            if (episode["prompt_source"] != "libero_plus_language_bddl" or
                    episode["source_task_language_sha256"] != episode["prompt_sha256"] or
                    not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
                raise ValueError("Prompt audit lacks the exact language-variant BDDL identity")
        elif (episode["prompt_source"] != "original_training_manifest" or
              episode["language"] != instructions[episode["suite"]][original] or
              episode["language_bddl_sha256"] is not None):
            raise ValueError("Prompt audit changed the canonical original instruction")
        key = (episode["suite"], episode["task_name"])
        entry = {field: episode[field] for field in
                 ("suite", "task_name", "canonical_task_name", "language", "prompt_source", "prompt_sha256",
                  "language_bddl_sha256", "source_task_language_sha256")}
        if key in mapping and mapping[key] != entry:
            raise ValueError("A task's instruction differs between trials")
        mapping[key] = entry
    entries = [mapping[key] for key in sorted(mapping)]
    return {key: value for key, value in instruction_catalog.items() if key != "instructions"} | {
        "mapped_tasks": len(entries), "mapping_sha256": json_hash(entries),
        "source_counts": {source: sum(entry["prompt_source"] == source for entry in entries)
                          for source in sorted({entry["prompt_source"] for entry in entries})},
        "suite_counts": {suite: sum(entry["suite"] == suite for entry in entries)
                         for suite in sorted({entry["suite"] for entry in entries})}}


def make_plan(suites, classification, benchmark, trials, seed, offset=0, limit=None,
              instruction_catalog=None, language_loader=None):
    if trials < 1 or offset < 0 or (limit is not None and limit < 1):
        raise ValueError("Invalid trial count, offset or limit")
    if benchmark == "libero_plus" and trials != 1:
        raise ValueError("LIBERO-Plus has exactly one trial per saved task variant")
    plan, suite_counts = [], {}
    for suite_name, suite in suites.items():
        count = int(suite.get_num_tasks())
        suite_counts[suite_name] = count
        labels = classification_index(classification, suite_name) if benchmark == "libero_plus" else {}
        names = [suite.get_task(index).name for index in range(count)]
        if len(set(names)) != count:
            raise ValueError(f"Duplicate task names in {suite_name}")
        if benchmark == "libero_plus" and set(names) != set(labels):
            raise ValueError(f"Classification does not exactly cover the installed {suite_name} task suite")
        prompts = [resolve_task_prompt(suite.get_task(index), suite_name, benchmark,
                                       instruction_catalog, language_loader) for index in range(count)]
        for trial in range(trials):
            for task_index in range(count):
                task = suite.get_task(task_index)
                if not isinstance(task.language, str) or not task.language.strip():
                    raise ValueError(f"Task {task.name} lacks an actual language instruction")
                label = labels.get(task.name, {})
                prompt = prompts[task_index]
                if benchmark == "libero_plus" and ((label["category"] == "Language Instructions") !=
                                                   (prompt["prompt_source"] == "libero_plus_language_bddl")):
                    raise ValueError("Language perturbation filename and official classification disagree")
                episode_id = f"{suite_name}/{task.name}/init-{trial:03d}"
                plan.append({
                    "episode_id": episode_id, "suite": suite_name, "task_index": task_index,
                    "task_name": task.name, **prompt, "init_index": trial,
                    "category": label.get("category", "Original LIBERO"),
                    "difficulty_level": label.get("difficulty_level", "original"),
                    "classification_id": label.get("id"),
                    "env_seed": stable_seed(seed, "env", episode_id),
                    "policy_seed": stable_seed(seed, "policy", episode_id),
                    "max_episode_steps": MAX_STEPS[suite_name],
                })
    full_count = len(plan)
    plan = plan[offset:] if limit is None else plan[offset:offset + limit]
    if not plan:
        raise ValueError("No episodes selected")
    # Hash the resolved initial states, including aliases, before any rollout.
    for episode in plan:
        states = load_official_states(suites[episode["suite"]], episode["task_index"])
        if episode["init_index"] >= len(states):
            raise ValueError(f"Insufficient saved states for {episode['episode_id']}")
        episode["official_state_sha256"] = array_hash(states[episode["init_index"]])
    return plan, {"suite_counts": suite_counts, "available_episodes": full_count,
                  "selected_episodes": len(plan), "all_available_selected": len(plan) == full_count,
                  "all_four_suites": set(suites) == set(SUITES)}


def _step(env, action):
    value = env.step(action.tolist())
    if len(value) == 4:
        observation, reward, done, info = value
    elif len(value) == 5:
        observation, reward, terminated, truncated, info = value
        done = terminated or truncated
    else:
        raise ValueError("Expected a four- or five-value simulator step")
    return observation, bool(done)


def rollout_episode(client, env, episode, init_state, settings):
    env.seed(int(episode["env_seed"]))
    env.reset()
    raw = env.set_init_state(init_state)
    noop = np.asarray([0.] * 6 + [-1.], dtype=np.float32)
    for _ in range(settings["settle_steps"]):
        # This must be the wrapper result: raw simulator observations bypass
        # Plus sensor corruption. set_init_state may itself return clean data.
        raw, _ = _step(env, noop)
    if raw is None:
        raise RuntimeError("Simulator did not return a post-initialization observation")
    if env.check_success():
        raise RuntimeError("Official initial state succeeds before policy execution")
    image_size = settings.get("policy", {}).get("image_size")
    obs = extract_observation(raw, image_size=image_size)
    history = deque([obs] * 2, maxlen=2)
    action_hash = hashlib.sha256()
    steps, cycles, success, done = 0, 0, False, False
    max_steps = settings.get("smoke_max_steps") or episode["max_episode_steps"]
    while steps < max_steps and not (success or done):
        noise_seed = stable_seed(episode["policy_seed"], "control", cycles)
        if settings.get("smoke"):
            actions = np.tile(noop, (16, 1))
        else:
            actions = validate_actions(client.predict(history, episode["language"], noise_seed,
                                                       f"{episode['episode_id']}/control-{cycles}"))
        for action in actions[:8]:
            if steps >= max_steps:
                break
            action_hash.update(np.ascontiguousarray(action).tobytes())
            raw, done = _step(env, action)
            history.append(extract_observation(raw, image_size=image_size))
            steps += 1
            success = bool(env.check_success())
            if success or done:
                break
        cycles += 1
    return {**episode, "success": success, "executed_steps": steps, "control_cycles": cycles,
            "executed_actions_sha256": action_hash.hexdigest(), "error": None,
            "termination": "success" if success else ("horizon" if steps >= max_steps else "environment_done")}


def initialize_worker(settings):
    global _WORKER_SETTINGS, _WORKER_CLIENT, _WORKER_SUITES
    _WORKER_SETTINGS = settings
    _WORKER_SUITES = {}
    os.environ["LIBERO_CONFIG_PATH"] = settings["libero_config_path"]
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    torch.set_num_threads(1)
    from .rendering import configure_robosuite_egl, install_robosuite_egl_cleanup
    mapping = configure_robosuite_egl(settings["render_gpu_device_id"])
    install_robosuite_egl_cleanup()
    print(json.dumps({"event": "renderer_device_verified", **mapping}), flush=True)
    _WORKER_CLIENT = None if settings["smoke"] else PolicyClient(settings["server_url"], settings["policy"])


def evaluate_episode(episode):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    started, env = time.monotonic(), None
    try:
        random.seed(episode["env_seed"])
        np.random.seed(episode["env_seed"])
        import torch
        torch.manual_seed(episode["env_seed"])
        name = episode["suite"]
        if name not in _WORKER_SUITES:
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                _WORKER_SUITES[name] = benchmark.get_benchmark_dict()[name](task_order_index=0)
        suite = _WORKER_SUITES[name]
        task = suite.get_task(episode["task_index"])
        prompt = resolve_task_prompt(task, name, _WORKER_SETTINGS["benchmark"],
                                     _WORKER_SETTINGS["instruction_catalog"],
                                     lambda task: language_variant_prompt(task, get_libero_path("bddl_files")))
        if task.name != episode["task_name"] or any(episode.get(key) != value for key, value in prompt.items()):
            raise RuntimeError("Benchmark task or instruction source changed after manifest creation")
        state = load_official_states(suite, episode["task_index"])[episode["init_index"]]
        if array_hash(state) != episode["official_state_sha256"]:
            raise RuntimeError("Official initial state changed after manifest creation")
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_names=list(CAMERAS), camera_heights=_WORKER_SETTINGS["policy"].get("camera_render_size", 128),
            camera_widths=_WORKER_SETTINGS["policy"].get("camera_render_size", 128),
            render_gpu_device_id=_WORKER_SETTINGS["render_gpu_device_id"],
            horizon=episode["max_episode_steps"] + _WORKER_SETTINGS["settle_steps"] + 1,
        )
        result = rollout_episode(_WORKER_CLIENT, env, episode, state, _WORKER_SETTINGS)
    except Exception as error:
        result = {**episode, "success": False, "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return {**result, "wall_seconds": time.monotonic() - started}


def summarize_results(plan, results, official_scope=False, smoke=False):
    planned = {entry["episode_id"]: entry for entry in plan}
    seen = {}
    for result in results:
        eid = result["episode_id"]
        if eid not in planned or eid in seen:
            raise ValueError(f"Unknown or duplicate result {eid}")
        if any(result.get(key) != value for key, value in planned[eid].items()):
            raise ValueError(f"Result changed its planned inputs: {eid}")
        seen[eid] = result

    def aggregate(episodes):
        completed = [seen[e["episode_id"]] for e in episodes if e["episode_id"] in seen]
        valid = [result for result in completed if result.get("error") is None]
        successes = sum(bool(result["success"]) for result in valid)
        complete = len(valid) == len(episodes)
        return {"expected": len(episodes), "completed": len(valid),
                "errors": len(completed) - len(valid), "successes": successes,
                "success_rate": successes / len(episodes) if complete and not smoke else None}

    grouped = {}
    for key in ("suite", "category", "difficulty_level"):
        grouped[key] = {str(value): aggregate([e for e in plan if e[key] == value])
                        for value in sorted({e[key] for e in plan}, key=str)}
    grouped["category_by_difficulty"] = {
        category: {str(level): aggregate([e for e in plan if e["category"] == category and e["difficulty_level"] == level])
                   for level in sorted({e["difficulty_level"] for e in plan if e["category"] == category}, key=str)}
        for category in sorted({e["category"] for e in plan})
    }
    overall = aggregate(plan)
    complete = overall["completed"] == len(plan)
    return {"complete": complete, "smoke": smoke,
            "official_benchmark_complete": complete and official_scope and not smoke,
            "official_success_rate": overall["success_rate"] if official_scope and not smoke else None,
            "overall": overall, "breakdowns": grouped}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def freeze_manifest(output, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != manifest:
            raise RuntimeError("Resume refused: checkpoint, settings, benchmark, or episode plan changed")
    else:
        if (output / "results.jsonl").exists():
            raise RuntimeError("Results exist without their immutable manifest")
        write_json(path, manifest)
    return json_hash(manifest)


def load_results(path, manifest_sha256):
    if not Path(path).exists():
        return []
    latest = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        if result.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Result belongs to a different evaluation manifest")
        eid = result["episode_id"]
        previous = latest.get(eid)
        if previous is not None:
            if previous.get("error") is None:
                raise ValueError("A completed episode cannot be retried")
            if result.get("attempt", 1) != previous.get("attempt", 1) + 1:
                raise ValueError("Invalid episode retry sequence")
        elif result.get("attempt", 1) != 1:
            raise ValueError("Missing first episode attempt")
        latest[eid] = result
    return list(latest.values())


def smoke_metadata(image_size=224):
    return {"protocol_version": 1, "variant": "heading_gaussian", "checkpoint_sha256": "0" * 64,
            "training_benchmark": "none", "selection_benchmark": "none", "image_size": image_size,
            "camera_render_size": 128, "image_resize": "bicubic",
            "image_orientation": "vertical_flip", "n_obs_steps": 2, "camera_names": list(CAMERAS),
            "state_dim": 9, "action_dim": 7, "prediction_horizon": 16, "execution_horizon": 8,
            "policy_kind": "noop_smoke_only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("libero", "libero_plus"), default="libero_plus")
    parser.add_argument("--libero-config-path", required=True, help="Explicit isolated LIBERO config directory")
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--policy-metadata", type=Path, help="Metadata JSON for offline --dry-run only")
    parser.add_argument("--training-manifest", type=Path, required=True,
                        help="Frozen dataset_manifest.json from the selected policy export")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--classification", type=Path, help="Defaults to the installed Plus classification JSON")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and save plan without creating simulators")
    parser.add_argument("--smoke", action="store_true", help="Run noop simulator checks; never produces benchmark scores")
    parser.add_argument("--smoke-max-steps", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.smoke_max_steps < 1 or len(set(args.suites)) != len(args.suites):
        parser.error("workers, smoke steps and unique suites must be positive")
    if args.policy_metadata and not args.dry_run:
        parser.error("--policy-metadata is only valid with --dry-run")
    config_path = Path(args.libero_config_path).resolve()
    if not (config_path / "config.yaml").is_file():
        parser.error("Explicit LIBERO config.yaml must already exist; global config is never modified")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path)
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark, get_libero_path
    import libero.libero as libero_module
    import contextlib
    # Plus prints thousands of task indices at suite construction.
    with contextlib.redirect_stdout(io.StringIO()):
        suites = {name: benchmark.get_benchmark_dict()[name](task_order_index=0) for name in args.suites}
    classification_path = args.classification or Path(libero_module.__file__).parent / "benchmark/task_classification.json"
    classification = json.loads(classification_path.read_text()) if args.benchmark == "libero_plus" else {}
    if args.benchmark == "libero" and any(suite.get_num_tasks() != 10 for suite in suites.values()):
        raise RuntimeError("Original LIBERO evaluation requires the original 10-task suites, not LIBERO-Plus")
    if args.smoke:
        metadata = smoke_metadata()
    elif args.policy_metadata:
        metadata = json.loads(args.policy_metadata.read_text())
    else:
        metadata = PolicyClient(args.server_url).metadata
    validate_metadata(metadata, args.benchmark, args.smoke)
    if not args.smoke and not metadata.get("dataset_manifest_sha256"):
        raise ValueError("The policy must identify its frozen training manifest")
    instruction_catalog = load_instruction_catalog(
        args.training_manifest, None if args.smoke else metadata["dataset_manifest_sha256"])
    trials = args.trials if args.trials is not None else (1 if args.benchmark == "libero_plus" else 50)
    plan, scope = make_plan(suites, classification, args.benchmark, trials, args.seed, args.offset, args.limit,
                            instruction_catalog, lambda task: language_variant_prompt(task, get_libero_path("bddl_files")))
    prompt_audit = audit_prompt_mapping(plan, instruction_catalog)
    official_scope = scope["all_available_selected"] and scope["all_four_suites"]
    if args.benchmark == "libero":
        official_scope = official_scope and trials == 50
    settings = {"benchmark": args.benchmark, "libero_config_path": str(config_path),
                "server_url": args.server_url, "policy": metadata, "settle_steps": 10,
                "camera_render_size": metadata.get("camera_render_size", 128),
                "image_resize": metadata.get("image_resize", "bicubic"),
                "instruction_catalog": instruction_catalog,
                "render_gpu_device_id": args.render_gpu_device_id, "smoke": args.smoke,
                "smoke_max_steps": args.smoke_max_steps if args.smoke else None}
    manifest = {"protocol_version": PROTOCOL_VERSION, "settings": settings, "scope": scope, "plan": plan,
                "prompt_audit": prompt_audit,
                "seed": args.seed, "trials": trials,
                "benchmark_module": str(Path(libero_module.__file__).resolve()),
                "benchmark_source_sha256": file_hash(benchmark.__file__),
                "classification_sha256": file_hash(classification_path) if classification else None,
                "libero_paths": {key: get_libero_path(key) for key in ("bddl_files", "init_states", "assets")},
                "config_sha256": file_hash(config_path / "config.yaml")}
    # Lock before creating the manifest as well as before writing results.
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_hash = freeze_manifest(args.output, manifest)
    write_json(args.output / "prompt_audit.json", prompt_audit)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "manifest_sha256": manifest_hash, **scope}, indent=2))
        return 0
    # Single writer per output directory; parallelize simulator workers inside
    # this coordinator rather than sharing its JSONL among independent jobs.
    results_path = args.output / "results.jsonl"
    results = load_results(results_path, manifest_hash)
    summarize_results(plan, results, official_scope, args.smoke)  # Validate resumed inputs before execution.
    completed = {result["episode_id"] for result in results if result.get("error") is None}
    latest = {result["episode_id"]: result for result in results}
    pending = [episode for episode in plan if episode["episode_id"] not in completed]
    write_json(args.output / "summary.json", summarize_results(plan, results, official_scope, args.smoke))
    with results_path.open("a", buffering=1) as stream:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(settings,)) as pool:
            futures = [pool.submit(evaluate_episode, episode) for episode in pending]
            last_summary = time.monotonic()
            for future in as_completed(futures):
                result = future.result()
                previous = latest.get(result["episode_id"])
                result["attempt"] = 1 if previous is None else previous.get("attempt", 1) + 1
                result["manifest_sha256"] = manifest_hash
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                latest[result["episode_id"]] = result
                results = list(latest.values())
                if len(results) % 10 == 0 or time.monotonic() - last_summary >= 30:
                    summary = summarize_results(plan, results, official_scope, args.smoke)
                    write_json(args.output / "summary.json", summary)
                    last_summary = time.monotonic()
                print(json.dumps({"completed": len(results), "total": len(plan), "episode_id": result["episode_id"],
                                  "success": result["success"], "error": result["error"]}), flush=True)
    summary = summarize_results(plan, results, official_scope, args.smoke)
    write_json(args.output / "summary.json", summary)
    return 0 if summary["complete"] else 1
