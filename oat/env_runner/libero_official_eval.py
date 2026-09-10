"""Reproducible single-checkpoint LIBERO evaluation with official initial states.

Each episode owns its simulator and policy RNG stream. Neither worker assignment
nor completion order changes the task, state, or action-noise sequence. Simulator
state is used to initialize/report evaluation only, never as policy conditioning.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from collections import deque

import numpy as np


PROTOCOL_VERSION = 1
_POLICY = None
_SETTINGS = None


def stable_seed(seed: int, *parts) -> int:
    payload = json.dumps([int(seed), *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_official_states(suite, task_index):
    # LIBERO's legacy helper omits weights_only=False; these bundled files
    # contain NumPy arrays, which modern PyTorch otherwise refuses to load.
    import torch
    from libero.libero import get_libero_path
    task = suite.get_task(task_index)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(path, map_location="cpu", weights_only=False)


def verify_repository_imports(repository_root):
    import sys
    expected = Path(repository_root).resolve() / "oat"
    actual = {}
    for name, module in tuple(sys.modules.items()):
        if name == "oat" or name.startswith("oat."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(expected):
                    raise RuntimeError(f"repository import contamination: {name} loaded from {path}")
                actual[name] = str(path)
    return actual


def make_episode_plan(task_names, n_per_task, seed, init_start=0):
    if n_per_task < 1 or init_start < 0:
        raise ValueError("n_per_task must be positive and init_start nonnegative")
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("task names must be nonempty and unique")
    # Interleave tasks so an interrupted pilot still covers the entire suite.
    return [
        {
            "episode_id": f"{task_name}/init-{init_index:03d}",
            "task_index": task_index,
            "task_name": task_name,
            "init_index": init_index,
            "env_seed": stable_seed(seed, "environment", task_name, init_index),
            "policy_seed": stable_seed(seed, "policy", task_name, init_index),
        }
        for init_index in range(init_start, init_start + n_per_task)
        for task_index, task_name in enumerate(task_names)
    ]



def assign_video_paths(plan, video_per_task, video_directory):
    """Select the first N planned episodes of each task, before any outcomes."""
    if video_per_task < 0:
        raise ValueError("video_per_task must be nonnegative")
    counts = {}
    assigned = []
    for episode in plan:
        episode = dict(episode)
        task = episode["task_name"]
        rank = counts.get(task, 0)
        counts[task] = rank + 1
        if rank < video_per_task:
            filename = f"task-{episode['task_index']:02d}_init-{episode['init_index']:03d}.mp4"
            episode["video_path"] = str(Path(video_directory) / filename)
        assigned.append(episode)
    return assigned


class ObservationVideoWriter:
    """Encode already-extracted frames; errors cannot alter episode decisions."""

    def __init__(self, path, fps=20):
        self.path = Path(path)
        self.fps = fps
        self.container = None
        self.stream = None
        self.frame_count = 0
        self.error = None
        self.closed = False

    def write_observation(self, observation):
        if self.error is not None or self.closed:
            return
        try:
            frame = observation["agentview_rgb"]
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("video requires an existing uint8 HxWx3 agentview_rgb observation")
            import av
            if self.container is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.container = av.open(str(self.path), mode="w")
                self.stream = self.container.add_stream("libx264", rate=self.fps)
                self.stream.width = frame.shape[1]
                self.stream.height = frame.shape[0]
                self.stream.pix_fmt = "yuv420p"
                self.stream.options = {"crf": "23", "preset": "veryfast"}
                self.stream.codec_context.thread_count = 1
            # Keep the encoder isolated from observation storage used by the policy.
            encoded = av.VideoFrame.from_ndarray(frame.copy(order="C"), format="rgb24")
            for packet in self.stream.encode(encoded):
                self.container.mux(packet)
            self.frame_count += 1
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.stream is not None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
        except Exception as error:
            if self.error is None:
                self.error = f"{type(error).__name__}: {error}"
        finally:
            if self.container is not None:
                try:
                    self.container.close()
                except Exception as error:
                    if self.error is None:
                        self.error = f"{type(error).__name__}: {error}"
            self.container = None
            self.stream = None

    def result_fields(self):
        return {
            "video_path": str(self.path) if self.path.exists() and self.frame_count else None,
            "video_frames": self.frame_count,
            "video_error": self.error,
        }

def summarize_results(plan, results):
    expected = {episode["episode_id"]: episode for episode in plan}
    if len(expected) != len(plan):
        raise ValueError("duplicate episodes in plan")
    seen = {}
    for result in results:
        episode_id = result["episode_id"]
        if episode_id not in expected or episode_id in seen:
            raise ValueError(f"unknown or duplicate episode: {episode_id}")
        if result["task_name"] != expected[episode_id]["task_name"]:
            raise ValueError("result task differs from episode plan")
        seen[episode_id] = result
    per_task = {}
    for task_name in sorted({entry["task_name"] for entry in plan}):
        task_plan = [entry for entry in plan if entry["task_name"] == task_name]
        task_results = [seen[entry["episode_id"]] for entry in task_plan if entry["episode_id"] in seen]
        valid = [result for result in task_results if result.get("error") is None]
        successes = sum(bool(result["success"]) for result in valid)
        per_task[task_name] = {
            "expected": len(task_plan), "completed": len(valid),
            "errors": len(task_results) - len(valid), "successes": successes,
            "success_rate": successes / len(valid) if valid else None,
        }
    complete = len(seen) == len(plan) and all(result.get("error") is None for result in seen.values())
    successes = sum(entry["successes"] for entry in per_task.values())
    count = sum(entry["completed"] for entry in per_task.values())
    # A partial run must not expose its biased partial mean as a benchmark SR.
    balanced = float(np.mean([entry["success_rate"] for entry in per_task.values()])) if complete else None
    interval = None
    if complete:
        # Wilson interval describes episode sampling uncertainty, not seed/model variance.
        p = successes / count
        z = 1.959963984540054
        denominator = 1 + z * z / count
        center = (p + z * z / (2 * count)) / denominator
        radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
        interval = [max(0., center - radius), min(1., center + radius)]
    return {
        "complete": complete, "expected_episodes": len(plan), "completed_episodes": count,
        "error_episodes": sum(result.get("error") is not None for result in seen.values()),
        "successes": successes, "mean_success_rate": balanced,
        "episode_wilson_95_interval": interval, "per_task": per_task,
    }


def load_policy(checkpoint, device):
    """Load only the selected policy state; auxiliary models remain embedded."""
    import dill
    import hydra
    import torch
    from oat.common.hydra_util import register_new_resolvers
    register_new_resolvers()
    with open(checkpoint, "rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    state_name = "ema_model" if bool(cfg.training.get("use_ema", False)) else "model"
    if state_name not in payload["state_dicts"]:
        raise ValueError(f"checkpoint is missing its configured {state_name} weights")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][state_name], strict=True)
    policy.to(device).eval().requires_grad_(False)
    return policy, cfg, state_name


def initialize_worker(settings):
    global _POLICY, _SETTINGS
    import torch
    torch.set_num_threads(int(settings["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if file_sha256(settings["checkpoint"]) != settings["checkpoint_sha256"]:
        raise RuntimeError("evaluation checkpoint snapshot changed")
    _POLICY, cfg, _ = load_policy(settings["checkpoint"], settings["device"])
    verify_repository_imports(settings["repository_root"])
    _SETTINGS = dict(settings)
    runner = cfg.task.policy.env_runner
    _SETTINGS.update({
        "image_size": int(runner.get("image_size", 128)),
        "camera_names": list(runner.get("camera_names", ["agentview", "robot0_eye_in_hand"])),
        "state_ports": list(runner.get("state_ports", ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"])),
        "n_obs_steps": int(_POLICY.n_obs_steps),
    })


def observation_window(history, ports, device, dtype):
    import torch
    output = {}
    for key in ports:
        value = history[-1][key]
        if isinstance(value, np.ndarray):
            window = np.stack([entry[key] for entry in history], axis=0)[None]
            output[key] = torch.as_tensor(window, device=device, dtype=dtype)
        elif isinstance(value, str):
            output[key] = [value]
        else:
            raise TypeError(f"unsupported observation value for {key}: {type(value)}")
    return output


def rollout_episode(policy, env, episode, init_state, settings):
    """Run a preconstructed env; optional frame capture never enters policy inputs."""
    video = ObservationVideoWriter(episode["video_path"]) if episode.get("video_path") else None
    try:
        result = _rollout_episode(policy, env, episode, init_state, settings, video)
    finally:
        if video is not None:
            video.close()
    if video is not None:
        result.update(video.result_fields())
    return result


def _rollout_episode(policy, env, episode, init_state, settings, video=None):
    import torch
    env.env.seed(int(episode["env_seed"]))
    env.env.reset()
    env.env.set_init_state(init_state)
    env.done = False
    env.cur_step = 0
    settle_action = np.array([0.] * 6 + [-1.], dtype=np.float32)
    for _ in range(int(settings["settle_steps"])):
        env.env.step(settle_action)
    # Fetch after settling; old LiberoEnv.reset returns a stale pre-settle frame.
    obs = env._extract_obs()
    if video is not None:
        video.write_observation(obs)
    initial_hash = array_sha256(env.env.get_sim_state())
    if env.env.check_success():
        raise RuntimeError("official initial state already succeeds after settling")
    history = deque([obs] * int(settings["n_obs_steps"]), maxlen=int(settings["n_obs_steps"]))
    policy.reset()
    if callable(getattr(policy, "set_source_seed", None)):
        policy.set_source_seed(int(episode["policy_seed"]))
    action_hash = hashlib.sha256()
    max_steps = int(settings["max_episode_steps"])
    steps = 0
    control_cycles = 0
    success = False
    terminated = False
    while steps < max_steps and not (success or terminated):
        noise_seed = stable_seed(episode["policy_seed"], "control", control_cycles)
        torch.manual_seed(noise_seed)
        if policy.device.type == "cuda":
            torch.cuda.manual_seed(noise_seed)
        obs_batch = observation_window(history, policy.get_observation_ports(), policy.device, policy.dtype)
        with torch.inference_mode():
            action = policy.predict_action(obs_batch)["action"].detach().cpu().numpy()
        if action.ndim != 3 or action.shape[0] != 1 or action.shape[-1] != 7 or action.shape[1] < 1:
            raise ValueError(f"expected (1,T,7) action, got {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError("policy returned nonfinite actions")
        for act in action[0]:
            if steps >= max_steps:
                break
            action_hash.update(np.ascontiguousarray(act).tobytes())
            obs, reward, terminated, truncated, _ = env.step(act)
            if video is not None:
                video.write_observation(obs)
            history.append(obs)
            steps += 1
            success = bool(reward >= 1.)
            terminated = bool(terminated or truncated)
            if success or terminated:
                break
        control_cycles += 1
    return {
        **episode, "success": success, "executed_steps": steps,
        "control_cycles": control_cycles, "initial_state_sha256": initial_hash,
        "executed_actions_sha256": action_hash.hexdigest(),
        "termination": "success" if success else ("horizon" if steps >= max_steps else "environment_done"),
        "error": None,
    }


def evaluate_episode(episode):
    import torch
    from libero.libero import benchmark
    from oat.env.libero.env import LiberoEnv
    started = time.monotonic()
    env = None
    try:
        random.seed(int(episode["env_seed"]))
        np.random.seed(int(episode["env_seed"]))
        torch.manual_seed(int(episode["env_seed"]))
        suite = benchmark.get_benchmark_dict()[_SETTINGS["suite"]]()
        task = suite.get_task(int(episode["task_index"]))
        if task.name != episode["task_name"]:
            raise RuntimeError("installed LIBERO task ordering changed")
        states = load_official_states(suite, int(episode["task_index"]))
        init_state = np.asarray(states[int(episode["init_index"])])
        if array_sha256(init_state) != episode["official_state_sha256"]:
            raise RuntimeError("official LIBERO initial state changed after planning")
        env = LiberoEnv(
            task_name=task.name, seed=int(episode["env_seed"]),
            image_size=_SETTINGS["image_size"], camera_names=_SETTINGS["camera_names"],
            state_ports=_SETTINGS["state_ports"], max_episode_steps=_SETTINGS["max_episode_steps"],
        )
        result = rollout_episode(_POLICY, env, episode, init_state, _SETTINGS)
    except Exception as error:
        import traceback
        result = {**episode, "success": False, "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    result["wall_seconds"] = time.monotonic() - started
    return result
