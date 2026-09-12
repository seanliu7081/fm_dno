"""Inline MimicGen rollouts sharded over the same devices used for training.

Each training rank owns policy inference on its existing CUDA device and a
temporary pool of simulator workers rendering on that same physical GPU.
The workspace gathers ``_eval_counts`` across ranks after every shard finishes.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
import random
import time

import dill
import numpy as np
import torch

from oat.env.mimicgen.env import (
    DEFAULT_CAMERAS, DEFAULT_STATE_PORTS, MimicGenEnv,
    physical_gpu_index, require_delta_controller,
)
from oat.env_runner.base_runner import BaseRunner
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper


def stable_seed(seed, *parts):
    payload = json.dumps([int(seed), *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def validate_manifest(manifest, n_per_task):
    if manifest.get("schema_version") != 1 or manifest.get("action_mode") != "delta":
        raise ValueError("MimicGen evaluation requires a version-1 delta-action manifest")
    tasks = manifest.get("tasks", [])
    if not tasks or len({task["task_name"] for task in tasks}) != len(tasks):
        raise ValueError("MimicGen task names must be nonempty and unique")
    if len({task["task_uid"] for task in tasks}) != len(tasks):
        raise ValueError("MimicGen scalar task_uids must be unique")
    for task in tasks:
        train, evaluation = task["train_demo_names"], task["eval_demo_names"]
        train_hashes, eval_hashes = task["train_initial_state_sha256"], task["eval_initial_state_sha256"]
        if not train or len(train) != len(train_hashes):
            raise ValueError("training demo names and initial-state hashes must match")
        if len(evaluation) < n_per_task or len(evaluation) != len(eval_hashes):
            raise ValueError(f"{task['task_name']} has insufficient held-out initial states")
        if len(set(train)) != len(train) or len(set(evaluation)) != len(evaluation):
            raise ValueError("duplicate train or held-out demonstration names")
        if set(train) & set(evaluation) or set(train_hashes) & set(eval_hashes):
            raise ValueError("held-out demonstrations or initial states overlap training")
        if len(set(eval_hashes)) != len(eval_hashes):
            raise ValueError("held-out initial states must be unique within each task")
        if int(task["horizon"]) <= 0:
            raise ValueError("task rollout horizon must be positive")
        require_delta_controller(task["env_meta"])
    return tasks


def make_episode_plan(tasks, n_per_task, seed, rank=0, world_size=1):
    if n_per_task < 1 or world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid episode count or distributed rank")
    plan = []
    for task in tasks:
        # Shard within each task: both GPUs evaluate every task (25 tests each
        # with a 50-test, two-GPU protocol). Sharding an interleaved task list
        # by global index would accidentally assign whole tasks to one GPU.
        for index in range(rank, n_per_task, world_size):
            demo = task["eval_demo_names"][index]
            plan.append({
                "episode_id": f"{task['task_name']}/{demo}",
                "task_name": task["task_name"], "task_uid": int(task["task_uid"]),
                "demo_name": demo, "eval_index": index,
                "env_seed": stable_seed(seed, "environment", task["task_name"], demo),
                "policy_seed": stable_seed(seed, "policy", task["task_name"], demo),
                "initial_state_sha256": task["eval_initial_state_sha256"][index],
            })
    return plan


class _EpisodeChunkEnv(MultiStepWrapper):
    """Keep completed episodes frozen until the next explicit vector reset.

AsyncVectorEnv normally auto-resets a completed slot on its next step. Suppress
that behavior and expose the true terminal status in info, so successful slots
cannot silently start another episode while other slots are still running.
"""
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["episode_done"] = bool(terminated or truncated)
        # A slot can succeed on its first primitive action. Keep its info shape
        # scalar rather than a shorter history than the other vector slots.
        info["steps"] = int(np.asarray(info["steps"])[-1])
        return obs, reward, False, False, info


def _make_env(task, eval_hdf5, image_size, native_image_size, camera_names, state_ports,
              max_episode_steps, physical_gpu, n_obs_steps, n_action_steps):
    return _EpisodeChunkEnv(
        MimicGenEnv(task, eval_hdf5, image_size=image_size, native_image_size=native_image_size,
                    camera_names=camera_names, state_ports=state_ports,
                    max_episode_steps=max_episode_steps,
                    render_gpu_device_id=physical_gpu),
        n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps or task["horizon"],
        reward_agg_method="max",
    )


def _select_episode(env, episode):
    env.env.set_episode(episode)


@contextmanager
def _preserve_rng(device, seed):
    """Inline evaluation must not alter the subsequent training RNG stream."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class MimicGenRunner(BaseRunner):
    def __init__(self, output_dir, manifest_path, n_per_task=50,
                 n_parallel_envs=4, n_obs_steps=2, n_action_steps=8,
                 image_size=128, camera_names=DEFAULT_CAMERAS,
                 state_ports=DEFAULT_STATE_PORTS, seed=20260912,
                 max_episode_steps=None, rank=0, world_size=1, device=None):
        super().__init__(output_dir)
        for value in (n_per_task, n_parallel_envs, n_obs_steps, n_action_steps,
                      image_size, seed, rank, world_size):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("MimicGen runner counts and seeds must be integers")
        if min(n_per_task, n_parallel_envs, n_obs_steps, n_action_steps, image_size, world_size) < 1:
            raise ValueError("MimicGen runner counts must be positive")
        if not 0 <= rank < world_size or n_per_task < world_size or seed < 0:
            raise ValueError("invalid distributed rank, test count or seed")
        self.manifest_path = Path(manifest_path).resolve()
        raw_manifest = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        self.manifest = json.loads(raw_manifest)
        self.tasks = validate_manifest(self.manifest, n_per_task)
        if self.manifest.get("image_size", image_size) != image_size:
            raise ValueError("evaluation image size differs from training dataset manifest")
        self.native_image_size = int(self.manifest.get("native_image_size", 84))
        eval_path = Path(self.manifest["eval_hdf5"])
        self.eval_hdf5 = str(eval_path if eval_path.is_absolute() else self.manifest_path.parent / eval_path)
        if not Path(self.eval_hdf5).is_file():
            raise FileNotFoundError(self.eval_hdf5)
        self.plan = make_episode_plan(self.tasks, n_per_task, seed, rank, world_size)
        self.n_per_task, self.n_parallel_envs = n_per_task, n_parallel_envs
        self.n_obs_steps, self.n_action_steps = n_obs_steps, n_action_steps
        self.image_size, self.camera_names, self.state_ports = image_size, tuple(camera_names), tuple(state_ports)
        self.max_episode_steps = max_episode_steps
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.device = torch.device(device) if device is not None else None
        self.env = None
        self.run_index = 0

    def _vector(self, task, count, device):
        factory = partial(
            _make_env, task=task, eval_hdf5=self.eval_hdf5,
            image_size=self.image_size, native_image_size=self.native_image_size, camera_names=self.camera_names,
            state_ports=self.state_ports, max_episode_steps=self.max_episode_steps,
            physical_gpu=physical_gpu_index(device), n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
        )
        # Factory construction creates spaces only, never a simulator. Each
        # worker's first reset imports robosuite after setting its EGL device.
        return AsyncVectorEnv([factory] * count, dummy_env_fn=factory,
                              shared_memory=False, context="spawn")

    @torch.inference_mode()
    def run(self, policy, epoch=None, global_step=None, **kwargs):
        device = torch.device(policy.device)
        if self.device is not None and device != self.device:
            raise ValueError(f"evaluation policy is on {device}, expected training device {self.device}")
        self.close()
        self.run_index += 1
        label = f"epoch_{int(epoch) + 1:03d}" if epoch is not None else f"run_{self.run_index:03d}"
        directory = Path(self.output_dir) / "rollouts" / label / f"rank_{self.rank}"
        directory.mkdir(parents=True, exist_ok=True)
        # Preserve artifacts from an interrupted evaluation/resume attempt.
        attempt = 0
        while (directory / f"attempt_{attempt:03d}").exists():
            attempt += 1
        directory = directory / f"attempt_{attempt:03d}"
        directory.mkdir()
        protocol = {
            "manifest_path": str(self.manifest_path), "manifest_sha256": self.manifest_sha256,
            "rank": self.rank, "world_size": self.world_size,
            "device": str(device), "physical_gpu": physical_gpu_index(device),
            "n_per_task": self.n_per_task, "seed": self.seed,
            "epoch": epoch, "global_step": global_step,
            "n_obs_steps": self.n_obs_steps, "n_action_steps": self.n_action_steps,
            "episodes": self.plan,
        }
        (directory / "manifest.json").write_text(json.dumps(protocol, indent=2) + "\n")
        rows = []
        started = time.monotonic()
        was_training = getattr(policy, "training", False)
        try:
            policy.eval()
            with _preserve_rng(device, stable_seed(self.seed, "rank", self.rank)):
                with (directory / "episodes.jsonl").open("w", buffering=1) as output:
                    for task in self.tasks:
                        task_plan = [episode for episode in self.plan if episode["task_name"] == task["task_name"]]
                        count = min(self.n_parallel_envs, len(task_plan))
                        self.env = self._vector(task, count, device)
                        try:
                            for start in range(0, len(task_plan), count):
                                active = task_plan[start:start + count]
                                padded = active + [active[-1]] * (count - len(active))
                                functions = [dill.dumps(partial(_select_episode, episode=episode)) for episode in padded]
                                self.env.call_each("run_dill_function", args_list=[(fn,) for fn in functions])
                                obs, _ = self.env.reset()
                                policy.reset()
                                success = np.zeros(count, dtype=bool)
                                done = np.zeros(count, dtype=bool)
                                steps = np.zeros(count, dtype=int)
                                horizon = self.max_episode_steps or task["horizon"]
                                # Shared-heading policies accept explicit Gaussian draws,
                                # making each test's source noise independent of sharding.
                                explicit_noise = "noise" in inspect.signature(policy.predict_action).parameters
                                generators = [torch.Generator(device=device).manual_seed(episode["policy_seed"])
                                              for episode in padded] if explicit_noise else None
                                for _ in range(0, horizon, self.n_action_steps):
                                    inputs = {key: torch.as_tensor(obs[key], device=device, dtype=policy.dtype)
                                              for key in policy.get_observation_ports()}
                                    inference = dict(kwargs)
                                    if explicit_noise:
                                        inference["noise"] = torch.stack([
                                            torch.randn(policy.horizon, policy.action_dim, generator=generator,
                                                        device=device, dtype=policy.dtype)
                                            for generator in generators
                                        ])
                                    actions = policy.predict_action(inputs, **inference)["action"].detach().cpu().numpy()
                                    if actions.shape != (count, self.n_action_steps, 7) or not np.isfinite(actions).all():
                                        raise ValueError("rollout policy must produce finite [batch, action_steps, 7] delta actions")
                                    obs, rewards, _, _, infos = self.env.step(actions)
                                    success |= np.asarray(rewards) >= 1
                                    step_values = np.asarray(infos["steps"])
                                    steps[:] = step_values[:, -1] if step_values.ndim > 1 else step_values
                                    done |= np.asarray(infos["episode_done"], dtype=bool)
                                    if done[:len(active)].all():
                                        break
                                if not done[:len(active)].all():
                                    raise RuntimeError("MimicGen rollout did not terminate within its configured horizon")
                                for index, episode in enumerate(active):
                                    row = {**episode, "rank": self.rank, "device": str(device),
                                           "success": bool(success[index]), "steps": min(int(steps[index]), horizon),
                                           "error": None}
                                    rows.append(row)
                                    output.write(json.dumps(row) + "\n")
                        finally:
                            self.close()
            if len(rows) != len(self.plan):
                raise RuntimeError("evaluation shard did not complete its episode plan")
            counts = {}
            for task in self.tasks:
                task_rows = [row for row in rows if row["task_name"] == task["task_name"]]
                counts[f"{task['task_name']}/mean_success_rate"] = [sum(row["success"] for row in task_rows), len(task_rows)]
            total = [sum(row["success"] for row in rows), len(rows)]
            counts["mean_success_rate"] = total
            counts["test/mean_score"] = total
            summary = {"complete": True, "rank": self.rank, "episodes": len(rows),
                       "successes": total[0], "counts": counts,
                       "elapsed_seconds": time.monotonic() - started}
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            return {"_eval_counts": counts,
                    f"eval/rank_{self.rank}/episodes": len(rows),
                    f"eval/rank_{self.rank}/elapsed_seconds": summary["elapsed_seconds"]}
        except BaseException as error:
            (directory / "error.json").write_text(json.dumps({"error": repr(error), "completed_episodes": len(rows)}) + "\n")
            raise
        finally:
            self.close()
            policy.train(was_training)

    def close(self):
        env, self.env = self.env, None
        if env is not None:
            env.close(timeout=5)
