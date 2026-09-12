"""Gymnasium adapter for MimicGen's robosuite delta-action environments.

Simulator imports are deliberately lazy: training ranks and vector-space probes
must not create an EGL context. Only spawned rollout workers create simulators.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
import random

import gymnasium as gym
import h5py
import numpy as np


STATE_SHAPES = {
    "robot0_joint_pos": (7,),
    "robot0_eef_pos": (3,),
    "robot0_eef_quat": (4,),
    "robot0_gripper_qpos": (2,),
}
DEFAULT_STATE_PORTS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
DEFAULT_CAMERAS = ("agentview", "robot0_eye_in_hand")


def initial_state_sha256(state):
    return hashlib.sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes()).hexdigest()


def physical_gpu_index(device, visible_devices=None):
    """Map a rank-local CUDA ordinal to the physical EGL device ordinal."""
    name = str(device)
    if not name.startswith("cuda"):
        return None
    logical = int(name.split(":")[1]) if ":" in name else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if visible:
        devices = [item.strip() for item in visible.split(",")]
        if logical >= len(devices) or not devices[logical].isdigit():
            raise ValueError("MimicGen EGL needs integer CUDA_VISIBLE_DEVICES ordinals")
        return int(devices[logical])
    return logical


def require_delta_controller(env_meta):
    controller = env_meta.get("env_kwargs", {}).get("controller_configs", {})
    if not isinstance(controller, dict) or controller.get("type") != "OSC_POSE":
        raise ValueError("MimicGen delta actions require an OSC_POSE controller")
    if controller.get("control_delta", True) is not True:
        raise ValueError("MimicGen evaluation refuses an absolute-action controller")


class MimicGenEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task, eval_hdf5, image_size=128, native_image_size=84,
                 camera_names=DEFAULT_CAMERAS, state_ports=DEFAULT_STATE_PORTS,
                 max_episode_steps=None, render_gpu_device_id=0):
        super().__init__()
        require_delta_controller(task["env_meta"])
        self.task = deepcopy(task)
        self.task_name = task["task_name"]
        self.task_uid = int(task["task_uid"])
        self.eval_hdf5 = str(eval_hdf5)
        self.image_size = int(image_size)
        self.native_image_size = int(native_image_size)
        self.camera_names = tuple(camera_names)
        self.state_ports = tuple(state_ports)
        self.max_episode_steps = int(max_episode_steps or task["horizon"])
        self.render_gpu_device_id = render_gpu_device_id
        self.env = None
        self.episode = None
        self.cur_step = 0
        self.success = False
        self.done = False
        spaces = {
            port: gym.spaces.Box(-np.inf, np.inf, STATE_SHAPES[port], np.float32)
            for port in self.state_ports
        }
        spaces.update({
            f"{camera}_rgb": gym.spaces.Box(0, 255, (image_size, image_size, 3), np.uint8)
            for camera in self.camera_names
        })
        spaces["task_uid"] = gym.spaces.Box(0, 255, (1,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(-1., 1., (7,), np.float32)

    def set_episode(self, episode):
        if episode["task_name"] != self.task_name:
            raise ValueError("episode belongs to a different MimicGen task")
        if episode["demo_name"] not in self.task["eval_demo_names"]:
            raise ValueError("episode is not in the held-out split")
        self.episode = dict(episode)

    def _build(self):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        if self.render_gpu_device_id is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.render_gpu_device_id)
        import mimicgen  # noqa: F401; registers MimicGen environment classes.
        import robosuite

        kwargs = deepcopy(self.task["env_meta"]["env_kwargs"])
        kwargs.update(
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            use_object_obs=True, camera_names=list(self.camera_names),
            camera_heights=self.native_image_size, camera_widths=self.native_image_size,
            camera_depths=False, ignore_done=True, hard_reset=False,
            horizon=self.max_episode_steps,
            render_gpu_device_id=self.render_gpu_device_id or 0,
        )
        # Preserve the dataset's controller scales while explicitly using delta control.
        kwargs["controller_configs"]["control_delta"] = True
        self.env = robosuite.make(self.task["env_meta"]["env_name"], **kwargs)
        if self.env.action_dim != 7:
            raise ValueError(f"expected seven delta-action channels, got {self.env.action_dim}")

    def _extract_obs(self, raw):
        result = {port: np.asarray(raw[port], dtype=np.float32) for port in self.state_ports}
        # robomimic image datasets are already vertically corrected. Raw robosuite
        # images require exactly one flip to match those files and LIBERO zarr.
        for camera in self.camera_names:
            image = np.ascontiguousarray(raw[f"{camera}_image"][::-1], dtype=np.uint8)
            if self.image_size != self.native_image_size:
                import cv2
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            result[f"{camera}_rgb"] = image
        result["task_uid"] = np.array([self.task_uid], dtype=np.float32)
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.episode is None:
            raise RuntimeError("set_episode must select a held-out state before reset")
        episode = self.episode
        np.random.seed(episode["env_seed"])
        random.seed(episode["env_seed"])
        with h5py.File(self.eval_hdf5, "r") as data:
            group = data[f"data/{self.task_name}/{episode['demo_name']}"]
            state = np.asarray(group["states"], dtype=np.float64)
            model = group.attrs["model_file"]
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("held-out initial state must be a finite flat simulator state")
        digest = initial_state_sha256(state)
        if digest != episode["initial_state_sha256"]:
            raise ValueError("held-out initial state hash differs from the evaluation manifest")
        if digest in self.task["train_initial_state_sha256"]:
            raise ValueError("held-out initial state overlaps a training initial state")
        if self.env is None:
            self._build()
        model = model.decode("utf-8") if isinstance(model, bytes) else str(model)
        # MimicGen's editor resolves robosuite, mimicgen, and task-zoo asset paths
        # and applies task-specific changes needed to replay the released files.
        model = self.env.edit_model_xml(model)
        self.env.reset_from_xml_string(model)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        # Resetting raw MuJoCo state leaves OSC pose caches and its nullspace
        # joint reference at the default reset pose. Restore both from the
        # held-out robot state so the first delta action starts at that pose.
        for robot in self.env.robots:
            robot.controller.update(force=True)
            robot.controller.update_initial_joints(robot._joint_positions)
            robot.controller.reset_goal()
        restored = self.env.sim.get_state().flatten()
        if not np.allclose(restored, state, atol=1e-10, rtol=0):
            raise RuntimeError("simulator failed to restore the held-out initial state")
        self.cur_step, self.success, self.done = 0, False, False
        raw = self.env._get_observations(force_update=True)
        self.last_obs = self._extract_obs(raw)
        return self.last_obs, {"initial_state_sha256": digest}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("MimicGen action must contain seven finite delta controls")
        if not self.done:
            raw, _, _, _ = self.env.step(np.clip(action, -1., 1.))
            success = self.env._check_success()
            self.success = bool(success["task"] if isinstance(success, dict) else success)
            self.cur_step += 1
            self.done = self.success or self.cur_step >= self.max_episode_steps
            self.last_obs = self._extract_obs(raw)
        return self.last_obs, float(self.success), self.done, False, {"steps": self.cur_step}

    def close(self):
        env, self.env = self.env, None
        if env is not None:
            env.close()
