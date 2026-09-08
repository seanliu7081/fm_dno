import multiprocessing
import os
import signal

import gymnasium as gym
import numpy as np
import pytest

from oat.gymnasium_util.async_vector_env import AsyncVectorEnv


class TinyEnv(gym.Env):
    metadata = {}
    render_mode = None

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    def echo(self, value):
        return value

    def fail(self):
        raise ValueError("runtime worker failure")

    def exit_without_reply(self):
        os._exit(17)


def failing_env():
    raise LookupError("environment initialization failed")


def make_vector(n_envs=2):
    return AsyncVectorEnv(
        [TinyEnv] * n_envs, dummy_env_fn=TinyEnv,
        shared_memory=False, context="spawn",
    )


def assert_closed(env):
    assert env.closed
    assert all(not process.is_alive() for process in env.processes)
    assert all(pipe is None or pipe.closed for pipe in env.parent_pipes)


def test_call_each_and_normal_close():
    env = make_vector()
    try:
        assert env.call_each("echo", args_list=[(1,), (2,)]) == (1, 2)
        obs, _ = env.reset()
        assert obs.shape == (2, 1)
        assert env.step(np.zeros((2, 1), dtype=np.float32))[0].shape == (2, 1)
    finally:
        env.close(timeout=5)
    assert_closed(env)


def test_initialization_exception_preserves_traceback_and_cleans_workers():
    previous_children = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(LookupError, match="environment initialization failed") as caught:
        AsyncVectorEnv(
            [TinyEnv, failing_env], dummy_env_fn=TinyEnv,
            shared_memory=False, context="spawn",
        )
    assert "failing_env" in str(caught.value.__cause__)
    assert "Original traceback from Worker-1" in str(caught.value.__cause__)
    assert {process.pid for process in multiprocessing.active_children()} <= previous_children


def test_runtime_exception_preserves_type_traceback_and_can_close():
    env = make_vector()
    try:
        with pytest.raises(ValueError, match="runtime worker failure") as caught:
            env.call_each("fail")
        assert "in fail" in str(caught.value.__cause__)
        assert "Original traceback from Worker-" in str(caught.value.__cause__)
    finally:
        env.close(timeout=5)
    assert_closed(env)


def test_terminated_worker_reports_pid_exitcode_signal_and_can_close():
    env = make_vector()
    process = env.processes[0]
    process.terminate()
    process.join(timeout=5)
    try:
        with pytest.raises(RuntimeError) as caught:
            env.call_each("echo", args_list=[(1,), (2,)])
        message = str(caught.value)
        assert "Worker-0" in message
        assert f"pid={process.pid}" in message
        assert f"exitcode={-signal.SIGTERM}" in message
        assert "signal=SIGTERM" in message
    finally:
        env.close(timeout=5)
    assert_closed(env)


def test_worker_exit_during_call_reports_exitcode_and_can_close():
    env = make_vector(1)
    try:
        with pytest.raises(RuntimeError, match="exitcode=17"):
            env.call_each("exit_without_reply")
    finally:
        env.close(timeout=5)
    assert_closed(env)


def test_close_after_external_worker_termination():
    env = make_vector()
    env.processes[0].terminate()
    env.processes[0].join(timeout=5)
    env.close(timeout=5)
    assert_closed(env)


def test_close_cleans_workers_when_pending_call_raises():
    env = make_vector()
    env.call_async("fail")
    env.close(timeout=5)
    assert_closed(env)
