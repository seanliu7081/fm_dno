"""Rollout workers must not survive into the next training epoch."""

import numpy as np
import pytest
import torch

from oat.env_runner import libero_runner
from oat.env_runner.libero_dno_runner import OrbitDnoLiberoRunner


class FakeVectorEnv:
    def __init__(self, env_fns, **kwargs):
        self.num_envs = len(env_fns)
        self.context = kwargs['context']
        self.closed = False
        self.init_calls = []

    def call_each(self, name, args_list):
        self.init_calls.append((name, args_list))

    def reset(self):
        return {'state': np.zeros((self.num_envs, 2, 3), dtype=np.float32)}, {}

    def step(self, action):
        obs, info = self.reset()
        return obs, np.ones(self.num_envs), np.ones(self.num_envs, dtype=bool), False, info

    def render(self):
        return [None] * self.num_envs

    def close(self, timeout):
        self.closed = True


class FakePolicy:
    device = torch.device('cpu')
    dtype = torch.float32

    def __init__(self, fail=False):
        self.fail = fail
        self.grad_modes = []

    def get_policy_name(self):
        return 'fake'

    def get_observation_ports(self):
        return ['state']

    def reset(self):
        pass

    def predict_action(self, obs):
        self.grad_modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        if self.fail:
            raise ValueError('policy failed during rollout')
        return {'action': torch.zeros(obs['state'].shape[0], 1, 7)}


@pytest.fixture
def vector_envs(monkeypatch):
    created = []

    def factory(*args, **kwargs):
        env = FakeVectorEnv(*args, **kwargs)
        created.append(env)
        return env

    monkeypatch.setattr(libero_runner, 'get_subtasks', lambda name: ['task_a', 'task_b'])
    monkeypatch.setattr(libero_runner, 'AsyncVectorEnv', factory)
    return created


def make_runner(tmp_path, runner_class):
    return runner_class(
        output_dir=str(tmp_path), task_name='fake', n_test=4, n_test_vis=0,
        n_parallel_envs=2, n_action_steps=1, max_episode_steps=1,
    )


@pytest.mark.parametrize('runner_class', [libero_runner.LiberoRunner, OrbitDnoLiberoRunner])
def test_workers_are_lazy_and_released_between_evaluations(tmp_path, vector_envs, runner_class):
    runner = make_runner(tmp_path, runner_class)
    assert not vector_envs
    runner.close()  # Safe before the first evaluation.
    policy = FakePolicy()
    first = runner.run(policy)
    assert first['mean_success_rate'] == 1.0
    assert vector_envs[0].closed
    assert runner.env is None
    second = runner.run(policy)
    assert second == first
    assert len(vector_envs) == 2
    assert all(env.closed and env.context == 'spawn' for env in vector_envs)
    assert vector_envs[0].init_calls == vector_envs[1].init_calls
    runner.close()  # Safe after cleanup as well.
    expected = (True, False) if runner_class is OrbitDnoLiberoRunner else (False, True)
    assert all(mode == expected for mode in policy.grad_modes)


@pytest.mark.parametrize('runner_class', [libero_runner.LiberoRunner, OrbitDnoLiberoRunner])
def test_policy_exception_releases_workers(tmp_path, vector_envs, runner_class):
    runner = make_runner(tmp_path, runner_class)
    with pytest.raises(ValueError, match='policy failed during rollout'):
        runner.run(FakePolicy(fail=True))
    assert vector_envs[0].closed
    assert runner.env is None
    assert runner.run(FakePolicy())['mean_success_rate'] == 1.0
    assert len(vector_envs) == 2
