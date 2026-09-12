"""Held-out split, GPU sharding and inline runner lifecycle checks."""
from copy import deepcopy
import json
import random

import cv2
import gymnasium as gym
import h5py
import numpy as np
import pytest
import torch

from oat.env.mimicgen.env import MimicGenEnv, physical_gpu_index
from oat.env_runner import mimicgen_runner as module


def task(index=0):
    return {
        'task_name': f'task_{index}', 'task_uid': index, 'horizon': 1,
        'train_demo_names': ['demo_0'],
        'eval_demo_names': [f'demo_{i}' for i in range(1, 5)],
        'train_initial_state_sha256': ['train'],
        'eval_initial_state_sha256': [f'eval_{i}' for i in range(1, 5)],
        'env_meta': {'env_name': 'Fake', 'env_kwargs': {'controller_configs': {'type': 'OSC_POSE', 'control_delta': True}}},
    }


def manifest(tmp_path):
    value = {'schema_version': 1, 'action_mode': 'delta', 'eval_hdf5': 'eval.hdf5', 'tasks': [task(0), task(1)]}
    (tmp_path / 'manifest.json').write_text(json.dumps(value))
    with h5py.File(tmp_path / 'eval.hdf5', 'w'):
        pass
    return tmp_path / 'manifest.json'


def test_shards_cover_every_task_without_repeated_tests():
    tasks = [task(index) for index in range(6)]
    for item in tasks:
        item['eval_demo_names'] = [f'demo_{i}' for i in range(100, 150)]
        item['eval_initial_state_sha256'] = [f'hash_{i}' for i in range(50)]
    plans = [module.make_episode_plan(tasks, 50, 123, rank, 2) for rank in range(2)]
    ids = [set(row['episode_id'] for row in plan) for plan in plans]
    assert not ids[0] & ids[1]
    assert len(ids[0] | ids[1]) == 300
    for plan in plans:
        assert len(plan) == 150
        for item in tasks:
            assert sum(row['task_name'] == item['task_name'] for row in plan) == 25
    combined = {row['episode_id']: row for plan in plans for row in plan}
    unsharded = {row['episode_id']: row for row in module.make_episode_plan(tasks, 50, 123)}
    assert combined == unsharded


@pytest.mark.parametrize('kind', ['name', 'hash', 'duplicate'])
def test_manifest_rejects_training_overlap_and_duplicate_test_states(kind):
    value = {'schema_version': 1, 'action_mode': 'delta', 'tasks': [task()]}
    item = value['tasks'][0]
    if kind == 'name':
        item['eval_demo_names'][0] = item['train_demo_names'][0]
    elif kind == 'hash':
        item['eval_initial_state_sha256'][0] = item['train_initial_state_sha256'][0]
    else:
        item['eval_initial_state_sha256'][1] = item['eval_initial_state_sha256'][0]
    with pytest.raises(ValueError, match='overlap|unique'):
        module.validate_manifest(value, 4)


def test_physical_gpu_mapping_is_local_to_the_training_pair(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    assert physical_gpu_index('cuda:0') == 2
    assert physical_gpu_index('cuda:1') == 3
    assert physical_gpu_index('cpu') is None
    with pytest.raises(ValueError):
        physical_gpu_index('cuda:2')


class FakePolicy(torch.nn.Module):
    device = torch.device('cpu')
    dtype = torch.float32
    horizon = 1
    action_dim = 7

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.noise = []

    def reset(self):
        pass

    def get_observation_ports(self):
        return ['robot0_eef_pos']

    def predict_action(self, observations, noise=None):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        if self.fail:
            raise RuntimeError('inference failed')
        self.noise.append(noise.clone())
        torch.rand(1)
        np.random.rand()
        random.random()
        return {'action': torch.zeros(observations['robot0_eef_pos'].shape[0], 1, 7)}


class FakeVector:
    def __init__(self, count):
        self.count = count
        self.closed = False
        self.selections = []

    def call_each(self, name, args_list):
        self.selections.extend(args_list)

    def reset(self):
        return {'robot0_eef_pos': np.zeros((self.count, 2, 3), dtype=np.float32)}, {}

    def step(self, actions):
        obs, _ = self.reset()
        return obs, np.ones(self.count), np.zeros(self.count), np.zeros(self.count), {'episode_done': np.ones(self.count, dtype=bool), 'steps': np.ones((self.count, 1), dtype=int)}

    def close(self, timeout):
        self.closed = True


@pytest.fixture
def runner(tmp_path, monkeypatch):
    instance = module.MimicGenRunner(tmp_path, manifest(tmp_path), n_per_task=4,
                                    n_parallel_envs=2, n_action_steps=1, rank=1, world_size=2, device='cpu')
    created = []

    def vector(task, count, device):
        result = FakeVector(count)
        created.append(result)
        return result

    monkeypatch.setattr(instance, '_vector', vector)
    return instance, created, tmp_path


def test_inline_runner_counts_rng_and_worker_cleanup(runner):
    instance, created, tmp_path = runner
    assert not created
    torch.manual_seed(91)
    np.random.seed(91)
    random.seed(91)
    expected_torch = torch.get_rng_state()
    expected_numpy = np.random.get_state()
    expected_python = random.getstate()
    policy = FakePolicy()
    result = instance.run(policy, epoch=49, global_step=300)
    assert result['_eval_counts']['test/mean_score'] == [4, 4]
    assert result['_eval_counts']['task_0/mean_success_rate'] == [2, 2]
    assert len(created) == 2 and all(env.closed for env in created)
    assert instance.env is None and policy.training
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert np.array_equal(np.random.get_state()[1], expected_numpy[1])
    assert random.getstate() == expected_python
    assert (tmp_path / 'rollouts/epoch_050/rank_1/attempt_000/summary.json').is_file()
    instance.run(policy, epoch=49, global_step=300)
    assert len(created) == 4 and all(env.closed for env in created)
    assert torch.equal(policy.noise[0], policy.noise[2])
    assert (tmp_path / 'rollouts/epoch_050/rank_1/attempt_001/summary.json').is_file()


def test_failure_releases_workers_and_preserves_incomplete_evidence(runner):
    instance, created, tmp_path = runner
    with pytest.raises(RuntimeError, match='inference failed'):
        instance.run(FakePolicy(fail=True), epoch=49)
    assert len(created) == 1 and created[0].closed and instance.env is None
    assert (tmp_path / 'rollouts/epoch_050/rank_1/attempt_000/error.json').is_file()
    assert not (tmp_path / 'rollouts/epoch_050/rank_1/attempt_000/summary.json').exists()


class OneStepEnv(gym.Env):
    observation_space = gym.spaces.Dict({'value': gym.spaces.Box(-1, 1, (1,), np.float32)})
    action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

    def __init__(self):
        self.calls = 0

    def reset(self, **kwargs):
        return {'value': np.zeros(1, np.float32)}, {}

    def step(self, action):
        self.calls += 1
        return {'value': np.ones(1, np.float32)}, 1., True, False, {'steps': self.calls}


def test_completed_vector_slot_is_frozen_without_autoreset():
    env = OneStepEnv()
    wrapper = module._EpisodeChunkEnv(env, 2, 8, max_episode_steps=10)
    wrapper.reset()
    for _ in range(3):
        _, reward, terminated, truncated, info = wrapper.step(np.zeros((8, 7)))
        assert reward == 1. and not terminated and not truncated and info['episode_done']
    assert env.calls == 1


def test_render_preprocessing_matches_native_dataset_resize(tmp_path):
    env = MimicGenEnv(task(), tmp_path / 'unused.hdf5', image_size=4, native_image_size=2)
    assert env.env is None
    raw = {key: np.zeros(shape, np.float32) for key, shape in [('robot0_eef_pos',(3,)),('robot0_eef_quat',(4,)),('robot0_gripper_qpos',(2,))]}
    source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    for camera in env.camera_names:
        raw[camera+'_image'] = source[::-1]
    obs = env._extract_obs(raw)
    expected = cv2.resize(source,(4,4),interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(obs['agentview_rgb'], expected)
    assert obs['task_uid'].shape == (1,)
    env.close()


def test_restored_episode_refreshes_first_action_controller_reference(tmp_path):
    from types import SimpleNamespace
    from oat.env.mimicgen.env import initial_state_sha256
    state = np.array([2.0, 3.0, 4.0])
    path = tmp_path / 'eval.hdf5'
    with h5py.File(path, 'w') as f:
        group = f.create_group('data/task_0/demo_1')
        group.create_dataset('states', data=state)
        group.attrs['model_file'] = '<mujoco />'
    item = task()
    item['eval_initial_state_sha256'][0] = initial_state_sha256(state)
    env = MimicGenEnv(item, path, image_size=2, native_image_size=2)

    class Controller:
        cached_position = 0.
        initial_joint = 0.
        goal = 0.

        def update(self, force=False):
            if force:
                self.cached_position = simulation.state[0]

        def update_initial_joints(self, joints):
            self.initial_joint = joints[0]

        def reset_goal(self):
            self.goal = self.cached_position

    controller = Controller()

    class Simulation:
        state = np.zeros(3)

        def reset(self):
            self.state = np.zeros(3)

        def set_state_from_flattened(self, value):
            self.state = value.copy()

        def forward(self):
            pass

        def get_state(self):
            return SimpleNamespace(flatten=lambda: self.state)

    simulation = Simulation()
    raw = {key: np.zeros(shape, np.float32) for key, shape in [('robot0_eef_pos', (3,)), ('robot0_eef_quat', (4,)), ('robot0_gripper_qpos', (2,))]}
    raw.update({camera+'_image': np.zeros((2, 2, 3), np.uint8) for camera in env.camera_names})
    robot = SimpleNamespace(controller=controller, _joint_positions=state)

    def first_step(action):
        # Stale OSC caches would incorrectly interpret this first delta action
        # relative to the default reset pose instead of the held-out pose.
        assert controller.cached_position == controller.goal == state[0]
        assert controller.initial_joint == state[0]
        return raw, 1., True, {}

    env.env = SimpleNamespace(
        sim=simulation, robots=[robot], edit_model_xml=lambda xml: xml,
        reset_from_xml_string=lambda xml: None,
        _get_observations=lambda **kwargs: raw, step=first_step,
        _check_success=lambda: True, close=lambda: None,
    )
    env.set_episode({'task_name': 'task_0', 'demo_name': 'demo_1',
                     'env_seed': 91, 'initial_state_sha256': initial_state_sha256(state)})
    env.reset()
    assert env.step(np.zeros(7))[1] == 1.
    env.close()
