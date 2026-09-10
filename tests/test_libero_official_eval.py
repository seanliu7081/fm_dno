"""Protocol tests avoid MuJoCo/GPU and exercise results that affect reported SR."""
from collections import deque

import numpy as np
import pytest
import torch

from oat.env_runner.libero_official_eval import (
    array_sha256, make_episode_plan, rollout_episode, stable_seed, summarize_results,
)


def test_episode_mapping_is_stable_for_larger_runs_and_distinct_rngs():
    pilot = make_episode_plan(["task_a", "task_b"], 2, 42)
    full = make_episode_plan(["task_a", "task_b"], 50, 42)
    assert pilot == full[:4]
    assert len({entry["episode_id"] for entry in full}) == 100
    assert all(entry["env_seed"] != entry["policy_seed"] for entry in full)
    assert make_episode_plan(["task_a", "task_b"], 2, 42, init_start=2) == full[4:8]
    assert stable_seed(42, "a", 1) != stable_seed(42, "a", 2)


def test_incomplete_or_error_evaluations_do_not_report_benchmark_sr():
    plan = make_episode_plan(["a", "b"], 2, 42)
    results = [{**entry, "success": entry["task_name"] == "a", "error": None} for entry in plan]
    partial = summarize_results(plan, results[:2])
    assert partial["mean_success_rate"] is None
    assert not partial["complete"]
    complete = summarize_results(plan, results)
    assert complete["complete"]
    assert complete["mean_success_rate"] == 0.5
    assert complete["successes"] == 2
    low, high = complete["episode_wilson_95_interval"]
    assert low < 0.5 < high
    results[0]["error"] = "simulator error"
    assert summarize_results(plan, results)["mean_success_rate"] is None
    assert summarize_results(plan, results)["error_episodes"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        summarize_results(plan, [results[0], results[0]])


class FakeControl:
    def __init__(self):
        self.value = -10
        self.seed_value = None
        self.reset_count = 0

    def seed(self, value):
        self.seed_value = value

    def reset(self):
        self.reset_count += 1
        self.value = 0

    def set_init_state(self, state):
        self.value = float(state[0])

    def step(self, action):
        self.value += 1

    def get_sim_state(self):
        return np.array([self.value])

    def check_success(self):
        return False


class FakeEnv:
    def __init__(self, success_after=2):
        self.env = FakeControl()
        self.success_after = success_after
        self.executed = 0

    def _extract_obs(self):
        return {"state": np.array([self.env.value], dtype=np.float32)}

    def step(self, action):
        self.executed += 1
        self.env.step(action)
        success = self.executed == self.success_after
        return self._extract_obs(), float(success), success, False, {}


class FakePolicy:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.observations = []
        self.reset_count = 0

    def get_observation_ports(self):
        return ["state"]

    def reset(self):
        self.reset_count += 1

    def set_source_seed(self, seed):
        self.source_seed = seed

    def predict_action(self, obs):
        self.observations.append(obs["state"].clone())
        return {"action": torch.randn(1, 3, 7)}


def settings(max_steps=10):
    return {"settle_steps": 2, "n_obs_steps": 2, "max_episode_steps": max_steps}


def test_rollout_uses_official_state_settled_observation_and_early_success():
    episode = make_episode_plan(["a"], 1, 42)[0]
    env = FakeEnv()
    policy = FakePolicy()
    result = rollout_episode(policy, env, episode, np.array([5.]), settings())
    assert env.env.reset_count == 1
    assert env.env.seed_value == episode["env_seed"]
    torch.testing.assert_close(policy.observations[0], torch.tensor([[[7.], [7.]]]))
    assert policy.reset_count == 1
    assert policy.source_seed == episode["policy_seed"]
    assert result["success"]
    assert result["executed_steps"] == 2  # Chunk has three actions; stop at success.
    assert result["initial_state_sha256"] == array_sha256(np.array([7.]))


def test_noise_is_independent_of_other_episodes_and_horizon_is_exact():
    episode = make_episode_plan(["a"], 1, 42)[0]
    first = rollout_episode(FakePolicy(), FakeEnv(success_after=99), episode, np.array([5.]), settings(5))
    torch.manual_seed(100)
    torch.randn(1024)
    second = rollout_episode(FakePolicy(), FakeEnv(success_after=99), episode, np.array([5.]), settings(5))
    assert first == second
    assert first["executed_steps"] == 5
    assert first["control_cycles"] == 2
    assert first["termination"] == "horizon"
    assert not first["success"]


def test_strict_loader_restores_one_actual_heading_policy_and_embedded_ema(tmp_path):
    import copy
    import dill
    import hydra
    from omegaconf import OmegaConf
    from oat.env_runner.libero_official_eval import load_policy
    from oat.model.common.normalizer import LinearNormalizer

    shape = {"obs": {"state": {"shape": [3], "type": "state"}}, "action": {"shape": [7]}}
    cfg = OmegaConf.create({
        "training": {"use_ema": True},
        "policy": {
            "_target_": "oat.policy.flow_policy_heading_zero.HeadingZeroFlowPolicy",
            "shape_meta": shape,
            "obs_encoder": {"_target_": "oat.perception.state_encoder.ProjectionStateEncoder", "shape_meta": shape},
            "horizon": 4, "n_action_steps": 2, "n_obs_steps": 2,
            "embed_dim": 16, "n_layers": 1, "n_heads": 2,
            "dropout": 0., "heading_hidden_dim": 8, "num_inference_steps": 2,
        },
    })
    model = hydra.utils.instantiate(cfg.policy)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(20, 7), "state": torch.randn(20, 3)}, last_n_dims=1, mode="limits")
    model.set_normalizer(normalizer)
    model.set_heading_xy_rms(0.4)
    ema_model = copy.deepcopy(model).eval()
    with torch.no_grad():
        next(ema_model.parameters()).add_(0.01)
    payload = {"cfg": cfg, "state_dicts": {"model": model.state_dict(), "ema_model": ema_model.state_dict()}, "pickles": {}}
    path = tmp_path / "self_contained.ckpt"
    torch.save(payload, path, pickle_module=dill)
    loaded, loaded_cfg, state_name = load_policy(path, "cpu")
    assert state_name == "ema_model"
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    for key, value in ema_model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])
    obs = {"state": torch.randn(1, 2, 3)}
    torch.manual_seed(21)
    expected = ema_model.predict_action(obs)["action"]
    torch.manual_seed(21)
    actual = loaded.predict_action(obs)["action"]
    torch.testing.assert_close(expected, actual)
    # Missing weights must fail instead of silently using raw/model parameters.
    del payload["state_dicts"]["ema_model"]
    torch.save(payload, path, pickle_module=dill)
    with pytest.raises(ValueError, match="ema_model"):
        load_policy(path, "cpu")


def test_video_selection_is_fixed_before_outcomes_and_default_is_unchanged(tmp_path):
    from oat.env_runner.libero_official_eval import assign_video_paths
    plan = make_episode_plan(["a", "b"], 3, 42, init_start=10)
    assert assign_video_paths(plan, 0, tmp_path) == plan
    recorded = assign_video_paths(plan, 2, tmp_path)
    assert [row["init_index"] for row in recorded if "video_path" in row] == [10, 10, 11, 11]
    assert len({row["video_path"] for row in recorded if "video_path" in row}) == 4
    assert all("video_path" not in row for row in plan)  # Input plan is not mutated.


class FakeRgbEnv(FakeEnv):
    def _extract_obs(self):
        obs = super()._extract_obs()
        obs["agentview_rgb"] = np.full((16, 16, 3), int(self.env.value), dtype=np.uint8)
        return obs


def test_video_capture_preserves_action_and_state_hashes_using_existing_frames(tmp_path):
    import av
    episode = make_episode_plan(["a"], 1, 42)[0]
    baseline = rollout_episode(FakePolicy(), FakeRgbEnv(), episode, np.array([5.]), settings())
    recorded_episode = {**episode, "video_path": str(tmp_path / "episode.mp4")}
    recorded = rollout_episode(FakePolicy(), FakeRgbEnv(), recorded_episode, np.array([5.]), settings())
    for key in ("success", "executed_steps", "control_cycles", "initial_state_sha256", "executed_actions_sha256"):
        assert recorded[key] == baseline[key]
    assert recorded["video_error"] is None
    assert recorded["video_frames"] == 3  # One settled observation + two executed steps.
    with av.open(recorded["video_path"]) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    assert frames[0].width == 16 and frames[0].height == 16


def test_video_failure_is_separate_from_policy_outcome(tmp_path):
    episode = {**make_episode_plan(["a"], 1, 42)[0], "video_path": str(tmp_path / "missing_camera.mp4")}
    result = rollout_episode(FakePolicy(), FakeEnv(), episode, np.array([5.]), settings())
    assert result["success"]
    assert result["error"] is None
    assert result["video_error"].startswith("KeyError:")
    assert result["video_path"] is None


def test_policy_failure_still_releases_video_resources(tmp_path, monkeypatch):
    from oat.env_runner import libero_official_eval
    writers = []

    class FakeWriter:
        def __init__(self, path):
            self.closed = False
            writers.append(self)

        def write_observation(self, observation):
            assert "state" in observation

        def close(self):
            self.closed = True

    class BrokenPolicy(FakePolicy):
        def predict_action(self, obs):
            raise RuntimeError("deliberate policy failure")

    monkeypatch.setattr(libero_official_eval, "ObservationVideoWriter", FakeWriter)
    episode = {**make_episode_plan(["a"], 1, 42)[0], "video_path": str(tmp_path / "partial.mp4")}
    with pytest.raises(RuntimeError, match="deliberate policy failure"):
        rollout_episode(BrokenPolicy(), FakeEnv(), episode, np.array([5.]), settings())
    assert len(writers) == 1 and writers[0].closed
