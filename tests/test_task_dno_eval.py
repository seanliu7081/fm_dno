"""Evaluation bookkeeping checks use a deterministic fake simulator, no GPUs."""
from collections import deque
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "eval_task_dno_libero", Path(__file__).parents[1] / "scripts/eval_task_dno_libero.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class FakeEnv:
    def __init__(self, terminate_at=3):
        self.terminate_at = terminate_at
        self.t = 0
        self.seed_value = None
        self.seed_calls = []
        model = SimpleNamespace(body_pos=np.zeros((2, 3)), body_quat=np.zeros((2, 4)))
        sim = SimpleNamespace(model=model, get_state=lambda: np.array([self.seed_value, self.t]))
        self.env = SimpleNamespace(seed=self._seed, sim=sim)

    def _seed(self, seed):
        self.seed_calls.append(seed)
        self.seed_value = seed

    def _extract_obs(self):
        return {"robot0_eef_pos": np.array([self.seed_value, self.t, 0.0], dtype=np.float32),
                "image": np.full((2, 2, 3), self.t % 256, dtype=np.uint8), "prompt": "pick"}

    def reset(self):
        self.t = -1
        stale = self._extract_obs()
        self.t = 0
        return stale, {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.terminate_at
        return self._extract_obs(), float(done), done, False, {}


class FakePolicy:
    n_obs_steps = 2
    n_action_steps = 8
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.obs = []
        self.resets = []

    def reset(self, seed=None):
        self.resets.append(seed)

    def clear_log(self):
        self.obs.clear()

    def predict_action(self, obs, task_context):
        assert not torch.is_inference_mode_enabled()
        self.obs.append(obs)
        assert task_context["phase"].dtype == torch.int64
        assert task_context["active"].dtype == torch.bool
        return {"action": torch.ones(1, 8, 7), "cond": torch.ones(1, 2, 4),
                "base_noise": torch.zeros(1, 16, 7), "optimized_noise": torch.ones(1, 16, 7),
                "context_features": torch.zeros(1, 9), "teacher_improved": torch.tensor([True]),
                "initial_objective": torch.tensor([0.2]), "final_objective": torch.tensor([0.1])}

    def summarize(self):
        return {"dno/nfe": 10.0}


def context_builder(env, state):
    state["calls"] = state.get("calls", 0) + 1
    return {"eef_pos": env._extract_obs()["robot0_eef_pos"],
            "goal_pos": np.zeros(3), "phase": np.int64(0), "active": True}


def test_actual_prefix_stops_on_success_and_teacher_records_real_transition():
    env, policy = FakeEnv(terminate_at=3), FakePolicy()
    result, rows = evaluation.run_episode(
        env, policy, context_builder, mode="dno", episode_id=4,
        seed=1004, max_episode_steps=16, collect_teacher=True,
    )
    assert env.seed_calls == [1004]
    assert policy.resets == [1004]
    assert result["success"] and result["env_steps"] == 3
    assert result["episode_return"] == 1.0
    assert len(rows) == 1 and rows[0]["executed_steps"] == 3
    assert rows[0]["env_t"] == 0 and rows[0]["next_env_t"] == 3
    assert rows[0]["chunk_return"] == 1 and rows[0]["episode_success"]
    assert rows[0]["next_eef_pos"][1] == 3
    assert rows[0]["executed_action"][3:].eq(0).all()
    # First policy input is after settling, not the stale -1 reset observation.
    assert policy.obs[0]["robot0_eef_pos"][0, :, 1].eq(0).all()


def test_last_chunk_is_cut_at_real_horizon_and_reported_as_truncated():
    env, policy = FakeEnv(terminate_at=99), FakePolicy()
    result, rows = evaluation.run_episode(
        env, policy, context_builder, mode="dno", episode_id=0,
        seed=42, max_episode_steps=10, collect_teacher=True,
    )
    assert result["env_steps"] == 10 and result["truncated"] and not result["success"]
    assert [row["executed_steps"].item() for row in rows] == [8, 2]
    assert [row["next_env_t"].item() for row in rows] == [8, 10]
    assert rows[-1]["truncated"]
    assert policy.obs[1]["robot0_eef_pos"][0, :, 1].tolist() == [7, 8]


def test_matching_reset_hash_enforced_before_any_policy_action():
    env, policy = FakeEnv(), FakePolicy()
    first, _ = evaluation.run_episode(env, policy, context_builder, mode="baseline",
                                      episode_id=0, seed=42, max_episode_steps=8)
    second, _ = evaluation.run_episode(env, policy, context_builder, mode="dno",
                                       episode_id=0, seed=42, max_episode_steps=8,
                                       expected_state_hash=first["initial_state_sha256"])
    assert first["initial_state_sha256"] == second["initial_state_sha256"]
    with pytest.raises(RuntimeError, match="Unmatched reset"):
        evaluation.run_episode(env, policy, context_builder, mode="dno",
                               episode_id=0, seed=43, max_episode_steps=8,
                               expected_state_hash=first["initial_state_sha256"])


def test_objective_failure_propagates_instead_of_running_without_goal():
    def unsupported(env, state):
        raise ValueError("unsupported task")
    with pytest.raises(ValueError, match="unsupported task"):
        evaluation.run_episode(FakeEnv(), FakePolicy(), unsupported, mode="dno",
                               episode_id=0, seed=42, max_episode_steps=8)


def test_atomic_result_writer_never_overwrites(tmp_path):
    path = tmp_path / "results.json"
    evaluation.write_json(path, {"completed": 1})
    with pytest.raises(FileExistsError):
        evaluation.write_json(path, {"completed": 2})
    assert path.read_text().find('"completed": 1') >= 0
    assert list(tmp_path.iterdir()) == [path]


def test_config_cli_precedence_and_forward_nfe_budget(tmp_path):
    config = tmp_path / "pilot.yaml"
    config.write_text("episodes: 2\nnum_candidates: 2\nnum_grad_steps: 1\n")
    args = evaluation.parse_args(["-c", "policy.ckpt", "-o", "results", "--config", str(config),
                                  "--episodes", "3"])
    assert args.episodes == 3 and args.best_of_k_candidates == 6
    with pytest.raises(SystemExit):
        evaluation.parse_args(["-c", "p.ckpt", "-o", "out", "--modes", "amortized"])


def test_best_of_k_cannot_export_residual_teachers(capsys):
    with pytest.raises(SystemExit):
        evaluation.parse_args(["-c", "policy.ckpt", "-o", "results",
                               "--modes", "best_of_k", "--teacher-mode", "best_of_k"])
    error = capsys.readouterr().err
    assert "--teacher-mode" in error and "invalid choice" in error
    assert "dno" in error and "amortized_dno" in error


def test_best_of_k_teacher_config_is_also_rejected(tmp_path, capsys):
    config = tmp_path / "invalid_teacher.yaml"
    config.write_text("modes: best_of_k\nteacher_mode: best_of_k\n")
    with pytest.raises(SystemExit):
        evaluation.parse_args(["-c", "policy.ckpt", "-o", "results", "--config", str(config)])
    assert "cannot teach residual updates" in capsys.readouterr().err


def test_libero_horizon_done_is_recorded_as_timeout_not_task_terminal():
    class HorizonDoneEnv(FakeEnv):
        def step(self, action):
            obs, _, done, truncated, info = super().step(action)
            return obs, 0.0, done, truncated, info

    result, rows = evaluation.run_episode(
        HorizonDoneEnv(terminate_at=10), FakePolicy(), context_builder,
        mode="dno", episode_id=0, seed=42, max_episode_steps=10, collect_teacher=True,
    )
    assert result["env_steps"] == 10 and not result["success"]
    assert not result["terminated"] and result["truncated"]
    assert result["env_reported_terminated"] and result["time_limit_reached"]
    assert not rows[-1]["terminated"] and rows[-1]["truncated"]
    assert rows[-1]["env_reported_terminated"] and rows[-1]["time_limit_reached"]
    assert rows[-1]["env_t"] == 8 and rows[-1]["next_env_t"] == 10
    assert not rows[0]["time_limit_reached"]


def test_success_on_final_allowed_step_remains_terminal():
    result, rows = evaluation.run_episode(
        FakeEnv(terminate_at=3), FakePolicy(), context_builder,
        mode="dno", episode_id=0, seed=42, max_episode_steps=3, collect_teacher=True,
    )
    assert result["success"] and result["terminated"] and not result["truncated"]
    assert result["time_limit_reached"] and result["env_reported_terminated"]
    assert rows[-1]["terminated"] and not rows[-1]["truncated"]
