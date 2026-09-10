"""CPU contract tests: real snapshot serialization, fake isolated evaluator."""
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import dill
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

import oat.env_runner.libero_official_runner as bridge
from oat.env_runner.libero_official_eval import (
    PROTOCOL_VERSION, assign_video_paths, file_sha256, make_episode_plan, summarize_results,
)


@pytest.fixture
def setup(tmp_path):
    model = torch.nn.Linear(2, 1)
    model.register_buffer("normalizer", torch.tensor([0.25, 0.75]))
    with torch.no_grad():
        model.weight.fill_(4)
        model.bias.fill_(-2)
    cfg = OmegaConf.create({"policy": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 1},
                            "training": {"use_ema": True}, "task": {"policy": {"env_runner": {}}}})
    runner = bridge.LiberoOfficialRunner(tmp_path, n_per_task=1, video_per_task=0)
    return runner, model, cfg


def fake_evaluator(monkeypatch, mutate=None, returncode=0):
    calls = []

    def run(command, *, cwd, env, stdout, stderr, check):
        calls.append((command, env))
        options = dict(zip(command[2::2], command[3::2]))
        output = Path(options["--output"])
        output.mkdir()
        checkpoint = Path(options["--checkpoint"])
        shutil.copyfile(checkpoint, output / "policy.ckpt")
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
        names = [f"task_{index}" for index in range(10)]
        plan = make_episode_plan(names, int(options["--n-per-task"]), int(options["--seed"]), int(options["--init-start"]))
        plan = assign_video_paths(plan, int(options["--video-per-task"]), output / "videos")
        for entry in plan:
            entry["official_state_sha256"] = "a" * 64
        weight = "ema_model" if payload["cfg"].training.use_ema else "model"
        manifest = {
            "protocol_version": PROTOCOL_VERSION, "checkpoint_sha256": file_sha256(checkpoint),
            "source_checkpoint": str(checkpoint), "checkpoint": str(output / "policy.ckpt"),
            "weights": weight, "policy_config": OmegaConf.to_container(payload["cfg"].policy, resolve=True),
            "training_position": {name: dill.loads(value) for name, value in payload["pickles"].items()},
            "device": "cuda:0", "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "egl_device": env["MUJOCO_EGL_DEVICE_ID"], "episodes": plan,
        }
        for name in ("suite", "n_per_task", "init_start", "seed", "workers", "max_episode_steps", "settle_steps", "video_per_task"):
            value = options["--" + name.replace("_", "-")]
            manifest[name] = value if name == "suite" else int(value)
        rows = [{**entry, "success": entry["task_index"] % 2 == 0, "error": None,
                 "initial_state_sha256": "b" * 64, "executed_actions_sha256": "c" * 64} for entry in plan]
        summary = summarize_results(plan, rows)
        summary.update(checkpoint_sha256=manifest["checkpoint_sha256"], weights=weight)
        if mutate:
            mutate(manifest, rows, summary, output)
        (output / "manifest.json").write_text(json.dumps(manifest))
        (output / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (output / "summary.json").write_text(json.dumps(summary))
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(bridge.subprocess, "run", run)
    return calls


def test_exact_selected_snapshot_scalar_metrics_and_parent_rng_unchanged(setup, monkeypatch):
    runner, selected_ema, cfg = setup
    calls = fake_evaluator(monkeypatch)
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    environment_before = dict(os.environ)
    training_before = selected_ema.training
    metrics = runner.run_checkpoint(selected_ema, cfg, epoch=14, global_step=29084)
    assert random.getstate() == py_before
    now_np = np.random.get_state()
    assert now_np[0] == np_before[0] and np.array_equal(now_np[1], np_before[1]) and now_np[2:] == np_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert dict(os.environ) == environment_before
    assert selected_ema.training == training_before
    assert metrics["mean_success_rate"] == 0.5
    assert metrics["eval/successes"] == 5 and metrics["eval/episodes"] == 10
    assert metrics["eval/completed_epochs"] == 15
    assert len(metrics) == 14 and all(type(value) in (int, float) for value in metrics.values())
    saved = Path(runner.output_dir) / "rollouts/epoch_015/policy.ckpt"
    payload = torch.load(saved, map_location="cpu", pickle_module=dill)
    assert set(payload["state_dicts"]) == {"ema_model"}
    for name, value in selected_ema.state_dict().items():
        assert torch.equal(value, payload["state_dicts"]["ema_model"][name])
    assert saved.stat().st_mode & 0o222 == 0
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "1"
    assert calls[0][1]["PYTHONPATH"] == str(bridge.ROOT)
    assert calls[0][0][1] == str(bridge.ROOT / "scripts/eval_heading_policy.py")


def test_reuses_verified_result_and_refuses_changed_model_config_or_position(setup, monkeypatch):
    runner, model, cfg = setup
    calls = fake_evaluator(monkeypatch)
    expected = runner.run_checkpoint(model, cfg, 14, 29084)
    cfg.training.resume = True  # Training-only config is irrelevant to inference.
    assert runner.run_checkpoint(model, cfg, 14, 29084) == expected
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="position"):
        runner.run_checkpoint(model, cfg, 14, 29085)
    cfg.policy.out_features = 2
    with pytest.raises(RuntimeError, match="configuration"):
        runner.run_checkpoint(model, cfg, 14, 29084)
    cfg.policy.out_features = 1
    with torch.no_grad():
        model.normalizer[0] += 1
    with pytest.raises(RuntimeError, match="tensor differs"):
        runner.run_checkpoint(model, cfg, 14, 29084)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["exit", "missing", "episode_error", "video_error", "duplicate", "wrong_seed", "wrong_summary", "wrong_hash", "wrong_weights", "wrong_plan"])
def test_invalid_results_never_return_metrics(setup, monkeypatch, failure):
    runner, model, cfg = setup
    def mutate(manifest, rows, summary, output):
        if failure == "missing": rows.pop()
        if failure == "episode_error": rows[0]["error"] = "simulator failed"
        if failure == "video_error": rows[0]["video_error"] = "encoder failed"
        if failure == "duplicate": rows[-1] = rows[0]
        if failure == "wrong_seed": rows[0]["policy_seed"] += 1
        if failure == "wrong_summary": summary["successes"] += 1
        if failure == "wrong_hash": manifest["checkpoint_sha256"] = "d" * 64
        if failure == "wrong_weights": manifest["weights"] = "model"
        if failure == "wrong_plan": manifest["episodes"][0]["init_index"] += 1
    fake_evaluator(monkeypatch, mutate, 7 if failure == "exit" else 0)
    with pytest.raises((RuntimeError, ValueError)):
        runner.run_checkpoint(model, cfg, 14, 29084)
    assert not list(Path(runner.output_dir).rglob("runner_verified.json"))


def test_incomplete_attempt_preserved_then_retry_and_corrupt_reuse_refused(setup, monkeypatch):
    runner, model, cfg = setup
    fake_evaluator(monkeypatch, lambda manifest, rows, summary, output: rows.pop())
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.run_checkpoint(model, cfg, 14, 29084)
    first = Path(runner.output_dir) / "rollouts/epoch_015/attempt_001/episodes.jsonl"
    first_bytes = first.read_bytes()
    calls = fake_evaluator(monkeypatch)
    runner.run_checkpoint(model, cfg, 14, 29084)
    assert first.read_bytes() == first_bytes and len(calls) == 1
    second = first.parent.parent / "attempt_002"
    assert (second / "runner_verified.json").exists()
    (second / "episodes.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.run_checkpoint(model, cfg, 14, 29084)
    assert len(calls) == 1


def test_missing_artifact_refusal_and_symlink_safety(setup, monkeypatch):
    runner, model, cfg = setup
    fake_evaluator(monkeypatch)
    runner.run_checkpoint(model, cfg, 14, 29084)
    summary = Path(runner.output_dir) / "rollouts/epoch_015/attempt_001/summary.json"
    summary.unlink()
    with pytest.raises(RuntimeError, match="missing evaluation artifact"):
        runner.run_checkpoint(model, cfg, 14, 29084)
    (Path(runner.output_dir) / "rollouts/epoch_030").symlink_to(summary.parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        runner.run_checkpoint(model, cfg, 29, 58169)


def test_raw_model_selection_and_context_requirement(setup, monkeypatch):
    runner, model, cfg = setup
    cfg.training.use_ema = False
    fake_evaluator(monkeypatch)
    runner.run_checkpoint(model, cfg, 0, 1)
    record = json.loads((Path(runner.output_dir) / "rollouts/epoch_001/snapshot.json").read_text())
    assert record["weights"] == "model"
    with pytest.raises(RuntimeError, match="run_checkpoint"):
        runner.run(model)
    assert runner.close() is None
