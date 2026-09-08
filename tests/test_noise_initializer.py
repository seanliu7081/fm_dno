import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from oat.dno.noise_initializer import (
    NoiseInitializer,
    build_context_features,
    load_noise_initializer,
    save_noise_initializer,
)
from scripts.train_noise_initializer import load_archive, split_by_episode


IDENTITY = {"source_policy": "F", "base_checkpoint_sha256": "a" * 64}


def inputs(batch=4):
    return torch.randn(batch, 2, 6), torch.randn(batch, 4, 3), {"context_features": torch.randn(batch, 9)}


def test_context_features_match_geometry_gripper_and_phase():
    context = {"goal_pos": [0.5, 0.2, 0.3], "eef_pos": [0.1, 0.2, 0.4], "desired_gripper": -1, "phase": 2}
    actual = build_context_features(context, batch_size=3)
    expected = torch.tensor([0.4, 0, -0.1, -1, 0, 0, 1, 0, 0]).expand(3, -1)
    torch.testing.assert_close(actual, expected)
    absent = build_context_features({"goal_pos": [0, 0, 0], "eef_pos": [0, 0, 0]})
    assert torch.equal(absent, torch.zeros(1, 9))
    with pytest.raises(ValueError, match="phase"):
        build_context_features({**context, "phase": 1.5})


def test_zero_initialization_exactly_preserves_source_and_has_gradients():
    torch.manual_seed(3)
    model = NoiseInitializer(6, 4, 3)
    cond, base, context = inputs()
    model.fit_input_normalizer(cond, base, context)
    predicted = model(cond, base, context)
    assert torch.equal(predicted, base)
    loss = (predicted - (base + 0.1)).square().mean()
    loss.backward()
    assert model.net[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_residual_bound_even_for_large_network_logits():
    model = NoiseInitializer(6, 4, 3, max_residual_rms=0.2)
    with torch.no_grad():
        model.net[-1].bias.fill_(1e5)
    cond, base, context = inputs()
    residual = model(cond, base, context) - base
    assert bool((residual.square().mean((1, 2)).sqrt() <= 0.200001).all())


def test_checkpoint_roundtrip_and_identity_guard(tmp_path):
    model = NoiseInitializer(6, 4, 3, hidden_dims=[16], max_residual_rms=0.3)
    cond, base, context = inputs()
    with torch.no_grad():
        model.net[-1].bias.fill_(0.1)
    model.fit_input_normalizer(cond, base, context)
    path = tmp_path / "selector.pt"
    save_noise_initializer(path, model, identity=IDENTITY)
    restored = load_noise_initializer(path, expected_identity=IDENTITY)
    assert not restored.training
    torch.testing.assert_close(restored(cond, base, context), model(cond, base, context))
    with pytest.raises(ValueError, match="source_policy"):
        load_noise_initializer(path, expected_identity={**IDENTITY, "source_policy": "E"})
    with pytest.raises(ValueError, match="base_checkpoint_sha256"):
        load_noise_initializer(path, expected_identity={**IDENTITY, "base_checkpoint_sha256": "b" * 64})
    old = torch.load(path, weights_only=True)
    old["format_version"] = 1
    torch.save(old, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_noise_initializer(path, expected_identity=IDENTITY)


def test_small_sample_normalization_keeps_noise_raw_and_unseen_phase_bounded():
    model = NoiseInitializer(6, 4, 3)
    cond = torch.ones(2, 2, 6) * 3
    # A constant source draw must never be normalized by tiny empirical variance.
    base = torch.ones(2, 4, 3) * 2
    train_context = {"goal_pos": [0, 0, 0], "eef_pos": [0, 0, 0], "desired_gripper": -1, "phase": 0}
    model.fit_input_normalizer(cond, base, train_context)
    cond_end, context_start = 12, 24
    torch.testing.assert_close(model.input_mean[cond_end:context_start], torch.zeros(12))
    torch.testing.assert_close(model.input_std[cond_end:context_start], torch.ones(12))
    torch.testing.assert_close(model.input_std[:cond_end], torch.full((12,), 0.1))
    torch.testing.assert_close(model.input_std[context_start:context_start + 3], torch.full((3,), 0.05))
    changed_context = {**train_context, "desired_gripper": 1, "phase": 4, "goal_pos": [0.1, 0, 0]}
    new_base = base + 20
    features = model.normalized_input_features(cond + 100, new_base, changed_context)
    # No rescaling or clipping of the actual source noise, even at large values.
    assert torch.equal(features[:, cond_end:context_start], new_base.flatten(1))
    assert torch.equal(features[:, :cond_end], torch.full((2, 12), 10.0))
    torch.testing.assert_close(features[:, context_start:context_start + 3], torch.tensor([2., 0, 0]).expand(2, -1))
    assert torch.equal(features[:, -6:], torch.tensor([1., 0, 0, 0, 0, 1]).expand(2, -1))
    assert torch.equal(model(cond + 100, new_base, changed_context), new_base)


def test_split_keeps_all_chunks_of_each_episode_together():
    ids = ["a", "b", "a", "c", "b", "d", "c", "d"]
    train, val, train_episodes, val_episodes = split_by_episode(ids, val_ratio=0.25)
    assert set(train_episodes).isdisjoint(val_episodes)
    assert {ids[i] for i in train.tolist()} == set(train_episodes)
    assert {ids[i] for i in val.tolist()} == set(val_episodes)
    assert len(train) + len(val) == len(ids)
    assert torch.equal(train, split_by_episode(ids, val_ratio=0.25)[0])
    with pytest.raises(ValueError, match="two retained episodes"):
        split_by_episode([1, 1, 1])


def make_archive(path, count=32):
    torch.manual_seed(11)
    cond, base, context = inputs(count)
    archive = {
        "cond": cond,
        "base_noise": base,
        "optimized_noise": base + 0.1,
        "context_features": context["context_features"],
        "episode_id": torch.arange(count) // 4,
        "accepted": torch.ones(count, dtype=torch.bool),
        "episode_success": torch.ones(count, dtype=torch.bool),
        "meta": IDENTITY,
    }
    torch.save(archive, path)
    return archive


def test_archive_filters_proxy_acceptance_and_real_success_separately(tmp_path):
    path = tmp_path / "teacher.pt"
    archive = make_archive(path)
    archive["accepted"][0] = False
    archive["episode_success"][1] = False
    torch.save(archive, path)
    assert load_archive(path)[3]["retained_rows"] == 31
    assert load_archive(path, successful_only=True)[3]["retained_rows"] == 30
    assert load_archive(path, successful_only=True, include_rejected=True)[3]["retained_rows"] == 31


def test_cpu_training_cli_saves_model_and_disjoint_split(tmp_path):
    archive = tmp_path / "teacher.pt"
    make_archive(archive)
    output = tmp_path / "run"
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/train_noise_initializer.py"), "--data", str(archive),
         "--output-dir", str(output), "--epochs", "5", "--device", "cpu", "--hidden-dims", "16",
         "--batch-size", "8", "--lr", "0.01", "--successful-only"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    run = json.loads((output / "run.json").read_text())
    assert set(run["train_episodes"]).isdisjoint(run["validation_episodes"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["best_val_mse"] < summary["val_baseline_mse"]
    assert (output / "last.pt").is_file()
    model = load_noise_initializer(output / "best.pt", expected_identity=IDENTITY)
    assert model.cond_dim == 6
