"""Exact zero angular jitter must survive training, inference, and checkpoint load."""

import contextlib
import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oat.policy.flow_policy_shared_heading import SharedHeadingFlowPolicy
from scripts import evaluate_mini_heading_prior_sampling as sampling
from scripts.mini_heading_prior import angular_jitter, set_variant
from test_shared_heading_policy import (
    SHAPE_META, TinyEncoder, batch as observation_batch, make_source_policy,
    single_cpu_thread, source_prediction,
)


@pytest.fixture(autouse=True)
def fixed_cpu_seed():
    torch.manual_seed(983)


def zero_policy():
    policy = make_source_policy(blocks=((0, 1),)).eval()
    policy.set_source_jitter("zero")
    return policy


def test_zero_jitter_aligns_source_exactly_without_advancing_angular_rng():
    policy = zero_policy()
    noise = torch.randn(3, 4, 7)
    prediction = source_prediction([0.2, -1.7, 2.4])
    state = copy.deepcopy(policy._source_rng.bit_generator.state)
    actual = policy.transform_source(noise, prediction)
    repeated = policy.transform_source(noise, prediction)
    expected = policy.transform_source(noise, prediction, angular_jitter=torch.zeros(3))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    assert policy._source_rng.bit_generator.state == state
    raw = policy.normalizer["action"].unnormalize(actual)
    resultant = raw[..., :2].sum(1)
    direction = torch.stack((prediction["theta"].cos(), prediction["theta"].sin()), -1)
    torch.testing.assert_close(resultant / resultant.norm(dim=-1, keepdim=True), direction, rtol=1e-5, atol=1e-6)


def test_default_vonmises_mode_retains_exact_private_random_sequence():
    policy = make_source_policy(blocks=((0, 1),)).eval()
    assert policy.source_jitter == "vonmises"
    policy.set_source_seed(73)
    generator = np.random.default_rng(73)
    noise, prediction = torch.randn(3, 4, 7), source_prediction([0.2, -1.7, 2.4])
    for _ in range(2):
        jitter = torch.tensor(generator.vonmises(0., policy.source_kappa, size=3), dtype=torch.float32)
        expected = policy.transform_source(noise, prediction, angular_jitter=jitter)
        actual = policy.transform_source(noise, prediction)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert policy._source_rng.bit_generator.state == generator.bit_generator.state


def test_explicit_diagnostic_jitter_intentionally_overrides_zero_default():
    policy = zero_policy()
    noise, prediction = torch.randn(3, 4, 7), source_prediction([0.2, -1.7, 2.4])
    jitter = torch.tensor([0.3, -0.4, 0.6])
    before = copy.deepcopy(policy._source_rng.bit_generator.state)
    overridden = policy.transform_source(noise, prediction, angular_jitter=jitter)
    unperturbed = policy.transform_source(noise, prediction)
    assert not torch.equal(overridden, unperturbed)
    assert policy._source_rng.bit_generator.state == before
    policy.set_source_jitter("vonmises")
    torch.testing.assert_close(overridden, policy.transform_source(noise, prediction, angular_jitter=jitter), rtol=0, atol=0)


def test_zero_default_matches_training_and_inference_without_explicit_jitter():
    policy, data = zero_policy(), observation_batch()
    noise, t = torch.randn_like(data["action"]), torch.rand(3)
    training = policy.loss_components(data, noise=noise, t=t)
    inference = policy.predict_action(data["obs"], noise=noise, return_source=True)
    explicit = policy.loss_components(data, noise=noise, t=t, angular_jitter=torch.zeros(3))
    for key in ("source", "source_active"):
        torch.testing.assert_close(training[key], inference[key], rtol=0, atol=0)
    torch.testing.assert_close(training["loss"], explicit["loss"], rtol=0, atol=0)
    torch.testing.assert_close(training["flow_loss"], explicit["flow_loss"], rtol=0, atol=0)
    explicit_inference = policy.predict_action(data["obs"], noise=noise, angular_jitter=torch.zeros(3))
    torch.testing.assert_close(inference["action_pred"], explicit_inference["action_pred"], rtol=0, atol=0)


def test_zero_jitter_does_not_change_existing_confidence_gate():
    policy = zero_policy()
    noise = torch.randn(3, 4, 7)
    prediction = source_prediction([0.2, -1.7, 2.4], [.49, .5, .9])
    source, active = policy._transform_source(noise, prediction)
    assert active.tolist() == [False, True, True]
    torch.testing.assert_close(source[0], noise[0], rtol=0, atol=0)


def test_runner_helper_zero_and_legacy_vonmises_preserve_global_rng():
    torch_state = torch.random.get_rng_state().clone()
    np_state = np.random.get_state()
    for kappa in (0., 4., 100.):
        actual = angular_jitter(177, 32, "cpu", kappa, mode="zero")
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        torch.testing.assert_close(actual, torch.zeros(32), rtol=0, atol=0)
    expected = torch.tensor(np.random.default_rng(177).vonmises(0., 4., size=32), dtype=torch.float32)
    torch.testing.assert_close(angular_jitter(177, 32, "cpu", 4.), expected, rtol=0, atol=0)
    torch.testing.assert_close(angular_jitter(177, 32, "cpu", 4., mode="vonmises"), expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    current = np.random.get_state()
    assert current[0] == np_state[0] and current[2:] == np_state[2:]
    np.testing.assert_array_equal(current[1], np_state[1])


def test_invalid_jitter_modes_are_rejected():
    policy = zero_policy()
    with pytest.raises(ValueError, match="source_jitter"):
        policy.set_source_jitter("uniform")
    with pytest.raises(ValueError):
        angular_jitter(177, 3, "cpu", 4., mode="uniform")


def test_variant_restores_zero_or_legacy_default_without_adding_checkpoint_tensors():
    policy = make_source_policy()
    tensors = copy.deepcopy(policy.state_dict())
    args = SimpleNamespace(source_kappa=4., min_source_confidence=.5, source_jitter="zero")
    set_variant(policy, "condition_prior_xy", args)
    assert policy.heading_mode == "condition" and policy.source_mode == "heading"
    assert policy.source_vector_blocks == ((0, 1),) and policy.source_jitter == "zero"
    legacy = SimpleNamespace(source_kappa=4., min_source_confidence=.5)
    set_variant(policy, "condition_prior_xy", legacy)
    assert policy.source_jitter == "vonmises"
    for key, value in tensors.items():
        torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)


def loader_fixture_policy(normalizer, xy_rms, source_jitter="vonmises"):
    with contextlib.redirect_stdout(io.StringIO()):
        policy = SharedHeadingFlowPolicy(
            shape_meta=SHAPE_META, obs_encoder=TinyEncoder(), horizon=16,
            n_action_steps=8, n_obs_steps=2, embed_dim=8, n_layers=1,
            n_heads=2, dropout=0, num_inference_steps=10, heading_hidden_dim=8,
            heading_mode="condition", heading_xy_rms=xy_rms,
            source_mode="heading", source_vector_blocks=((0, 1),),
            source_jitter=source_jitter,
        )
    policy.set_normalizer(normalizer)
    with torch.no_grad():
        policy.obs_encoder.head[-1].bias[2] = 4
    return policy.eval()


def write_loader_fixture(tmp_path, metadata_mode):
    normalizer = make_source_policy().normalizer
    legacy = metadata_mode is None
    mode = "vonmises" if legacy else metadata_mode
    original = loader_fixture_policy(normalizer, .8, source_jitter=mode)
    args = {"steps": 1, "source_kappa": 4., "min_source_confidence": .5}
    if not legacy:
        args["source_jitter"] = mode
    manifest = {"args": args, "shape_meta": SHAPE_META}
    payload = {"seed": 42, "mode": "condition_prior_xy", "step": 1,
               "heading_mode": "condition", "source_mode": "heading",
               "source_kappa": 4., "min_source_confidence": .5,
               "source_vector_blocks": ((0, 1),), "heading_xy_rms": .8,
               "shape_meta": SHAPE_META, "manifest": manifest,
               "state_dict": original.state_dict()}
    if not legacy:
        payload["source_jitter"] = mode
    path = tmp_path / "seed42" / "condition_prior_xy" / "ema.pt"
    path.parent.mkdir(parents=True)
    torch.save(payload, path)
    torch.save(normalizer.state_dict(), tmp_path / "normalizer.pt")
    return original, manifest, path


@pytest.mark.parametrize("metadata_mode", (None, "vonmises", "zero"))
def test_checkpoint_loader_restores_exact_zero_and_legacy_defaults(tmp_path, monkeypatch, metadata_mode):
    original, manifest, _ = write_loader_fixture(tmp_path, metadata_mode)
    monkeypatch.setattr(sampling, "build_policy", lambda shape_meta, args, normalizer, xy_rms:
                        loader_fixture_policy(normalizer, xy_rms))
    loaded, config = sampling.load_policy(tmp_path, manifest, {"heading_xy_rms": .8}, 42, "condition_prior_xy", "cpu")
    expected_mode = "vonmises" if metadata_mode is None else metadata_mode
    assert loaded.source_jitter == expected_mode
    assert config["source_jitter"] == expected_mode
    original.set_source_seed(991)
    loaded.set_source_seed(991)
    observations = {"state": torch.randn(3, 2, 3)}
    noise = torch.randn(3, 16, 7)
    # Match inference modes so CPU attention chooses the same kernel.
    with torch.no_grad():
        expected = original.predict_action(observations, noise=noise, return_source=True)
        actual = loaded.predict_action(observations, noise=noise, return_source=True)
    for key in ("source", "source_active", "action_pred"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_checkpoint_loader_rejects_disagreement_with_manifest(tmp_path):
    _, manifest, path = write_loader_fixture(tmp_path, "zero")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["source_jitter"] = "vonmises"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="jitter"):
        sampling.load_policy(tmp_path, manifest, {"heading_xy_rms": .8}, 42, "condition_prior_xy", "cpu")
