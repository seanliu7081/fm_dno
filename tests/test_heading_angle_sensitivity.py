"""CPU invariants for the inference-only oracle source-angle diagnostic."""

import copy
import math

import pytest
import torch

from scripts.evaluate_heading_angle_sensitivity import (
    ARMS, evaluate_batch, sample_from_condition, source_angles,
)
from test_shared_heading_policy import (
    batch as observation_batch, make_source_policy, single_cpu_thread,
)


@pytest.fixture(autouse=True)
def fixed_cpu_seed():
    torch.manual_seed(913)


def tiny_policy():
    policy = make_source_policy(blocks=((0, 1),)).eval()
    # Make the heading condition observable to the transformer, instead of relying
    # on its deliberately neutral initialization in the training ablation.
    with torch.no_grad():
        policy.model.cond_obs_emb.weight[:, -3:].normal_(0, 0.1)
    return policy


def wrap(angle):
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def test_predicted_arm_exactly_replays_native_inference():
    policy, data = tiny_policy(), observation_batch()
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.1, -0.4, 0.3])
    with torch.no_grad():
        native = policy.predict_action(data["obs"], noise=noise, angular_jitter=jitter, return_source=True)
        cond, prediction = policy.obs_encoder.encode_with_heading(data["obs"])
        actual = sample_from_condition(policy, cond, prediction, noise, jitter, prediction["theta"])
    for key in ("action", "action_pred", "source", "source_active"):
        torch.testing.assert_close(actual[key], native[key], rtol=0, atol=0)


def test_every_arm_receives_same_condition_and_encodes_observations_once():
    policy, data = tiny_policy(), observation_batch()
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.1, -0.4, 0.3])
    captured = []
    hook = policy.model.register_forward_pre_hook(lambda module, args: captured.append(args[2]))
    try:
        result = evaluate_batch(policy, data, noise, jitter, torch.tensor([1., -1., 1.]))
    finally:
        hook.remove()
    assert policy.obs_encoder.policy_encoder.calls == 1
    assert tuple(result["arms"]) == ARMS
    assert len(captured) == len(ARMS) * policy.num_inference_steps
    for condition in captured:
        torch.testing.assert_close(condition, result["condition"], rtol=0, atol=0)
    assert not result["condition"].requires_grad
    for arm in ARMS:
        assert not result["arms"][arm]["action_pred"].requires_grad


def test_future_action_labels_never_modify_condition_or_predicted_control():
    policy, data = tiny_policy(), observation_batch()
    data["action"][:, :, :2] = torch.tensor([0.4, 0.2])
    changed_labels = {"obs": data["obs"], "action": data["action"].clone()}
    changed_labels["action"][:, :, :2] *= -1
    noise, jitter, signs = torch.randn_like(data["action"]), torch.zeros(3), torch.ones(3)
    before = evaluate_batch(policy, data, noise, jitter, signs)
    after = evaluate_batch(policy, changed_labels, noise, jitter, signs)
    torch.testing.assert_close(before["condition"], after["condition"], rtol=0, atol=0)
    for key in before["prediction"]:
        torch.testing.assert_close(before["prediction"][key], after["prediction"][key], rtol=0, atol=0)
    for key in ("action_pred", "source", "source_active"):
        torch.testing.assert_close(before["arms"]["predicted"][key], after["arms"]["predicted"][key], rtol=0, atol=0)
    assert not torch.allclose(before["arms"]["oracle"]["source"], after["arms"]["oracle"]["source"])


def test_signed_angle_offsets_share_sign_and_preserve_inputs():
    prediction = {"theta": torch.tensor([0.7, -0.8, 1.2]), "confidence": torch.tensor([0.9, 0.6, 0.99])}
    target = {"theta": torch.tensor([3.1, -3.1, 0.5]), "valid": torch.ones(3, dtype=torch.bool)}
    signs = torch.tensor([1., -1., 1.])
    prediction_before, target_before = copy.deepcopy(prediction), copy.deepcopy(target)
    actual = source_angles(prediction, target, signs)
    assert tuple(actual) == ARMS
    torch.testing.assert_close(actual["predicted"], prediction["theta"], rtol=0, atol=0)
    torch.testing.assert_close(actual["oracle"], target["theta"], rtol=0, atol=0)
    for degrees in (15, 30, 60):
        torch.testing.assert_close(wrap(actual[f"oracle_{degrees}"] - target["theta"]),
                                   signs * math.radians(degrees), rtol=0, atol=5e-7)
    for dictionary, saved in ((prediction, prediction_before), (target, target_before)):
        for key in dictionary:
            torch.testing.assert_close(dictionary[key], saved[key], rtol=0, atol=0)


def test_invalid_or_nonfinite_ground_truth_preserves_predicted_source_angle():
    prediction = {"theta": torch.tensor([0.2, -0.6, 1.4]), "confidence": torch.ones(3)}
    target = {"theta": torch.tensor([2.4, float("nan"), -2.]),
              "valid": torch.tensor([False, True, True])}
    actual = source_angles(prediction, target, torch.tensor([1., -1., 1.]))
    for arm in ARMS:
        torch.testing.assert_close(actual[arm][:2], prediction["theta"][:2], rtol=0, atol=0)
    assert actual["oracle"][2] == target["theta"][2]


def test_invalid_target_fallback_replays_actions_and_source_bit_for_bit():
    policy, data = tiny_policy(), observation_batch()
    data["action"][1, :, :2] = 0
    noise = torch.randn_like(data["action"])
    result = evaluate_batch(policy, data, noise, torch.tensor([0.1, -0.4, 0.3]), torch.tensor([1., -1., 1.]))
    assert not result["target"]["valid"][1]
    for arm in ARMS:
        for key in ("source", "action_pred", "source_active"):
            torch.testing.assert_close(result["arms"][arm][key][1],
                                       result["arms"]["predicted"][key][1], rtol=0, atol=0)


def test_angle_override_never_changes_original_confidence_gate():
    policy, data = tiny_policy(), observation_batch()
    with torch.no_grad():
        cond, prediction = policy.obs_encoder.encode_with_heading(data["obs"])
    prediction["confidence"] = torch.tensor([0.49, 0.5, float("nan")])
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.2, 0.3, -0.4])
    target = {"theta": torch.tensor([1.2, 2.3, -1.4]), "valid": torch.ones(3, dtype=torch.bool)}
    for theta in source_angles(prediction, target, torch.ones(3)).values():
        result = sample_from_condition(policy, cond, prediction, noise, jitter, theta)
        assert result["source_active"].tolist() == [False, True, False]
        torch.testing.assert_close(result["source"][[0, 2]],
                                   (policy.prior_noise_scale * noise)[[0, 2]], rtol=0, atol=0)
        assert torch.isfinite(result["action_pred"]).all()


def test_nonfinite_predicted_heading_cannot_be_activated_by_oracle():
    policy, data = tiny_policy(), observation_batch()
    with torch.no_grad():
        cond, prediction = policy.obs_encoder.encode_with_heading(data["obs"])
    prediction["theta"][0] = float("nan")
    target = {"theta": torch.tensor([0.4, 1.2, -1.3]), "valid": torch.ones(3, dtype=torch.bool)}
    noise, jitter = torch.randn_like(data["action"]), torch.zeros(3)
    for theta in source_angles(prediction, target, torch.ones(3)).values():
        assert torch.isnan(theta[0])
        result = sample_from_condition(policy, cond, prediction, noise, jitter, theta)
        assert not result["source_active"][0]
        torch.testing.assert_close(result["source"][0],
                                   policy.prior_noise_scale * noise[0], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ARMS)
def test_oracle_source_rotates_only_raw_xy_and_adds_same_recorded_jitter(arm):
    policy, data = tiny_policy(), observation_batch()
    with torch.no_grad():
        policy.normalizer.params_dict["action"]["scale"].copy_(torch.tensor([1., 3., 2., 4., 5., 6., 7.]))
        policy.normalizer.params_dict["action"]["offset"].copy_(torch.tensor([.2, -.3, .1, .4, -.2, .5, -.1]))
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.2, -0.3, 0.4])
    result = evaluate_batch(policy, data, noise, jitter, torch.tensor([1., -1., 1.]))
    assert result["arms"][arm]["source_active"].all()
    source = result["arms"][arm]["source"]
    raw_before = policy.normalizer["action"].unnormalize(noise * policy.prior_noise_scale)
    raw_after = policy.normalizer["action"].unnormalize(source)
    torch.testing.assert_close(raw_after[..., :2].square().sum(-1),
                               raw_before[..., :2].square().sum(-1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(source[..., 2:], (noise * policy.prior_noise_scale)[..., 2:], rtol=0, atol=0)
    resultant = raw_after[..., :2].sum(1)
    expected_theta = result["source_angles"][arm] + jitter
    expected_direction = torch.stack((expected_theta.cos(), expected_theta.sin()), -1)
    torch.testing.assert_close(resultant / resultant.norm(dim=-1, keepdim=True),
                               expected_direction, rtol=1e-5, atol=1e-6)
