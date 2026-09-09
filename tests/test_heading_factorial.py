"""CPU checks for independent heading-condition and source-jitter factors."""

import copy

import pytest
import torch

from scripts.evaluate_heading_angle_sensitivity import evaluate_batch as oracle_evaluate_batch
from scripts.evaluate_heading_factorial import ARMS, evaluate_batch, make_conditions
from test_heading_angle_sensitivity import tiny_policy
from test_shared_heading_policy import batch as observation_batch, single_cpu_thread


@pytest.fixture(autouse=True)
def fixed_cpu_seed():
    torch.manual_seed(947)


def test_original_condition_jitter_cell_exactly_replays_previous_oracle():
    policy, data = tiny_policy(), observation_batch()
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.1, -0.4, 0.3])
    expected = oracle_evaluate_batch(policy, data, noise, jitter, torch.ones(3))["arms"]["oracle"]
    actual = evaluate_batch(policy, data, noise, jitter)["arms"]["predicted_jitter"]
    for key in ("action", "action_pred", "source", "source_active"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_condition_override_changes_only_two_direction_channels():
    policy, data = tiny_policy(), observation_batch()
    with torch.no_grad():
        original, prediction = policy.obs_encoder.encode_with_heading(data["obs"])
    target = policy.heading_targets(data["action"])
    saved = copy.deepcopy((original, prediction, target))
    conditions = make_conditions(original, prediction, target)
    assert tuple(conditions) == ("predicted", "gt")
    torch.testing.assert_close(conditions["predicted"], original, rtol=0, atol=0)
    torch.testing.assert_close(conditions["gt"][..., :-3], original[..., :-3], rtol=0, atol=0)
    torch.testing.assert_close(conditions["gt"][..., -1], original[..., -1], rtol=0, atol=0)
    expected = target["direction"][:, None, :].expand(-1, policy.n_obs_steps, -1)
    torch.testing.assert_close(conditions["gt"][..., -3:-1], expected, rtol=0, atol=0)
    assert conditions["gt"].data_ptr() != original.data_ptr()
    torch.testing.assert_close(original, saved[0], rtol=0, atol=0)
    for dictionary, previous in ((prediction, saved[1]), (target, saved[2])):
        for key in dictionary:
            torch.testing.assert_close(dictionary[key], previous[key], rtol=0, atol=0)


def test_nonfinite_or_invalid_heading_does_not_replace_condition():
    cond = torch.randn(4, 2, 9)
    prediction = {"theta": torch.tensor([0.2, 0.4, float("nan"), 0.6])}
    target_theta = torch.tensor([0.7, 1.2, 2.4, float("nan")])
    target = {"theta": target_theta,
              "direction": torch.stack((target_theta.cos(), target_theta.sin()), -1),
              "valid": torch.tensor([True, False, True, True])}
    result = make_conditions(cond, prediction, target)
    torch.testing.assert_close(result["gt"][1:], cond[1:], rtol=0, atol=0)
    torch.testing.assert_close(result["gt"][0, :, -3:-1], target["direction"][0].expand(2, -1), rtol=0, atol=0)
    assert torch.isfinite(result["gt"]).all()


def test_evaluator_encodes_once_and_passes_factor_condition_to_each_euler_step():
    policy, data = tiny_policy(), observation_batch()
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.2, -0.3, 0.4])
    captured = []
    hook = policy.model.register_forward_pre_hook(lambda module, args: captured.append(args[2]))
    try:
        result = evaluate_batch(policy, data, noise, jitter)
    finally:
        hook.remove()
    assert policy.obs_encoder.policy_encoder.calls == 1
    assert tuple(result["arms"]) == ARMS
    assert len(captured) == len(ARMS) * policy.num_inference_steps
    for index, arm in enumerate(ARMS):
        expected = result["conditions"][arm.split("_")[0]]
        for actual in captured[index * policy.num_inference_steps:(index + 1) * policy.num_inference_steps]:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not result["arms"][arm]["action_pred"].requires_grad


@pytest.mark.parametrize("jitter_factor", ("jitter", "zero"))
def test_condition_factor_cannot_change_source_or_source_gate(jitter_factor):
    policy, data = tiny_policy(), observation_batch()
    result = evaluate_batch(policy, data, torch.randn_like(data["action"]), torch.tensor([0.2, -0.3, 0.4]))
    for key in ("source", "source_active"):
        torch.testing.assert_close(result["arms"]["predicted_" + jitter_factor][key],
                                   result["arms"]["gt_" + jitter_factor][key], rtol=0, atol=0)


def test_invalid_target_falls_back_to_original_angle_jitter_and_condition_in_every_cell():
    policy, data = tiny_policy(), observation_batch()
    data["action"][1, :, :2] = 0
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.2, -0.7, 0.4])
    native = policy.predict_action(data["obs"], noise=noise, angular_jitter=jitter, return_source=True)
    result = evaluate_batch(policy, data, noise, jitter)
    assert not result["target"]["valid"][1]
    torch.testing.assert_close(result["conditions"]["gt"][1], result["conditions"]["predicted"][1], rtol=0, atol=0)
    for arm in ARMS:
        for key in ("source", "action_pred", "source_active"):
            torch.testing.assert_close(result["arms"][arm][key][1], native[key][1], rtol=0, atol=0)


def test_gate_remains_original_but_does_not_disable_valid_gt_condition_factor():
    policy, data = tiny_policy(), observation_batch()
    with torch.no_grad():
        policy.obs_encoder.head[-1].weight.zero_()
        policy.obs_encoder.head[-1].bias.copy_(torch.tensor([0.4, 0.7, -6.]))
    data["action"][:, :, :2] = torch.tensor([1., 0.])
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.2, -0.3, 0.4])
    result = evaluate_batch(policy, data, noise, jitter)
    assert result["target"]["valid"].all()
    assert not torch.equal(result["conditions"]["gt"], result["conditions"]["predicted"])
    for arm in ARMS:
        assert not result["arms"][arm]["source_active"].any()
        torch.testing.assert_close(result["arms"][arm]["source"], policy.prior_noise_scale * noise, rtol=0, atol=0)
    # A gated source must not incorrectly gate the separate conditioning factor.
    assert not torch.equal(result["arms"]["gt_zero"]["action_pred"],
                           result["arms"]["predicted_zero"]["action_pred"])


@pytest.mark.parametrize("arm", ARMS)
def test_xy_geometry_and_requested_zero_or_original_jitter_direction(arm):
    policy, data = tiny_policy(), observation_batch()
    with torch.no_grad():
        policy.normalizer.params_dict["action"]["scale"].copy_(torch.tensor([1., 3., 2., 4., 5., 6., 7.]))
        policy.normalizer.params_dict["action"]["offset"].copy_(torch.tensor([.2, -.3, .1, .4, -.2, .5, -.1]))
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.2, -0.3, 0.4])
    result = evaluate_batch(policy, data, noise, jitter)
    assert result["target"]["valid"].all() and result["arms"][arm]["source_active"].all()
    source = result["arms"][arm]["source"]
    before = policy.normalizer["action"].unnormalize(policy.prior_noise_scale * noise)
    after = policy.normalizer["action"].unnormalize(source)
    torch.testing.assert_close(after[..., :2].square().sum(-1), before[..., :2].square().sum(-1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(source[..., 2:], (policy.prior_noise_scale * noise)[..., 2:], rtol=0, atol=0)
    theta = result["target"]["theta"] + (jitter if arm.endswith("jitter") else torch.zeros_like(jitter))
    expected_direction = torch.stack((theta.cos(), theta.sin()), -1)
    resultant = after[..., :2].sum(1)
    torch.testing.assert_close(resultant / resultant.norm(dim=-1, keepdim=True), expected_direction, rtol=1e-5, atol=1e-6)


def test_future_labels_do_not_change_predicted_condition_or_prediction():
    policy, data = tiny_policy(), observation_batch()
    data["action"][:, :, :2] = torch.tensor([0.4, 0.2])
    other = {"obs": data["obs"], "action": data["action"].clone()}
    other["action"][:, :, :2] *= -1
    noise, jitter = torch.randn_like(data["action"]), torch.tensor([0.2, -0.3, 0.4])
    before, after = evaluate_batch(policy, data, noise, jitter), evaluate_batch(policy, other, noise, jitter)
    torch.testing.assert_close(before["conditions"]["predicted"], after["conditions"]["predicted"], rtol=0, atol=0)
    for key in before["prediction"]:
        torch.testing.assert_close(before["prediction"][key], after["prediction"][key], rtol=0, atol=0)
    assert not torch.equal(before["conditions"]["gt"][..., -3:-1], after["conditions"]["gt"][..., -3:-1])
