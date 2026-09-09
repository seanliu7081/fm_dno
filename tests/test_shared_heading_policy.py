"""CPU checks for the controlled shared-encoder heading experiment."""

import contextlib
import copy
import io

import pytest
import torch
from torch import nn

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.policy.flow_policy import FlowPolicy
from oat.policy.flow_policy_shared_heading import HEADING_MODES, SharedHeadingFlowPolicy


SHAPE_META = {
    "action": {"shape": [7]},
    "obs": {"state": {"shape": [3], "type": "state"}},
}


class TinyEncoder(BaseObservationEncoder):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.normalizer = LinearNormalizer()
        self.calls = 0

    def modalities(self):
        return ["state"]

    def output_feature_dim(self):
        return 4

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, obs):
        self.calls += 1
        return self.projection(self.normalizer["state"].normalize(obs["state"]))


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(41)


def make_policy(mode="baseline"):
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(24, 7), "state": torch.randn(24, 3)})
    with contextlib.redirect_stdout(io.StringIO()):
        policy = SharedHeadingFlowPolicy(
            shape_meta=SHAPE_META, obs_encoder=TinyEncoder(), horizon=4,
            n_action_steps=2, n_obs_steps=2, embed_dim=8, n_layers=1,
            n_heads=2, dropout=0, num_inference_steps=2,
            heading_hidden_dim=8, heading_mode=mode, heading_xy_rms=0.8,
        )
    policy.set_normalizer(normalizer)
    return policy


def batch():
    return {"obs": {"state": torch.randn(3, 2, 3)}, "action": torch.randn(3, 4, 7)}


def gradients_nonzero(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


@pytest.mark.parametrize("mode", HEADING_MODES)
def test_supervision_gradients_reach_shared_encoder_only_when_enabled(mode):
    policy = make_policy(mode)
    result = policy.loss_components(batch())
    assert policy.obs_encoder.policy_encoder.calls == 1
    result["weighted_heading_loss"].backward()
    assert gradients_nonzero(policy.obs_encoder.head)
    assert gradients_nonzero(policy.obs_encoder.policy_encoder) == (mode != "baseline")
    assert not gradients_nonzero(policy.model)


@pytest.mark.parametrize("mode", HEADING_MODES)
def test_flow_gradients_reach_heading_head_only_through_predicted_condition(mode):
    policy = make_policy(mode)
    # Initialization is deliberately neutral. Once these columns learn nonzero
    # weights, flow loss must be able to update a condition-mode heading head.
    with torch.no_grad():
        policy.model.cond_obs_emb.weight[:, -3:].normal_(0, 0.1)
    policy.loss_components(batch())["flow_loss"].backward()
    assert gradients_nonzero(policy.obs_encoder.policy_encoder)
    assert gradients_nonzero(policy.obs_encoder.head) == (mode == "condition")


def test_shared_initialization_matches_across_all_modes():
    initial = make_policy()
    data = batch()
    noise = torch.randn_like(data["action"])
    t = torch.rand(3)
    expected = initial.loss_components(data, noise=noise, t=t)["flow_loss"]
    for mode in HEADING_MODES:
        policy = copy.deepcopy(initial)
        policy.set_heading_mode(mode)
        actual = policy.loss_components(data, noise=noise, t=t)["flow_loss"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def make_matching_vanilla(shared):
    with contextlib.redirect_stdout(io.StringIO()):
        vanilla = FlowPolicy(
            shape_meta=SHAPE_META, obs_encoder=copy.deepcopy(shared.obs_encoder.policy_encoder),
            horizon=4, n_action_steps=2, n_obs_steps=2, embed_dim=8,
            n_layers=1, n_heads=2, dropout=0, num_inference_steps=2,
        )
    model_state = copy.deepcopy(shared.model.state_dict())
    model_state["cond_obs_emb.weight"] = model_state["cond_obs_emb.weight"][:, :-3]
    vanilla.model.load_state_dict(model_state)
    vanilla.set_normalizer(shared.normalizer)
    return vanilla


def test_baseline_matches_vanilla_forward_inference_and_clipped_update():
    shared = make_policy()
    vanilla = make_matching_vanilla(shared)
    data = batch()
    # Both forward methods draw the same IID noise and flow times.
    torch.manual_seed(123)
    shared_result = shared.loss_components(data)
    torch.manual_seed(123)
    vanilla_loss = vanilla(data)
    torch.testing.assert_close(shared_result["flow_loss"], vanilla_loss, rtol=1e-6, atol=1e-7)
    torch.manual_seed(124)
    shared_prediction = shared.predict_action(data["obs"])
    torch.manual_seed(124)
    vanilla_prediction = vanilla.predict_action(data["obs"])
    torch.testing.assert_close(shared_prediction["action"], vanilla_prediction["action"])

    optimizer_args = dict(policy_lr=1e-3, obs_enc_lr=1e-3, weight_decay=0, betas=(0.9, 0.999))
    shared_optimizer = shared.get_optimizer(**optimizer_args)
    vanilla_optimizer = vanilla.get_optimizer(**optimizer_args)
    shared_result["loss"].backward()
    vanilla_loss.backward()
    core = list(shared.model.parameters()) + list(shared.obs_encoder.policy_encoder.parameters())
    # Clipping head and core together could contaminate the baseline update.
    torch.nn.utils.clip_grad_norm_(core, 0.1)
    torch.nn.utils.clip_grad_norm_(shared.obs_encoder.head.parameters(), 0.1)
    torch.nn.utils.clip_grad_norm_(vanilla.parameters(), 0.1)
    shared_optimizer.step()
    vanilla_optimizer.step()
    torch.testing.assert_close(
        shared.obs_encoder.policy_encoder.projection.weight, vanilla.obs_encoder.projection.weight,
        rtol=1e-5, atol=1e-7,
    )
    for name, expected in vanilla.model.named_parameters():
        actual = dict(shared.model.named_parameters())[name]
        if name == "cond_obs_emb.weight":
            actual = actual[:, :-3]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize("mode", HEADING_MODES)
def test_optimizer_covers_head_and_preserves_fixed_normalizers(mode):
    policy = make_policy(mode)
    before = copy.deepcopy(policy.normalizer.state_dict())
    optimizer = policy.get_optimizer(1e-3, 1e-3, 0.01, (0.9, 0.999))
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len({id(p) for p in optimized})
    assert {id(p) for p in optimized} == {id(p) for p in policy.parameters() if p.requires_grad}
    assert all(id(p) in {id(q) for q in optimized} for p in policy.obs_encoder.head.parameters())
    assert all(not p.requires_grad for p in policy.normalizer.parameters())
    policy(batch()).backward()
    optimizer.step()
    for name, value in before.items():
        torch.testing.assert_close(policy.normalizer.state_dict()[name], value, rtol=0, atol=0)
    assert policy.heading_xy_rms not in set(policy.parameters())
    assert policy.heading_xy_rms.item() == pytest.approx(0.8)


def test_raw_heading_targets_are_independent_of_minmax_and_mask_holds():
    policy = make_policy()
    actions = torch.zeros(3, 4, 7)
    actions[0, :, :2] = torch.tensor([0.3, 0.4])
    actions[2, 0, 0] = 1
    actions[2, 1, 0] = -1
    target = policy.heading_targets(actions)
    torch.testing.assert_close(target["direction"][0], torch.tensor([0.6, 0.8]))
    assert target["confidence"][0].item() == pytest.approx(1.25)
    assert target["valid"].tolist() == [True, False, False]
    with torch.no_grad():
        policy.normalizer.params_dict["action"]["scale"].mul_(10)
        policy.normalizer.params_dict["action"]["offset"].add_(20)
    repeated = policy.heading_targets(actions)
    for name in target:
        torch.testing.assert_close(repeated[name], target[name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", HEADING_MODES)
def test_zero_heading_and_zero_prediction_are_finite(mode):
    policy = make_policy(mode)
    for parameter in policy.obs_encoder.head.parameters():
        nn.init.zeros_(parameter)
    data = batch()
    data["action"].zero_()
    result = policy.loss_components(data)
    assert result["heading_loss"].item() == 0
    assert result["valid_fraction"].item() == 0
    assert torch.isfinite(result["loss"])
    torch.testing.assert_close(
        result["prediction"]["direction"], torch.tensor([[1., 0.]]).expand(3, -1),
    )
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_condition_and_inference_depend_on_observations_only():
    policy = make_policy("condition")
    data = batch()
    other = {"obs": data["obs"], "action": -data["action"]}
    first = policy.loss_components(data)["prediction"]
    second = policy.loss_components(other)["prediction"]
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    assert policy.predict_action(data["obs"])["action"].shape == (3, 2, 7)


def test_invalid_configuration_is_rejected():
    policy = make_policy()
    with pytest.raises(ValueError, match="heading_mode"):
        policy.set_heading_mode("unknown")
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="heading_xy_rms"):
            policy.set_heading_xy_rms(value)
