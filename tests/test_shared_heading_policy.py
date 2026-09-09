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


def source_prediction(theta, confidence=None):
    theta = torch.as_tensor(theta, dtype=torch.float32)
    return {"theta": theta, "confidence": torch.ones_like(theta) if confidence is None
            else torch.as_tensor(confidence, dtype=torch.float32)}


def make_source_policy(mode="condition", blocks=((0, 1), (3, 4))):
    policy = make_policy(mode)
    policy.set_source_vector_blocks(blocks)
    policy.set_source_mode("heading")
    with torch.no_grad():
        policy.obs_encoder.head[-1].bias[2] = 4
    return policy


@pytest.mark.parametrize("blocks", [((0, 1),), ((0, 1), (3, 4))])
def test_heading_source_aligns_realized_raw_direction_and_preserves_geometry(blocks):
    policy = make_source_policy(blocks=blocks)
    with torch.no_grad():
        policy.normalizer.params_dict["action"]["scale"].copy_(torch.tensor([1., 3., 2., 4., 5., 6., 7.]))
        policy.normalizer.params_dict["action"]["offset"].copy_(torch.tensor([.2, -.3, .1, .4, -.2, .5, -.1]))
    z = torch.randn(3, 4, 7)
    prediction = source_prediction([0., .7, -1.3])
    jitter = torch.tensor([.2, -.1, .3])
    output = policy.transform_source(z, prediction, angular_jitter=jitter)
    raw_before = policy.normalizer["action"].unnormalize(z)
    raw_after = policy.normalizer["action"].unnormalize(output)
    resultant = raw_after[..., :2].sum(1)
    expected_angle = prediction["theta"] + jitter
    expected_direction = torch.stack((expected_angle.cos(), expected_angle.sin()), -1)
    torch.testing.assert_close(
        resultant / resultant.norm(dim=-1, keepdim=True), expected_direction,
        rtol=1e-5, atol=1e-6,
    )
    for block in blocks:
        torch.testing.assert_close(
            raw_before[..., list(block)].square().sum(-1),
            raw_after[..., list(block)].square().sum(-1), rtol=1e-5, atol=1e-6,
        )
    untouched = [i for i in range(7) if i not in {d for b in blocks for d in b}]
    torch.testing.assert_close(output[..., untouched], z[..., untouched], rtol=0, atol=0)
    # A raw rotation conjugated by anisotropic Min-Max is not orthogonal in
    # normalized coordinates. The experiment must not claim normalized norm preservation.
    assert not torch.allclose(z.square().sum((1, 2)), output.square().sum((1, 2)))


def test_source_confidence_gate_includes_threshold_and_ignores_invalid_heading():
    policy = make_source_policy()
    z = torch.randn(4, 4, 7)
    prediction = source_prediction([1., 1., 1., float("nan")], [.8, .49, .5, 1.])
    output, active = policy._transform_source(z, prediction, torch.zeros(4))
    assert active.tolist() == [True, False, True, False]
    torch.testing.assert_close(output[~active], z[~active], rtol=0, atol=0)
    assert torch.isfinite(output).all()


def test_source_gradient_detaches_reference_but_retains_finite_noise_gradients():
    policy = make_source_policy()
    z = torch.randn(3, 4, 7, requires_grad=True)
    theta = torch.tensor([0., .7, -1.3], requires_grad=True)
    confidence = torch.ones(3, requires_grad=True)
    output = policy.transform_source(z, {"theta": theta, "confidence": confidence}, torch.zeros(3))
    output.square().sum().backward()
    assert theta.grad is None and confidence.grad is None
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.normalizer.parameters())


@pytest.mark.parametrize("mode", HEADING_MODES)
def test_source_does_not_add_head_gradient_path_to_flow_loss(mode):
    policy = make_source_policy(mode)
    with torch.no_grad():
        policy.model.cond_obs_emb.weight[:, -3:].normal_(0, .1)
    result = policy.loss_components(batch(), angular_jitter=torch.zeros(3))
    assert result["source_active"].all()
    result["flow_loss"].backward()
    assert gradients_nonzero(policy.obs_encoder.policy_encoder)
    assert gradients_nonzero(policy.obs_encoder.head) == (mode == "condition")


@pytest.mark.parametrize("blocks", [((0, 1),), ((0, 1), (3, 4))])
def test_training_and_inference_use_identical_explicit_source(blocks):
    policy = make_source_policy(blocks=blocks).eval()
    data = batch()
    noise, jitter, t = torch.randn_like(data["action"]), torch.tensor([.1, .3, -.2]), torch.rand(3)
    training = policy.loss_components(data, noise=noise, t=t, angular_jitter=jitter)
    inference = policy.predict_action(data["obs"], noise=noise, angular_jitter=jitter, return_source=True)
    repeated = policy.predict_action(data["obs"], noise=noise, angular_jitter=jitter, return_source=True)
    torch.testing.assert_close(training["source"], inference["source"], rtol=0, atol=0)
    torch.testing.assert_close(training["source_active"], inference["source_active"], rtol=0, atol=0)
    torch.testing.assert_close(inference["action_pred"], repeated["action_pred"], rtol=0, atol=0)
    for key in training["prediction"]:
        torch.testing.assert_close(training["prediction"][key], inference["prediction"][key], rtol=0, atol=0)


def test_private_angular_rng_is_reproducible_and_does_not_advance_global_rngs():
    import numpy as np

    policy = make_source_policy()
    z = torch.randn(3, 4, 7)
    prediction = source_prediction([0., .7, -1.3])
    torch_state = torch.random.get_rng_state().clone()
    np_state = np.random.get_state()
    policy.set_source_seed(17)
    first = policy.transform_source(z, prediction)
    second = policy.transform_source(z, prediction)
    assert not torch.equal(first, second)
    policy.set_source_seed(17)
    repeated = policy.transform_source(z, prediction)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    np_actual = np.random.get_state()
    assert np_actual[0] == np_state[0] and np_actual[2:] == np_state[2:]
    np.testing.assert_array_equal(np_actual[1], np_state[1])


def test_heading_source_changes_angular_law_by_aligning_the_realized_chunk():
    policy = make_source_policy(blocks=((0, 1),))
    with torch.no_grad():
        policy.normalizer.params_dict["action"]["scale"].fill_(1.)
        policy.normalizer.params_dict["action"]["offset"].zero_()
    policy.set_source_seed(712)
    z = torch.randn(2048, 4, 7)
    output = policy.transform_source(z, source_prediction(torch.zeros(2048)))
    before = z[..., :2].sum(1)
    after = output[..., :2].sum(1)
    before_cosine = (before[:, 0] / before.norm(dim=-1)).mean()
    after_cosine = (after[:, 0] / after.norm(dim=-1)).mean()
    assert before_cosine.abs() < .1
    assert after_cosine > .8  # kappa=4 concentration; independent Gaussian rotation would stay ~0.


def test_iid_mode_ignores_extra_jitter_and_preserves_original_rng_and_keys():
    policy = make_policy("condition").eval()
    data = batch()
    torch.manual_seed(118)
    plain = policy.loss_components(data)
    plain_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(118)
    extra = policy.loss_components(data, angular_jitter=torch.ones(3))
    torch.testing.assert_close(plain["flow_loss"], extra["flow_loss"], rtol=0, atol=0)
    torch.testing.assert_close(plain["source"], extra["source"], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), plain_rng, rtol=0, atol=0)
    assert not extra["source_active"].any()
    assert set(policy.predict_action(data["obs"])) == {"action", "action_pred"}
    # Sampling explicit angular noise for the heading arm cannot shift common
    # PyTorch dropout, flow-time, or base-Gaussian draws in subsequent operations.
    heading = copy.deepcopy(policy)
    heading.set_source_mode("heading")
    torch.manual_seed(119)
    policy.loss_components(data)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(119)
    heading.loss_components(data)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)


def test_source_zero_raw_resultant_stays_finite_and_unchanged():
    policy = make_source_policy()
    raw = torch.zeros(3, 4, 7)
    raw[:, 0, 0] = 1
    raw[:, 1, 0] = -1
    z = policy.normalizer["action"].normalize(raw).requires_grad_()
    output, active = policy._transform_source(z, source_prediction([1., 2., 3.]), torch.zeros(3))
    assert not active.any()
    torch.testing.assert_close(output, z, rtol=0, atol=0)
    output.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_source_metadata_adds_no_checkpoint_tensors_and_validates_blocks():
    policy = make_policy()
    original_state = copy.deepcopy(policy.state_dict())
    policy.set_source_mode("heading")
    policy.set_source_vector_blocks(((0, 1),))
    policy.load_state_dict(original_state, strict=True)
    assert set(policy.state_dict()) == set(original_state)
    for blocks in ((), ((0,),), ((0, 1), (1, 3)), ((0, 1), (3, 7)), ((3, 4),), ((0, 1.5),)):
        with pytest.raises(ValueError, match="source_vector_blocks"):
            policy.set_source_vector_blocks(blocks)
    with pytest.raises(ValueError, match="source_mode"):
        policy.set_source_mode("unknown")
