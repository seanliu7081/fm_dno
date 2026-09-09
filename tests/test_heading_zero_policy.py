"""Dedicated full-training zero-jitter policy parity and lifecycle checks."""

import contextlib
import copy
import io
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from oat.policy.base_policy import BasePolicy
from oat.policy.flow_policy_heading_zero import HeadingZeroFlowPolicy
from scripts.mini_shared_heading import optimizer_for
from test_shared_heading_policy import (
    SHAPE_META, TinyEncoder, batch as observation_batch, gradients_nonzero,
    make_source_policy, single_cpu_thread,
)


@pytest.fixture(autouse=True)
def fixed_cpu_seed():
    torch.manual_seed(1013)


def make_policy(**overrides):
    normalizer = make_source_policy(blocks=((0, 1),)).normalizer
    settings = dict(shape_meta=SHAPE_META, obs_encoder=TinyEncoder(), horizon=4,
                    n_action_steps=2, n_obs_steps=2, embed_dim=8, n_layers=1,
                    n_heads=2, dropout=0, num_inference_steps=2,
                    heading_hidden_dim=8, heading_xy_rms=.8)
    settings.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = HeadingZeroFlowPolicy(**settings)
    policy.set_normalizer(normalizer)
    with torch.no_grad():
        policy.obs_encoder.head[-1].bias[2] = 4
    return policy


def test_fixed_defaults_and_equivalent_explicit_configuration():
    policy = make_policy(heading_mode="condition", source_mode="heading",
                         source_jitter="zero", source_vector_blocks=[[0, 1]])
    assert policy.heading_mode == "condition" and policy.source_mode == "heading"
    assert policy.source_jitter == "zero" and policy.source_vector_blocks == ((0, 1),)
    assert "heading_zero" in policy.get_policy_name()


@pytest.mark.parametrize("override", [
    {"heading_mode": "auxiliary"}, {"source_mode": "iid"},
    {"source_jitter": "vonmises"}, {"source_vector_blocks": ((0, 1), (3, 4))},
    {"source_vector_blocks": ((3, 4),)}, {"source_vector_blocks": None},
])
def test_conflicting_variant_constructor_settings_are_rejected(override):
    with pytest.raises(ValueError, match=next(iter(override))):
        make_policy(**override)


def test_policy_matches_equivalent_mini_losses_sources_and_inference_exactly():
    old = make_source_policy("condition", blocks=((0, 1),)).eval()
    old.set_source_jitter("zero")
    policy = make_policy().eval()
    policy.load_state_dict(old.state_dict(), strict=True)
    data = observation_batch()
    noise, t = torch.randn_like(data["action"]), torch.rand(3)
    expected = old.loss_components(data, noise=noise, t=t)
    actual = policy.loss_components(data, noise=noise, t=t)
    for key in ("loss", "flow_loss", "heading_loss", "validity_loss", "source", "source_active"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    with torch.no_grad():
        original_actions = old.predict_action(data["obs"], noise=noise, return_source=True)
        new_actions = policy.predict_action(data["obs"], noise=noise, return_source=True)
    for key in ("action_pred", "action", "source", "source_active"):
        torch.testing.assert_close(new_actions[key], original_actions[key], rtol=0, atol=0)


def test_optimizer_partitions_head_core_and_preserves_configured_learning_rates():
    policy = make_policy()
    optimizer = policy.get_optimizer(policy_lr=5e-5, obs_enc_lr=1e-5,
                                     heading_lr=7e-4, weight_decay=.03, betas=(.9, .95))
    head = {id(p) for p in policy.obs_encoder.head.parameters() if p.requires_grad}
    encoder = {id(p) for p in policy.obs_encoder.policy_encoder.parameters() if p.requires_grad}
    flow = {id(p) for p in policy.model.parameters() if p.requires_grad}
    covered = []
    for group in optimizer.param_groups:
        ids = {id(p) for p in group["params"]}
        covered.extend(id(p) for p in group["params"])
        if ids & head:
            assert ids == head and group["lr"] == 7e-4 and group["weight_decay"] == 0
        elif ids & encoder:
            assert ids <= encoder and group["lr"] == 1e-5
        elif ids:
            assert ids <= flow and group["lr"] == 5e-5
    assert len(covered) == len(set(covered))
    assert set(covered) == {id(p) for p in policy.parameters() if p.requires_grad}
    assert optimizer.defaults["betas"] == (.9, .95)
    default = policy.get_optimizer(5e-5, 1e-5, 0., (.9, .95))
    head_group = next(g for g in default.param_groups if {id(p) for p in g["params"]} == head)
    assert head_group["lr"] == 1e-3


def test_clip_groups_are_ordered_disjoint_and_cover_trainable_parameters():
    policy = make_policy()
    groups = policy.get_gradient_clip_groups()
    assert tuple(groups) == ("core", "heading")
    assert {id(p) for p in groups["heading"]} == {id(p) for p in policy.obs_encoder.head.parameters()}
    flat = [p for group in groups.values() for p in group]
    assert len(flat) == len({id(p) for p in flat})
    assert {id(p) for p in flat} == {id(p) for p in policy.parameters() if p.requires_grad}
    assert all(p.requires_grad for p in flat)


def test_optimizer_and_separately_clipped_update_match_mini_protocol():
    policy = make_policy().eval()
    old = make_source_policy("condition", blocks=((0, 1),)).eval()
    old.set_source_jitter("zero")
    old.load_state_dict(policy.state_dict(), strict=True)
    args = SimpleNamespace(policy_lr=5e-5, encoder_lr=1e-5, head_lr=1e-3)
    expected_optimizer, old_core, old_head = optimizer_for(old, args)
    optimizer = policy.get_optimizer(args.policy_lr, args.encoder_lr, 0., (.9, .95))
    data = observation_batch()
    noise, t = torch.randn_like(data["action"]), torch.rand(3)
    old.loss_components(data, noise=noise, t=t)["loss"].backward()
    policy.loss_components(data, noise=noise, t=t)["loss"].backward()
    for group in (old_core, old_head):
        torch.nn.utils.clip_grad_norm_(group, 1., error_if_nonfinite=True)
    for group in policy.get_gradient_clip_groups().values():
        torch.nn.utils.clip_grad_norm_(group, 1., error_if_nonfinite=True)
    expected_optimizer.step()
    optimizer.step()
    for key, expected in old.state_dict().items():
        torch.testing.assert_close(policy.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("loss_name", ("flow_loss", "weighted_heading_loss"))
def test_head_and_shared_encoder_train_jointly(loss_name):
    policy = make_policy()
    with torch.no_grad():
        policy.model.cond_obs_emb.weight[:, -3:].normal_(0, .1)
    policy.loss_components(observation_batch())[loss_name].backward()
    assert gradients_nonzero(policy.obs_encoder.head)
    assert gradients_nonzero(policy.obs_encoder.policy_encoder)


def test_source_branch_remains_detached_from_predictor_and_uses_no_angular_jitter():
    policy, data = make_policy().eval(), observation_batch()
    _, prediction = policy.obs_encoder.encode_with_heading(data["obs"])
    noise = torch.randn_like(data["action"], requires_grad=True)
    before = copy.deepcopy(policy._source_rng.bit_generator.state)
    source = policy.transform_source(noise, prediction)
    assert policy._source_rng.bit_generator.state == before
    source.square().mean().backward()
    assert noise.grad is not None and torch.isfinite(noise.grad).all()
    assert not gradients_nonzero(policy.obs_encoder.head)
    assert not gradients_nonzero(policy.obs_encoder.policy_encoder)
    raw = policy.normalizer["action"].unnormalize(source.detach())
    resultant = raw[..., :2].sum(1)
    torch.testing.assert_close(resultant / resultant.norm(dim=-1, keepdim=True), prediction["direction"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(source[..., 2:], noise[..., 2:], rtol=0, atol=0)


def test_dataset_configuration_sets_checkpoint_buffer_without_changing_normalizer():
    policy = make_policy()
    previous = copy.deepcopy(policy.normalizer.state_dict())
    dataset = SimpleNamespace(get_heading_xy_rms=lambda: .37)
    policy.configure_from_dataset(dataset)
    assert policy.heading_xy_rms.item() == pytest.approx(.37)
    assert "heading_xy_rms" in policy.state_dict()
    assert "heading_xy_rms" not in dict(policy.named_parameters())
    for key, value in previous.items():
        torch.testing.assert_close(policy.normalizer.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="heading_xy_rms"):
        policy.configure_from_dataset(SimpleNamespace(get_heading_xy_rms=lambda: float("nan")))


def test_workspace_checkpoint_restores_zero_mode_and_rms_without_loading_dataset(tmp_path, monkeypatch):
    from oat.workspace.train_policy import TrainPolicyWorkspace

    cfg = OmegaConf.create({
        "_target_": "oat.workspace.train_policy.TrainPolicyWorkspace",
        "policy": {
            "_target_": "oat.policy.flow_policy_heading_zero.HeadingZeroFlowPolicy",
            "shape_meta": SHAPE_META,
            "obs_encoder": {"_target_": "test_shared_heading_policy.TinyEncoder"},
            "horizon": 4, "n_action_steps": 2, "n_obs_steps": 2,
            "embed_dim": 8, "n_layers": 1, "n_heads": 2, "dropout": 0,
            "num_inference_steps": 2, "heading_hidden_dim": 8,
        },
        "optimizer": {"policy_lr": 5e-5, "obs_enc_lr": 1e-5, "heading_lr": 1e-3,
                      "weight_decay": 0., "betas": [.9, .95]},
        "training": {"use_ema": True},
    })
    with contextlib.redirect_stdout(io.StringIO()):
        workspace = TrainPolicyWorkspace(cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    normalizer = make_source_policy().normalizer
    for policy in (workspace.model, workspace.ema_model):
        policy.set_normalizer(normalizer)
        policy.configure_from_dataset(SimpleNamespace(get_heading_xy_rms=lambda: .37))
        with torch.no_grad():
            policy.obs_encoder.head[-1].bias[2] = 4
        policy.eval()
    checkpoint = tmp_path / "policy.ckpt"
    workspace.save_checkpoint(path=checkpoint, use_thread=False)

    def reject_dataset_access(*args, **kwargs):
        raise AssertionError("Inference checkpoint restoration must not access a dataset")

    monkeypatch.setattr(HeadingZeroFlowPolicy, "configure_from_dataset", reject_dataset_access)
    with contextlib.redirect_stdout(io.StringIO()):
        restored = BasePolicy.from_checkpoint(str(checkpoint), output_dir=str(tmp_path))
    restored.eval()
    assert isinstance(restored, HeadingZeroFlowPolicy)
    assert restored.source_jitter == "zero" and restored.source_mode == "heading"
    assert restored.heading_mode == "condition" and restored.source_vector_blocks == ((0, 1),)
    assert restored.heading_xy_rms.item() == pytest.approx(.37)
    data = observation_batch()
    noise = torch.randn_like(data["action"])
    with torch.no_grad():
        expected = workspace.ema_model.predict_action(data["obs"], noise=noise, return_source=True)
        actual = restored.predict_action(data["obs"], noise=noise, return_source=True)
    for key in ("source", "source_active", "action_pred"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
