"""CPU checks for temporal U-Net flow matching and shared heading deployment."""

import contextlib
import io

import pytest
import torch
from omegaconf import OmegaConf

from oat.model.flow.unet import FlowUnet1D
from oat.policy.base_policy import BasePolicy
from oat.policy.flow_policy_heading_zero import HeadingZeroFlowPolicy
from test_shared_heading_policy import (
    SHAPE_META, TinyEncoder, batch as observation_batch, make_source_policy,
)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(912)


def make_backbone(horizon=7):
    return FlowUnet1D(
        input_dim=7, output_dim=7, horizon=horizon, n_obs_steps=2, cond_dim=7,
        down_dims=(16, 32, 64), diffusion_step_embed_dim=16,
    )


@pytest.mark.parametrize("horizon", (1, 3, 4, 7, 16))
def test_shapes_and_gradients_include_padded_horizons(horizon):
    model = make_backbone(horizon)
    sample = torch.randn(3, horizon, 7, requires_grad=True)
    cond = torch.randn(3, 2, 7, requires_grad=True)
    velocity = model(sample, torch.tensor([0., 231., 999.]), cond)
    assert velocity.shape == sample.shape
    assert torch.isfinite(velocity).all()
    velocity.square().mean().backward()
    for value in (sample, cond):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


def test_continuous_time_broadcast_and_ordered_observation_dependence():
    model = make_backbone().eval()
    sample, cond = torch.randn(3, 7, 7), torch.randn(3, 2, 7)
    with torch.no_grad():
        expected = model(sample, torch.full((3,), 0.75), cond)
        for timestep in (0.75, torch.tensor(0.75), torch.tensor([0.75])):
            torch.testing.assert_close(model(sample, timestep, cond), expected, rtol=0, atol=0)
        assert not torch.allclose(model(sample, 0.0, cond), expected)
        assert not torch.allclose(model(sample, 701.0, cond), expected)
        assert not torch.allclose(model(sample, 0.75, cond.flip(1)), expected)
        assert not torch.allclose(model(sample, 0.75, cond + .5), expected)


def test_neutral_heading_columns_still_learn_in_all_film_blocks():
    model = make_backbone().eval()
    model.zero_condition_columns(range(4, 7))
    sample, cond = torch.randn(3, 7, 7), torch.randn(3, 2, 7)
    changed = cond.clone()
    changed[..., -3:] += 10
    torch.testing.assert_close(model(sample, 200., cond), model(sample, 200., changed), rtol=0, atol=0)
    model(sample, 200., cond).square().mean().backward()
    indices = [16 + step * 7 + feature for step in range(2) for feature in range(4, 7)]
    from oat.model.diffusion.conditional_unet1d import ConditionalResidualBlock1D
    for module in model.modules():
        if isinstance(module, ConditionalResidualBlock1D):
            assert module.cond_encoder[1].weight.grad[:, indices].abs().sum() > 0


def policy_config(backbone_type="unet"):
    return {
        "_target_": "oat.policy.flow_policy_heading_zero.HeadingZeroFlowPolicy",
        "shape_meta": SHAPE_META,
        "obs_encoder": {"_target_": "test_shared_heading_policy.TinyEncoder"},
        "horizon": 4, "n_action_steps": 2, "n_obs_steps": 2,
        "embed_dim": 8, "n_layers": 2, "n_heads": 2, "dropout": 0.,
        "num_inference_steps": 2, "heading_hidden_dim": 8,
        "backbone_type": backbone_type,
        "backbone_kwargs": {"down_dims": [16, 32], "diffusion_step_embed_dim": 16}
        if backbone_type == "unet" else ({"num_register_tokens": 2} if backbone_type == "starvla_dit" else {}),
    }


def make_policy(backbone_type="unet"):
    parameters = policy_config(backbone_type)
    parameters.pop("_target_")
    parameters["obs_encoder"] = TinyEncoder()
    with contextlib.redirect_stdout(io.StringIO()):
        policy = HeadingZeroFlowPolicy(**parameters)
    policy.set_normalizer(make_source_policy(blocks=((0, 1),)).normalizer)
    policy.set_heading_xy_rms(.8)
    with torch.no_grad():
        policy.obs_encoder.head[-1].bias[2] = 4
    return policy


@pytest.mark.parametrize("backbone_type", ("transformer", "starvla_dit", "unet"))
def test_heading_zero_loss_source_and_optimizer_support_all_backbones(backbone_type):
    policy = make_policy(backbone_type)
    data = observation_batch()
    noise = torch.randn_like(data["action"])
    optimizer = policy.get_optimizer(1e-3, 1e-3, 0., (.9, .95))
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in policy.parameters() if parameter.requires_grad
    }
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        components = policy.loss_components(data, noise=noise)
        assert torch.isfinite(components["loss"])
        components["loss"].backward()
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in policy.obs_encoder.head.parameters())
        optimizer.step()
    policy.eval()
    with torch.no_grad():
        train_source = policy.loss_components(data, noise=noise)["source"]
        output = policy.predict_action(data["obs"], noise=noise, return_source=True)
    torch.testing.assert_close(output["source"], train_source, rtol=0, atol=0)
    assert output["action"].shape == (3, 2, 7)
    assert output["action_pred"].shape == (3, 4, 7)
    assert torch.isfinite(output["action_pred"]).all()


def test_heading_condition_receives_flow_gradient_after_neutral_columns_learn():
    policy = make_policy()
    data = observation_batch()
    optimizer = policy.get_optimizer(1e-3, 1e-3, 0., (.9, .95))
    policy.loss_components(data)["flow_loss"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    policy.loss_components(data)["flow_loss"].backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in policy.obs_encoder.head.parameters())


def test_workspace_checkpoint_restores_unet_heading_and_noise_prior(tmp_path):
    from oat.workspace.train_policy import TrainPolicyWorkspace

    cfg = OmegaConf.create({
        "_target_": "oat.workspace.train_policy.TrainPolicyWorkspace",
        "policy": policy_config(),
        "optimizer": {"policy_lr": 1e-3, "obs_enc_lr": 1e-3, "heading_lr": 1e-3,
                      "weight_decay": 0., "betas": [.9, .95]},
        "training": {"use_ema": True},
    })
    with contextlib.redirect_stdout(io.StringIO()):
        workspace = TrainPolicyWorkspace(cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    trained = make_policy()
    trained.configure_from_dataset(type("Dataset", (), {"get_heading_xy_rms": lambda self: .37})())
    data = observation_batch()
    optimizer = trained.get_optimizer(1e-3, 1e-3, 0., (.9, .95))
    trained(data).backward()
    optimizer.step()
    for policy in (workspace.model, workspace.ema_model):
        policy.load_state_dict(trained.state_dict())
        policy.eval()
    checkpoint = tmp_path / "policy.ckpt"
    workspace.save_checkpoint(path=checkpoint, use_thread=False)
    with contextlib.redirect_stdout(io.StringIO()):
        restored = BasePolicy.from_checkpoint(str(checkpoint), output_dir=str(tmp_path)).eval()
    assert isinstance(restored.model, FlowUnet1D)
    assert restored.heading_xy_rms.item() == pytest.approx(.37)
    assert restored.source_jitter == "zero" and restored.source_mode == "heading"
    noise = torch.randn_like(data["action"])
    with torch.no_grad():
        expected = workspace.ema_model.predict_action(data["obs"], noise=noise, return_source=True)
        actual = restored.predict_action(data["obs"], noise=noise, return_source=True)
    for key in ("source", "source_active", "action", "action_pred"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_bfloat16_autocast_keeps_finite_velocity_and_gradient():
    model = make_backbone(horizon=4)
    sample, cond = torch.randn(3, 4, 7), torch.randn(3, 2, 7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        velocity = model(sample, torch.tensor([.5, 251., 900.]), cond)
        loss = velocity.float().square().mean()
    assert velocity.shape == sample.shape and torch.isfinite(velocity).all()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
