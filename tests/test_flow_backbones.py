"""Training and checkpoint compatibility for the retained flow backbones."""
import pytest
import torch
import contextlib
import io
from hydra.utils import get_class
from oat.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from oat.policy.flow_policy import FlowPolicy
from test_shared_heading_policy import SHAPE_META, TinyEncoder, batch, make_source_policy
from test_unet_flow import policy_config

BACKBONES = ("transformer", "unet", "starvla_dit")


def make_policy(backbone, family="zero", **overrides):
    cfg = policy_config(backbone)
    if family == "plain":
        cfg["_target_"] = "oat.policy.flow_policy.FlowPolicy"
        cfg.pop("heading_hidden_dim")
    elif family == "gaussian":
        cfg["_target_"] = "oat.policy.flow_policy_heading_gaussian.HeadingGaussianFlowPolicy"
    cfg.update(overrides)
    cls = get_class(cfg.pop("_target_"))
    cfg["obs_encoder"] = TinyEncoder()
    policy = cls(**cfg)
    policy.set_normalizer(make_source_policy(blocks=((0, 1),)).normalizer)
    if family != "plain":
        policy.set_heading_xy_rms(.8)
    return policy


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("family", ("plain", "zero", "gaussian"))
def test_training_and_checkpoint_roundtrip(backbone, family, tmp_path):
    torch.manual_seed(12)
    policy = make_policy(backbone, family)
    encoder = policy.obs_encoder if family == "plain" else policy.obs_encoder.policy_encoder
    before = encoder.projection.weight.detach().clone()
    optimizer = policy.get_optimizer(1e-2, 1e-2, 0., (.9, .999))
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in policy.parameters() if p.requires_grad}
    data = batch()
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = policy(data)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        optimizer.step()
    assert not torch.equal(before, encoder.projection.weight)
    policy.eval()
    torch.manual_seed(31)
    with torch.no_grad():
        expected = policy.predict_action(data["obs"])
    assert expected["action"].shape == (3, 2, 7)
    assert expected["action_pred"].shape == (3, 4, 7)
    path = tmp_path / "policy.pt"
    torch.save(policy.state_dict(), path)
    restored = make_policy(backbone, family).eval()
    restored.load_state_dict(torch.load(path, weights_only=False), strict=True)
    torch.manual_seed(31)
    with torch.no_grad():
        actual = restored.predict_action(data["obs"])
    for key in expected:
        assert torch.isfinite(actual[key]).all()
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("explicit_selector", (False, True), ids=("default", "explicit"))
def test_legacy_backbone_keeps_exact_initial_state(explicit_selector):
    # Construct the observation encoder before resetting the seed: only backbone
    # initialization must consume the same random stream as the legacy model.
    encoder = TinyEncoder()
    torch.manual_seed(91)
    expected = TransformerForDiffusion(
        input_dim=7, output_dim=7, horizon=4, n_obs_steps=2, cond_dim=4,
        n_layer=2, n_head=2, n_emb=8, p_drop_emb=0.0, p_drop_attn=0.0,
        causal_attn=False, time_as_cond=True, obs_as_cond=True,
    )
    torch.manual_seed(91)
    extra = {"backbone_type": "transformer"} if explicit_selector else {}
    with contextlib.redirect_stdout(io.StringIO()):
        policy = FlowPolicy(
            shape_meta=SHAPE_META, obs_encoder=encoder, horizon=4,
            n_action_steps=2, n_obs_steps=2, embed_dim=8, n_layers=2,
            n_heads=2, dropout=0.0, num_inference_steps=2, **extra,
        )
    assert type(policy.model) is TransformerForDiffusion
    assert policy.model.state_dict().keys() == expected.state_dict().keys()
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(policy.model.state_dict()[name], value, rtol=0, atol=0)



@pytest.mark.parametrize("backbone", ("typo_dit", "mixed_dit"))
def test_unsupported_backbone_fails(backbone):
    with pytest.raises(ValueError, match="backbone"):
        make_policy(backbone)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbone_dimensions_cannot_be_overridden(backbone):
    with pytest.raises(ValueError):
        make_policy(backbone, backbone_kwargs={"n_emb": 16})


@pytest.mark.parametrize("backbone", BACKBONES)
def test_unknown_backbone_option_fails(backbone):
    with pytest.raises((TypeError, ValueError)):
        make_policy(backbone, backbone_kwargs={"not_a_backbone_option": True})
