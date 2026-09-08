"""CPU integration checks for selectable flow and heading-flow backbones.

These use tiny image/state encoders and need no simulator, datasets, or vision
packages. Run with ``python -m pytest tests/test_flow_backbones.py -q``.
"""

import contextlib
import io
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from oat.model.common.normalizer import LinearNormalizer
from oat.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.heading_predictor import ObservationHeadingPredictor
from oat.policy.flow_policy import FlowPolicy
from oat.policy.flow_policy_learned_canon import LearnedCanonicalPhaseFlowPolicy
from oat.symmetry.so2_chunk import SO2ChunkSpec


SHAPE_META = {
    "action": {"shape": [7]},
    "obs": {
        "robot0_eef_pos": {"shape": [3], "type": "state"},
        "camera": {"shape": [3, 4, 4], "type": "rgb"},
    },
}
ACTION_SPEC = SO2ChunkSpec.translation_only()
BACKBONES = ("mixed_dit", "starvla_dit")


class TinyObservationEncoder(BaseObservationEncoder):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.normalizer = LinearNormalizer()
        self.projection = nn.Linear(4, 4)
        self.dropout = nn.Dropout(dropout)

    def modalities(self):
        return ["state", "rgb"]

    def output_feature_dim(self):
        return 4

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, obs_dict):
        state = self.normalizer["robot0_eef_pos"].normalize(obs_dict["robot0_eef_pos"])
        image_mean = obs_dict["camera"].mean(dim=(-3, -2, -1))
        features = torch.cat((state, image_mean.unsqueeze(-1)), dim=-1)
        return self.dropout(self.projection(features))


def make_normalizer():
    result = LinearNormalizer()
    values = torch.linspace(-1.0, 1.0, 32)[:, None]
    result.fit({
        "action": values.expand(32, 7).clone(),
        "robot0_eef_pos": values.expand(32, 3).clone(),
    })
    return result


def make_observations(batch_size=3):
    return {
        "robot0_eef_pos": torch.randn(batch_size, 2, 3),
        "camera": torch.rand(batch_size, 2, 3, 4, 4),
    }


def make_policy(
    backbone_type="transformer", *, heading_condition=False,
    reference_checkpoint=None, initialize=True, **overrides,
):
    parameters = {
        "shape_meta": SHAPE_META,
        "obs_encoder": TinyObservationEncoder(),
        "horizon": 4,
        "n_action_steps": 2,
        "n_obs_steps": 2,
        "embed_dim": 8,
        "n_layers": 2,
        "n_heads": 2,
        "dropout": 0.0,
        "num_inference_steps": 2,
        "backbone_type": backbone_type,
        "backbone_kwargs": {"num_register_tokens": 2} if backbone_type == "starvla_dit" else {},
    }
    policy_class = FlowPolicy
    if heading_condition:
        policy_class = LearnedCanonicalPhaseFlowPolicy
        parameters.update({
            "reference_model": ObservationHeadingPredictor(
                TinyObservationEncoder(dropout=0.5), hidden_dim=8,
            ),
            "reference_checkpoint": str(reference_checkpoint),
            "reference_use": "condition",
            "action_spec": {"block_weights": [1.0, 0.0]},
            "coupling": {"mode": "iid"},
        })
    parameters.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = policy_class(**parameters)
        if initialize:
            policy.set_normalizer(make_normalizer())
            if heading_condition:
                policy.prepare_for_training()
    return policy


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(12)


@pytest.fixture
def reference_checkpoint(tmp_path):
    predictor = ObservationHeadingPredictor(TinyObservationEncoder(dropout=0.5), hidden_dim=8)
    predictor.set_normalizer(make_normalizer(), ACTION_SPEC)
    artifact = tmp_path / "heading_reference.pt"
    predictor.save_checkpoint(artifact, action_spec=ACTION_SPEC, horizon=4)
    return artifact


@pytest.mark.parametrize("backbone_type", BACKBONES)
@pytest.mark.parametrize("heading_condition", (False, True), ids=("plain", "heading"))
def test_training_optimizer_and_inference(backbone_type, heading_condition, reference_checkpoint):
    policy = make_policy(
        backbone_type, heading_condition=heading_condition,
        reference_checkpoint=reference_checkpoint,
    )
    policy.train()
    encoder = policy.obs_encoder.policy_encoder if heading_condition else policy.obs_encoder
    before_encoder = encoder.projection.weight.detach().clone()
    before_model = {name: parameter.detach().clone()
                    for name, parameter in policy.model.named_parameters()}
    if heading_condition:
        reference = policy.reference_model
        before_reference = {name: parameter.detach().clone()
                            for name, parameter in reference.named_parameters()}
        assert not any(module.training for module in reference.modules())
        assert not any(parameter.requires_grad for parameter in reference.parameters())

    optimizer = policy.get_optimizer(1e-2, 1e-2, 0.0, (0.9, 0.999))
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimized) == len({id(parameter) for parameter in optimized})
    assert {id(parameter) for parameter in optimized} == {
        id(parameter) for parameter in policy.parameters() if parameter.requires_grad
    }
    batch = {"obs": make_observations(), "action": torch.randn(3, 4, 7)}
    # Zero-initialized outputs/gates delay gradients into earlier layers. Exercise
    # several real updates before checking that observation conditioning learns.
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = policy(batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [parameter.grad for parameter in optimized if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()

    assert not torch.equal(encoder.projection.weight, before_encoder)
    assert any(not torch.equal(parameter, before_model[name])
               for name, parameter in policy.model.named_parameters())
    if heading_condition:
        assert encoder.output_feature_dim() + 3 == policy.obs_feature_dim
        assert all(parameter.grad is None for parameter in reference.parameters())
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(parameter, before_reference[name], rtol=0, atol=0)

    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(batch["obs"])
    assert result["action"].shape == (3, 2, 7)
    assert result["action_pred"].shape == (3, 4, 7)
    assert all(torch.isfinite(value).all() for value in result.values())


@pytest.mark.parametrize("backbone_type", BACKBONES)
def test_heading_dno_rollout_retains_noise_gradients(backbone_type, reference_checkpoint):
    policy = make_policy(
        backbone_type, heading_condition=True, reference_checkpoint=reference_checkpoint,
    )
    obs = make_observations()
    optimizer = policy.get_optimizer(1e-2, 1e-2, 0.0, (0.9, 0.999))
    # Exercise gradients through a learned velocity field, including mixed DiT's
    # attention branches after their initially zero gates have started learning.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        policy({"obs": obs, "action": torch.randn(3, 4, 7)}).backward()
        optimizer.step()
    policy.eval()
    policy.requires_grad_(False)
    noise = torch.randn(3, 4, 7, requires_grad=True)
    result = policy.predict_action_from_noise(obs, noise, n_steps=3)
    assert result["action"].shape == (3, 2, 7)
    assert result["action_pred"].shape == (3, 4, 7)
    result["action_pred"].square().mean().backward()
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()
    assert noise.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in policy.reference_model.parameters())


@pytest.mark.parametrize("backbone_type", BACKBONES)
@pytest.mark.parametrize("heading_condition", (False, True), ids=("plain", "heading"))
def test_checkpoint_roundtrip_embeds_heading_reference(
    backbone_type, heading_condition, reference_checkpoint, tmp_path,
):
    policy = make_policy(
        backbone_type, heading_condition=heading_condition,
        reference_checkpoint=reference_checkpoint,
    ).eval()
    obs = make_observations()
    # Train first so mixed DiT's initially zero output cannot hide broken loads.
    optimizer = policy.get_optimizer(1e-2, 1e-2, 0.0, (0.9, 0.999))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        policy({"obs": obs, "action": torch.randn(3, 4, 7)}).backward()
        optimizer.step()

    torch.manual_seed(31)
    with torch.no_grad():
        expected = policy.predict_action(obs)
    checkpoint = tmp_path / "policy.pt"
    torch.save(policy.state_dict(), checkpoint)
    reference_checkpoint.unlink()
    restored = make_policy(
        backbone_type, heading_condition=heading_condition,
        reference_checkpoint=reference_checkpoint, initialize=False,
    )
    restored.load_state_dict(torch.load(checkpoint, weights_only=False), strict=True)
    restored.eval()
    if heading_condition:
        restored.prepare_for_training()
        assert bool(restored.obs_encoder.reference_loaded)
        assert not any(parameter.requires_grad for parameter in restored.reference_model.parameters())
        assert not any(module.training for module in restored.reference_model.modules())
    torch.manual_seed(31)
    with torch.no_grad():
        actual = restored.predict_action(obs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("explicit_selector", (False, True), ids=("default", "explicit"))
def test_legacy_backbone_keeps_exact_initial_state(explicit_selector):
    # Construct the observation encoder before resetting the seed: only backbone
    # initialization must consume the same random stream as the legacy model.
    encoder = TinyObservationEncoder()
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


@pytest.mark.parametrize("backbone_type", BACKBONES)
@pytest.mark.parametrize("heading_condition", (False, True), ids=("plain", "heading"))
def test_training_config_composes(backbone_type, heading_condition):
    prefix = "train_flowpolicy_heading_condition" if heading_condition else "train_flowpolicy"
    config_name = f"{prefix}_{backbone_type}"
    config_directory = str(Path(__file__).resolve().parents[1] / "oat" / "config")
    with initialize_config_dir(config_dir=config_directory, version_base="1.3"):
        cfg = compose(config_name=config_name)
        policy_config = OmegaConf.to_container(cfg.policy, resolve=True)
    assert cfg.name == config_name
    assert policy_config["backbone_type"] == backbone_type
    assert policy_config["embed_dim"] == 256
    assert policy_config["n_layers"] == 4
    assert policy_config["n_heads"] == 4
    assert policy_config["horizon"] == 16
    assert policy_config["num_inference_steps"] == 10
    if backbone_type == "starvla_dit":
        assert policy_config["backbone_kwargs"]["num_register_tokens"] == 32
    if heading_condition:
        assert policy_config["_target_"] == (
            "oat.policy.flow_policy_learned_canon.LearnedCanonicalPhaseFlowPolicy"
        )
        assert policy_config["reference_use"] == "condition"
    else:
        assert policy_config["_target_"] == "oat.policy.flow_policy.FlowPolicy"


def test_unknown_backbone_fails_at_construction():
    with pytest.raises(ValueError, match="backbone"):
        make_policy("typo_dit")


@pytest.mark.parametrize("backbone_type", ("transformer",) + BACKBONES)
def test_backbone_kwargs_cannot_override_policy_dimensions(backbone_type):
    with pytest.raises(ValueError):
        make_policy(backbone_type, backbone_kwargs={"n_emb": 16})


@pytest.mark.parametrize("backbone_type", BACKBONES)
def test_unrecognized_backbone_kwargs_fail_at_construction(backbone_type):
    with pytest.raises((TypeError, ValueError)):
        make_policy(backbone_type, backbone_kwargs={"not_a_backbone_option": True})
