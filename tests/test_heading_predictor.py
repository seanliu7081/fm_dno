import math

import pytest
import torch
import torch.nn as nn

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.heading_predictor import ObservationHeadingPredictor
from oat.perception.state_encoder import ProjectionStateEncoder
from oat.symmetry.so2_chunk import SO2ChunkSpec


SPEC = SO2ChunkSpec.translation_only()
SHAPE_META = {"obs": {"state": {"type": "state", "shape": [3]}}}


def make_predictor(**kwargs):
    return ObservationHeadingPredictor(
        ProjectionStateEncoder(SHAPE_META, out_dim=4), hidden_dim=8, **kwargs
    )


def make_normalizer():
    # Deliberately unequal planar scales/offsets to exercise SO(2) normalization.
    actions = torch.arange(70, dtype=torch.float32).reshape(10, 7) / 20
    actions[:, 1] *= 3
    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions, "state": torch.randn(20, 3)})
    return normalizer


def test_targets_mask_holds_reversals_and_keep_world_direction():
    predictor = make_predictor()
    normalizer = make_normalizer()
    old_scale = normalizer.params_dict["action"]["scale"].clone()
    predictor.set_normalizer(normalizer, SPEC)
    actions = torch.zeros(4, 2, 7)
    actions[0, :, 0] = 1
    actions[1, :, 1] = 1
    actions[3, :, 0] = torch.tensor([1.0, -1.0])
    targets = predictor.targets(actions, SPEC)
    torch.testing.assert_close(targets["theta"][:2], torch.tensor([0.0, math.pi / 2]))
    assert targets["valid"].tolist() == [True, True, False, False]
    assert not targets["theta"].requires_grad
    torch.testing.assert_close(normalizer.params_dict["action"]["scale"], old_scale)
    scales = predictor.normalizer.params_dict["action"]["scale"]
    assert scales[0] == scales[1]
    assert predictor.normalizer.params_dict["action"]["offset"][:2].tolist() == [0, 0]


def test_all_invalid_loss_is_finite_and_still_trains_validity():
    predictor = make_predictor()
    predictor.set_normalizer(make_normalizer(), SPEC)
    batch = {"obs": {"state": torch.randn(5, 2, 3)}, "action": torch.zeros(5, 2, 7)}
    result = predictor.loss(batch, SPEC)
    assert result["heading_loss"].item() == 0
    assert result["valid_fraction"].item() == 0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.head.parameters())
    assert predictor.head[-1].bias.grad[2] > 0


def test_circular_loss_handles_wrap_boundary():
    predictor = make_predictor()
    predictor.set_normalizer(make_normalizer(), SPEC)
    with torch.no_grad():
        predictor.head[-1].weight.zero_()
        predictor.head[-1].bias.copy_(torch.tensor([-1.0, 1e-3, 0.0]))
    actions = torch.zeros(2, 2, 7)
    actions[:, :, 0] = -1
    actions[:, :, 1] = -1e-3
    result = predictor.loss({"obs": {"state": torch.randn(2, 2, 3)}, "action": actions}, SPEC)
    assert result["heading_loss"] < 1e-5
    assert result["heading_error_deg"] < 0.2


class StochasticEncoder(BaseObservationEncoder):
    def __init__(self):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(0.8)

    def output_feature_dim(self):
        return 3

    def forward(self, obs):
        x = obs["state"]
        return self.dropout(self.batchnorm(x.flatten(0, 1))).reshape_as(x)


def test_freeze_keeps_entire_encoder_eval_under_parent_train():
    predictor = ObservationHeadingPredictor(StochasticEncoder(), hidden_dim=8).freeze()
    parent = nn.Sequential(predictor)
    parent.train()
    assert not any(module.training for module in predictor.modules())
    assert not any(param.requires_grad for param in predictor.parameters())
    observations = {"state": torch.randn(8, 2, 3)}
    mean = predictor.obs_encoder.batchnorm.running_mean.clone()
    first, second = predictor(observations), predictor(observations)
    torch.testing.assert_close(first["direction"], second["direction"], rtol=0, atol=0)
    torch.testing.assert_close(mean, predictor.obs_encoder.batchnorm.running_mean)


def test_checkpoint_round_trip_weights_normalizers_and_metadata(tmp_path):
    source = make_predictor()
    source.set_normalizer(make_normalizer(), SPEC, vector_mode="global_rms")
    source.freeze()
    observations = {"state": torch.randn(5, 2, 3)}
    checkpoint = tmp_path / "reference.pt"
    source.save_checkpoint(
        checkpoint, action_spec=SPEC, horizon=4,
        model_config={"_target_": "test.ExplicitArchitecture"}, metadata={"reference_mode": "state"},
    )
    restored = make_predictor().freeze()
    metadata = restored.load_checkpoint(
        checkpoint, action_spec=SPEC, horizon=4, normalizer_vector_mode="global_rms"
    )
    restored.train()
    assert metadata["reference_mode"] == "state"
    assert not any(param.requires_grad for param in restored.parameters())
    assert not restored.obs_encoder.training
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(source(observations)["theta"], restored(observations)["theta"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="action_spec"):
        make_predictor().load_checkpoint(checkpoint, action_spec=SO2ChunkSpec.libero_osc_pose())
    with pytest.raises(ValueError, match="horizon"):
        make_predictor().load_checkpoint(checkpoint, horizon=8)
    with pytest.raises(ValueError, match="normalizer_vector_mode"):
        make_predictor().load_checkpoint(checkpoint, normalizer_vector_mode="rms")


def test_forward_requires_expected_history_and_returns_unit_direction():
    predictor = make_predictor()
    predictor.set_normalizer(make_normalizer(), SPEC)
    with torch.no_grad():
        predictor.head[-1].weight.zero_()
        predictor.head[-1].bias.zero_()
    result = predictor({"state": torch.randn(3, 2, 3)})
    torch.testing.assert_close(result["direction"].norm(dim=-1), torch.ones(3))
    assert torch.isfinite(result["theta"]).all()
    with pytest.raises(ValueError, match="observation steps"):
        predictor({"state": torch.randn(3, 1, 3)})


def test_factory_matches_trainer_state_encoder_and_does_not_mutate_config(tmp_path):
    import copy
    import hydra
    from omegaconf import OmegaConf
    from oat.perception.heading_predictor import build_heading_predictor
    from oat.perception.fused_obs_encoder import FusedObservationEncoder

    shape = {
        'obs': {
            'image': {'type': 'rgb', 'shape': [84, 84, 3]},
            'state': {'type': 'state', 'shape': [3]},
        },
        'action': {'shape': [7]},
    }
    original = copy.deepcopy(shape)
    state_config = {
        '_target_': 'oat.perception.state_encoder.ProjectionStateEncoder', 'out_dim': 4,
    }
    predictor = build_heading_predictor(
        shape_meta=OmegaConf.create(shape), reference_mode='state',
        state_encoder=state_config, hidden_dim=8,
        vision_encoder={'_target_': 'nonexistent.MustNeverBeInstantiated'},
    )
    assert shape == original
    assert isinstance(predictor.obs_encoder, FusedObservationEncoder)
    assert predictor.obs_encoder.vision_encoder is None
    assert predictor.obs_encoder.state_ports == ['state']
    filtered = copy.deepcopy(shape)
    filtered['obs'].pop('image')
    trainer = ObservationHeadingPredictor(FusedObservationEncoder(
        shape_meta=filtered, state_encoder=state_config,
    ), hidden_dim=8)
    trainer.set_normalizer(make_normalizer(), SPEC)
    checkpoint = tmp_path / 'factory.pt'
    trainer.save_checkpoint(checkpoint, action_spec=SPEC, horizon=4)
    predictor.load_checkpoint(checkpoint, action_spec=SPEC, horizon=4)
    observations = {'state': torch.randn(3, 2, 3)}
    torch.testing.assert_close(trainer(observations)['theta'], predictor(observations)['theta'])
    instantiated = hydra.utils.instantiate({
        '_target_': 'oat.perception.heading_predictor.build_heading_predictor',
        '_recursive_': False,
        'shape_meta': shape, 'reference_mode': 'state', 'hidden_dim': 8,
        'state_encoder': state_config,
    })
    assert instantiated._encoder_signature() == trainer._encoder_signature()


def test_checkpoint_rejects_changed_input_order_with_identical_tensor_shapes(tmp_path):
    shape = {'obs': {
        'first': {'type': 'state', 'shape': [2]},
        'second': {'type': 'state', 'shape': [2]},
    }}
    reversed_shape = {'obs': dict(reversed(list(shape['obs'].items())))}
    source = ObservationHeadingPredictor(ProjectionStateEncoder(shape), hidden_dim=8)
    restored = ObservationHeadingPredictor(ProjectionStateEncoder(reversed_shape), hidden_dim=8)
    normalizer = make_normalizer()
    normalizer.fit({'first': torch.randn(10, 2), 'second': torch.randn(10, 2)})
    source.set_normalizer(normalizer, SPEC)
    checkpoint = tmp_path / 'port_order.pt'
    source.save_checkpoint(checkpoint, action_spec=SPEC, horizon=4)
    assert all(source.state_dict()[key].shape == value.shape
               for key, value in restored.state_dict().items())
    with pytest.raises(ValueError, match='encoder_signature'):
        restored.load_checkpoint(checkpoint, action_spec=SPEC)
