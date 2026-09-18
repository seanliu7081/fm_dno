"""Real-robot state schemas keep the spatial encoder's LIBERO default intact."""

import copy

import pytest
import torch

from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.perception.spatial_token_encoder import SpatialTokenObservationEncoder


LEGACY_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FRUIT_KEYS = ["robot0_eef_pos", "robot0_eef_rot6d", "robot0_gripper_qpos"]


def shape_meta(real_robot=False):
    states = zip(FRUIT_KEYS, (3, 6, 1)) if real_robot else zip(LEGACY_KEYS, (3, 4, 2))
    return {"obs": {
        "agentview_rgb": {"shape": [128, 128, 3], "type": "rgb"},
        "robot0_eye_in_hand_rgb": {"shape": [128, 128, 3], "type": "rgb"},
        **{key: {"shape": [size], "type": "state"} for key, size in states},
        "task_uid": {"shape": [1], "type": "state"},
    }, "action": {"shape": [7]}}


def make_encoder(meta, **kwargs):
    return SpatialTokenObservationEncoder(
        meta, n_obs_steps=2, crop_shape=(112, 112), token_dim=16,
        task_ids=[0], **kwargs,
    )


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("real_robot,state_keys,expected_dim", [
    (False, None, 9),
    (True, FRUIT_KEYS, 10),
])
def test_actual_forward_retains_image_geometry_with_both_robot_state_schemas(
        real_robot, state_keys, expected_dim):
    meta = shape_meta(real_robot)
    encoder = make_encoder(meta, state_keys=state_keys).eval()
    assert encoder.state_projection[0].in_features == expected_dim
    normalizer = LinearNormalizer()
    for key in encoder.state_ports:
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    for key in encoder.rgb_ports:
        normalizer[key] = SingleFieldLinearNormalizer.create_fit(
            torch.tensor([[0., 0., 0.], [255., 255., 255.]])
        )
    encoder.set_normalizer(normalizer)
    observations = {
        key: torch.zeros((1, 2, *info["shape"]),
                         dtype=torch.uint8 if info["type"] == "rgb" else torch.float32)
        for key, info in meta["obs"].items()
    }
    observations["task_uid"] = observations["task_uid"].long()
    with torch.inference_mode():
        output = encoder(observations)
    assert encoder.grid_shape == (4, 4)
    assert output.shape == (1, 2 * 2 * 16 + 2 + 1, 16)
    assert output.shape[1] == encoder.output_token_count()
    assert torch.isfinite(output).all()
    assert all(not parameter.requires_grad for parameter in encoder.normalizer.parameters())


def test_legacy_default_checkpoint_restores_with_explicit_legacy_keys():
    default = make_encoder(shape_meta())
    explicit = make_encoder(shape_meta(), state_keys=LEGACY_KEYS)
    explicit.load_state_dict(default.state_dict(), strict=True)
    assert default.state_ports == explicit.state_ports == LEGACY_KEYS
    assert default.state_projection[0].weight.shape == (16, 9)


def test_default_still_rejects_real_robot_state_schema_without_explicit_keys():
    with pytest.raises(ValueError, match="robot0_eef_quat must have shape"):
        make_encoder(shape_meta(real_robot=True))


@pytest.mark.parametrize("keys", [
    [], ["robot0_eef_pos", "robot0_eef_pos"], [""], [1],
    "robot0_eef_pos", ["missing"], ["task_uid"], ["agentview_rgb"],
])
def test_invalid_state_keys_fail_before_model_construction(keys):
    with pytest.raises(ValueError, match="state_keys|Unknown state"):
        make_encoder(shape_meta(real_robot=True), state_keys=keys)


@pytest.mark.parametrize("shape", [[], [0], [-1], [2, 3], [2.5], [True]])
def test_custom_state_keys_require_positive_vector_shapes(shape):
    meta = copy.deepcopy(shape_meta(real_robot=True))
    meta["obs"]["robot0_eef_rot6d"]["shape"] = shape
    with pytest.raises(ValueError, match="positive one-dimensional shape"):
        make_encoder(meta, state_keys=FRUIT_KEYS)
