"""Physical camera-coordinate invariants used by task-image command overlays."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('comparison_projection', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_comparison_projection.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_camera_optical_axis_and_image_orientation():
    # An identity MuJoCo camera looks down -Z with +Y pointing image-up.
    transform = _module.camera_transform([0, 0, 0], np.eye(3), 90., 100, 200)
    points = np.array([[0., 0., -1.], [1., 0., -1.], [0., 1., -1.], [0., 0., 1.]])
    pixels, depth = _module.project_world(points, transform)
    np.testing.assert_allclose(pixels[:3], [[100, 50], [150, 50], [100, 0]], atol=1e-12)
    np.testing.assert_allclose(depth, [1, 1, 1, -1])


def test_projection_is_rigid_frame_invariant_with_chunk_batch():
    points = np.array([[[.1, .2, -2.], [.3, .4, -3.]]])
    shift = np.array([4., -3., 2.])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    original = _module.camera_transform(np.zeros(3), np.eye(3), 45, 128, 128)
    moved = _module.camera_transform(shift, rotation, 45, 128, 128)
    expected_pixels, expected_depth = _module.project_world(points, original)
    pixels, depth = _module.project_world(points @ rotation.T + shift, moved)
    np.testing.assert_allclose(pixels, expected_pixels)
    np.testing.assert_allclose(depth, expected_depth)


def test_episode_digest_matches_converter_float32_but_not_different_actions():
    actions = np.array([[.123456789, 1.], [-1., .25]], dtype=np.float64)
    assert _module.float32_action_digest(actions) == _module.float32_action_digest(actions.astype(np.float32))
    different = actions.copy(); different[0, 0] += .01
    assert _module.float32_action_digest(actions) != _module.float32_action_digest(different)
    assert _module.float32_action_digest(actions) != _module.float32_action_digest(actions.ravel())
