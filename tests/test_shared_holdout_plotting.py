"""Check scientific plotting invariants that could otherwise mislead viewers."""
import numpy as np
import pytest

from scripts.plot_shared_holdout_comparison import (
    angle_error, head16, save_heatmap_arrays, source_limit,
)


def test_smq_heads_are_never_averaged_into_an_h16_arrow():
    heading = {"heading_direction": np.array([[[1., 0.]]])}
    smq = {"heading_direction": np.array([[[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]])}
    np.testing.assert_array_equal(head16(heading, 0), [1., 0.])
    assert head16(smq, 0) is None


def test_source_limits_include_all_models_phases_and_extreme_draws():
    first = np.zeros((3, 1024, 16, 7))
    last = np.zeros_like(first)
    # The last draw of the last observation must survive any displayed subset.
    last[-1, -1, -1, 0] = 100.
    models = [{"source_raw": first}, {"source_raw": last}]
    assert source_limit(models, [0, 1, 2], "source_raw", False) > 100.
    assert source_limit(models, [0, 1, 2], "source_raw", True) > 100.
    assert source_limit(models, [0, 1, 2], "source_raw", True, segment=3) > 100.
    assert source_limit(models, [0, 1, 2], "source_raw", True, segment=0) == .1


def test_signed_errors_wrap_across_the_angle_boundary():
    theta = np.deg2rad([-179., 179.])
    vectors = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    np.testing.assert_allclose(angle_error(vectors, vectors[::-1]), [2., -2.], atol=1e-10)


def test_density_mass_does_not_renormalize_out_of_view_draws(tmp_path):
    heatmaps = {
        "density": np.array([[[[.1, .1], [0., 0.]]]]),
        "visible": np.array([[[True, False, False, False]]]),
        "camera_valid": np.ones((1, 1, 4), dtype=bool),
        "pixels": np.zeros((1, 1, 4, 2)), "gt_path": np.zeros((1, 2, 2)),
        "crop": [[0, 2, 2, 0]], "density_max": .1, "bandwidth_px": 2.,
        "translation_scale": .05, "steps": 1,
    }
    summary = save_heatmap_arrays(tmp_path, heatmaps, "test")
    assert summary["visible_fraction"] == [[.25]]
    assert summary["density_mass"] == [[.2]]
    heatmaps["density"] *= 2
    with pytest.raises(ValueError, match="cannot create endpoint probability mass"):
        save_heatmap_arrays(tmp_path, heatmaps, "invalid")
