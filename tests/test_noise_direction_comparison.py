"""Direction validity and paired whole-episode aggregation checks."""
import numpy as np
import pytest

from scripts.analyze_noise_direction_comparison import (
    EpisodeBootstrap, analyze_model, directional_scores, paired_window_difference, source_shape_metrics,
)


def test_direction_angles_wrap_and_invalid_prediction_cannot_improve_accuracy():
    angle = np.deg2rad([-179., 90., 0., 0.])
    predictions = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    predictions[2] = 0.
    target_angle = np.deg2rad([179., 0., 0., 0.])
    target = np.stack((np.cos(target_angle), np.sin(target_angle)), axis=-1)
    scores = directional_scores(predictions, target, [True, True, True, False], .05)
    np.testing.assert_allclose(scores["angle_deg"][:2], [2., 90.], atol=1e-10)
    assert np.isnan(scores["angle_deg"][2:]).all()
    np.testing.assert_equal(scores["within15"], [1., 0., 0., np.nan])
    np.testing.assert_equal(scores["coverage_on_gt_valid"], [1., 1., 0., np.nan])


def test_raw_xy_resultant_horizon_and_threshold_match_heading_target():
    gt = np.zeros((1, 16, 7))
    gt[:, :8, 0] = 1.
    gt[:, 8:, 1] = 1.
    pred = np.repeat(gt[:, None], 2, axis=1)
    pred[:, :, :, 0] = 1.
    pred[:, :, :, 1] = 0.
    archive = {"action_pred": pred, "source_raw": pred, "source_norm": pred, "heading_direction": np.array([[0., 1.]]),
               "heading_confidence": np.array([.8]), "source_active": np.ones((1, 2))}
    scores, shared = analyze_model(archive, gt, .5, .05)
    np.testing.assert_allclose(scores["direction_h16_angle_deg"], 45.)
    np.testing.assert_allclose(scores["direction_h8_angle_deg"], 0.)
    np.testing.assert_allclose(scores["head_angle_deg"], 45.)
    np.testing.assert_allclose(shared["gt_confidence_h16"], np.sqrt(128) / (.5 * 4))
    # Huge unrelated gripper/rotation errors do not alter XY direction.
    archive["action_pred"][:, :, :, 3:] = 100.
    changed, _ = analyze_model(archive, gt, .5, .05)
    np.testing.assert_equal(changed["direction_h16_angle_deg"], scores["direction_h16_angle_deg"])
    assert np.all(changed["action_h16_mse"] > scores["action_h16_mse"])


def test_baseline_head_is_unavailable_instead_of_zero_accuracy():
    gt = np.zeros((1, 16, 7)); gt[:, :, 0] = 1.
    archive = {"action_pred": gt[:, None], "source_raw": gt[:, None], "source_norm": gt[:, None],
               "heading_direction": np.full((1, 2), np.nan), "heading_confidence": np.full(1, np.nan),
               "source_active": np.zeros((1, 1))}
    scores, _ = analyze_model(archive, gt, .5, .05, baseline=True)
    assert np.isnan(scores["head_within30"]).all()
    assert np.isnan(scores["source_active"]).all()
    np.testing.assert_equal(scores["direction_h16_angle_deg"], [[0.]])


def test_pairing_excludes_the_same_invalid_draw_in_both_arms():
    left = np.array([[1., np.nan, 9.], [np.nan, np.nan, np.nan]])
    right = np.array([[4., 1000., np.nan], [2., 3., 4.]])
    difference, coverage = paired_window_difference(left, right)
    np.testing.assert_equal(difference, [-3., np.nan])
    np.testing.assert_allclose(coverage, [1 / 3, 0.])


def test_task_balance_and_whole_episode_resampling_match_exact_reference():
    tasks = np.array([1, 1, 1, 2])
    episodes = np.array([10, 10, 11, 20])
    values = np.array([0., 0., 9., 20.])
    estimator = EpisodeBootstrap(tasks, episodes, repetitions=1000, seed=9)
    actual = estimator.estimate(values)
    assert actual["mean"] == pytest.approx((3. + 20.) / 2.)
    assert actual["mean"] != pytest.approx(values.mean())
    # Episode 10 always carries BOTH zero-valued windows. Reconstruct the
    # exact clustered distribution; frame-level resampling would differ.
    multiplicities = estimator.strata[0][-1]
    first_task = multiplicities[:, 1] * 9 / (multiplicities[:, 0] * 2 + multiplicities[:, 1])
    reference = (first_task + 20) / 2
    np.testing.assert_equal(actual["ci95_episode_bootstrap"], np.quantile(reference, [.025, .975]))
    assert estimator.estimate(values) == EpisodeBootstrap(tasks, episodes, 1000, 9).estimate(values)


def test_bootstrap_does_not_silently_drop_task_without_valid_directions():
    estimator = EpisodeBootstrap([1, 1, 2], [10, 11, 20], repetitions=100, seed=3)
    result = estimator.estimate([1., 2., np.nan])
    assert np.isnan(result["mean"])
    assert result["eligible_tasks"] == 1
    assert result["ci95_episode_bootstrap"] is None
    assert result["finite_bootstrap_replicates"] == 0


def test_source_sum_ray_can_coexist_with_broad_per_step_noise():
    source = np.zeros((1, 3, 16, 7))
    source[0, :, :, 0] = np.arange(1, 4)[:, None]
    source[0, :, ::2, 1] = 10.
    source[0, :, 1::2, 1] = -10.
    metrics = source_shape_metrics(source, source)
    np.testing.assert_allclose(metrics["source_raw_sum16_major_variance_fraction"], 1.)
    np.testing.assert_allclose(metrics["source_raw_sum16_direction_concentration"], 1.)
    assert np.all(metrics["source_raw_per_step_cov_yy"] > 0.)
    np.testing.assert_allclose(metrics["source_raw_sum16_cov_yy"], 0.)


def test_source_to_head_diagnostics_exclude_inactive_fallback_draws():
    gt = np.zeros((1, 16, 7)); gt[:, :, 0] = 1.
    source = np.repeat(gt[:, None], 2, axis=1)
    source[:, 1] *= -1.
    archive = {"action_pred": source, "source_raw": source, "source_norm": source,
               "heading_direction": np.array([[1., 0.]]), "heading_confidence": np.array([.9]),
               "source_active": np.array([[True, False]])}
    metrics, _ = analyze_model(archive, gt, .5, .05)
    np.testing.assert_equal(metrics["source_head_angle_deg"], [[0., np.nan]])
    np.testing.assert_equal(metrics["source_fallback"], [[0., 1.]])
    np.testing.assert_equal(metrics["source_gt_angle_deg"], [[0., 180.]])
