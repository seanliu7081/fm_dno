"""Scientific invariants for the shared-held-out, segmented-head comparison."""
import numpy as np
import pytest

from scripts.analyze_shared_holdout_comparison import (
    COMMON_XY_RMS, TARGET_CONFIDENCE, EpisodeBootstrap, analyze_model,
    directional_scores, paired_window_difference, segment_resultants, summarize_draws,
)


def make_archive(gt, segments=4, draws=2):
    pred = np.broadcast_to(gt[:, None], (len(gt), draws, 16, 7)).copy()
    head = segment_resultants(gt, segments)
    norm = np.linalg.norm(head, axis=-1, keepdims=True)
    head = np.divide(head, norm, out=np.zeros_like(head), where=norm > 0)
    return {"action_pred": pred, "source_raw": pred.copy(), "source_norm": pred.copy(),
            "heading_direction": head, "heading_confidence": np.ones((len(gt), segments)),
            "source_active": np.ones((len(gt), draws, segments), dtype=bool)}


def turning_ground_truth():
    gt = np.zeros((1, 16, 7))
    for segment, (x, y) in enumerate([(1., 0.), (0., 1.), (-1., 0.), (0., -1.)]):
        gt[0, 4 * segment:4 * (segment + 1), :2] = [x, y]
    return gt


def test_four_local_smq_heads_are_valid_when_global_direction_cancels():
    gt = turning_ground_truth()
    scores, shared = analyze_model(make_archive(gt), gt, "smq_dit")
    assert not shared["gt_valid_h16"].item()
    assert shared["gt_valid_segment4"].all()
    assert np.isnan(scores["direction_h16_angle_deg"]).all()
    np.testing.assert_allclose(scores["own_segment_head_angle_deg"], 0.)
    np.testing.assert_allclose(scores["direction_segment4_angle_deg"], 0.)
    assert scores["own_segment_head_angle_deg"].shape == (1, 2, 4)
    assert not any(key.startswith("head_") for key in scores)


def test_smq_head_and_source_gate_apply_to_corresponding_segment_only():
    gt = turning_ground_truth()
    archive = make_archive(gt)
    archive["heading_direction"][0, 1] *= -1
    archive["source_raw"][0, 1, 8:12] *= -1
    archive["source_active"][0, 1, 2] = False
    scores, _ = analyze_model(archive, gt, "smq_dit")
    np.testing.assert_allclose(scores["own_segment_head_angle_deg"][0, 0], [0, 180, 0, 0])
    np.testing.assert_allclose(scores["own_segment_source_head_angle_deg"][0, 0], [0, 180, 0, 0])
    assert np.isnan(scores["own_segment_source_head_angle_deg"][0, 1, 2])
    assert scores["source_gt_segment4_angle_deg"][0, 1, 2] == pytest.approx(180)
    assert scores["own_segment_source_active"][0, 1, 2] == 0
    assert scores["own_segment_source_fallback"][0, 1, 2] == 1


def test_same_action_threshold_and_four_segment_metric_for_all_three_models():
    gt = np.zeros((1, 16, 7)); gt[..., 0] = 1
    smq = make_archive(gt, 4)
    heading = make_archive(gt, 1)
    threshold = TARGET_CONFIDENCE * COMMON_XY_RMS * 2
    for archive in (smq, heading):
        archive["action_pred"][:, 0, :4, 0] = (threshold * .5) / 4
    scores_smq, _ = analyze_model(smq, gt, "smq_dit")
    for model in ("heading_zero_dit", "heading_gaussian_dit"):
        scores_heading, _ = analyze_model(heading, gt, model)
        for key in scores_smq:
            if not key.startswith("own_segment_"):
                np.testing.assert_equal(scores_smq[key], scores_heading[key])
    assert np.isnan(scores_smq["direction_segment4_angle_deg"][0, 0, 0])
    assert scores_smq["direction_segment4_within30"][0, 0, 0] == 0
    assert scores_smq["direction_segment4_coverage_on_gt_valid"][0, 0, 0] == 0
    assert summarize_draws(scores_smq["direction_segment4_within30"]).item() == pytest.approx(.875)


def test_pair_segment_validity_before_averaging_to_avoid_mismatched_targets():
    left = np.array([[[2., np.nan, 100., np.nan], [6., 10., np.nan, np.nan]]])
    right = np.array([[[1., 1000., np.nan, np.nan], [2., 2., 90., np.nan]]])
    paired, coverage = paired_window_difference(left, right)
    # Draw 0 uses only segment 0: +1; draw 1 uses segments 0,1: mean(+4,+8)=6.
    np.testing.assert_allclose(paired, [3.5])
    np.testing.assert_allclose(coverage, [3 / 8])
    assert not np.isclose((summarize_draws(left) - summarize_draws(right)).item(), 3.5)


def test_whole_episode_bootstrap_reuses_paired_multiplicities_and_single_episode_strata():
    tasks = [31, 31, 31, 35]
    episodes = [10, 10, 20, 30]
    left = np.array([[0., 2.], [2., 4.], [8., 10.], [20., 22.]])
    right = np.ones((4, 2))
    difference, _ = paired_window_difference(left, right)
    estimator = EpisodeBootstrap(tasks, episodes, repetitions=2000, seed=42)
    result = estimator.estimate(difference)
    assert result["mean"] == pytest.approx(((0 + 2 + 8) / 3 + 20) / 2)
    mult = estimator.strata[0][-1]
    task31 = (mult[:, 0] * 2 + mult[:, 1] * 8) / (mult[:, 0] * 2 + mult[:, 1])
    np.testing.assert_allclose(result["ci95_episode_bootstrap"], np.quantile((task31 + 20) / 2, [.025, .975]))
    assert np.all(estimator.strata[1][-1] == 1)
    # The same episode draws apply in both model arms and their paired effects.
    np.testing.assert_equal(estimator.estimate(difference), EpisodeBootstrap(tasks, episodes, 2000, 42).estimate(difference))


def test_direction_wrap_and_zero_prediction_coverage_are_not_silent_successes():
    angles = np.deg2rad([-179., 0.])
    pred = np.column_stack((np.cos(angles), np.sin(angles)))
    pred[1] = 0
    target_angle = np.deg2rad(179.)
    target = np.array([np.cos(target_angle), np.sin(target_angle)])
    scores = directional_scores(pred, target, [True, True], .05)
    assert scores["angle_deg"][0] == pytest.approx(2.)
    assert np.isnan(scores["angle_deg"][1])
    np.testing.assert_equal(scores["within30"], [1., 0.])
    np.testing.assert_equal(scores["coverage_on_gt_valid"], [1., 0.])


def test_smq_rejects_global_heading_substitution():
    gt = turning_ground_truth()
    with pytest.raises(ValueError, match="4 saved heading segments"):
        analyze_model(make_archive(gt, 1), gt, "smq_dit")


def test_shared_selection_rejects_training_leakage_and_changed_original_frame():
    from copy import deepcopy
    from scripts.analyze_shared_holdout_comparison import EXPECTED_TASK_WINDOWS, MODEL_IDS, validate_selection

    task_episodes = {31: [31, 32, 33], 35: [350], 36: [360, 361], 38: [380], 39: [390]}
    windows, mapping = [], []
    for task, count in EXPECTED_TASK_WINDOWS.items():
        ids = task_episodes[task]
        for episode in ids:
            mapping.append({"server_episode_index": episode, "smq_episode_index": episode + 500,
                            "task_uid": task, "length": 1000, "heading_split": "validation", "smq_split": "validation"})
        for frame in range(count):
            index = len(windows)
            episode = ids[frame % len(ids)]
            windows.append({"window_index": index, "original_window_index": index, "episode_index": episode,
                            "smq_episode_index": episode + 500, "task_uid": task, "frame_index": frame + 1})
    selection = {"models": dict.fromkeys(MODEL_IDS), "windows": windows}
    audit = {"reusable_saved_window_indices": list(range(239)), "episode_mapping": mapping}
    original = deepcopy(selection)
    tasks, episodes = validate_selection(selection, audit, original)
    assert len(tasks) == 239 and len(set(episodes)) == 8
    changed = deepcopy(selection)
    changed["windows"][0]["frame_index"] += 1
    with pytest.raises(ValueError, match="differs from the original saved window"):
        validate_selection(changed, audit, original)
    leaked = deepcopy(audit)
    leaked["episode_mapping"][0]["smq_split"] = "training"
    with pytest.raises(ValueError, match="seen in training"):
        validate_selection(selection, leaked, original)
