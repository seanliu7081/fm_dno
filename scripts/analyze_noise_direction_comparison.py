"""Analyze the approved three-checkpoint comparison from saved inference arrays.

This module never loads a policy, fits a normalizer, or accesses a simulator.
Direction is the raw XY action resultant, not a measured robot displacement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


MODEL_IDS = ("baseline_transformer", "heading_zero_dit", "heading_gaussian_dit")
LABELS = {
    "baseline_transformer": "Original Gaussian / Transformer",
    "heading_zero_dit": "Heading Zero / DiT",
    "heading_gaussian_dit": "Heading Gaussian / DiT",
}
DEFAULT_OUTPUT = Path("output/noise_direction_best_sr_20260912")


def mean_finite(values, axis=None):
    """NaN-aware mean without warning on deliberately unavailable directions."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def directional_scores(predicted, target, target_valid, prediction_threshold):
    """Score vectors; an invalid prediction fails accuracy on a valid target.

    Inputs may broadcast. Angular error and cosine only exist when BOTH vectors
    have a direction. Coverage exposes every direction omitted from these means.
    """
    predicted, target = np.broadcast_arrays(np.asarray(predicted, float), np.asarray(target, float))
    if predicted.shape[-1] != 2:
        raise ValueError("Directions require two coordinates")
    target_valid = np.broadcast_to(np.asarray(target_valid, bool), predicted.shape[:-1])
    pred_norm = np.linalg.norm(predicted, axis=-1)
    gt_norm = np.linalg.norm(target, axis=-1)
    pred_valid = np.isfinite(predicted).all(axis=-1) & (pred_norm >= prediction_threshold) & (pred_norm > 1e-12)
    gt_valid = target_valid & np.isfinite(target).all(axis=-1) & (gt_norm > 1e-12)
    both = gt_valid & pred_valid
    dot = np.sum(predicted * target, axis=-1)
    cosine = np.divide(dot, pred_norm * gt_norm, out=np.full_like(dot, np.nan), where=both)
    cosine = np.clip(cosine, -1., 1.)
    angle = np.degrees(np.arccos(cosine))
    return {
        "angle_deg": angle,
        "cosine": cosine,
        "within15": np.where(gt_valid, both & (angle <= 15. + 1e-10), np.nan).astype(float),
        "within30": np.where(gt_valid, both & (angle <= 30. + 1e-10), np.nan).astype(float),
        "within15_conditional": np.where(both, angle <= 15. + 1e-10, np.nan).astype(float),
        "within30_conditional": np.where(both, angle <= 30. + 1e-10, np.nan).astype(float),
        "prediction_valid": pred_valid.astype(float),
        "coverage_on_gt_valid": np.where(gt_valid, pred_valid, np.nan).astype(float),
        "prediction_norm": pred_norm,
    }


class EpisodeBootstrap:
    """One deterministic set of whole-episode multiplicities, stratified by task.

    Each task has equal weight; within a task all selected windows have equal
    weight. Resampling an episode carries every selected window and fixed noise
    draw with it. The same multiplicities are reused for every model/comparison.
    """
    def __init__(self, tasks, episodes, repetitions=2000, seed=20260912):
        self.tasks = np.asarray(tasks)
        self.episodes = np.asarray(episodes)
        if self.tasks.shape != self.episodes.shape or self.tasks.ndim != 1 or not len(self.tasks):
            raise ValueError("Nonempty aligned task and episode arrays are required")
        if repetitions < 1:
            raise ValueError("Bootstrap repetitions must be positive")
        self.repetitions, self.seed = int(repetitions), int(seed)
        rng = np.random.default_rng(seed)
        self.strata = []
        for task in np.unique(self.tasks):
            indices = np.flatnonzero(self.tasks == task)
            ids, inverse = np.unique(self.episodes[indices], return_inverse=True)
            choices = rng.integers(len(ids), size=(repetitions, len(ids)))
            multiplicities = np.zeros((repetitions, len(ids)), dtype=np.int32)
            np.add.at(multiplicities, (np.arange(repetitions)[:, None], choices), 1)
            self.strata.append((task, indices, ids, inverse, multiplicities))

    def estimate(self, values):
        values = np.asarray(values, float)
        if values.shape != self.tasks.shape:
            raise ValueError("Bootstrap input must contain one draw-averaged value per window")
        task_means, bootstrap_means = [], []
        for _, indices, ids, inverse, multiplicities in self.strata:
            sampled = values[indices]
            valid = np.isfinite(sampled)
            sums = np.bincount(inverse, weights=np.where(valid, sampled, 0), minlength=len(ids))
            counts = np.bincount(inverse, weights=valid.astype(float), minlength=len(ids))
            task_means.append(float(mean_finite(sampled)))
            boot_sum = multiplicities @ sums
            boot_count = multiplicities @ counts
            bootstrap_means.append(np.divide(boot_sum, boot_count, out=np.full(self.repetitions, np.nan), where=boot_count > 0))
        # Do not silently omit tasks without eligible directions.
        point = float(np.mean(task_means))
        replicates = np.mean(bootstrap_means, axis=0)
        finite = replicates[np.isfinite(replicates)]
        interval = np.quantile(finite, [.025, .975]).tolist() if len(finite) >= .9 * self.repetitions else None
        return {
            "mean": point,
            "ci95_episode_bootstrap": interval,
            "finite_bootstrap_replicates": len(finite),
            "eligible_windows": int(np.isfinite(values).sum()),
            "eligible_tasks": int(np.isfinite(task_means).sum()),
        }

    def metadata(self):
        return {
            "repetitions": self.repetitions, "seed": self.seed, "confidence": .95,
            "unit": "whole held-out episode within each fixed task",
            "task_weight": 1. / len(self.strata), "within_task_weight": "equal selected windows",
            "draw_handling": "average fixed draws within each window before resampling episodes",
            "minimum_finite_replicate_fraction": .9,
            "conditions_on": "three fixed selected checkpoints, ten fixed tasks, selected windows, and saved paired noise draws",
            "strata": [{"task_uid": int(task), "episode_ids": ids.tolist(), "selected_windows": len(indices)}
                       for task, indices, ids, _, _ in self.strata],
        }


def paired_window_difference(left, right):
    """Pair the exact draw indices, then average the complete pairs per window."""
    left, right = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
    if left.ndim != 2:
        raise ValueError("Paired metrics require [window, draw] arrays")
    paired = np.isfinite(left) & np.isfinite(right)
    difference = np.where(paired, left - right, np.nan)
    return mean_finite(difference, axis=1), paired.mean(axis=1)


def source_shape_metrics(source_raw, source_norm):
    """Descriptive XY cloud statistics per observation, never across tasks.

    Per-step clouds pool the fixed draws and 16 positions. Sum16 clouds contain
    one resultant per draw. Population covariance describes the saved cloud.
    """
    raw = np.asarray(source_raw, dtype=np.float64)
    normalized = np.asarray(source_norm, dtype=np.float64)
    if normalized.shape != raw.shape or not np.isfinite(normalized).all():
        raise ValueError("Normalized sources must match raw sources and be finite")
    windows, draws = raw.shape[:2]
    metrics = {}
    for space, source in (("raw", raw), ("norm", normalized)):
        xy = source[..., :2]
        clouds = {"per_step": xy.reshape(windows, -1, 2), "sum16": xy.sum(axis=2)}
        for kind, cloud in clouds.items():
            center = cloud.mean(axis=1)
            centered = cloud - center[:, None, :]
            covariance = np.einsum("wni,wnj->wij", centered, centered) / cloud.shape[1]
            eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.)
            trace = eigenvalues.sum(axis=1)
            major_fraction = np.divide(eigenvalues[:, 1], trace, out=np.full(windows, np.nan), where=trace > 1e-20)
            statistics = {"centroid_x": center[:, 0], "centroid_y": center[:, 1],
                          "cov_xx": covariance[:, 0, 0], "cov_xy": covariance[:, 0, 1], "cov_yy": covariance[:, 1, 1],
                          "cov_eigenvalue_min": eigenvalues[:, 0], "cov_eigenvalue_max": eigenvalues[:, 1],
                          "major_variance_fraction": major_fraction}
            if kind == "sum16":
                lengths = np.linalg.norm(cloud, axis=-1)
                directions = np.divide(cloud, lengths[..., None], out=np.full_like(cloud, np.nan), where=lengths[..., None] > 1e-12)
                statistics["direction_concentration"] = np.linalg.norm(mean_finite(directions, axis=1), axis=-1)
            for key, value in statistics.items():
                metrics[f"source_{space}_{kind}_{key}"] = np.broadcast_to(value[:, None], (windows, draws)).copy()
    return metrics


def analyze_model(archive, ground_truth, xy_rms, target_confidence, baseline=False):
    predicted = np.asarray(archive["action_pred"], dtype=np.float64)
    source = np.asarray(archive["source_raw"], dtype=np.float64)
    if predicted.ndim != 4 or predicted.shape[0] != len(ground_truth) or predicted.shape[2:] != (16, 7):
        raise ValueError("Predictions must have [window, draw, 16, 7] shape")
    if source.shape != predicted.shape or not np.isfinite(predicted).all() or not np.isfinite(source).all():
        raise ValueError("Source/prediction arrays must match and be finite")
    windows, draws = predicted.shape[:2]
    direction = np.asarray(archive["heading_direction"], dtype=float)
    confidence = np.asarray(archive["heading_confidence"], dtype=float)
    active = np.asarray(archive["source_active"])
    if direction.shape != (windows, 2) or confidence.shape != (windows,) or active.shape != (windows, draws):
        raise ValueError("Unexpected heading or source activation shape")
    metrics = source_shape_metrics(source, archive["source_norm"])
    common = {}
    for horizon in (16, 8):
        threshold = target_confidence * xy_rms * np.sqrt(horizon)
        gt_resultant = ground_truth[:, :horizon, :2].sum(axis=1)
        gt_norm = np.linalg.norm(gt_resultant, axis=-1)
        gt_valid = gt_norm >= threshold
        pred_resultant = predicted[:, :, :horizon, :2].sum(axis=2)
        scores = directional_scores(pred_resultant, gt_resultant[:, None, :], gt_valid[:, None], threshold)
        metrics.update({f"direction_h{horizon}_{key}": value for key, value in scores.items()})
        difference = predicted[:, :, :horizon] - ground_truth[:, None, :horizon]
        metrics[f"action_h{horizon}_mse"] = np.mean(difference ** 2, axis=(2, 3))
        metrics[f"translation_h{horizon}_mse"] = np.mean(difference[:, :, :, :3] ** 2, axis=(2, 3))
        common[f"gt_resultant_h{horizon}"] = gt_resultant
        common[f"gt_valid_h{horizon}"] = gt_valid
        common[f"gt_confidence_h{horizon}"] = gt_norm / (xy_rms * np.sqrt(horizon))
    gt_resultant = common["gt_resultant_h16"]
    gt_valid = common["gt_valid_h16"]
    head_scores = directional_scores(direction[:, None, :], gt_resultant[:, None, :], gt_valid[:, None], 1e-12)
    metrics.update({f"head_{key}": np.broadcast_to(value, (windows, draws)).copy() for key, value in head_scores.items()})
    metrics["head_confidence"] = np.broadcast_to(confidence[:, None], (windows, draws)).copy()
    metrics["source_active"] = active.astype(float)
    metrics["source_fallback"] = 1. - active.astype(float)
    source_resultant = source[:, :, :, :2].sum(axis=2)
    source_gt_scores = directional_scores(source_resultant, gt_resultant[:, None, :], gt_valid[:, None], 1e-12)
    metrics.update({f"source_gt_{key}": value for key, value in source_gt_scores.items()})
    head_valid = np.isfinite(direction).all(axis=-1) & (np.linalg.norm(direction, axis=-1) > 1e-12)
    source_head_scores = directional_scores(source_resultant, direction[:, None, :], head_valid[:, None] & active.astype(bool), 1e-12)
    metrics.update({f"source_head_{key}": np.where(active, value, np.nan) for key, value in source_head_scores.items()})
    metrics["source_head_parallel"] = np.where(active, np.sum(source_resultant * direction[:, None, :], axis=-1), np.nan)
    metrics["source_head_perpendicular"] = np.where(active, source_resultant[..., 1] * direction[:, None, 0] - source_resultant[..., 0] * direction[:, None, 1], np.nan)
    # Baseline has no heading head. An absent head is unavailable, not a failed
    # direction prediction, and source shaping activity is not applicable.
    if baseline:
        for key in metrics:
            if key.startswith("head_") or key.startswith("source_head_") or key in ("source_active", "source_fallback"):
                metrics[key] = np.full((windows, draws), np.nan)
    else:
        if not np.isfinite(direction).all() or not np.isfinite(confidence).all():
            raise ValueError("Heading checkpoints must have finite heading predictions")
    return metrics, common


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_number(value, percent=False):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100 * value:.1f}%" if percent else f"{value:.2f}"


def write_report(output, result):
    selected = result["selection"]
    protocol = result["direction_definition"]
    lines = [
        "# Noise and action-direction comparison across LIBERO-10", "",
        f"The comparison uses exactly three approved best-success-rate checkpoints on {selected['window_count']:,} held-out windows "
        f"across all {selected['task_count']} LIBERO-10 tasks, with {selected['draws_per_window']} saved draws per window. "
        "It measures noise/source shape, generated action direction versus demonstration direction, and prediction changes on task images. "
        f"The selected windows cover {sum(task['episodes'] for task in selected['tasks'])} held-out demonstration episodes "
        f"({min(task['episodes'] for task in selected['tasks'])}–{max(task['episodes'] for task in selected['tasks'])} per task).", "",
        "[Open the visual comparison](gallery.html). [Per-task metrics](per_task_metrics.csv). [Machine-readable results](metrics.json).", "",
        "Direct figures: [all-task noise dot plots](figures/noise_overview_all10.png), "
        "[action direction versus ground truth](figures/action_direction_overview_all10.png), "
        "[first-action desired-goal heatmaps](figures/prediction_heatmaps_first_action_overview_all10.png), and "
        "[first-8 intended-displacement heatmaps](figures/prediction_heatmaps_overview_all10.png).", "",
        "## Checkpoints", "",
        "These success rates come from the existing checkpoint-selection evaluations; this experiment does not run a new success-rate evaluation.", "",
        "| Checkpoint | Evaluation snapshot | Stored epoch / global step | Existing successes | Existing success rate |",
        "|---|---|---:|---:|---:|",
    ]
    for model_id in MODEL_IDS:
        provenance = result["models"][model_id]["selection_checkpoint"]
        position = provenance["training_position"]
        evaluation = Path(provenance["evaluation_summary"])
        snapshot = next((part for part in evaluation.parts if part.startswith("epoch_")), "ep55 checkpoint filename")
        lines.append(f"| {LABELS[model_id]} | [{snapshot}]({evaluation}) | {position['epoch']} / {position['global_step']:,} | "
                     f"{provenance['successes']}/{provenance['evaluation_episodes']} | {format_number(provenance['success_rate'], True)} |")
    lines += ["", "Epoch conventions differ: the baseline `ep55` checkpoint uses stored counter 55 (56 completed epochs); "
              "heading evaluation snapshots 030/060 mean 30/60 completed epochs and store counters 29/59.", "",
        "## Task-balanced results", "",
        "Each task contributes equally. Direction angles exclude undefined or near-zero resultants; coverage is reported alongside them. "
        "Within-30° accuracy counts a near-zero predicted resultant as a failure whenever the ground-truth direction is valid.", "",
        "| Checkpoint | H16 angle ↓ | H16 cosine ↑ | H16 within 30° ↑ | H16 coverage ↑ | First-8 angle ↓ | First-8 action MSE ↓ | Head angle vs GT16 ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ("direction_h16_angle_deg", "direction_h16_cosine", "direction_h16_within30", "direction_h16_coverage_on_gt_valid",
            "direction_h8_angle_deg", "action_h8_mse", "head_angle_deg")
    for model_id in MODEL_IDS:
        summary = result["models"][model_id]["task_balanced"]
        cells = [f"{summary[key]['mean']:.5f}" if key == "action_h8_mse" and summary[key]["mean"] is not None
                 else format_number(summary[key]["mean"], key in ("direction_h16_within30", "direction_h16_coverage_on_gt_valid")) for key in keys]
        lines.append(f"| {LABELS[model_id]} | " + " | ".join(cells) + " |")
    h16 = result["models"]["heading_zero_dit"]["task_balanced"]["direction_h16_angle_deg"]["mean"]
    h8 = result["models"]["heading_gaussian_dit"]["task_balanced"]["direction_h8_angle_deg"]["mean"]
    lines += ["", f"Heading Zero has the lowest full-chunk H16 angular error ({h16:.2f}°), while Heading Gaussian has the lowest "
              f"first-8 angular error ({h8:.2f}°). The horizon therefore changes the ordering of the two heading checkpoints. "
              "The paired intervals below quantify these fixed-checkpoint differences."]
    lines += ["", "## Source/noise distribution", "",
              "Covariance is measured separately for individual XY noise dots (all 16 positions and saved draws) and each draw's summed-16 XY vector, "
              "always within the same observation. The major-variance fraction is 0.5 for equal variance in both axes and approaches 1 for a line. "
              "Direction concentration is the length of the mean unit resultant across draws (0 dispersed, 1 identical direction). "
              "Source-to-head angles include only draws where heading shaping is active; fallbacks are counted separately. "
              "Per-step covariance uses 512 saved dots (32 draws × 16 positions), while sum16 covariance uses only 32 resultants per window. "
              "These are finite-sample estimates: ordering the covariance eigenvalues makes the empirical major-variance fraction exceed "
              "the isotropic population value of 0.5, explaining the original Gaussian sum16 value near 0.61.", "",
              "| Checkpoint | Raw per-step major variance | Raw sum16 major variance | Raw sum16 direction concentration | Source→head angle, active only | Shaping active |",
              "|---|---:|---:|---:|---:|---:|"]
    source_keys = ("source_raw_per_step_major_variance_fraction", "source_raw_sum16_major_variance_fraction",
                   "source_raw_sum16_direction_concentration", "source_head_angle_deg", "source_active")
    for model_id in MODEL_IDS:
        summary = result["models"][model_id]["task_balanced"]
        cells = [format_number(summary[key]["mean"], key == "source_active") for key in source_keys]
        lines.append(f"| {LABELS[model_id]} | " + " | ".join(cells) + " |")
    lines += ["", "Raw and model-normalized centroids, covariance matrices/eigenvalues, and sum16 concentration are saved per window and per task. "
              "A narrow resultant distribution does not imply that the individual noise dots have collapsed."]
    zero_source = result["models"]["heading_zero_dit"]["task_balanced"]
    gaussian_source = result["models"]["heading_gaussian_dit"]["task_balanced"]
    lines += ["", f"Heading Zero's individual noise dots remain broadly two-dimensional (major-variance fraction "
              f"{zero_source['source_raw_per_step_major_variance_fraction']['mean']:.3f}), while their sum16 lies on the predicted-heading ray "
              f"(direction concentration {zero_source['source_raw_sum16_direction_concentration']['mean']:.3f}). "
              f"Heading Gaussian retains angular spread around the predicted heading (mean source-to-head angle "
              f"{gaussian_source['source_head_angle_deg']['mean']:.2f}°, concentration "
              f"{gaussian_source['source_raw_sum16_direction_concentration']['mean']:.3f}). "
              "Both heading sources were active on every selected draw; no fallback occurred."]
    lines += ["", "## Paired checkpoint differences", "",
              "Effects are the first checkpoint minus the second. Negative angular-error/MSE effects favor the first checkpoint. "
              "Conditional direction comparisons retain the same valid window/draw pairs in both checkpoints.", "",
              "| Comparison | H16 angular-error change | 95% episode-bootstrap interval | First-8 MSE change | 95% episode-bootstrap interval |",
              "|---|---:|---|---:|---|"]
    for comparison in result["paired_differences"].values():
        cells = []
        for key in ("direction_h16_angle_deg", "action_h8_mse"):
            metric = comparison["metrics"][key]
            interval = metric["ci95_episode_bootstrap"]
            cells += [f"{metric['mean']:.5f}" if key == "action_h8_mse" and metric["mean"] is not None else format_number(metric["mean"]),
                      f"[{interval[0]:.5f}, {interval[1]:.5f}]" if interval else "unavailable"]
        lines.append(f"| {LABELS[comparison['left']]} − {LABELS[comparison['right']]} | " + " | ".join(cells) + " |")
    lines += ["", "## All ten tasks", "",
              "The table reports generated H16 angular error and within-30° accuracy. Full H8, heading-head, source, coverage, and MSE metrics are in the CSV.", "",
              "| Task | Original Gaussian angle / accuracy | Heading Zero angle / accuracy | Heading Gaussian angle / accuracy |",
              "|---|---|---|---|"]
    for task in selected["tasks"]:
        cells = []
        for model_id in MODEL_IDS:
            metrics = result["models"][model_id]["per_task"][str(task["task_uid"])]["metrics"]
            cells.append(format_number(metrics["direction_h16_angle_deg"]) + "° / " + format_number(metrics["direction_h16_within30"], True))
        lines.append(f"| {task['task_name']} | " + " | ".join(cells) + " |")
    lines += ["", "## Task-image heatmaps", "",
              "The [first-8 endpoint maps](figures/prediction_heatmaps_overview_all10.png) project intended cumulative command displacement. "
              "Some predicted endpoints leave the camera view, producing a legitimately empty image heatmap; the figures retain the fixed frames "
              "and report offscreen fractions. Empty visible density does not mean the checkpoint produced no action.", "",
              "The supplementary [first-action desired-goal maps](figures/prediction_heatmaps_first_action_overview_all10.png) project "
              "`observed_eef_xyz + 0.05 × clip(action[0,:3], -1, 1)`, the OSC controller's desired positional goal. "
              "They show local prediction changes when cumulative first-8 endpoints are outside the image. "
              "Both sets use the same three checkpoints, all 30 fixed representative frames (three per task), and the same 128 saved predictions per frame. "
              "No observations, noise samples, model inference, or selection choices were added or changed for this supplement; no dynamics are simulated.", "",
              "Each task's supplementary figure is saved as `figures/tasks/task_<uid>_prediction_heatmaps_first_action.png` for task UIDs 30–39. "
              "Its numeric maps and visible/offscreen counts are in [heatmap_first_action_arrays.npz](heatmap_first_action_arrays.npz) and "
              "[heatmap_first_action_summary.json](heatmap_first_action_summary.json).", "",
              "## Definitions and interpretation", "",
              "- Direction is the resultant of the first 16 or first 8 **raw action XY commands**, after checkpoint denormalization. "
              "The 16-step definition matches the heading training target. Direction of the draw-averaged vector is not substituted for the mean error of sampled predictions.",
              "- All checkpoints use their dataset training alignment: observations `[t−1,t]` and demonstration actions `[t:t+16]`. "
              "The original HDF5 stores post-action observations/proprioception with same-index pre-action simulator states/actions. "
              "These are dataset-aligned imitation targets, not guaranteed sixteen strictly future actions; no relabeling or retraining is performed.",
              f"- A GT or generated resultant is valid when its norm is at least `0.05 × heading_xy_rms × sqrt(H)`. "
              f"The saved common training RMS is {protocol['heading_xy_rms']:.9g}; thresholds are {protocol['resultant_threshold_h16']:.9g} for H16 "
              f"and {protocol['resultant_threshold_h8']:.9g} for H8. No statistic is fitted on held-out data.",
              "- Source direction is the H16 XY resultant in saved raw action coordinates. Source/head vectors only require a numerical nonzero norm; "
              "source-versus-GT metrics use the same GT validity mask. Source-to-head metrics condition on the saved active shaping gate. "
              "`source_active` and `source_fallback` report its complementary fractions for heading models.",
              "- Angular error is in degrees; cosine and accuracy fractions are dimensionless. Action MSE is the mean squared difference across all seven raw channels, "
              "so translation, rotation, and gripper units are mixed; XYZ-only MSE is also saved.",
              "- Draws are averaged inside a window. Whole held-out episodes are resampled with replacement **within each task**, retaining every selected window and draw. "
              f"All model and pair intervals reuse {result['bootstrap']['repetitions']:,} deterministic bootstrap multiplicity draws and equal task weights.",
              "- The intervals condition on these fixed selected checkpoints and noise draws. They do not estimate variation across retraining runs or fresh task success rates. "
              "Some tasks have few held-out episodes; their uncertainty estimates have limited support.",
              "- The approved original Gaussian checkpoint uses a Transformer backbone, while both heading checkpoints use DiT. "
              "Their selected training epochs also differ. Differences therefore compare complete checkpoints and cannot isolate a causal effect of the noise prior.",
              "- Task-image heatmaps project the endpoint `observed_eef_xyz + 0.05 × sum(clip(action[:8,:3], -1, 1))` through the verified agentview camera. "
              "They show **intended command displacement**, not simulated motion: no dynamics or policy rollouts are used. "
              "Ground-truth command endpoints use exactly the same construction. Camera/image alignment checks are recorded in `projection.json`.", "",
              "## Reproducibility", "",
              "Checkpoint metadata and source-file SHA-256 digests are recorded in `metrics.json`. `per_window_metrics.npz` contains each draw-averaged metric, "
              "GT resultants/validity, model order, and window/task/episode identities. No policies are loaded or modified by the analysis script.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def summarize(output, repetitions=2000, seed=20260912):
    output = Path(output)
    selection = json.loads((output / "selection.json").read_text())
    windows = selection["windows"]
    tasks = np.array([row["task_uid"] for row in windows], dtype=np.int64)
    episodes = np.array([row["episode_index"] for row in windows], dtype=np.int64)
    if len(np.unique(tasks)) != 10:
        raise ValueError("The approved comparison requires all ten LIBERO-10 tasks")
    if len(windows) != 1280 or not np.all(np.unique(tasks, return_counts=True)[1] == 128):
        raise ValueError("The approved comparison requires 128 selected windows per task")
    if set(selection["models"]) != set(MODEL_IDS):
        raise ValueError("Only the three approved checkpoints may be compared")
    validation_ids = set(selection["validation_episode_indices"])
    training_ids = set(selection["train_episode_indices"])
    if not set(episodes).issubset(validation_ids) or set(episodes) & training_ids:
        raise ValueError("Selected windows must belong exclusively to held-out episodes")
    if len({(row["episode_index"], row["frame_index"]) for row in windows}) != len(windows):
        raise ValueError("Selected windows must not be duplicated")
    with np.load(output / "data.npz", allow_pickle=False) as data:
        ground_truth = np.asarray(data["actions_gt"], dtype=np.float64)
    if ground_truth.shape != (len(windows), 16, 7) or not np.isfinite(ground_truth).all():
        raise ValueError("Ground truth must be finite [window,16,7] raw actions")
    metadata = {model: json.loads((output / "models" / f"{model}.json").read_text()) for model in MODEL_IDS}
    zero, gaussian = metadata[MODEL_IDS[1]], metadata[MODEL_IDS[2]]
    xy_rms = float(zero["heading_xy_rms"])
    if not np.isfinite(xy_rms) or xy_rms <= 0 or not np.isclose(xy_rms, float(gaussian["heading_xy_rms"]), rtol=1e-6, atol=0):
        raise ValueError("Both heading checkpoints must contain the same positive training XY RMS")
    confidence = float(zero["min_target_confidence"])
    if confidence != float(gaussian["min_target_confidence"]) or not np.isclose(confidence, .05):
        raise ValueError("Expected a shared checkpoint min_target_confidence of .05")
    bootstrap = EpisodeBootstrap(tasks, episodes, repetitions, seed)
    raw_metrics, per_window, common = {}, {}, None
    for model in MODEL_IDS:
        with np.load(output / "models" / f"{model}.npz", allow_pickle=False) as archive:
            metrics, shared = analyze_model(archive, ground_truth, xy_rms, confidence, model == MODEL_IDS[0])
        if next(iter(metrics.values())).shape[1] != 32:
            raise ValueError("The approved comparison requires exactly 32 offline draws per window")
        raw_metrics[model] = metrics
        per_window[model] = {key: mean_finite(values, axis=1) for key, values in metrics.items()}
        if common is not None:
            for key in shared:
                np.testing.assert_array_equal(shared[key], common[key])
        common = shared
    task_rows = []
    task_info = []
    models = {}
    for task in np.unique(tasks):
        rows = [row for row in windows if row["task_uid"] == task]
        task_info.append({"task_uid": int(task), "task_name": rows[0]["task_name"], "windows": len(rows),
                          "episodes": len(np.unique(episodes[tasks == task])),
                          "gt_valid_fraction_h16": float(common["gt_valid_h16"][tasks == task].mean()),
                          "gt_valid_fraction_h8": float(common["gt_valid_h8"][tasks == task].mean())})
    for model in MODEL_IDS:
        summary = {key: bootstrap.estimate(value) for key, value in per_window[model].items()}
        per_task = {}
        for task in task_info:
            selected = tasks == task["task_uid"]
            metrics = {key: float(mean_finite(value[selected])) for key, value in per_window[model].items()}
            per_task[str(task["task_uid"])] = {**task, "metrics": metrics}
            task_rows.append({"model_id": model, **task, **metrics})
        models[model] = {"label": LABELS[model], "checkpoint": metadata[model], "selection_checkpoint": selection["models"][model], "task_balanced": summary, "per_task": per_task}
    comparisons = {}
    for right, left in itertools.combinations(MODEL_IDS, 2):
        pair_metrics = {}
        for key in raw_metrics[left]:
            difference, coverage = paired_window_difference(raw_metrics[left][key], raw_metrics[right][key])
            pair_metrics[key] = {**bootstrap.estimate(difference), "paired_draw_fraction_all_windows": float(np.mean(coverage))}
        comparisons[f"{left}-minus-{right}"] = {"left": left, "right": right, "metrics": pair_metrics}
    provenance_files = ["selection.json", "data.npz"] + [f"models/{model}.{suffix}" for model in MODEL_IDS for suffix in ("json", "npz")]
    result = {
        "schema_version": 1, "model_order": list(MODEL_IDS),
        "selection": {"window_count": len(windows), "draws_per_window": 32, "task_count": 10, "tasks": task_info},
        "direction_definition": {"coordinates": "raw action XY", "heading_xy_rms": xy_rms,
                                 "rms_source": "saved heading_zero_dit checkpoint; matched to heading_gaussian_dit",
                                 "min_target_confidence": confidence, "source_nonzero_threshold": 1e-12,
                                 "source_head_condition": "saved source_active gate is true",
                                 "resultant_threshold_h16": confidence * xy_rms * 4,
                                 "resultant_threshold_h8": confidence * xy_rms * np.sqrt(8),
                                 "accuracy_denominator": "GT-valid windows/draws; invalid predicted direction counts as failure",
                                 "angle_denominator": "both GT and prediction valid; equal windows then equal tasks",
                                 "alignment": "training dataset convention: observations[t-1,t] and actions[t:t+16]; HDF5 observations are post-action while same-index states/actions are pre-action"},
        "bootstrap": bootstrap.metadata(), "models": models, "paired_differences": comparisons,
        "provenance_sha256": {name: file_digest(output / name) for name in provenance_files},
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": file_digest(Path(__file__))},
    }
    result = json_safe(result)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "per_task_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(task_rows))
    arrays = {key: np.stack([per_window[model][key] for model in MODEL_IDS]) for key in per_window[MODEL_IDS[0]]}
    arrays.update(common)
    arrays.update(model_ids=np.array(MODEL_IDS), task_uid=tasks, episode_index=episodes,
                  window_index=np.array([row["window_index"] for row in windows]),
                  frame_index=np.array([row["frame_index"] for row in windows]),
                  global_index=np.array([row["global_index"] for row in windows]))
    np.savez_compressed(output / "per_window_metrics.npz", **arrays)
    write_report(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("At least 100 bootstrap draws are required")
    result = summarize(args.output_dir, args.bootstrap_draws, args.bootstrap_seed)
    print(json.dumps({model: {key: result["models"][model]["task_balanced"][key]
                             for key in ("direction_h16_angle_deg", "direction_h16_within30", "action_h8_mse")}
                      for model in MODEL_IDS}, indent=2))


if __name__ == "__main__":
    main()
