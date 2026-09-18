"""Compare three fixed checkpoints only on demonstrations held out by all three.

Analysis uses saved predictions, never model loading, normalization fitting, or
simulator rollouts. SMQ's four local headings are not collapsed into a global head.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_noise_direction_comparison import (
        EpisodeBootstrap, directional_scores, file_digest, format_number,
        json_safe, mean_finite, source_shape_metrics,
    )
except ModuleNotFoundError:  # Support ``python scripts/analyze_...py``.
    from analyze_noise_direction_comparison import (
        EpisodeBootstrap, directional_scores, file_digest, format_number,
        json_safe, mean_finite, source_shape_metrics,
    )

MODEL_IDS = ("heading_zero_dit", "heading_gaussian_dit", "smq_dit")
LABELS = {"heading_zero_dit": "Heading Zero / DiT", "heading_gaussian_dit": "Heading Gaussian / DiT", "smq_dit": "SMQ / DiT"}
DEFAULT_OUTPUT = Path("output/shared_holdout_heading_smq_20260914")
DEFAULT_AUDIT = Path("output/smq_intake_20260914/episode_split_audit.json")
COMMON_XY_RMS = 0.32195183634757996
TARGET_CONFIDENCE = .05
EXPECTED_TASK_WINDOWS = {31: 77, 35: 64, 36: 30, 38: 26, 39: 42}
MISSING_TASKS = [30, 32, 33, 34, 37]


def segment_resultants(actions, segment_count):
    """Sum raw XY commands separately inside contiguous equal-size segments."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape[-2:] != (16, 7) or segment_count not in (1, 4):
        raise ValueError("Expected 16x7 action chunks and either one or four segments")
    return actions[..., :2].reshape(*actions.shape[:-2], segment_count, 16 // segment_count, 2).sum(axis=-2)


def paired_window_difference(left, right):
    """Pair identical windows, draws, and (if present) local segment indices.

    Segment direction eligibility can differ between models. Pair before averaging
    segments, then average fixed draws, so both arms use the same valid targets.
    """
    left, right = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
    if left.ndim not in (2, 3):
        raise ValueError("Paired scores require [window,draw] or [window,draw,segment]")
    paired = np.isfinite(left) & np.isfinite(right)
    difference = np.where(paired, left - right, np.nan)
    if left.ndim == 3:
        difference = mean_finite(difference, axis=2)
    return mean_finite(difference, axis=1), paired.mean(axis=tuple(range(1, paired.ndim)))


def summarize_draws(values):
    """Equal eligible local segments within each draw; then equal eligible draws."""
    values = np.asarray(values)
    if values.ndim == 3:
        values = mean_finite(values, axis=2)
    if values.ndim != 2:
        raise ValueError("Metrics must have window/draw dimensions")
    return mean_finite(values, axis=1)


def analyze_model(archive, ground_truth, model_id, xy_rms=COMMON_XY_RMS, target_confidence=TARGET_CONFIDENCE):
    """Return common-comparable scores, own-head diagnostics, and shared GT.

    The first result contains draw-level common scores and descriptive own-head
    scores. Own-head scores retain a segment axis: K=1/H16 for heading policies,
    K=4/H4 for SMQ. Their prefix prevents treating them as equivalent head targets.
    """
    if model_id not in MODEL_IDS:
        raise ValueError("Unknown checkpoint")
    predicted = np.asarray(archive["action_pred"], dtype=np.float64)
    source = np.asarray(archive["source_raw"], dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if predicted.ndim != 4 or predicted.shape[0] != len(gt) or predicted.shape[2:] != (16, 7):
        raise ValueError("Predictions must have [window, draw, 16, 7] shape")
    if gt.shape != (len(gt), 16, 7) or not np.isfinite(gt).all():
        raise ValueError("Ground truth must be finite raw actions [window,16,7]")
    if source.shape != predicted.shape or not np.isfinite(predicted).all() or not np.isfinite(source).all():
        raise ValueError("Sources and predictions must match and be finite")
    windows, draws = predicted.shape[:2]
    segments = 4 if model_id == "smq_dit" else 1
    direction = np.asarray(archive["heading_direction"], dtype=float)
    confidence = np.asarray(archive["heading_confidence"], dtype=float)
    active = np.asarray(archive["source_active"])
    if direction.shape != (windows, segments, 2) or confidence.shape != (windows, segments) or active.shape != (windows, draws, segments):
        raise ValueError(f"Expected {segments} saved heading segments for {model_id}")
    if not np.isfinite(direction).all() or not np.isfinite(confidence).all() or not np.isin(active, [0, 1]).all():
        raise ValueError("Head predictions must be finite and activation gates boolean")
    active = active.astype(bool)
    metrics = source_shape_metrics(source, archive["source_norm"])
    common = {}
    for horizon in (16, 8):
        threshold = target_confidence * xy_rms * np.sqrt(horizon)
        gt_resultant = gt[:, :horizon, :2].sum(axis=1)
        gt_norm = np.linalg.norm(gt_resultant, axis=-1)
        gt_valid = (gt_norm >= threshold) & (gt_norm > 1e-12)
        pred_resultant = predicted[:, :, :horizon, :2].sum(axis=2)
        scores = directional_scores(pred_resultant, gt_resultant[:, None], gt_valid[:, None], threshold)
        metrics.update({f"direction_h{horizon}_{key}": value for key, value in scores.items()})
        difference = predicted[:, :, :horizon] - gt[:, None, :horizon]
        metrics[f"action_h{horizon}_mse"] = np.mean(difference ** 2, axis=(2, 3))
        metrics[f"translation_h{horizon}_mse"] = np.mean(difference[..., :3] ** 2, axis=(2, 3))
        common[f"gt_resultant_h{horizon}"] = gt_resultant
        common[f"gt_valid_h{horizon}"] = gt_valid
        common[f"gt_confidence_h{horizon}"] = gt_norm / (xy_rms * np.sqrt(horizon))
    # Same four contiguous segments and same validity threshold for ALL policies.
    gt_segments = segment_resultants(gt, 4)
    segment_threshold = target_confidence * xy_rms * 2
    gt_segment_norm = np.linalg.norm(gt_segments, axis=-1)
    gt_segment_valid = (gt_segment_norm >= segment_threshold) & (gt_segment_norm > 1e-12)
    segment_scores = directional_scores(segment_resultants(predicted, 4), gt_segments[:, None], gt_segment_valid[:, None], segment_threshold)
    metrics.update({f"direction_segment4_{key}": value for key, value in segment_scores.items()})
    common.update(gt_resultant_segment4=gt_segments, gt_valid_segment4=gt_segment_valid,
                  gt_confidence_segment4=gt_segment_norm / (xy_rms * 2))
    source_scores = directional_scores(source[..., :2].sum(axis=2), common["gt_resultant_h16"][:, None], common["gt_valid_h16"][:, None], 1e-12)
    metrics.update({f"source_gt_{key}": value for key, value in source_scores.items()})
    source_segment_scores = directional_scores(segment_resultants(source, 4), gt_segments[:, None], gt_segment_valid[:, None], 1e-12)
    metrics.update({f"source_gt_segment4_{key}": value for key, value in source_segment_scores.items()})
    # A heading target is exactly its own prediction interval, never another head's.
    own_gt = segment_resultants(gt, segments)
    own_threshold = target_confidence * xy_rms * np.sqrt(16 // segments)
    own_gt_valid = np.linalg.norm(own_gt, axis=-1) >= own_threshold
    head_scores = directional_scores(direction[:, None], own_gt[:, None], own_gt_valid[:, None], 1e-12)
    metrics.update({f"own_segment_head_{key}": np.broadcast_to(value, (windows, draws, segments)).copy() for key, value in head_scores.items()})
    metrics["own_segment_head_confidence"] = np.broadcast_to(confidence[:, None], (windows, draws, segments)).copy()
    metrics["own_segment_source_active"] = active.astype(float)
    metrics["own_segment_source_fallback"] = (~active).astype(float)
    own_source = segment_resultants(source, segments)
    head_norm = np.linalg.norm(direction, axis=-1)
    head_valid = head_norm > 1e-12
    head_unit = np.divide(direction, head_norm[..., None], out=np.zeros_like(direction), where=head_valid[..., None])
    head_source_scores = directional_scores(own_source, direction[:, None], head_valid[:, None] & active, 1e-12)
    metrics.update({f"own_segment_source_head_{key}": np.where(active, value, np.nan) for key, value in head_source_scores.items()})
    metrics["own_segment_source_head_parallel"] = np.where(active, (own_source * head_unit[:, None]).sum(axis=-1), np.nan)
    metrics["own_segment_source_head_perpendicular"] = np.where(active, own_source[..., 1] * head_unit[:, None, :, 0] - own_source[..., 0] * head_unit[:, None, :, 1], np.nan)
    return metrics, common


def validate_selection(selection, audit, original_selection=None):
    """Refuse any added, duplicated, reordered, or non-shared-held-out window."""
    windows = selection["windows"]
    tasks = np.asarray([row["task_uid"] for row in windows], dtype=np.int64)
    episodes = np.asarray([row["episode_index"] for row in windows], dtype=np.int64)
    if set(selection["models"]) != set(MODEL_IDS):
        raise ValueError("Only Heading Zero, Heading Gaussian, and SMQ are in scope")
    counts = dict(zip(*np.unique(tasks, return_counts=True)))
    if counts != EXPECTED_TASK_WINDOWS or len(set(episodes)) != 8:
        raise ValueError("Expected exactly 239 existing windows across 8 shared episodes and 5 tasks")
    if [row["window_index"] for row in windows] != list(range(len(windows))):
        raise ValueError("Shared windows must be reindexed consecutively")
    original = [row["original_window_index"] for row in windows]
    if original != audit["reusable_saved_window_indices"]:
        raise ValueError("Window subset must exactly equal the split audit in original order")
    if len({(row["episode_index"], row["frame_index"]) for row in windows}) != len(windows):
        raise ValueError("Duplicated windows are not allowed")
    mapping = {row["server_episode_index"]: row for row in audit["episode_mapping"]}
    for row in windows:
        if original_selection is not None:
            original = original_selection["windows"][row["original_window_index"]]
            for key in ("task_uid", "episode_index", "frame_index"):
                if row[key] != original[key]:
                    raise ValueError(f"Selected {key} differs from the original saved window")
        match = mapping[row["episode_index"]]
        if match["heading_split"] != "validation" or match["smq_split"] != "validation":
            raise ValueError("A selected episode was seen in training")
        if row["smq_episode_index"] != match["smq_episode_index"] or row["task_uid"] != match["task_uid"]:
            raise ValueError("SMQ mapping or task mismatch")
        if not 1 <= row["frame_index"] <= match["length"] - 16:
            raise ValueError("An observation/action window extends beyond its episode")
    return tasks, episodes


def write_report(output, result):
    selected, models = result["selection"], result["models"]
    lines = ["# Shared held-out comparison: Heading Zero, Heading Gaussian, and SMQ", "",
        "This offline comparison uses exactly **239 existing windows from 8 demonstrations across 5 LIBERO-10 tasks**, "
        "held out from training by all three checkpoints. Each window has 32 paired saved noise draws. "
        "The five tasks contribute equal weight; windows contribute equally within each task. "
        "The original comparison contained 1,280 windows; its other 1,041 windows are excluded because SMQ used their demonstrations in training.", "",
        "[Visual gallery](gallery.html) · [Per-task metrics](per_task_metrics.csv) · [Machine-readable results](metrics.json)", "",
        "Figures: [source clouds](figures/noise_overview_shared5.png), [action direction](figures/action_direction_overview_shared5.png), "
        "[paired differences](figures/paired_model_comparison.png), [angular-error distributions](figures/action_angle_distributions.png), "
        "[first-action image heatmaps](figures/prediction_heatmaps_first_action_overview_shared5.png), "
        "[first-8 image heatmaps](figures/prediction_heatmaps_first8_overview_shared5.png).", "",
        "## Exact held-out scope", "", "| Task UID | Task | Windows | Demonstrations |", "|---|---|---:|---:|"]
    for task in selected["tasks"]:
        lines.append(f"| {task['task_uid']} | {task['task_name']} | {task['windows']} | {task['episodes']} |")
    lines += ["", "Tasks 30, 32, 33, 34, and 37 have no demonstrations held out by all three and are excluded. "
        "All 500 demonstrations were matched uniquely by action and three proprioception hashes. RGB images were not hashed; "
        "inference here supplies exactly the same server-side RGB observations to each checkpoint. "
        "The split audit and input-file SHA-256 digests are recorded in `metrics.json`.", "",
        "## Fixed checkpoints", "", "| Model | Stored epoch / global step | Weights | Checkpoint |", "|---|---:|---|---|"]
    for model in MODEL_IDS:
        meta = models[model]["checkpoint"]
        sel = models[model]["selection_checkpoint"]
        pos = sel.get("training_position", meta.get("training_position", {}))
        lines.append(f"| {LABELS[model]} | {pos.get('epoch', 'see metadata')} / {pos.get('global_step', 'see metadata')} | "
                     f"{meta.get('weights', sel.get('weights', 'see metadata'))} | `{meta.get('checkpoint', sel.get('checkpoint', 'see metadata'))}` |")
    lines += ["", "These are selected trained checkpoints, not matched retraining runs. Their training epochs, architecture details, "
        "spatial observation encoders, and numerical precision differ; complete configurations and inference metadata are retained. "
        "All three use DiT-based policies, but SMQ additionally uses its spatial/motion representation and four local headings. "
        "This comparison cannot isolate a causal effect of the source distribution. **No retraining or new simulator success-rate evaluation is performed.**", "",
        "## Common action metrics", "",
        "Angles omit undefined predicted directions and report coverage. Within-30° accuracy treats invalid predictions as failures on valid GT targets. "
        "H16 and H8 sum raw action XY commands over the first 16 and first 8 steps. The common four-segment metric scores "
        "steps 0–3, 4–7, 8–11, and 12–15 separately for every model, then averages eligible segments per draw and draws per window.", "",
        "| Model | H16 angle ↓ | H16 within 30° ↑ | H16 coverage ↑ | H8 angle ↓ | Four-segment angle ↓ | H8 action MSE ↓ |", "|---|---:|---:|---:|---:|---:|---:|"]
    keys = ("direction_h16_angle_deg", "direction_h16_within30", "direction_h16_coverage_on_gt_valid", "direction_h8_angle_deg", "direction_segment4_angle_deg", "action_h8_mse")
    for model in MODEL_IDS:
        summary = models[model]["task_balanced"]
        cells = [f"{summary[key]['mean']:.5f}" if key.endswith("_mse") and summary[key]["mean"] is not None else format_number(summary[key]["mean"], key.endswith(("within30", "coverage_on_gt_valid"))) for key in keys]
        lines.append(f"| {LABELS[model]} | " + " | ".join(cells) + " |")
    lines += ["", "## Common source-distribution metrics", "",
        "All three sources are compared in raw action coordinates after their own saved checkpoint denormalization. "
        "Per-step clouds contain 512 dots (32 draws × 16 positions); resultant clouds contain 32 H16 sums. "
        "Covariances are population covariances of these finite saved clouds within each observation. The major-variance fraction ranges "
        "from 0.5 (equal eigenvalues) to 1 (a line); finite-sample isotropic clouds usually exceed 0.5. "
        "Direction concentration is the norm of the mean unit resultant (0 dispersed, 1 aligned). Normalized-space diagnostics "
        "are also saved but use each model's own normalization and should not be read as a shared physical scale.", "",
        "| Model | Raw per-step major variance | Raw H16 major variance | Raw H16 direction concentration | Source→GT16 angle |", "|---|---:|---:|---:|---:|"]
    for model in MODEL_IDS:
        summary = models[model]["task_balanced"]
        keys = ("source_raw_per_step_major_variance_fraction", "source_raw_sum16_major_variance_fraction", "source_raw_sum16_direction_concentration", "source_gt_angle_deg")
        lines.append(f"| {LABELS[model]} | " + " | ".join(format_number(summary[key]["mean"]) for key in keys) + " |")
    lines += ["", "## Own-head diagnostics (different targets; descriptive only)", "",
        "Heading Zero and Heading Gaussian predict one direction for the full H16 chunk. SMQ predicts four directions, each targeting "
        "its corresponding four-step segment. SMQ heads are scored against those local GT sums and local source sums. "
        "No global SMQ head is synthesized. The scalar SMQ diagnostics below average eligible local segments per observation; "
        "they are not equivalent to the heading models' H16 head score. No cross-model paired effect is computed for these unlike head targets. "
        "Source→head scores condition on the saved activation gate for that same local segment, and active/fallback rates retain all segments.", "",
        "| Model | Head targets | Own-target head angle | Own-target head coverage | Active source→own head angle | Active segments |", "|---|---|---:|---:|---:|---:|"]
    for model in MODEL_IDS:
        summary = models[model]["task_balanced"]
        keys = ("own_segment_head_angle_deg", "own_segment_head_coverage_on_gt_valid", "own_segment_source_head_angle_deg", "own_segment_source_active")
        cells = [format_number(summary[key]["mean"], key.endswith(("coverage_on_gt_valid", "source_active"))) for key in keys]
        lines.append(f"| {LABELS[model]} | {'4 × H4' if model == 'smq_dit' else '1 × H16'} | " + " | ".join(cells) + " |")
    lines += ["", "Detailed local-segment values are saved in `per_window_metrics.npz` under `<model>__<metric>__per_segment`, "
        "with dimensions `[window, segment]` after averaging draws. Per-segment task summaries are in `metrics.json`.", "",
        "## Paired fixed-checkpoint differences", "",
        "Effects are the first checkpoint minus the second. Negative angle/MSE differences favor the first. "
        "Exactly matching draw and segment indices are paired before averaging; angle differences retain only pairs valid for both models.", "",
        "| Comparison | H16 angle change | 95% episode interval | H8 MSE change | 95% episode interval |", "|---|---:|---|---:|---|"]
    for comparison in result["paired_differences"].values():
        cells = []
        for key in ("direction_h16_angle_deg", "action_h8_mse"):
            metric = comparison["metrics"][key]
            interval = metric["ci95_episode_bootstrap"]
            cells += [f"{metric['mean']:.5f}" if metric["mean"] is not None else "—", f"[{interval[0]:.5f}, {interval[1]:.5f}]" if interval else "unavailable"]
        lines.append(f"| {LABELS[comparison['left']]} − {LABELS[comparison['right']]} | " + " | ".join(cells) + " |")
    lines += ["", f"All summaries and comparisons reuse {result['bootstrap']['repetitions']:,} deterministic whole-episode bootstrap samples within each task. "
        "Sampling an episode carries all its selected windows and saved draws. The five fixed tasks remain equally weighted. "
        "**Only 8 episodes support these intervals, and tasks 35, 38, and 39 each have just one episode.** "
        "Those single-episode strata cannot contribute estimated between-episode variability, so intervals are descriptive and have limited uncertainty support. "
        "They condition on fixed checkpoints, tasks, windows, and noise draws, not retraining variability or full LIBERO-10 performance.", "",
        "## Definitions and reproducibility", "",
        f"- The common training XY RMS is `{COMMON_XY_RMS}` from the existing heading checkpoint; GT and action prediction validity "
        f"use `0.05 × RMS × sqrt(H)`: H16 `{TARGET_CONFIDENCE * COMMON_XY_RMS * 4:.10g}`, "
        f"H8 `{TARGET_CONFIDENCE * COMMON_XY_RMS * np.sqrt(8):.10g}`, and H4 `{TARGET_CONFIDENCE * COMMON_XY_RMS * 2:.10g}`. "
        "No scale is fitted on the shared held-out subset. Source and head vectors only require numerical nonzero norm (1e−12).",
        "- Direction is an action-command resultant, not measured robot displacement. H16/H8 action MSE averages all seven raw channels; "
        "XYZ-only translation MSE is also reported. Mixed-channel MSE combines translation, rotation, and gripper units.",
        "- Dataset alignment is preserved: observations `[t−1,t]` and actions `[t:t+16]`. The original HDF5 observations are post-action, "
        "while same-index simulator states/actions are pre-action. These are dataset-aligned imitation targets; no relabeling occurs.",
        "- Image heatmaps use 15 representative windows (3 per retained task), all selected within these 239 windows. "
        "They show projected intended action-command endpoints, not simulated trajectories or achieved goals. "
        "First-action endpoints use `eef_xyz + 0.05 × clip(action[0,:3],−1,1)`; first-8 endpoints sum the first eight clipped XYZ commands. "
        "Offscreen counts and projection metadata accompany the visual artifacts.",
        "- `selection.json` preserves original window indices and both datasets' episode indices. `metrics.json` records input hashes, "
        "checkpoint metadata, paired bootstrap settings, and per-task/per-segment summaries. `per_window_metrics.npz` preserves window identities "
        "and shared ground-truth resultants/validity. The analysis script only reads saved inference arrays.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def summarize(output, repetitions=2000, seed=20260914, split_audit=DEFAULT_AUDIT):
    output, split_audit = Path(output), Path(split_audit)
    selection = json.loads((output / "selection.json").read_text())
    audit = json.loads(split_audit.read_text())
    original_provenance = audit["inputs"]["comparison_selection"]
    original_path = Path(original_provenance["path"])
    if file_digest(original_path) != original_provenance["sha256"]:
        raise ValueError("The original window selection changed after the split audit")
    original_selection = json.loads(original_path.read_text())
    tasks, episodes = validate_selection(selection, audit, original_selection)
    windows = selection["windows"]
    with np.load(output / "data.npz", allow_pickle=False) as data:
        ground_truth = np.asarray(data["actions_gt"], dtype=float)
    metadata = {model: json.loads((output / "models" / f"{model}.json").read_text()) for model in MODEL_IDS}
    for model in MODEL_IDS[:2]:
        rms = float(metadata[model]["heading_xy_rms"])
        if not np.isclose(rms, COMMON_XY_RMS, rtol=1e-6, atol=0):
            raise ValueError("The saved heading training RMS differs from the common protocol")
    bootstrap = EpisodeBootstrap(tasks, episodes, repetitions, seed)
    raw_metrics, per_window, common = {}, {}, None
    for model in MODEL_IDS:
        with np.load(output / "models" / f"{model}.npz", allow_pickle=False) as archive:
            metrics, shared = analyze_model(archive, ground_truth, model)
        if next(iter(metrics.values())).shape[1] != 32:
            raise ValueError("Expected exactly 32 saved draws per window")
        raw_metrics[model] = metrics
        per_window[model] = {key: summarize_draws(value) for key, value in metrics.items()}
        if common is not None:
            for key in shared:
                np.testing.assert_array_equal(shared[key], common[key])
        common = shared
    task_rows, task_info, models = [], [], {}
    for task in np.unique(tasks):
        mask = tasks == task
        row = windows[int(np.flatnonzero(mask)[0])]
        task_info.append({"task_uid": int(task), "task_name": row.get("task_name", f"Task {task}"), "windows": int(mask.sum()),
                          "episodes": len(np.unique(episodes[mask])), "episode_indices": np.unique(episodes[mask]).tolist(),
                          "gt_valid_fraction_h16": float(common["gt_valid_h16"][mask].mean()),
                          "gt_valid_fraction_h8": float(common["gt_valid_h8"][mask].mean()),
                          "gt_valid_fraction_segment4": float(common["gt_valid_segment4"][mask].mean())})
    for model in MODEL_IDS:
        summary = {key: bootstrap.estimate(value) for key, value in per_window[model].items()}
        per_task, per_segment = {}, {}
        for task in task_info:
            mask = tasks == task["task_uid"]
            metrics = {key: float(mean_finite(value[mask])) for key, value in per_window[model].items()}
            per_task[str(task["task_uid"])] = {**task, "metrics": metrics}
            task_rows.append({"model_id": model, **{k: v for k, v in task.items() if k != "episode_indices"}, **metrics})
        for key, values in raw_metrics[model].items():
            if values.ndim == 3:
                averaged_draws = mean_finite(values, axis=1)
                per_segment[key] = {"task_balanced": [bootstrap.estimate(averaged_draws[:, k]) for k in range(values.shape[2])],
                    "per_task": {str(task["task_uid"]): mean_finite(averaged_draws[tasks == task["task_uid"]], axis=0) for task in task_info}}
        models[model] = {"label": LABELS[model], "checkpoint": metadata[model], "selection_checkpoint": selection["models"][model],
                         "own_heading_segment_count": 4 if model == "smq_dit" else 1,
                         "own_heading_horizon": 4 if model == "smq_dit" else 16,
                         "task_balanced": summary, "per_task": per_task, "per_segment": per_segment}
    comparisons = {}
    for right, left in itertools.combinations(MODEL_IDS, 2):
        pair_metrics = {}
        for key in raw_metrics[left]:
            # Own-head intervals exist within a model, but unlike targets are not contrasted.
            if key.startswith("own_segment_"):
                continue
            difference, coverage = paired_window_difference(raw_metrics[left][key], raw_metrics[right][key])
            pair_metrics[key] = {**bootstrap.estimate(difference), "paired_draw_fraction_all_windows": float(coverage.mean()),
                                 "pairing_unit": "same window/draw/segment" if raw_metrics[left][key].ndim == 3 else "same window/draw"}
        comparisons[f"{left}-minus-{right}"] = {"left": left, "right": right, "metrics": pair_metrics}
    bootstrap_meta = bootstrap.metadata()
    bootstrap_meta.update(conditions_on="three fixed selected checkpoints, five fixed shared tasks, 239 selected windows, and saved paired noise draws",
                          episode_count=8, single_episode_task_uids=[35, 38, 39],
                          limitations="Only eight episodes; three tasks have a single episode and contribute no estimated between-episode variance")
    provenance_files = ["selection.json", "data.npz"] + [f"models/{model}.{suffix}" for model in MODEL_IDS for suffix in ("json", "npz")]
    result = {"schema_version": 1, "model_order": list(MODEL_IDS),
        "selection": {"window_count": len(windows), "draws_per_window": 32, "task_count": 5, "episode_count": 8,
                      "tasks": task_info, "missing_task_uids": MISSING_TASKS, "original_window_count": 1280,
                      "excluded_original_windows_in_smq_training": 1041, "representative_count": len(selection.get("representatives", []))},
        "direction_definition": {"coordinates": "raw action XY", "heading_xy_rms": COMMON_XY_RMS,
            "rms_source": "saved heading_zero_dit training RMS, matched to heading_gaussian_dit, fixed for all three checkpoints",
            "min_target_confidence": TARGET_CONFIDENCE, "resultant_threshold_h16": TARGET_CONFIDENCE * COMMON_XY_RMS * 4,
            "resultant_threshold_h8": TARGET_CONFIDENCE * COMMON_XY_RMS * np.sqrt(8), "resultant_threshold_h4": TARGET_CONFIDENCE * COMMON_XY_RMS * 2,
            "source_nonzero_threshold": 1e-12, "own_head_targets": {"heading_zero_dit": "1xH16", "heading_gaussian_dit": "1xH16", "smq_dit": "4xH4"},
            "source_head_condition": "saved source_active gate for the corresponding local segment",
            "accuracy_denominator": "GT-valid segments/windows; invalid predicted directions count as failure",
            "angle_denominator": "both GT and prediction valid; equal eligible segments then draws/windows and equal five tasks",
            "segment_aggregation": "common 4xH4 for action/source; own 1xH16 or4xH4 for descriptive head diagnostics; pair segments before averaging",
            "alignment": "observations[t-1,t], actions[t:t+16], unchanged dataset convention"},
        "bootstrap": bootstrap_meta, "models": models, "paired_differences": comparisons,
        "split_audit": {"path": str(split_audit.resolve()), "sha256": file_digest(split_audit), "original_selection": original_provenance, "rgb_hashes_checked": audit["rgb_hashes_checked"],
                        "matched_episodes": audit["matched_episodes"], "matching_fields": audit["matching_fields"]},
        "provenance_sha256": {name: file_digest(output / name) for name in provenance_files},
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": file_digest(Path(__file__))}}
    result = json_safe(result)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "per_task_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(task_rows))
    arrays = {key: np.stack([per_window[model][key] for model in MODEL_IDS]) for key in per_window[MODEL_IDS[0]]}
    arrays.update(common)
    for model in MODEL_IDS:
        for key, value in raw_metrics[model].items():
            if value.ndim == 3:
                arrays[f"{model}__{key}__per_segment"] = mean_finite(value, axis=1)
    arrays.update(model_ids=np.array(MODEL_IDS), task_uid=tasks, episode_index=episodes,
                  window_index=np.array([row["window_index"] for row in windows]),
                  original_window_index=np.array([row["original_window_index"] for row in windows]),
                  smq_episode_index=np.array([row["smq_episode_index"] for row in windows]),
                  frame_index=np.array([row["frame_index"] for row in windows]))
    np.savez_compressed(output / "per_window_metrics.npz", **arrays)
    write_report(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("At least 100 bootstrap draws are required")
    result = summarize(args.output_dir, args.bootstrap_draws, args.bootstrap_seed, args.split_audit)
    print(json.dumps({model: {key: result["models"][model]["task_balanced"][key]
                      for key in ("direction_h16_angle_deg", "direction_h8_angle_deg", "direction_segment4_angle_deg", "action_h8_mse")}
                      for model in MODEL_IDS}, indent=2))


if __name__ == "__main__":
    main()
