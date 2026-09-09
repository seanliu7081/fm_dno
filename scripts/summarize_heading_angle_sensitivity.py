"""Report fixed-model source-angle sensitivity from saved paired evaluations.

All intervals condition on the saved training seeds and latent draws. Ground truth
headings use future actions and are an offline intervention, not deployable inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("predicted", "oracle", "oracle_15", "oracle_30", "oracle_60")
ANGLES = (0, 15, 30, 60)
LABELS = {"predicted": "Predicted heading", "oracle": "GT heading",
          "oracle_15": "GT ±15°", "oracle_30": "GT ±30°", "oracle_60": "GT ±60°"}
METRICS = ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "raw_prefix8_rotation_mse",
           "raw_prefix8_gripper_mse", "generated_heading_mae_deg")
TITLES = {"raw_prefix8_mse": "Prefix-8 action MSE", "raw_prefix8_xyz_mse": "Prefix-8 XYZ MSE",
          "raw_prefix8_rotation_mse": "Prefix-8 rotation MSE",
          "raw_prefix8_gripper_mse": "Prefix-8 gripper MSE",
          "generated_heading_mae_deg": "Generated heading MAE (°)"}
STRATA = ((0., 15., "0–15°"), (15., 30., "15–30°"),
          (30., 60., "30–60°"), (60., 180.00001, ">60°"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_at(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def metric_array(archive: dict, arm: str, key: str) -> np.ndarray:
    return archive[f"{arm}__{key}"]


def safe_mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def metric_mask(archive: dict, key: str) -> np.ndarray:
    return archive["valid"].astype(bool) if key == "generated_heading_mae_deg" else np.ones(len(archive["valid"]), bool)


def validate_archives(archives: dict[int, dict], seeds: list[int]) -> tuple[int, int]:
    reference = archives[seeds[0]]
    windows = len(reference["valid"])
    if not windows or len(set(seeds)) != len(seeds):
        raise ValueError("Require nonempty windows and unique training seeds")
    draws = metric_array(reference, ARMS[0], METRICS[0]).shape[0]
    if not draws:
        raise ValueError("Require at least one saved noise draw")
    for seed in seeds:
        data = archives[seed]
        for name in ("episode", "frame", "task", "valid"):
            if data[name].shape != (windows,) or not np.array_equal(data[name], reference[name]):
                raise ValueError(f"Window identity/validity differs across seeds: {seed}, {name}")
        if data["predicted_heading_error_deg"].shape != (windows,):
            raise ValueError("Predictor errors must be one value per held-out observation")
        valid = data["valid"].astype(bool)
        if not valid.any() or not np.isfinite(data["predicted_heading_error_deg"][valid]).all():
            raise ValueError("No finite eligible predictor-heading diagnostics")
        for arm in ARMS:
            for key in METRICS:
                values = metric_array(data, arm, key)
                if values.shape != (draws, windows) or not np.isfinite(values).all():
                    raise ValueError(f"Missing, nonfinite, or mismatched metric: {seed}, {arm}, {key}")
                if (values < -1e-7).any():
                    raise ValueError(f"Negative error metric: {seed}, {arm}, {key}")
                if arm != "predicted" and not np.allclose(values[:, ~valid], metric_array(data, "predicted", key)[:, ~valid], rtol=0., atol=1e-7):
                    raise ValueError("Invalid-label fallback differs from the predicted source")
        if data["source_active"].shape not in ((windows,), (draws, windows)):
            raise ValueError("Source activity must be per window or per draw/window")
    return draws, windows


def bootstrap_paired(archives: dict[int, dict], seeds: list[int], arm: str, key: str,
                     repetitions: int, random_seed: int, valid_only: bool = False) -> dict:
    """Average fixed draws/seeds, then resample paired episode clusters by task."""
    reference = archives[seeds[0]]
    mask = reference["valid"].astype(bool) if valid_only else metric_mask(reference, key)
    tasks, episodes = reference["task"][mask], reference["episode"][mask]
    if not len(tasks):
        raise ValueError("No observations eligible for paired comparison")
    delta = np.stack([(metric_array(archives[seed], arm, key) - metric_array(archives[seed], "predicted", key)).mean(0)[mask]
                      for seed in seeds])
    averaged = delta.mean(0)
    rng = np.random.default_rng(random_seed)
    bootstrap = np.zeros(repetitions)
    strata = []
    for task in np.unique(tasks):
        selected = tasks == task
        clusters, inverse = np.unique(episodes[selected], return_inverse=True)
        sums = np.bincount(inverse, weights=averaged[selected])
        counts = np.bincount(inverse).astype(float)
        choices = rng.integers(len(clusters), size=(repetitions, len(clusters)))
        weight = selected.mean()
        bootstrap += weight * sums[choices].sum(1) / counts[choices].sum(1)
        strata.append({"task": int(task), "episode_ids": clusters.tolist(),
                       "windows": int(selected.sum()), "fixed_weight": float(weight)})
    effects = delta.mean(1)
    return {"effect_treatment_minus_predicted": float(averaged.mean()),
            "ci95_episode_bootstrap": np.quantile(bootstrap, [.025, .975]).tolist(),
            "per_seed_effect": {str(seed): float(effects[i]) for i, seed in enumerate(seeds)},
            "sign_consistent_across_seeds": bool(np.all(effects >= 0) or np.all(effects <= 0)),
            "eligible_windows": int(mask.sum()), "eligible_episodes": len(np.unique(episodes)),
            "valid_only": valid_only or key == "generated_heading_mae_deg", "strata": strata}


def stratified_accuracy(archives: dict[int, dict], seeds: list[int], repetitions: int,
                        random_seed: int) -> list[dict]:
    """Strata are fixed independently by each checkpoint's predicted angle error.

    Shared task/episode bootstrap multiplicities preserve pairing between training
    seeds even though their error-bin membership differs. Within each seed/bin,
    task weights are fixed to the original eligible-window counts. Empty sampled
    cells yield unavailable bootstrap replicates; those are counted explicitly.
    """
    ref = archives[seeds[0]]
    tasks, episodes = ref["task"], ref["episode"]
    rng = np.random.default_rng(random_seed)
    multiplicities = {}
    for task in np.unique(tasks):
        clusters = np.unique(episodes[tasks == task])
        choices = rng.integers(len(clusters), size=(repetitions, len(clusters)))
        count = np.zeros((repetitions, len(clusters)), dtype=np.int32)
        np.add.at(count, (np.arange(repetitions)[:, None], choices), 1)
        multiplicities[int(task)] = (clusters, count)
    result = []
    for low, high, label in STRATA:
        per_seed, seed_bootstraps = [], []
        for seed in seeds:
            data = archives[seed]
            angle_error = data["predicted_heading_error_deg"]
            mask = data["valid"].astype(bool) & (angle_error >= low) & (angle_error < high)
            control = metric_array(data, "predicted", "raw_prefix8_mse").mean(0)
            treatment = metric_array(data, "oracle", "raw_prefix8_mse").mean(0)
            gain = control - treatment
            estimate = {"seed": seed, "windows": int(mask.sum()),
                        "episodes": len(np.unique(episodes[mask])),
                        "predicted_heading_mae_deg": safe_mean(angle_error[mask]),
                        "predicted_prefix8_mse": safe_mean(control[mask]),
                        "oracle_prefix8_mse": safe_mean(treatment[mask]),
                        "mse_reduction": safe_mean(gain[mask])}
            if mask.any():
                bootstrap = np.zeros(repetitions)
                for task in np.unique(tasks[mask]):
                    selected = mask & (tasks == task)
                    clusters, counts = multiplicities[int(task)]
                    sums = np.asarray([gain[selected & (episodes == ep)].sum() for ep in clusters])
                    sizes = np.asarray([(selected & (episodes == ep)).sum() for ep in clusters])
                    denominator = counts @ sizes
                    numerator = counts @ sums
                    means = np.divide(numerator, denominator, out=np.full(repetitions, np.nan), where=denominator != 0)
                    bootstrap += float(selected.sum() / mask.sum()) * means
                seed_bootstraps.append(bootstrap)
                finite = bootstrap[np.isfinite(bootstrap)]
                estimate["ci95_episode_bootstrap"] = np.quantile(finite, [.025, .975]).tolist() if len(finite) >= .9 * repetitions else None
                estimate["finite_bootstrap_replicates"] = len(finite)
            else:
                seed_bootstraps.append(np.full(repetitions, np.nan))
                estimate.update(ci95_episode_bootstrap=None, finite_bootstrap_replicates=0)
            per_seed.append(estimate)
        all_present = all(row["windows"] > 0 for row in per_seed)
        bootstrap_mean = np.mean(seed_bootstraps, axis=0)
        finite = bootstrap_mean[np.isfinite(bootstrap_mean)]
        result.append({"label": label, "lower_inclusive_deg": low, "upper_exclusive_deg": high,
                       "per_seed": per_seed,
                       "mean_mse_reduction": float(np.mean([r["mse_reduction"] for r in per_seed])) if all_present else None,
                       "ci95_episode_bootstrap": np.quantile(finite, [.025, .975]).tolist() if len(finite) >= .9 * repetitions else None,
                       "finite_bootstrap_replicates": len(finite), "interval_minimum_retained_fraction": .9,
                       "interpretation": "Positive reduction means the GT source angle lowers action MSE. Membership is fixed within each seed; seeds can contain different windows in the same bin."})
    return result


def observations(result: dict) -> list[str]:
    means, seeds = result["seed_means"], result["seeds"]
    control = means["predicted"]["raw_prefix8_mse"]
    oracle = means["oracle"]["raw_prefix8_mse"]
    pct = 100 * (control - oracle) / control
    effect = result["comparisons"]["oracle"]["raw_prefix8_mse"]
    details = "; ".join(f"seed {seed}: {effect['per_seed_effect'][str(seed)]:+.6g}" for seed in seeds)
    notes = [f"Replacing only the source direction with GT changes mean prefix-8 MSE from {control:.6g} to {oracle:.6g}: {pct:+.3f}% reduction. Paired differences (GT minus predicted): {details}."]
    if not effect["sign_consistent_across_seeds"]:
        notes.append("The GT replacement effect changes sign across the fixed training seeds; this does not support a consistent action improvement.")
    values = np.asarray([means[arm]["raw_prefix8_mse"] for arm in ARMS[1:]])
    monotonic = bool(np.all(np.diff(values) >= 0))
    notes.append("Mean action MSE " + ("rises monotonically" if monotonic else "does not rise monotonically")
                 + " across the controlled 0°, 15°, 30°, and 60° injected errors. These angles are in addition to the unchanged Von Mises dither.")
    notes.append("GT changes the source while the observation condition still contains the model prediction. This is sensitivity of an existing model to a source-only intervention; GT is not a guaranteed attainable upper bound, and a null/negative result does not prove heading precision is irrelevant.")
    return notes


def make_plots(result: dict, output_dir: Path, pdf: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = result["seeds"]
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    records = {(row["seed"], row["arm"]): row["mean_over_draws"] for row in result["per_seed"]}
    style = {"font.size": 10, "axes.spines.right": False, "axes.spines.top": False, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for axis, key in zip(axes.flat, METRICS):
            for index, seed in enumerate(seeds):
                color = colors[index % len(colors)]
                y = [records[(seed, arm)][key] for arm in ARMS[1:]]
                axis.plot(ANGLES, y, color=color, alpha=.58, marker="o", markersize=3, linewidth=1.1, label=f"GT ± error, seed {seed}")
                axis.axhline(records[(seed, "predicted")][key], color=color, alpha=.52, linestyle="--", linewidth=1., label=f"Predicted, seed {seed}")
            axis.plot(ANGLES, [result["seed_means"][arm][key] for arm in ARMS[1:]], color="#202020", linewidth=2.2,
                      marker="o", markersize=4, label="GT ± error, seed mean")
            axis.axhline(result["seed_means"]["predicted"][key], color="#202020", linestyle="--", linewidth=1.8, label="Predicted, seed mean")
            axis.set(title=TITLES[key], xlabel="Injected absolute error around GT (°)", xticks=ANGLES)
            axis.grid(alpha=.18)
        axes.flat[-1].axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False, fontsize=10)
        axes.flat[-1].text(.02, .25, f"{result['windows']} held-out windows × {result['noise_draws']} fixed draws\nCondition and κ=4 dither unchanged\nMSE: first 8 raw actions\nHeading: full 16-action resultant\nOffline diagnostic; no SR measurement", transform=axes.flat[-1].transAxes, va="top", linespacing=1.5)
        fig.suptitle("Fixed-model source-angle sensitivity", fontsize=15)
        fig.text(.5, .015, "Horizontal lines are the unmodified predicted-angle source. The x-axis applies only to GT interventions.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(output_dir / "angle_sensitivity.png", dpi=200)
        if pdf:
            fig.savefig(output_dir / "angle_sensitivity.pdf")
        plt.close(fig)
        fig, (axis, count_axis) = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={"width_ratios": [1.45, 1]})
        x = np.arange(len(STRATA))
        width = .65 / len(seeds)
        for index, seed in enumerate(seeds):
            positions = x + (index - (len(seeds) - 1) / 2) * width
            rows = [next(item for item in bucket["per_seed"] if item["seed"] == seed) for bucket in result["accuracy_strata"]]
            gains = np.asarray([row["mse_reduction"] if row["mse_reduction"] is not None else np.nan for row in rows])
            axis.plot(positions, gains, marker="o", color=colors[index % len(colors)], label=f"Seed {seed}")
            count_axis.bar(positions, [row["windows"] for row in rows], width=width, color=colors[index % len(colors)], alpha=.7, label=f"Seed {seed}")
        means = np.asarray([item["mean_mse_reduction"] if item["mean_mse_reduction"] is not None else np.nan for item in result["accuracy_strata"]])
        axis.plot(x, means, "s-", color="#202020", linewidth=2., label="Seed mean")
        for position, item in enumerate(result["accuracy_strata"]):
            interval = item["ci95_episode_bootstrap"]
            if interval is not None:
                axis.vlines(position, *interval, color="#202020", linewidth=1.6)
                axis.hlines(interval, position - .045, position + .045, color="#202020", linewidth=1.6)
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1.)
        axis.set(title="Action benefit from GT source direction", ylabel="Prefix-8 MSE reduction (predicted − GT)")
        count_axis.set(title="Eligible windows per seed", ylabel="Window count")
        for a in (axis, count_axis):
            a.set_xticks(x, [s[2] for s in STRATA])
            a.set_xlabel("Original predictor heading error (°)")
            a.grid(axis="y", alpha=.18)
        axis.legend(frameon=False)
        count_axis.legend(frameon=False)
        fig.suptitle("Does a less accurate heading leave more room for improvement?", fontsize=13)
        fig.text(.5, .025, "Positive: GT improves actions. Descriptive bins use seed-specific fixed masks; sparse-bin intervals are suppressed.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .94))
        fig.savefig(output_dir / "accuracy_strata.png", dpi=200)
        if pdf:
            fig.savefig(output_dir / "accuracy_strata.pdf")
        plt.close(fig)


def write_report(result: dict, path: Path) -> None:
    lines = ["# Fixed-model heading-angle sensitivity", "",
             f"Two mechanisms remain fixed: the model's predicted heading condition and its observation features. "
             f"Only the angle used to construct the translation-XY source is replaced. "
             f"Evaluation uses {result['windows']} held-out windows, {result['noise_draws']} paired latent draws, and training seeds {result['seeds']}.", "",
             "No retraining, best-draw selection, or success-rate evaluation is performed. The normalizer, crop, model weights, Gaussian samples, predicted-validity gate, κ=4 Von Mises dither, and 10-step Euler solver stay fixed. "
             "Injected errors are signed 15°, 30°, or 60° with the same fixed sign per draw/window across levels and seeds. GT labels use the 16-action XY resultant; invalid GT labels fall back to the predicted source.", "",
             "## Generated-action results", "", "Errors are averaged after scoring each generated action sample. Lower is better; action MSE uses raw first-eight-step actions, and generated heading uses the full 16-step resultant on valid target headings.", "",
             "| Seed | Source angle | Prefix-8 MSE | XYZ MSE | Rotation MSE | Gripper MSE | Generated heading MAE (°) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["per_seed"]:
        values = " | ".join(f"{row['mean_over_draws'][key]:.6g}" for key in METRICS)
        lines.append(f"| {row['seed']} | {LABELS[row['arm']]} | {values} |")
    for arm in ARMS:
        values = " | ".join(f"{result['seed_means'][arm][key]:.6g}" for key in METRICS)
        lines.append(f"| Mean | {LABELS[arm]} | {values} |")
    lines += ["", "## Interpretation", ""]
    for note in result["observations"]:
        lines += [note, ""]
    lines += ["## Paired effects versus predicted-angle source", "",
              "Negative effects mean lower error. Intervals resample whole held-out episodes within each task after averaging fixed latent draws and training seeds. All arm comparisons retain the same episode pairing and fixed task weights.", "",
              "| Source angle | Metric | Mean paired effect | 95% episode-bootstrap interval |",
              "| --- | --- | ---: | --- |"]
    for arm in ARMS[1:]:
        for key in METRICS:
            value = result["comparisons"][arm][key]
            low, high = value["ci95_episode_bootstrap"]
            lines.append(f"| {LABELS[arm]} | {TITLES[key]} | {value['effect_treatment_minus_predicted']:+.6g} | [{low:+.6g}, {high:+.6g}] |")
    lines += ["", f"There are {result['bootstrap']['repetitions']:,} bootstrap replicates. **Intervals condition on these fixed training seeds and noise samples. They do not estimate uncertainty over training runs or SR.** Valid-label-only action comparisons are also saved in `summary.json`.", "",
              "## Original heading accuracy and GT replacement", "",
              "Positive MSE reduction means the GT source improves actions. Bin membership uses each checkpoint's original predictor error and remains fixed for all interventions.", "",
              "| Error bin | Seed | Windows | Episodes | Mean predictor error (°) | Predicted MSE | GT MSE | MSE reduction |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for bucket in result["accuracy_strata"]:
        for row in bucket["per_seed"]:
            values = [row[key] for key in ("predicted_heading_mae_deg", "predicted_prefix8_mse", "oracle_prefix8_mse", "mse_reduction")]
            formatted = " | ".join("n/a" if value is None else f"{value:.6g}" for value in values)
            lines.append(f"| {bucket['label']} | {row['seed']} | {row['windows']} | {row['episodes']} | {formatted} |")
    lines += ["", "Stratum intervals use shared episode multiplicities across seeds with seed-specific fixed bin membership and task weights. Replicates with an empty sampled seed/task/bin are omitted and counted in the JSON. Intervals are suppressed when fewer than 90% of replicates remain; bin point estimates and counts are descriptive, and narrow bins can have limited episode coverage.", "",
              "## Invalid labels and source gate", ""]
    for row in result["eligibility"]:
        lines += [f"- Seed {row['seed']}: {row['valid_windows']}/{result['windows']} valid GT labels; {row['invalid_windows']} invalid-label fallbacks; source gate active in {row['active_fraction']:.3%} of draw/window pairs. Invalid-label intervention metrics match the predicted control within the validation tolerance. Mean original valid-target heading error: {row['predicted_heading_mae_deg']:.3f}°."]
    if result["source_diagnostics"]:
        lines += ["", "Source-angle errors (valid GT labels only) distinguish the supplied angle from the actual noise resultant after the unchanged dither and gate:", "",
                  "| Seed | Arm | Supplied angle MAE (°) | Realized noise heading MAE (°) |", "| --- | --- | ---: | ---: |"]
        for row in result["source_diagnostics"]:
            lines.append(f"| {row['seed']} | {LABELS[row['arm']]} | {row['source_angle_error_deg']:.6g} | {row['source_heading_error_deg']:.6g} |")
    lines += ["", "Future action labels appear only in this diagnostic source-angle override. The heading condition remains predicted, so a GT source can disagree with the direction supplied as condition. "
              "The checkpoint was trained on predicted-angle sources; this intervention changes its source distribution. A GT arm is therefore neither a deployable score nor a strict attainable performance upper bound. A negative result cannot alone establish that better heading prediction would be useless after matched training.", "",
              "## Plots", "", "![Angle sensitivity](angle_sensitivity.png)", "", "![Original-accuracy strata](accuracy_strata.png)", "",
              "All per-draw means, per-seed effects, conditional intervals, input hashes, and report-source hash are in `summary.json`. Evaluator protocol and provenance remain in `manifest.json`.", ""]
    path.write_text("\n".join(lines))


def summarize(run_dir: Path, repetitions: int = 2000, random_seed: int = 92026,
              pdf: bool = False) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    if not (run_dir / "evaluation_complete.json").is_file():
        raise ValueError("Evaluation is incomplete: evaluation_complete.json is required")
    complete = read_json(run_dir / "evaluation_complete.json")
    seeds = list(map(int, manifest["seeds"]))
    if not complete.get("complete") or complete["seeds"] != seeds or manifest["arms"] != list(ARMS):
        raise ValueError("Incomplete evaluation or mismatched declared seeds/arms")
    paths = {seed: run_dir / f"seed{seed}.npz" for seed in seeds}
    for seed, path in paths.items():
        checks = complete["checks"][str(seed)]
        if sha256(path) != checks["archive_sha256"] or not all(row["bit_exact"] for row in checks["predicted_control_replay"].values()):
            raise ValueError("Saved archive integrity or exact predicted-control replay failed")
    archives = {seed: arrays_at(path) for seed, path in paths.items()}
    draws, windows = validate_archives(archives, seeds)
    if int(manifest["draws"]) != draws or int(manifest["windows"]) != windows or complete["draws"] != draws or complete["windows"] != windows:
        raise ValueError("Manifest dimensions disagree with saved evaluations")
    per_seed, seed_means, eligibility, source_diagnostics = [], {}, [], []
    for seed in seeds:
        data = archives[seed]
        for arm in ARMS:
            per_draw = [{"draw": draw, **{key: float(metric_array(data, arm, key)[draw, metric_mask(data, key)].mean())
                                         for key in METRICS}} for draw in range(draws)]
            means = {key: float(np.mean([row[key] for row in per_draw])) for key in METRICS}
            per_seed.append({"seed": seed, "arm": arm, "mean_over_draws": means, "per_draw": per_draw})
            if all(f"{arm}__{key}" in data for key in ("source_angle_error_deg", "source_heading_error_deg")):
                source_diagnostics.append({"seed": seed, "arm": arm, **{key: float(metric_array(data, arm, key)[:, data["valid"].astype(bool)].mean()) for key in ("source_angle_error_deg", "source_heading_error_deg")}})
        valid = data["valid"].astype(bool)
        eligibility.append({"seed": seed, "valid_windows": int(valid.sum()), "invalid_windows": int((~valid).sum()),
                            "active_fraction": float(data["source_active"].mean()),
                            "predicted_heading_mae_deg": float(data["predicted_heading_error_deg"][valid].mean()),
                            "invalid_fallback_max_metric_difference": max(float(np.max(np.abs(metric_array(data, arm, key)[:, ~valid] - metric_array(data, "predicted", key)[:, ~valid]))) if (~valid).any() else 0.
                                                                          for arm in ARMS[1:] for key in METRICS)})
    for arm in ARMS:
        seed_means[arm] = {key: float(np.mean([row["mean_over_draws"][key] for row in per_seed if row["arm"] == arm])) for key in METRICS}
    comparisons = {arm: {key: bootstrap_paired(archives, seeds, arm, key, repetitions, random_seed + index)
                         for index, key in enumerate(METRICS)} for arm in ARMS[1:]}
    valid_comparisons = {arm: {key: bootstrap_paired(archives, seeds, arm, key, repetitions, random_seed + index, valid_only=True)
                               for index, key in enumerate(METRICS)} for arm in ARMS[1:]}
    result = {"seeds": seeds, "noise_draws": draws, "windows": windows, "arms": list(ARMS),
              "per_seed": per_seed, "seed_means": seed_means, "comparisons": comparisons,
              "valid_only_comparisons": valid_comparisons,
              "accuracy_strata": stratified_accuracy(archives, seeds, repetitions, random_seed + 100),
              "eligibility": eligibility, "source_diagnostics": source_diagnostics,
              "bootstrap": {"repetitions": repetitions, "random_seed": random_seed,
                            "unit": "whole validation episode, stratified by task",
                            "conditions_on": "fixed training seeds and saved noise draws"},
              "evaluation_completion": complete,
              "input_sha256": {str(path.name): sha256(path) for path in [run_dir / "manifest.json", run_dir / "evaluation_complete.json", *paths.values()]},
              "report_source_sha256": sha256(Path(__file__)),
              "limitations": ["No retraining or SR evaluation.", "Future GT action headings are diagnostic-only inputs.",
                              "Predicted condition remains fixed; GT source can change condition/source consistency.",
                              "GT is not a guaranteed attainable upper bound for the trained model.",
                              "Conditional episode intervals do not establish robustness over training seeds."]}
    result["observations"] = observations(result)
    source_copy = run_dir / "source/scripts" / Path(__file__).name
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    source_copy.write_bytes(Path(__file__).read_bytes())
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, run_dir / "report.md")
    make_plots(result, run_dir, pdf)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=92026)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("bootstrap draws must be positive")
    result = summarize(args.run_dir.resolve(), args.bootstrap_draws, args.bootstrap_seed, args.pdf)
    print(json.dumps({"run_dir": str(args.run_dir.resolve()), "seed_means": result["seed_means"],
                      "observations": result["observations"]}, indent=2))


if __name__ == "__main__":
    main()
