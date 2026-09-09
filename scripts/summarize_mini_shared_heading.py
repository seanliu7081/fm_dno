"""Summarize a completed shared-heading mini-test without loading policy weights.

Example: python scripts/summarize_mini_shared_heading.py --run-dir OUTPUT --pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MODES = ("baseline", "auxiliary", "condition")
LABELS = {
    "baseline": "Baseline + detached probe",
    "auxiliary": "Auxiliary supervision",
    "condition": "Auxiliary + heading condition",
}
# key -> saved array, identity prefix, whether the target-validity mask applies
METRICS = {
    "flow_mse": ("flow_mse", "", False),
    "heading_mae_deg": ("heading_mae_deg", "", True),
    "raw_prefix8_mse": ("sample_raw_prefix8_mse", "sample_", False),
    "generated_heading_mae_deg": ("sample_generated_heading_mae_deg", "sample_", True),
}
TITLES = {
    "flow_mse": "Flow-matching MSE",
    "heading_mae_deg": "Predictor heading MAE (degrees)",
    "raw_prefix8_mse": "Generated action prefix-8 MSE (raw units)",
    "generated_heading_mae_deg": "Generated chunk heading MAE (degrees)",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def generated_rows(rows: np.ndarray, per_task: int) -> np.ndarray:
    """Reproduce the runner's fixed sampling subset, retaining frame identities."""
    rng = np.random.default_rng(20260910)
    indices = []
    for task in np.unique(rows[:, 2]):
        episodes = np.unique(rows[rows[:, 2] == task, 0])
        for j, episode in enumerate(episodes):
            count = per_task // len(episodes) + (j < per_task % len(episodes))
            available = np.flatnonzero((rows[:, 2] == task) & (rows[:, 0] == episode))
            indices.extend(rng.choice(available, count, replace=False).tolist())
    return rows[np.asarray(indices, dtype=np.int64)]


def validate_rows(rows: np.ndarray, split: dict) -> None:
    if rows.ndim != 2 or rows.shape[1] != 3 or not len(rows):
        raise ValueError("Validation rows must contain episode, absolute frame, and task.")
    if len(np.unique(rows[:, :2], axis=0)) != len(rows):
        raise ValueError("Duplicate validation sample identities.")
    ends = np.asarray(split["episode_ends"], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    episodes, frames, tasks = rows.T
    if np.any(episodes < 0) or np.any(episodes >= len(ends)):
        raise ValueError("Invalid validation episode ID.")
    if np.any(frames < starts[episodes]) or np.any(frames >= ends[episodes]):
        raise ValueError("Validation frames cross an episode boundary.")
    if not np.array_equal(tasks, np.asarray(split["episode_tasks"])[episodes]):
        raise ValueError("Validation task identities differ from the episode mapping.")
    if not set(episodes).issubset(split["validation_episodes"]):
        raise ValueError("Validation rows contain episodes outside the held-out split.")
    if set(split["train_episodes"]) & set(split["validation_episodes"]):
        raise ValueError("Training and validation episode sets overlap.")


def load_archive(path: Path, rows: np.ndarray, sampled: np.ndarray) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for prefix, identities in (("", rows), ("sample_", sampled)):
        for name, column in (("episode", 0), ("task", 2)):
            if not np.array_equal(arrays[prefix + name], identities[:, column]):
                raise ValueError(f"Identity/order mismatch: {path}, {prefix + name}")
        mask = arrays[prefix + "valid"]
        if mask.shape != (len(identities),) or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Invalid target-validity mask: {path}, {prefix}")
    for key, (field, prefix, angular) in METRICS.items():
        values = arrays[field]
        count = len(sampled) if prefix else len(rows)
        if values.shape != (count,):
            raise ValueError(f"Invalid metric shape: {path}, {key}")
        eligible = arrays[prefix + "valid"].astype(bool) if angular else np.ones(count, bool)
        if not eligible.any() or not np.isfinite(values[eligible]).all():
            raise ValueError(f"No finite eligible observations: {path}, {key}")
    return arrays


def metric_mean(archive: dict, key: str) -> float:
    field, prefix, angular = METRICS[key]
    values = archive[field]
    mask = archive[prefix + "valid"].astype(bool) if angular else np.ones(len(values), bool)
    return float(values[mask].mean())


def heading_null_controls(manifest: dict, split: dict, validation_rows: np.ndarray,
                          expected_valid: np.ndarray) -> dict:
    """Fit heading-bias controls on saved training windows, using no images."""
    import zarr

    dataset = Path(manifest["args"]["dataset"])
    if not dataset.is_absolute():
        dataset = Path(__file__).resolve().parents[1] / dataset
    root = zarr.open(str(dataset), mode="r")
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    ends = np.asarray(split["episode_ends"], dtype=np.int64)
    if not np.array_equal(root["meta"]["episode_ends"][:], ends):
        raise ValueError("Dataset episode boundaries differ from the saved split.")
    train_rows = np.asarray(split["train_windows"], dtype=np.int64)
    train_split = {**split, "validation_episodes": split["train_episodes"],
                   "train_episodes": split["validation_episodes"]}
    validate_rows(train_rows, train_split)
    horizon, threshold = 16, .05
    rms = np.float32(split["heading_xy_rms"])

    def targets(rows):
        indices = np.minimum(rows[:, 1, None] + np.arange(horizon), ends[rows[:, 0], None] - 1)
        resultant = actions[indices, :2].sum(axis=1, dtype=np.float32)
        norms = np.linalg.norm(resultant, axis=1)
        direction = resultant / np.maximum(norms[:, None], np.float32(1e-8))
        valid = norms / (rms * np.float32(np.sqrt(horizon))) >= threshold
        return direction, valid

    train_direction, train_valid = targets(train_rows)
    validation_direction, validation_valid = targets(validation_rows)
    if not np.array_equal(validation_valid, expected_valid.astype(bool)):
        raise ValueError("Reconstructed heading-validity labels differ from saved evaluation labels.")
    if not train_valid.any():
        raise ValueError("No valid training headings for the circular-mean controls.")

    def circular_mean(selected):
        mean = train_direction[selected].astype(np.float64).mean(axis=0)
        concentration = float(np.linalg.norm(mean))
        # An undefined circular mean has no preferred direction; fix +X in advance.
        return (mean / concentration if concentration > 1e-12 else np.array([1., 0.])), concentration

    global_direction, global_concentration = circular_mean(train_valid)
    task_directions, task_fits = {}, {}
    for task in np.unique(validation_rows[:, 2]):
        selected = train_valid & (train_rows[:, 2] == task)
        if not selected.any():
            raise ValueError(f"Task {task} has no valid training headings for its circular mean.")
        direction, concentration = circular_mean(selected)
        task_directions[int(task)] = direction
        task_fits[str(task)] = {"training_valid_windows": int(selected.sum()),
                               "direction": direction.tolist(), "resultant_length": concentration}

    def evaluate_predictions(prediction):
        cross = prediction[:, 0] * validation_direction[:, 1] - prediction[:, 1] * validation_direction[:, 0]
        dot = (prediction * validation_direction).sum(axis=1)
        error = np.abs(np.arctan2(cross, dot)) * (180. / np.pi)
        return {"heading_mae_deg": float(error[validation_valid].mean()),
                "heading_cosine": float(dot[validation_valid].mean()),
                "valid_windows": int(validation_valid.sum()),
                "per_task": {str(task): {
                    "heading_mae_deg": float(error[validation_valid & (validation_rows[:, 2] == task)].mean()),
                    "heading_cosine": float(dot[validation_valid & (validation_rows[:, 2] == task)].mean()),
                } for task in np.unique(validation_rows[validation_valid, 2])}}

    return {
        "fit_scope": "valid headings from saved training windows only; no validation fitting",
        "training_windows": len(train_rows), "training_valid_windows": int(train_valid.sum()),
        "target": {"raw_xy_horizon": horizon, "min_target_confidence": threshold,
                   "fixed_train_episode_xy_rms": float(rms), "pad_actions_at_episode_end": True},
        "validation_validity_matches_archives": True,
        "global_circular_mean": {
            "direction": global_direction.tolist(), "resultant_length": global_concentration,
            **evaluate_predictions(np.broadcast_to(global_direction, validation_direction.shape)),
        },
        "per_task_circular_mean": {
            "task_fits": task_fits,
            **evaluate_predictions(np.stack([task_directions[int(task)] for task in validation_rows[:, 2]])),
        },
        "interpretation": "Controls quantify global and task-specific heading bias. Circular means minimize cosine loss, not angular MAE. Beating them supports sample-dependent readout beyond those biases, but does not isolate visual information from robot state.",
    }


def paired_bootstrap(archives: dict, seeds: list[int], treatment: str, control: str,
                     key: str, draws: int, rng: np.random.Generator) -> dict:
    """Resample episode sums/counts within tasks, with fixed original task weights.

    The same episode multiplicities apply to both arms and every fixed training
    seed. Averaging paired differences across seeds before resampling is equivalent
    to averaging the per-seed paired effects after these common cluster draws.
    """
    field, prefix, angular = METRICS[key]
    reference = archives[(seeds[0], control)]
    episodes, tasks = reference[prefix + "episode"], reference[prefix + "task"]
    mask = reference[prefix + "valid"].astype(bool) if angular else np.ones(len(episodes), bool)
    differences = np.stack([
        archives[(seed, treatment)][field] - archives[(seed, control)][field]
        for seed in seeds
    ])[:, mask]
    averaged = differences.mean(axis=0)
    episodes, tasks = episodes[mask], tasks[mask]
    samples = np.zeros(draws, dtype=np.float64)
    strata = []
    for task in np.unique(tasks):
        selected = tasks == task
        cluster_ids, inverse = np.unique(episodes[selected], return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        sums = np.bincount(inverse, weights=averaged[selected])
        weight = float(selected.sum() / len(tasks))
        choices = rng.integers(0, len(cluster_ids), size=(draws, len(cluster_ids)))
        samples += weight * (sums[choices].sum(axis=1) / counts[choices].sum(axis=1))
        strata.append({"task": int(task), "eligible_windows": int(selected.sum()),
                       "episode_ids": cluster_ids.tolist(), "task_weight": weight})
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "metric": key, "effect_treatment_minus_control": float(averaged.mean()),
        "ci95_episode_bootstrap": [float(low), float(high)],
        "per_seed_effect": {str(seed): float(differences[i].mean()) for i, seed in enumerate(seeds)},
        "eligible_windows": len(tasks), "episode_clusters": int(len(np.unique(episodes))),
        "strata": strata,
    }


def plot_curves(records: list[dict], seeds: list[int], steps: int, run_dir: Path,
                save_pdf: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#555555", "auxiliary": "#0072B2", "condition": "#D55E00"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
        for axis, key in zip(axes.flat, METRICS):
            for mode in MODES:
                curves = []
                for seed in seeds:
                    selected = sorted((r for r in records if r["seed"] == seed and r["mode"] == mode),
                                      key=lambda r: r["step"])
                    x = np.asarray([r["step"] for r in selected])
                    y = np.asarray([r[key] for r in selected])
                    if curves and not np.array_equal(x, curves[0][0]):
                        raise ValueError("Learning-curve evaluation steps differ between seeds.")
                    curves.append((x, y))
                    axis.plot(x, y, color=colors[mode], alpha=.28, linewidth=.9, linestyle="--")
                axis.plot(curves[0][0], np.mean([y for _, y in curves], axis=0),
                          color=colors[mode], linewidth=2, marker="o", markersize=3,
                          label=LABELS[mode])
            axis.set_title(TITLES[key], fontsize=11)
            axis.set_xlabel("Optimizer updates per arm")
            axis.grid(alpha=.18)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(.5, .04), fontsize=9)
        fig.suptitle(f"Shared heading: {steps:,} updates from scratch, {len(seeds)} fixed training seeds",
                     fontsize=13)
        fig.text(.5, .015, "Solid: mean over seeds. Dashed: individual seeds. Offline EMA evaluation; no success-rate measurement.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, .95))
        fig.savefig(run_dir / "learning_curves.png", dpi=220)
        if save_pdf:
            fig.savefig(run_dir / "learning_curves.pdf")
        plt.close(fig)


def interpret_results(result: dict) -> dict:
    means = result["seed_means"]
    records = {(r["seed"], r["mode"]): r for r in result["final_per_seed"]}
    reductions = {str(seed): 100. * (
        records[(seed, "baseline")]["raw_prefix8_mse"] - records[(seed, "condition")]["raw_prefix8_mse"]
    ) / records[(seed, "baseline")]["raw_prefix8_mse"] for seed in result["seeds"]}
    mean_reduction = 100. * (means["baseline"]["raw_prefix8_mse"] - means["condition"]["raw_prefix8_mse"]) / means["baseline"]["raw_prefix8_mse"]
    seed_text = "; ".join(f"seed {seed}: {value:+.3f}%" for seed, value in reductions.items())
    notes = [
        "Mean predictor heading MAE is "
        f"{means['baseline']['heading_mae_deg']:.2f}° for the detached baseline probe, "
        f"{means['auxiliary']['heading_mae_deg']:.2f}° with auxiliary supervision, and "
        f"{means['condition']['heading_mae_deg']:.2f}° with the heading condition. "
        "This measures heading readout from fused image/state features, not visual geometry alone.",
        f"Auxiliary supervision versus baseline gives flow MSE {means['auxiliary']['flow_mse']:.6f} versus "
        f"{means['baseline']['flow_mse']:.6f}, and raw prefix-8 MSE {means['auxiliary']['raw_prefix8_mse']:.6f} versus "
        f"{means['baseline']['raw_prefix8_mse']:.6f}. Better heading prediction need not improve action imitation.",
        f"Condition versus baseline reduces the mean raw prefix-8 MSE by {mean_reduction:+.3f}%. "
        f"Per-seed reductions are {seed_text}. Positive percentages mean lower error.",
    ]
    if mean_reduction > 0 and any(abs(v) < 1. for v in reductions.values()) and any(v > 5. for v in reductions.values()):
        notes.append("The mean prefix-error gain is uneven across training seeds: at least one seed changes by less than 1%, "
                     "while another improves by more than 5%. The averaged episode-bootstrap interval does not remove this seed dependence.")
    notes.append("These are final-budget offline results. They do not establish a robust success-rate improvement; "
                 "paired rollouts and more training seeds are required for that conclusion.")
    return {"condition_vs_baseline_prefix_reduction_percent": mean_reduction,
            "per_seed_condition_vs_baseline_prefix_reduction_percent": reductions, "notes": notes}


def markdown_report(result: dict) -> str:
    args = result["protocol"]["args"]
    keys = list(METRICS)
    lines = ["# Shared-heading mini-test", "",
             f"Completed **{result['final_step']:,} updates per arm**, from scratch, with fixed training seeds "
             f"{', '.join(map(str, result['seeds']))}. These are offline EMA measurements; no environment success rate was measured.", "",
             f"The protocol uses {args['train_per_task']} fixed training windows and {args['val_per_task']} "
             f"held-out validation windows per task, batch size {args['batch_size']}, and auxiliary weight {args['aux_weight']}. "
             f"Generated-action metrics use {args['sample_per_task']} fixed windows per task and one common latent sample per observation.", "",
             "The baseline trains a diagnostic heading head on detached shared features. Auxiliary supervision also updates the shared encoder; "
             "the condition arm additionally supplies predicted heading to the flow model.", "",
             "## Final measurements", "", "All four metrics are lower-is-better. Prefix action MSE covers eight executed-prefix positions; "
             "heading metrics describe the full 16-action chunk. Angular metrics exclude invalid target headings.", "",
             "| Seed | Arm | Flow MSE | Heading MAE (°) | Raw prefix-8 MSE | Generated heading MAE (°) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for record in result["final_per_seed"]:
        values = " | ".join(f"{record[key]:.6g}" for key in keys)
        lines.append(f"| {record['seed']} | {LABELS[record['mode']]} | {values} |")
    for mode in MODES:
        values = " | ".join(f"{result['seed_means'][mode][key]:.6g}" for key in keys)
        lines.append(f"| Mean | {LABELS[mode]} | {values} |")
    lines += ["", "## What these measurements show", ""]
    for note in result["interpretation"]["notes"]:
        lines.extend([note, ""])
    controls = result["heading_null_controls"]
    lines += ["", "## Heading-bias controls", "",
              f"These constant-direction controls are fitted only on valid headings from the {controls['training_windows']:,} saved training windows. "
              "The task control selects its fixed training-set direction using task ID; neither control uses images or robot state. "
              "Both are evaluated on the same valid held-out windows as the learned heading heads.", "",
              "| Control | Heading MAE (°), lower is better | Mean cosine, higher is better |",
              "| --- | ---: | ---: |"]
    for key, label in (("global_circular_mean", "Global circular mean"),
                       ("per_task_circular_mean", "Per-task circular mean")):
        control = controls[key]
        lines.append(f"| {label} | {control['heading_mae_deg']:.6g} | {control['heading_cosine']:.6g} |")
    lines += ["", controls["interpretation"]]
    lines += ["", "## Paired differences", "", "Differences are treatment minus control; negative values indicate lower error. "
              "Intervals are 95% percentile intervals from paired episode-cluster bootstrap draws stratified by task.", "",
              "| Comparison | Metric | Mean paired difference | 95% episode-bootstrap interval |",
              "| --- | --- | ---: | --- |"]
    for comparison in result["comparisons"]:
        for key, estimate in comparison["metrics"].items():
            lo, hi = estimate["ci95_episode_bootstrap"]
            lines.append(f"| {comparison['label']} | {TITLES[key]} | {estimate['effect_treatment_minus_control']:+.6g} | [{lo:+.6g}, {hi:+.6g}] |")
    lines += ["", f"Bootstrap uses {result['bootstrap']['draws']:,} draws. It resamples whole episode sums and counts within each task, "
              "retains the original eligible-window task weights, and applies the same episode draws to both arms and every fixed seed. "
              "For angular metrics, only episodes with eligible target headings contribute. Individual windows are not treated as independent episodes.", "",
              "**These intervals describe held-out episode sampling conditional on the listed training seeds. They do not measure robust training-seed uncertainty, "
              "task-population uncertainty, or success-rate uncertainty.**", "",
              "## Interpretation limits", ""]
    lines += [f"- {note}" for note in result["limitations"]]
    lines += ["", "## Learning curves", "", "![Offline learning curves](learning_curves.png)", "",
              "Machine-readable values, per-task bootstrap weights, sample-identity hashes, and protocol metadata are in `comparison.json`.", ""]
    return "\n".join(lines)


def summarize(run_dir: Path, draws: int = 2000, bootstrap_seed: int = 7031,
              save_pdf: bool = False) -> dict:
    if not (run_dir / "summary.json").is_file():
        raise ValueError("Run is incomplete: summary.json must exist before summarization.")
    manifest, summary = read_json(run_dir / "manifest.json"), read_json(run_dir / "summary.json")
    records, split = read_json(run_dir / "metrics.json"), read_json(run_dir / "split.json")
    args = manifest["args"]
    seeds = [int(seed) for seed in args["seeds"]]
    steps = int(args["steps"])
    expected = {(seed, mode) for seed in seeds for mode in MODES}
    final = summary["final"]
    identities = [(int(r["seed"]), r["mode"]) for r in final]
    if len(set(seeds)) != len(seeds) or set(identities) != expected or len(identities) != len(expected):
        raise ValueError("Final summary does not contain exactly one record per expected seed and arm.")
    if any(int(r["step"]) != steps for r in final):
        raise ValueError("Final records do not match the manifest's completed update budget.")
    rows = np.asarray(split["validation_windows"], dtype=np.int64)
    validate_rows(rows, split)
    sampled = generated_rows(rows, int(args["sample_per_task"]))
    archives = {}
    reference = None
    final_records = []
    for seed in seeds:
        for mode in MODES:
            path = run_dir / f"seed{seed}" / mode / f"eval_{steps:05d}.npz"
            arrays = load_archive(path, rows, sampled)
            if reference is not None:
                for field in ("valid", "sample_valid", "episode", "task", "sample_episode", "sample_task"):
                    if not np.array_equal(arrays[field], reference[field]):
                        raise ValueError(f"Paired evaluation metadata differs: {path}, {field}")
            reference = arrays
            archives[(seed, mode)] = arrays
            record = {"seed": seed, "mode": mode, "step": steps,
                      **{key: metric_mean(arrays, key) for key in METRICS}}
            reported = next(r for r in final if r["seed"] == seed and r["mode"] == mode)
            matching = [r for r in records if r["seed"] == seed and r["mode"] == mode and r["step"] == steps]
            if len(matching) != 1:
                raise ValueError("metrics.json must contain one final evaluation per seed and arm.")
            for source in (reported, matching[0]):
                for key in METRICS:
                    if not np.isclose(record[key], source[key], rtol=1e-6, atol=1e-9):
                        raise ValueError(f"Saved metric disagrees with archive: {seed}, {mode}, {key}")
            final_records.append(record)
    controls = heading_null_controls(manifest, split, rows, reference["valid"])
    rng = np.random.default_rng(bootstrap_seed)
    comparisons = []
    for treatment, control in (("auxiliary", "baseline"), ("condition", "auxiliary"), ("condition", "baseline")):
        comparisons.append({
            "label": f"{treatment} − {control}", "treatment": treatment, "control": control,
            "metrics": {key: paired_bootstrap(archives, seeds, treatment, control, key, draws, rng)
                        for key in METRICS},
        })
    limitations = list(manifest.get("limitations", [])) + [
        f"Training starts from scratch and ends after {steps:,} updates; learning curves describe this limited budget.",
        "The detached baseline probe tests heading predictability from fused image/state features. Since those features include robot state, better probe accuracy alone does not prove that the vision encoder learned new geometry.",
        "A single preset auxiliary weight was tested; this experiment does not establish the best supervision strength.",
        "Validation windows within an episode are correlated. Some tasks have few held-out episodes; bootstrap intervals can be unstable or degenerate for a sparsely represented task.",
        "There is no environment rollout or success-rate measurement. Lower imitation loss or heading error need not improve closed-loop success.",
        "Archives save ordered episode/task IDs but not frame IDs. Frame identities are reconstructed from split.json and the runner's fixed selection algorithm; their provenance is shared, not independently embedded in each archive.",
    ]
    result = {
        "run_dir": str(run_dir.resolve()), "final_step": steps, "seeds": seeds,
        "protocol": manifest, "elapsed_seconds": summary.get("elapsed_seconds"),
        "final_per_seed": final_records,
        "heading_null_controls": controls,
        "seed_means": {mode: {key: float(np.mean([r[key] for r in final_records if r["mode"] == mode]))
                               for key in METRICS} for mode in MODES},
        "comparisons": comparisons,
        "bootstrap": {"draws": draws, "seed": bootstrap_seed, "confidence": .95,
                      "unit": "validation episode, stratified by task",
                      "task_weights": "fixed original eligible-window fractions; angular metrics use target-valid windows",
                      "seed_treatment": "paired effects averaged over fixed seeds; seeds not resampled",
                      "interpretation": "held-out episode sampling conditional on fixed training seeds; not SR or robust training-seed uncertainty"},
        "identity_checks": {"validation_windows": len(rows), "generated_windows": len(sampled),
                            "validation_episode_clusters": int(len(np.unique(rows[:, 0]))),
                            "validation_identity_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
                            "generated_identity_sha256": hashlib.sha256(sampled.astype("<i8").tobytes()).hexdigest(),
                            "ordered_episode_task_and_validity_match_all_arms_and_seeds": True},
        "limitations": limitations,
    }
    result["interpretation"] = interpret_results(result)
    # Validate plots before writing the machine-readable and Markdown reports.
    plot_curves(records, seeds, steps, run_dir, save_pdf)
    (run_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (run_dir / "results.md").write_text(markdown_report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7031)
    parser.add_argument("--pdf", action="store_true", help="Also save the learning curves as vector PDF.")
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    try:
        result = summarize(args.run_dir, args.bootstrap_draws, args.bootstrap_seed, args.pdf)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"run_dir": result["run_dir"], "final_step": result["final_step"],
                      "outputs": ["comparison.json", "results.md", "learning_curves.png"]
                      + (["learning_curves.pdf"] if args.pdf else [])}))


if __name__ == "__main__":
    main()
