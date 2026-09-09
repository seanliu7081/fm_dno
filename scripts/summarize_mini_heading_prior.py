"""Summarize a completed IID / heading-prior mini-test without loading weights.

Usage: python scripts/summarize_mini_heading_prior.py --run-dir OUTPUT --pdf
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_mini_shared_heading import (
    heading_null_controls, load_archive, read_json, validate_rows,
)

MODES = ("condition", "condition_prior_xy", "condition_prior_both")
LABELS = {"condition": "Condition + IID", "condition_prior_xy": "Condition + XY prior",
          "condition_prior_both": "Condition + both-block prior"}
METRICS = {
    "raw_prefix8_mse": ("sample_raw_prefix8_mse", "sample_", False),
    "raw_prefix8_xyz_mse": ("sample_raw_prefix8_xyz_mse", "sample_", False),
    "generated_heading_mae_deg": ("sample_generated_heading_mae_deg", "sample_", True),
    "heading_mae_deg": ("heading_mae_deg", "", True),
    "flow_mse": ("flow_mse", "", False),
}
DISPLAY_METRICS = tuple(METRICS)
METRICS.update({
    "raw_prefix8_rotation_mse": ("sample_raw_prefix8_rotation_mse", "sample_", False),
    "raw_prefix8_gripper_mse": ("sample_raw_prefix8_gripper_mse", "sample_", False),
})
EFFECT_METRICS = tuple(key for key in METRICS if key != "flow_mse")
SOURCE_DIAGNOSTICS = ("source_active", "source_normalized_ms", "iid_normalized_ms",
                      "source_heading_to_prediction_deg", "source_to_target_mse")
TITLES = {
    "raw_prefix8_mse": "Raw prefix-8 action MSE",
    "raw_prefix8_xyz_mse": "Raw prefix-8 XYZ MSE",
    "generated_heading_mae_deg": "Generated heading MAE (degrees)",
    "heading_mae_deg": "Predictor heading MAE (degrees)",
    "flow_mse": "Native-path flow MSE (diagnostic only)",
    "raw_prefix8_rotation_mse": "Raw prefix-8 rotation MSE",
    "raw_prefix8_gripper_mse": "Raw prefix-8 gripper MSE",
}


def generated_rows(rows: np.ndarray, per_task: int) -> np.ndarray:
    """Match the runner's balanced sampling, including small-subset capacity limits."""
    rng = np.random.default_rng(20260910)
    selected = []
    for task in np.unique(rows[:, 2]):
        episodes = np.unique(rows[rows[:, 2] == task, 0])
        available = [np.flatnonzero((rows[:, 2] == task) & (rows[:, 0] == ep)) for ep in episodes]
        counts = [min(per_task // len(episodes) + (j < per_task % len(episodes)), len(idx))
                  for j, idx in enumerate(available)]
        remaining = per_task - sum(counts)
        while remaining:
            progressed = False
            for j, idx in enumerate(available):
                if counts[j] < len(idx) and remaining:
                    counts[j] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise ValueError(f"Too few validation windows for task {task}")
        for count, idx in zip(counts, available):
            selected.extend(rng.choice(idx, count, replace=False).tolist())
    return rows[np.asarray(selected, dtype=np.int64)]


def eligible(archive: dict, key: str) -> np.ndarray:
    field, prefix, angular = METRICS[key]
    mask = archive[prefix + "valid"].astype(bool) if angular else np.ones(len(archive[field]), bool)
    if archive[field].shape != mask.shape or not mask.any() or not np.isfinite(archive[field][mask]).all():
        raise ValueError(f"Invalid saved metric or no eligible observations: {key}")
    return mask


def paired_effect(archives: dict, seeds: list[int], treatment: str, control: str,
                  key: str, draws: int, rng: np.random.Generator) -> dict:
    """Paired episode-cluster bootstrap; task weights and training seeds stay fixed."""
    field, prefix, _ = METRICS[key]
    reference = archives[(seeds[0], control)]
    mask = eligible(reference, key)
    tasks, episodes = reference[prefix + "task"][mask], reference[prefix + "episode"][mask]
    delta = np.stack([archives[(s, treatment)][field][mask] - archives[(s, control)][field][mask]
                      for s in seeds])
    averaged = delta.mean(axis=0)
    samples = np.zeros(draws)
    strata = []
    for task in np.unique(tasks):
        selected = tasks == task
        cluster_ids, inverse = np.unique(episodes[selected], return_inverse=True)
        sums = np.bincount(inverse, weights=averaged[selected])
        counts = np.bincount(inverse).astype(float)
        choices = rng.integers(0, len(cluster_ids), size=(draws, len(cluster_ids)))
        weight = float(selected.sum() / len(tasks))
        samples += weight * sums[choices].sum(1) / counts[choices].sum(1)
        strata.append({"task": int(task), "episode_ids": cluster_ids.tolist(),
                       "eligible_windows": int(selected.sum()), "fixed_task_weight": weight})
    return {"effect_treatment_minus_control": float(averaged.mean()),
            "ci95_episode_bootstrap": np.quantile(samples, [.025, .975]).tolist(),
            "per_seed_effect": {str(seed): float(delta[i].mean()) for i, seed in enumerate(seeds)},
            "strata": strata, "eligible_windows": len(tasks),
            "eligible_episode_clusters": int(len(np.unique(episodes)))}


def source_geometry(manifest: dict, split: dict) -> dict:
    """Read action-only training ranges to document the actual affine source map."""
    import zarr
    dataset = Path(manifest["args"]["dataset"])
    root = zarr.open(str(dataset if dataset.is_absolute() else ROOT / dataset), mode="r")
    ends = np.asarray(split["episode_ends"], dtype=np.int64)
    mask = np.zeros(len(ends), dtype=bool)
    mask[split["train_episodes"]] = True
    actions = np.asarray(root["data"]["action"][:])[np.repeat(mask, np.diff(np.r_[0, ends]))]
    low, high = actions.min(0), actions.max(0)
    ranges = high - low
    degenerate = ranges < 1e-4
    scale = 2. / np.where(degenerate, 2., ranges)
    offset = -1. - scale * low
    offset[degenerate] = -low[degenerate]
    compatible = lambda i, j: bool(np.isclose(scale[i], scale[j], rtol=1e-6)
                                  and np.allclose(offset[[i, j]], 0., atol=1e-7))
    return {"action_scale": scale.tolist(), "action_offset": offset.tolist(),
            "xy_rotation_compatible": compatible(0, 1), "rotation_xy_rotation_compatible": compatible(3, 4),
            "source_map": "N(R_phi(N^-1(z))), with N(a)=S*a+b and phi from detached predicted raw XY heading plus angular dither minus realized raw-source XY heading",
            "primary_blocks": [[0, 1]], "sensitivity_blocks": [[0, 1], [3, 4]],
            "interpretation": "For unequal coordinate scales or nonzero offsets, raw-space rotation is affine/nonorthogonal in normalized coordinates and can change normalized source means, covariance, and norms. The both-block arm is a sensitivity test, not an equivalent implementation of the XY-only prior."}


def interpretations(result: dict) -> list[str]:
    records = {(r["seed"], r["mode"]): r for r in result["final_per_seed"]}
    notes = []
    for mode in MODES[1:]:
        percentages = {str(s): 100. * (records[(s, "condition")]["raw_prefix8_mse"]
                       - records[(s, mode)]["raw_prefix8_mse"]) / records[(s, "condition")]["raw_prefix8_mse"]
                       for s in result["seeds"]}
        mean_base = result["seed_means"]["condition"]["raw_prefix8_mse"]
        mean_arm = result["seed_means"][mode]["raw_prefix8_mse"]
        pct = 100. * (mean_base - mean_arm) / mean_base
        changes = "; ".join(f"seed {seed}: {value:+.3f}%" for seed, value in percentages.items())
        notes.append(f"{LABELS[mode]} versus IID: mean prefix-8 MSE reduction {pct:+.3f}%; {changes}. Positive percentages mean lower error.")
        if min(percentages.values()) < 0 < max(percentages.values()):
            notes.append(f"The prefix-error effect for {LABELS[mode]} changes sign between training seeds; its average must not be described as a consistent gain.")
        elif any(abs(v) < 1 for v in percentages.values()) and any(abs(v) > 5 for v in percentages.values()):
            notes.append(f"The prefix-error effect for {LABELS[mode]} is uneven: at least one seed changes by less than 1%, while another changes by more than 5%.")
    notes.append("Judge this source intervention primarily by generated-action errors and their per-seed consistency. Native-path flow MSE uses a different interpolation and velocity target in each source arm; a lower number is not a controlled improvement in the same regression problem.")
    notes.append("These final-budget offline measurements do not establish a success-rate gain, and two fixed seeds do not establish robust uncertainty over training runs.")
    return notes


def learning_curves(records: list[dict], seeds: list[int], steps: int, run_dir: Path, pdf: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"condition": "#555555", "condition_prior_xy": "#0072B2", "condition_prior_both": "#D55E00"}
    with plt.rc_context({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for axis, key in zip(axes.flat, DISPLAY_METRICS):
            for mode in MODES:
                curves = []
                for seed in seeds:
                    rows = sorted((r for r in records if r["mode"] == mode and r["seed"] == seed), key=lambda r: r["step"])
                    x, y = np.asarray([r["step"] for r in rows]), np.asarray([r[key] for r in rows])
                    if not len(x) or (curves and not np.array_equal(x, curves[0][0])):
                        raise ValueError("Missing or unequal learning-curve evaluation steps.")
                    curves.append((x, y))
                    axis.plot(x, y, color=colors[mode], alpha=.25, linewidth=.9, linestyle="--")
                axis.plot(curves[0][0], np.mean([y for _, y in curves], axis=0), color=colors[mode],
                          linewidth=2, marker="o", markersize=3, label=LABELS[mode])
            axis.set_title(TITLES[key], fontsize=11)
            axis.set_xlabel("Optimizer updates per arm")
            axis.grid(alpha=.18)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(.04, .85, "Primary: XY-only directional source\nSensitivity: rotate both raw XY blocks\n\nShared heading remains jointly trained.\nSource construction uses detached predictions.\n\nNative-path flow MSE is not a common\nregression objective across these arms.\n\nNo environment success-rate measurement.",
                           transform=axes.flat[-1].transAxes, va="top", fontsize=11, linespacing=1.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3, frameon=False)
        fig.suptitle(f"Heading source prior: {steps:,} updates per arm, {len(seeds)} fixed seeds", fontsize=14)
        fig.text(.5, .015, "Solid: mean over fixed seeds. Dashed: individual seeds. Offline EMA evaluation.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .085, 1, .95))
        fig.savefig(run_dir / "learning_curves.png", dpi=220)
        if pdf:
            fig.savefig(run_dir / "learning_curves.pdf")
        plt.close(fig)


def report(result: dict) -> str:
    args = result["protocol"]["args"]
    geometry = result["source_geometry"]
    lines = ["# Heading-prior mini-test", "",
             f"Final-budget comparison: **{result['final_step']:,} updates per arm**, training seeds {', '.join(map(str, result['seeds']))}. "
             "All arms jointly train the shared heading head and supply predicted heading as condition. The primary intervention adds a translation-XY directional source; the both-block arm is a sensitivity test.", "",
             f"Fixed data: {args['train_per_task']} training and {args['val_per_task']} validation windows per task; "
             f"{args['sample_per_task']} generated-action samples per task, batch size {args['batch_size']}. "
             "Normalization, initialization, observation processing, minibatches, Gaussian draws, and Euler budgets are matched by the training protocol. See manifest metadata for exact settings.", "",
             "## Final results", "", "Action MSE uses the first eight actions; heading MAE uses the 16-action XY resultant and excludes invalid target headings. "
             "All errors are lower-is-better. **Native-path flow MSE is diagnostic: the sources induce different flow targets and interpolation paths.**", "",
             "| Seed | Arm | Prefix-8 MSE | Prefix-8 XYZ MSE | Generated heading MAE (°) | Predictor heading MAE (°) | Native-path flow MSE |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["final_per_seed"]:
        values = " | ".join(f"{row[key]:.6g}" for key in DISPLAY_METRICS)
        lines.append(f"| {row['seed']} | {LABELS[row['mode']]} | {values} |")
    for mode in MODES:
        values = " | ".join(f"{result['seed_means'][mode][key]:.6g}" for key in DISPLAY_METRICS)
        lines.append(f"| Mean | {LABELS[mode]} | {values} |")
    lines += ["", "Rotation/gripper decomposition (means over the fixed seeds):", "",
              "| Arm | Prefix-8 rotation MSE | Prefix-8 gripper MSE |", "| --- | ---: | ---: |"]
    for mode in MODES:
        row = result["seed_means"][mode]
        lines.append(f"| {LABELS[mode]} | {row['raw_prefix8_rotation_mse']:.6g} | {row['raw_prefix8_gripper_mse']:.6g} |")
    lines += ["", "Source diagnostics (means over the fixed seeds; source-to-target MSE is another path diagnostic):", "",
              "| Arm | Gate active fraction | Source MS | IID MS | Source-to-predicted heading MAE (°) | Source-to-target MSE |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        diag = result["source_diagnostic_means"][mode]
        values = " | ".join(f"{diag[key]:.6g}" for key in SOURCE_DIAGNOSTICS)
        lines.append(f"| {LABELS[mode]} | {values} |")
    lines += ["", "## Interpretation", ""]
    for note in result["interpretation"]:
        lines += [note, ""]
    lines += ["## Paired episode-bootstrap effects", "", "Effects are treatment minus control; negative values mean lower error. "
              "No cross-arm flow-MSE effect is used as a common-objective comparison.", "",
              "| Comparison | Metric | Mean paired effect | 95% episode-bootstrap interval |", "| --- | --- | ---: | --- |"]
    for comparison in result["comparisons"]:
        for key, estimate in comparison["metrics"].items():
            low, high = estimate["ci95_episode_bootstrap"]
            lines.append(f"| {comparison['label']} | {TITLES[key]} | {estimate['effect_treatment_minus_control']:+.6g} | [{low:+.6g}, {high:+.6g}] |")
    lines += ["", f"{result['bootstrap']['draws']:,} paired bootstrap draws resample whole validation episodes within each task. "
              "Episode sums and counts preserve within-episode dependence, with original eligible-window task weights fixed. "
              "The same episode draws apply to both arms and all fixed seeds; only episodes with valid targets contribute to angular metrics. "
              "**Intervals describe held-out episode sampling conditional on these seeds. They do not describe robust training-seed uncertainty or success-rate uncertainty.**", "",
              "## Source geometry and controls", "",
              f"Train-only action min-max scales are `{geometry['action_scale']}`; offsets are `{geometry['action_offset']}`. "
              f"Translation XY already commutes with rotation: **{geometry['xy_rotation_compatible']}**. "
              f"The rotation-XY block commutes: **{geometry['rotation_xy_rotation_compatible']}**.", "",
              "With N(a)=S·a+b, the raw-rotation source is N(Rφ(N⁻¹(z)))=S·Rφ·S⁻¹(z−b)+b. "
              "The primary arm rotates translation XY. The sensitivity arm also rotates rotation XY; unequal scales and nonzero offsets there additionally change its normalized source mean, covariance, and norm. "
              "It therefore does not isolate exactly the same distribution change as the primary arm.", "",
              "Source angles come from current detached observation predictions, with the configured angular dither and predicted-validity gate. "
              "The heading head still trains through auxiliary supervision and conditioning. Detaching the source avoids gradients through the flow target, but the learned source distribution can still change across training updates. "
              "Predicted validity is not angular accuracy or calibrated angular uncertainty.", "",
              "## Limits", ""]
    lines += [f"- {note}" for note in result["limitations"]]
    lines += ["", "## Learning curves", "", "![Offline learning curves](learning_curves.png)", "",
              "Detailed per-seed effects, cluster weights, source diagnostics, and protocol metadata are in `comparison.json`.", ""]
    return "\n".join(lines)


def summarize(run_dir: Path, draws: int = 2000, bootstrap_seed: int = 7031, pdf: bool = False) -> dict:
    if not (run_dir / "summary.json").is_file():
        raise ValueError("Run incomplete: summary.json must exist before summarization.")
    manifest, summary = read_json(run_dir / "manifest.json"), read_json(run_dir / "summary.json")
    records, split = read_json(run_dir / "metrics.json"), read_json(run_dir / "split.json")
    args = manifest["args"]
    seeds, steps = list(map(int, args["seeds"])), int(args["steps"])
    final = summary["final"]
    expected = {(seed, mode) for seed in seeds for mode in MODES}
    keys = [(int(r["seed"]), r["mode"]) for r in final]
    if len(set(seeds)) != len(seeds) or set(keys) != expected or len(keys) != len(expected):
        raise ValueError("Summary must contain exactly the three expected arms for every training seed.")
    if any(int(r["step"]) != steps for r in final):
        raise ValueError("Final evaluations do not match the declared update budget.")
    rows = np.asarray(split["validation_windows"], dtype=np.int64)
    validate_rows(rows, split)
    sampled = generated_rows(rows, int(args["sample_per_task"]))
    archives, final_records, reference = {}, [], None
    for seed in seeds:
        for mode in MODES:
            path = run_dir / f"seed{seed}" / mode / f"eval_{steps:05d}.npz"
            archive = load_archive(path, rows, sampled)
            for prefix, expected_rows in (("", rows), ("sample_", sampled)):
                if not np.array_equal(archive[prefix + "frame"], expected_rows[:, 1]):
                    raise ValueError(f"Saved frame identities differ from split/subset: {path}, {prefix}")

            if reference is not None:
                for key in ("valid", "sample_valid", "episode", "frame", "task", "sample_episode", "sample_frame", "sample_task"):
                    if not np.array_equal(reference[key], archive[key]):
                        raise ValueError(f"Paired sample identities or target masks differ: {path}, {key}")
            reference = archive
            archives[(seed, mode)] = archive
            row = {"seed": seed, "mode": mode, "step": steps}
            for key, (field, _, _) in METRICS.items():
                row[key] = float(archive[field][eligible(archive, key)].mean())
            record = next(r for r in final if r["seed"] == seed and r["mode"] == mode)
            matching = [r for r in records if r["seed"] == seed and r["mode"] == mode and r["step"] == steps]
            if len(matching) != 1:
                raise ValueError("metrics.json must contain exactly one final record per seed/arm.")
            for source in (record, matching[0]):
                for key in METRICS:
                    if not np.isclose(row[key], source[key], rtol=1e-6, atol=1e-9):
                        raise ValueError(f"Metric/archive disagreement: {seed}, {mode}, {key}")
            for key in SOURCE_DIAGNOSTICS:
                values = archive[key]
                if values.shape != (len(rows),) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid source diagnostic: {path}, {key}")
                for source in (record, matching[0]):
                    if not np.isclose(values.mean(), source[key], rtol=1e-6, atol=1e-9):
                        raise ValueError(f"Source diagnostic/archive mismatch: {path}, {key}")
            row["diagnostics"] = {k: v for k, v in record.items()
                                  if k not in {*METRICS, "seed", "mode", "step", "per_task"}}
            final_records.append(row)
    rng = np.random.default_rng(bootstrap_seed)
    comparisons = []
    for treatment, control in ((MODES[1], MODES[0]), (MODES[2], MODES[0]), (MODES[2], MODES[1])):
        comparisons.append({"label": f"{treatment} − {control}", "treatment": treatment, "control": control,
                            "metrics": {key: paired_effect(archives, seeds, treatment, control, key, draws, rng)
                                        for key in EFFECT_METRICS}})
    result = {"run_dir": str(run_dir.resolve()), "final_step": steps, "seeds": seeds,
              "protocol": manifest, "elapsed_seconds": summary.get("elapsed_seconds"),
              "final_per_seed": final_records,
              "seed_means": {mode: {key: float(np.mean([r[key] for r in final_records if r["mode"] == mode]))
                                     for key in METRICS} for mode in MODES},
              "source_diagnostic_means": {mode: {key: float(np.mean([r["diagnostics"][key] for r in final_records if r["mode"] == mode]))
                                                        for key in SOURCE_DIAGNOSTICS} for mode in MODES},
              "comparisons": comparisons,
              "source_geometry": source_geometry(manifest, split),
              "heading_null_controls": heading_null_controls(manifest, split, rows, reference["valid"]),
              "identity_checks": {"validation_windows": len(rows), "generated_windows": len(sampled),
                                  "episode_clusters": int(len(np.unique(rows[:, 0]))),
                                  "validation_identity_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
                                  "generated_identity_sha256": hashlib.sha256(sampled.astype("<i8").tobytes()).hexdigest(),
                                  "ordered_episode_frame_task_and_validity_match": True},
              "bootstrap": {"draws": draws, "seed": bootstrap_seed, "unit": "episode stratified by task",
                            "seed_treatment": "paired effects averaged over fixed seeds; seeds not resampled",
                            "task_weights": "original eligible-window fractions held fixed"},
              "limitations": list(manifest.get("limitations", [])) + [
                  "No environment rollout or success-rate measurement. Offline action or heading error need not track task success.",
                  "Two fixed training seeds and a limited training budget do not establish stability across training runs.",
                  "Only one preset auxiliary weight, angular concentration, and gate threshold are tested.",
                  "The heading head receives fused image/state features; its accuracy cannot isolate visual geometry from robot state.",
                  "The learned source is nonstationary during training despite detaching source construction from backpropagation.",
                  "Native-path flow losses are different regression objectives and are not ranked as a shared metric.",
                  "Action metrics use a fixed small held-out subset and one latent draw per observation; episode-cluster intervals retain this sampling limitation.",
                  "Saved episode/frame/task identities and target-validity masks are checked against the split and fixed sampling subset in every arm and seed.",
              ]}
    result["interpretation"] = interpretations(result)
    learning_curves(records, seeds, steps, run_dir, pdf)
    (run_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (run_dir / "results.md").write_text(report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7031)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    try:
        result = summarize(args.run_dir, args.bootstrap_draws, args.bootstrap_seed, args.pdf)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({"run_dir": result["run_dir"], "final_step": result["final_step"],
                      "outputs": ["comparison.json", "results.md", "learning_curves.png"]
                      + (["learning_curves.pdf"] if args.pdf else [])}))


if __name__ == "__main__":
    main()
