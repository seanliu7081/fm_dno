"""Summarize the fixed-model GT-source condition × angular-jitter diagnostic.

Future-label interventions are diagnostic-only. All paired episode intervals
condition on the saved model seeds and noise draws, not training-run uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("predicted_jitter", "predicted_zero", "gt_jitter", "gt_zero")
METRICS = ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "raw_prefix8_rotation_mse",
           "raw_prefix8_gripper_mse", "generated_heading_mae_deg")
TITLES = {"raw_prefix8_mse": "Prefix-8 action MSE", "raw_prefix8_xyz_mse": "Prefix-8 XYZ MSE",
          "raw_prefix8_rotation_mse": "Prefix-8 rotation MSE",
          "raw_prefix8_gripper_mse": "Prefix-8 gripper MSE",
          "generated_heading_mae_deg": "Generated heading MAE (°)"}
LABELS = {"predicted_jitter": "Predicted condition / saved κ=4 jitter",
          "predicted_zero": "Predicted condition / zero jitter",
          "gt_jitter": "GT condition / saved κ=4 jitter",
          "gt_zero": "GT condition / zero jitter"}
CONTRASTS = {
    "zero_minus_jitter_predicted_condition": {
        "label": "Zero − jitter, predicted condition", "weights": [-1, 1, 0, 0]},
    "zero_minus_jitter_gt_condition": {
        "label": "Zero − jitter, GT condition", "weights": [0, 0, -1, 1]},
    "gt_minus_predicted_condition_with_jitter": {
        "label": "GT − predicted condition, jitter", "weights": [-1, 0, 1, 0]},
    "gt_minus_predicted_condition_zero_jitter": {
        "label": "GT − predicted condition, zero jitter", "weights": [0, -1, 0, 1]},
    "interaction": {
        "label": "Interaction: difference of differences", "weights": [1, -1, -1, 1]},
}
SOURCE_METRICS = ("source_angle_error_deg", "source_heading_error_deg")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def metric(data: dict, arm: str, key: str) -> np.ndarray:
    return data[f"{arm}__{key}"]


def eligible(data: dict, key: str, valid_only: bool = False) -> np.ndarray:
    return data["valid"].astype(bool) if valid_only or key == "generated_heading_mae_deg" else np.ones(len(data["valid"]), bool)


def validate(manifest: dict, completion: dict, archives: dict[int, dict]) -> None:
    seeds, draws, windows = manifest["seeds"], manifest["draws"], manifest["windows"]
    if manifest["arms"] != list(ARMS) or completion["arms"] != list(ARMS):
        raise ValueError("Unexpected factorial arms or ordering")
    if len(set(seeds)) != len(seeds) or min(draws, windows, len(seeds)) < 1:
        raise ValueError("Require nonempty draws/windows and unique model seeds")
    if not completion.get("complete") or any(completion[key] != manifest[key] for key in ("seeds", "draws", "windows")):
        raise ValueError("Incomplete or mismatched factorial evaluation")
    ref = archives[seeds[0]]
    for seed, data in archives.items():
        for key in ("episode", "frame", "task", "valid"):
            if data[key].shape != (windows,) or not np.array_equal(data[key], ref[key]):
                raise ValueError(f"Unpaired held-out identities or label validity: {seed}, {key}")
        if not data["valid"].any():
            raise ValueError("No valid GT headings")
        for key in ("epsilon", "angular_jitter"):
            if not np.array_equal(data[key], ref[key]):
                raise ValueError(f"Unpaired saved random input: {seed}, {key}")
        if data["epsilon"].shape != (draws, windows, 16, 7) or data["angular_jitter"].shape != (draws, windows):
            raise ValueError("Unexpected saved noise dimensions")
        invalid = ~data["valid"].astype(bool)
        for arm in ARMS:
            for key in (*METRICS, *SOURCE_METRICS):
                values = metric(data, arm, key)
                if values.shape != (draws, windows) or not np.isfinite(values).all() or (values < -1e-7).any():
                    raise ValueError(f"Malformed/nonfinite/negative metric: {seed}, {arm}, {key}")
                if not np.array_equal(values[:, invalid], metric(data, ARMS[0], key)[:, invalid]):
                    raise ValueError(f"Invalid-label fallback differs between arms: {seed}, {arm}, {key}")
            for key in ("source", "action_pred"):
                if metric(data, arm, key).shape != (draws, windows, 16, 7):
                    raise ValueError(f"Unexpected tensor shape: {seed}, {arm}, {key}")
                if not np.array_equal(metric(data, arm, key)[:, invalid], metric(data, ARMS[0], key)[:, invalid]):
                    raise ValueError(f"Invalid-label tensor fallback differs: {seed}, {arm}, {key}")
        original, intervention = data["conditioning_predicted"], data["conditioning_gt"]
        if original.shape != intervention.shape or original.shape[0] != windows or original.shape[-1] != 141:
            raise ValueError("Condition shape differs from original shared-heading model")
        if not np.array_equal(original[..., :-3], intervention[..., :-3]) or not np.array_equal(original[..., -1], intervention[..., -1]):
            raise ValueError("GT condition changes observation features or predicted validity")
        if not np.array_equal(original[invalid], intervention[invalid]):
            raise ValueError("Invalid-label GT condition fallback differs")


def paired_contrast(archives: dict[int, dict], seeds: list[int], weights: list[float],
                    key: str, repetitions: int, random_seed: int, valid_only: bool = False) -> dict:
    """A linear factorial contrast with shared episode resampling in every cell."""
    if len(weights) != len(ARMS) or not np.isclose(sum(weights), 0.):
        raise ValueError("A contrast must have one weight per cell and sum to zero")
    ref = archives[seeds[0]]
    mask = eligible(ref, key, valid_only)
    tasks, episodes = ref["task"][mask], ref["episode"][mask]
    delta = np.stack([sum(weight * metric(archives[seed], arm, key).astype(np.float64).mean(0)[mask]
                         for weight, arm in zip(weights, ARMS)) for seed in seeds])
    average = delta.mean(0)
    rng = np.random.default_rng(random_seed)
    estimates = np.zeros(repetitions)
    strata = []
    for task in np.unique(tasks):
        selected = tasks == task
        clusters, inverse = np.unique(episodes[selected], return_inverse=True)
        sums = np.bincount(inverse, weights=average[selected])
        counts = np.bincount(inverse).astype(float)
        choices = rng.integers(len(clusters), size=(repetitions, len(clusters)))
        weight = float(selected.mean())
        estimates += weight * sums[choices].sum(1) / counts[choices].sum(1)
        strata.append({"task": int(task), "episode_ids": clusters.tolist(),
                       "windows": int(selected.sum()), "fixed_task_weight": weight})
    effects = delta.mean(1)
    return {"effect": float(average.mean()), "ci95_episode_bootstrap": np.quantile(estimates, [.025, .975]).tolist(),
            "per_seed_effect": {str(seed): float(effects[i]) for i, seed in enumerate(seeds)},
            "same_sign_across_fixed_seeds": bool(np.all(effects >= 0) or np.all(effects <= 0)),
            "weights": dict(zip(ARMS, weights)), "eligible_windows": int(mask.sum()),
            "eligible_episodes": len(np.unique(episodes)),
            "valid_only": valid_only or key == "generated_heading_mae_deg", "strata": strata}


def interpret(result: dict) -> list[str]:
    notes = []
    for key in CONTRASTS:
        value = result["contrasts"][key]["metrics"]["raw_prefix8_mse"]
        low, high = value["ci95_episode_bootstrap"]
        seeds = "; ".join(f"seed {seed}: {effect:+.6g}" for seed, effect in value["per_seed_effect"].items())
        notes.append(f"{CONTRASTS[key]['label']}: action-MSE contrast {value['effect']:+.6g}, conditional 95% episode interval [{low:+.6g}, {high:+.6g}]; {seeds}.")
    notes += ["For simple effects, negative means the named change lowers error. The interaction is (GT-zero − GT-jitter) − (predicted-zero − predicted-jitter): negative means removing dither helps more (or hurts less) with GT condition, while positive means the opposite.",
              "All four cells center the source on GT for valid labels. Source-center angular error, realized noise-heading error after dither, and final generated heading error are different measurements. Differences between their MAEs are not a causal decomposition of error.",
              "GT direction condition and zero-dither sources were not used to train these checkpoints. This fixed-model intervention diagnoses sensitivity and interaction; it does not identify a deployable gain, a strict attainable upper bound, or an SR improvement."]
    return notes


def plots(result: dict, output: Path, pdf: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    seeds = result["seeds"]
    cells = {(row["seed"], row["arm"]): row["mean_over_draws"] for row in result["per_seed"]}
    colors = {"jitter": "#0072B2", "zero": "#D55E00"}
    with plt.rc_context({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for axis, key in zip(axes.flat, METRICS):
            for kind, arms in (("jitter", ("predicted_jitter", "gt_jitter")), ("zero", ("predicted_zero", "gt_zero"))):
                for seed in seeds:
                    axis.plot([0, 1], [cells[(seed, arm)][key] for arm in arms], color=colors[kind],
                              linestyle="--", alpha=.45, linewidth=1, marker="o", markersize=3)
                axis.plot([0, 1], [result["seed_means"][arm][key] for arm in arms], color=colors[kind],
                          linewidth=2.2, marker="o", markersize=5,
                          label="Saved κ=4 jitter" if kind == "jitter" else "Zero jitter")
            axis.set_xticks([0, 1], ["Predicted condition", "GT direction condition"])
            axis.set_title(TITLES[key])
            axis.set_xlim(-.15, 1.15)
            axis.grid(axis="y", alpha=.2)
        axes.flat[-1].axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False)
        axes.flat[-1].text(.03, .7, "All valid-label cells use GT source center.\nOnly direction condition and jitter change.\n\nSolid: mean of fixed training seeds\nDashed: individual seed\n\nMSE: first 8 raw actions\nHeading: full 16-action resultant\n\nOffline diagnostic; no retraining or SR.", transform=axes.flat[-1].transAxes, va="top", linespacing=1.5)
        fig.suptitle("GT-source diagnostic: heading condition × angular jitter", fontsize=15)
        fig.text(.5, .02, f"Same {result['windows']} observations × {result['draws']} paired noise draws × {len(seeds)} fixed checkpoints. Invalid GT labels use the original predicted setup in every cell.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .045, 1, .95))
        fig.savefig(output / "factorial_cells.png", dpi=200)
        if pdf:
            fig.savefig(output / "factorial_cells.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
        ordered = list(CONTRASTS)
        y = np.arange(len(ordered))[::-1]
        for axis, key in zip(axes, ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "generated_heading_mae_deg")):
            for position, contrast in zip(y, ordered):
                item = result["contrasts"][contrast]["metrics"][key]
                low, high = item["ci95_episode_bootstrap"]
                color = "#777777" if contrast == "interaction" else "#0072B2"
                axis.hlines(position, low, high, color=color, linewidth=2)
                axis.vlines([low, high], position - .065, position + .065, color=color, linewidth=1.3)
                axis.plot(item["effect"], position, "o", color=color)
                axis.scatter(list(item["per_seed_effect"].values()), np.full(len(seeds), position), marker="|", s=100,
                             color="#D55E00", zorder=3)
            axis.axvline(0, color="#999999", linestyle="--", linewidth=1.)
            axis.set_yticks(y, [CONTRASTS[k]["label"] for k in ordered] if axis is axes[0] else [""] * len(ordered))
            axis.set_title(TITLES[key])
            axis.set_xlabel("Paired contrast (raw metric units)")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.grid(axis="x", alpha=.2)
        fig.suptitle("Simple effects and interaction, conditional on fixed seeds and noise", fontsize=14)
        fig.text(.5, .045, "Dots: fixed-seed mean. Orange ticks: individual seeds. Bars: 95% paired episode-bootstrap intervals.\nSimple effect < 0: lower error. Interaction = (GT-zero − GT-jitter) − (predicted-zero − predicted-jitter).", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, .94))
        fig.savefig(output / "factorial_effects.png", dpi=200)
        if pdf:
            fig.savefig(output / "factorial_effects.pdf")
        plt.close(fig)


def write_report(result: dict, path: Path) -> None:
    lines = ["# GT-source factorial diagnostic", "",
             f"Fixed checkpoints: seeds {result['seeds']}; {result['windows']} held-out observations and {result['draws']} saved paired noise draws. "
             "No weights are updated. The two intervention factors are heading direction condition (predicted or GT) and angular dither (the original saved κ=4 samples or zero).", "",
             "**The source is centered on the future-label GT heading in every valid-label cell.** This means `predicted_jitter` has predicted condition but GT-centered source; it is the previous oracle-source control, not the original fully predicted deployment setup. "
             "GT condition replaces only the two heading direction components. Observation features, predicted validity, model weights, normalizer, base Gaussian samples, source gate, and 10-step Euler solver stay fixed. Invalid GT labels revert to the original predicted condition, predicted source angle, and original saved jitter in all four cells.", "",
             "## Four-cell results", "", "Errors are scored per generated sample and then averaged. Lower is better. Action MSE uses the first eight raw actions; generated heading MAE uses the 16-action resultant on valid target headings.", "",
             "| Seed | Cell | Action MSE | XYZ MSE | Rotation MSE | Gripper MSE | Generated heading MAE (°) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["per_seed"]:
        values = " | ".join(f"{row['mean_over_draws'][key]:.6g}" for key in METRICS)
        lines.append(f"| {row['seed']} | {LABELS[row['arm']]} | {values} |")
    for arm in ARMS:
        values = " | ".join(f"{result['seed_means'][arm][key]:.6g}" for key in METRICS)
        lines.append(f"| Mean | {LABELS[arm]} | {values} |")
    lines += ["", "## Simple effects and interaction", "",
              "Simple effects are zero minus jitter, or GT condition minus predicted condition. Negative means the named change lowers error. "
              "The interaction is `(GT-zero − GT-jitter) − (predicted-zero − predicted-jitter)`, so its sign describes whether the zero-jitter effect changes with condition.", "",
              "| Contrast | Metric | Effect | 95% episode-bootstrap interval |", "| --- | --- | ---: | --- |"]
    for name, contrast in result["contrasts"].items():
        for key, estimate in contrast["metrics"].items():
            low, high = estimate["ci95_episode_bootstrap"]
            lines.append(f"| {CONTRASTS[name]['label']} | {TITLES[key]} | {estimate['effect']:+.6g} | [{low:+.6g}, {high:+.6g}] |")
    lines += ["", f"{result['bootstrap']['repetitions']:,} bootstrap replicates resample whole validation episodes within task, retaining the same episode multiplicities in every cell and seed. "
              "Fixed latent draws and fixed seeds are averaged before resampling; original eligible-window task weights remain fixed. "
              "**Intervals are conditional on these checkpoints and saved noise draws. They do not estimate uncertainty over training runs or success rates.** Valid-target-only action effects and all per-seed effects are additionally saved in `summary.json`.", "",
              "## Interpretation", ""]
    for note in result["interpretation"]:
        lines += [note, ""]
    lines += ["## Source center, actual noise direction, and generated direction", "",
              "The source center has zero GT error on valid labels. With saved κ=4 dither, the actual noise resultant can still deviate substantially; zero dither removes that angular deviation only at initialization. "
              "The final generated heading is a separate measurement after the flow solver. **Subtracting these MAEs does not attribute a causal amount of error to the predictor, the noise, or the flow.**", "",
              "| Seed | Cell | Source-center MAE (°) | Realized noise heading MAE (°) | Generated heading MAE (°) |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in result["per_seed"]:
        diag = row["source_diagnostics_valid_only"]
        lines.append(f"| {row['seed']} | {LABELS[row['arm']]} | {diag['source_angle_error_deg']:.6g} | {diag['source_heading_error_deg']:.6g} | {row['mean_over_draws']['generated_heading_mae_deg']:.6g} |")
    lines += ["", "## Eligibility and limits", ""]
    for row in result["eligibility"]:
        lines.append(f"- Seed {row['seed']}: {row['valid_windows']}/{result['windows']} valid labels; {row['invalid_windows']} invalid-label fallbacks; source active in {row['source_active_fraction']:.3%} of draw/window pairs. Invalid-label metrics, source tensors, and generated action tensors match exactly across cells.")
    lines += ["", "Both GT direction condition and deterministic angular alignment change distributions relative to training. The saved Gaussian draw still varies under zero jitter, so this is not deterministic action generation. "
              "This experiment uses future action labels and cannot be deployed as measured. Results diagnose the existing checkpoints rather than predict what matched retraining would achieve; they neither establish a strict upper bound nor measure SR.", "",
              "## Plots", "", "![Four-cell interaction plots](factorial_cells.png)", "", "![Paired effects](factorial_effects.png)", "",
              "`summary.json` retains per-draw means, per-seed effects, all-window and valid-only intervals, source diagnostics, provenance hashes, and the exact report-source hash. Evaluator provenance remains in `manifest.json` and `evaluation_complete.json`.", ""]
    path.write_text("\n".join(lines))


def summarize(run_dir: Path, repetitions: int = 2000, random_seed: int = 92027,
              pdf: bool = False) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    completion = read_json(run_dir / "evaluation_complete.json")
    seeds = list(map(int, manifest["seeds"]))
    paths = {seed: run_dir / f"seed{seed}.npz" for seed in seeds}
    for seed, path in paths.items():
        checks = completion["checks"][str(seed)]
        if sha256(path) != checks["archive_sha256"]:
            raise ValueError(f"Saved factorial archive hash differs: {seed}")
        if not all(record["bit_exact"] for record in checks["control_replay"].values()):
            raise ValueError(f"The saved oracle-source control did not exactly replay: {seed}")
    archives = {seed: read_archive(path) for seed, path in paths.items()}
    validate(manifest, completion, archives)
    draws, windows = manifest["draws"], manifest["windows"]
    per_seed, eligibility = [], []
    for seed in seeds:
        data = archives[seed]
        for arm in ARMS:
            per_draw = [{"draw": draw, **{key: float(metric(data, arm, key)[draw, eligible(data, key)].astype(np.float64).mean())
                                         for key in METRICS}} for draw in range(draws)]
            means = {key: float(np.mean([row[key] for row in per_draw])) for key in METRICS}
            source = {key: float(metric(data, arm, key)[:, data["valid"].astype(bool)].astype(np.float64).mean()) for key in SOURCE_METRICS}
            per_seed.append({"seed": seed, "arm": arm, "mean_over_draws": means,
                             "per_draw": per_draw, "source_diagnostics_valid_only": source})
        eligibility.append({"seed": seed, "valid_windows": int(data["valid"].sum()),
                            "invalid_windows": int((~data["valid"].astype(bool)).sum()),
                            "source_active_fraction": float(data["source_active"].mean()),
                            "invalid_fallback_exact": True})
    seed_means = {arm: {key: float(np.mean([row["mean_over_draws"][key] for row in per_seed if row["arm"] == arm]))
                        for key in METRICS} for arm in ARMS}
    contrasts, valid_only = {}, {}
    for name, spec in CONTRASTS.items():
        contrasts[name] = {"label": spec["label"], "metrics": {key: paired_contrast(archives, seeds, spec["weights"], key, repetitions, random_seed + index)
                                                                       for index, key in enumerate(METRICS)}}
        valid_only[name] = {"label": spec["label"], "metrics": {key: paired_contrast(archives, seeds, spec["weights"], key, repetitions, random_seed + index, True)
                                                                      for index, key in enumerate(METRICS)}}
    result = {"seeds": seeds, "draws": draws, "windows": windows, "arms": list(ARMS),
              "per_seed": per_seed, "seed_means": seed_means, "contrasts": contrasts,
              "valid_only_contrasts": valid_only, "eligibility": eligibility,
              "bootstrap": {"repetitions": repetitions, "random_seed": random_seed,
                            "unit": "paired whole validation episode within task",
                            "conditions_on": "fixed checkpoints and saved noise samples"},
              "evaluation_completion": completion,
              "input_sha256": {path.name: sha256(path) for path in [run_dir / "manifest.json", run_dir / "evaluation_complete.json", *paths.values()]},
              "report_source_sha256": sha256(Path(__file__)),
              "limitations": ["No retraining or SR measurement", "GT requires future action labels", "GT condition and zero jitter are distribution changes relative to training",
                              "No causal error decomposition by subtracting angular MAEs", "Not a strict attainable performance upper bound", "Intervals condition on fixed model seeds and latent draws"]}
    result["interpretation"] = interpret(result)
    snapshot = run_dir / "source/scripts" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, run_dir / "report.md")
    plots(result, run_dir, pdf)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=92027)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("bootstrap count must be positive")
    result = summarize(args.run_dir.resolve(), args.bootstrap_draws, args.bootstrap_seed, args.pdf)
    print(json.dumps({"run_dir": str(args.run_dir.resolve()), "seed_means": result["seed_means"],
                      "interpretation": result["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
