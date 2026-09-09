"""Compare newly trained predicted-heading zero-jitter flow against saved controls.

The three deployable arms receive observations only. Original saved Gaussian
samples are paired across checkpoints; old controls must replay bit-exactly under
the current backward-compatible inference code before results are accepted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.evaluate_mini_heading_prior_sampling import (
    METRICS, ROOT, angular_error, batch_at, load_policy, load_selected_cache,
    read_json, sample_indices_for, sha256,
)
from scripts.summarize_mini_heading_prior import paired_effect

ARMS = ("iid", "prior_kappa4", "prior_zero")
MODES = {"iid": "condition", "prior_kappa4": "condition_prior_xy", "prior_zero": "condition_prior_xy"}
LABELS = {"iid": "Condition + IID", "prior_kappa4": "Condition + XY prior, κ=4",
          "prior_zero": "Condition + XY prior, zero jitter (retrained)"}
TITLES = {"raw_prefix8_mse": "Prefix-8 action MSE", "raw_prefix8_xyz_mse": "Prefix-8 XYZ MSE",
          "raw_prefix8_rotation_mse": "Prefix-8 rotation MSE", "raw_prefix8_gripper_mse": "Prefix-8 gripper MSE",
          "generated_heading_mae_deg": "Generated heading MAE (°)", "predictor_mae_full_validation_deg": "Predictor heading MAE, full validation (°)"}
DIAGNOSTICS = ("source_normalized_ms", "iid_normalized_ms", "source_heading_to_prediction_deg")
CRITICAL_FILES = ("scripts/mini_heading_prior.py", "scripts/mini_shared_heading.py",
                  "oat/policy/flow_policy_shared_heading.py")


def archive_at(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def exact_tensors(a: dict, b: dict, label: str) -> None:
    if a.keys() != b.keys():
        raise ValueError(f"Tensor keys differ: {label}")
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0., atol=0., msg=f"{label}: {key}")


def same_rows(archive: dict, rows: np.ndarray, prefix: str = "") -> None:
    for name, column in (("episode", 0), ("frame", 1), ("task", 2)):
        if not np.array_equal(archive[prefix + name], rows[:, column]):
            raise ValueError(f"Saved observation identities differ: {prefix + name}")


def primary_record(run: Path, manifest: dict, split: dict, seed: int, arm: str,
                   rows: np.ndarray, reference_full_valid: np.ndarray | None) -> tuple[dict, np.ndarray]:
    step, mode = manifest["args"]["steps"], MODES[arm]
    records = [r for r in read_json(run / "summary.json")["final"]
               if r["seed"] == seed and r["mode"] == mode and r["step"] == step]
    if len(records) != 1:
        raise ValueError(f"Missing unique completed endpoint: {run}, seed{seed}, {mode}")
    path = run / f"seed{seed}" / mode / f"eval_{step:05d}.npz"
    data = archive_at(path)
    full_rows = np.asarray(split["validation_windows"], dtype=np.int64)
    same_rows(data, full_rows)
    same_rows(data, rows, "sample_")
    valid = data["valid"].astype(bool)
    if len(valid) != 1000 or not valid.any() or (reference_full_valid is not None and not np.array_equal(valid, reference_full_valid)):
        raise ValueError("Full validation target mask differs from the matched 1,000-window evaluation")
    values = {}
    for key in METRICS:
        mask = data["sample_valid"].astype(bool) if key == "generated_heading_mae_deg" else slice(None)
        values[key] = float(data["sample_" + key][mask].mean())
        if not np.isclose(values[key], records[0][key], rtol=1e-8, atol=1e-10):
            raise ValueError(f"Primary summary disagrees with saved arrays: {arm}, {key}")
    heading_mae = float(data["heading_mae_deg"][valid].mean())
    if not np.isclose(heading_mae, records[0]["heading_mae_deg"], rtol=1e-8, atol=1e-9):
        raise ValueError("Full-validation predictor MAE differs from primary summary")
    return {"seed": seed, "arm": arm, "step": step, "metrics": values,
            "predictor_mae_full_validation_deg": heading_mae,
            "validation_windows": len(valid), "valid_heading_windows": int(valid.sum()),
            "single_noise_generated_windows": len(rows), "archive": str(path), "archive_sha256": sha256(path)}, valid


def prepare(old_run: Path, zero_run: Path, paired_run: Path, reference_run: Path) -> dict:
    """Read-only protocol audit; require both new training seeds to be complete."""
    for run in (old_run, zero_run, paired_run, reference_run):
        if not (run / "summary.json").is_file():
            raise ValueError(f"Run is incomplete: {run}")
    plan = read_json(zero_run / "comparison_plan.json")
    if plan["training_endpoint"] != 5000 or plan["seeds"] != [42, 43] or plan["generated_action_draws"] != 4:
        raise ValueError("Comparison differs from the predeclared plan")
    old, new = read_json(old_run / "manifest.json"), read_json(zero_run / "manifest.json")
    pair = read_json(paired_run / "manifest.json")
    old_split, new_split = read_json(old_run / "split.json"), read_json(zero_run / "split.json")
    if old_split != new_split or old["shape_meta"] != new["shape_meta"]:
        raise ValueError("Train/validation splits, heading RMS, or observation/action shapes changed")
    ignored = {"output_dir", "modes", "source_jitter"}
    if {k:v for k,v in old["args"].items() if k not in ignored} != {k:v for k,v in new["args"].items() if k not in ignored}:
        raise ValueError("Training arguments differ beyond output directory, selected modes, and source jitter")
    if old["args"].get("source_jitter", "vonmises") != "vonmises" or new["args"].get("source_jitter") != "zero":
        raise ValueError("Expected old κ=4 default and explicitly trained zero-jitter source")
    if new["args"]["modes"] != ["condition_prior_xy"] or new["args"]["seeds"] != [42, 43] or new["args"]["steps"] != 5000:
        raise ValueError("Require the authorized 5,000-step two-seed zero-jitter training run")
    if pair["seeds"] != [42, 43] or pair["draws"] != 4 or pair["windows"] != 200:
        raise ValueError("Require the original paired four streams on 200 held-out observations")
    if pair["primary_manifest_sha256"] != sha256(old_run / "manifest.json") or pair["primary_summary_sha256"] != sha256(old_run / "summary.json"):
        raise ValueError("Paired random samples belong to a different old training experiment")
    old_norm = torch.load(old_run / "normalizer.pt", map_location="cpu", weights_only=True)
    new_norm = torch.load(zero_run / "normalizer.pt", map_location="cpu", weights_only=True)
    exact_tensors(old_norm, new_norm, "cross-run normalizer")
    code_history = []
    for name in CRITICAL_FILES:
        current_hash = sha256(ROOT / name)
        if current_hash != new["source_sha256"][name]:
            raise ValueError(f"Current inference/training code differs from new training snapshot: {name}")
        for run, manifest in ((old_run, old), (zero_run, new)):
            if sha256(run / "source" / name) != manifest["source_sha256"][name]:
                raise ValueError(f"Historical source snapshot integrity failed: {run}, {name}")
        code_history.append({"file": name, "old_sha256": old["source_sha256"][name],
                             "new_sha256": current_hash, "changed": current_hash != old["source_sha256"][name]})
    rows_all = np.asarray(old_split["validation_windows"], dtype=np.int64)
    rows = rows_all[sample_indices_for(rows_all, old["args"]["sample_per_task"])]
    if not np.array_equal(rows, np.asarray(pair["selected_rows"], dtype=np.int64)):
        raise ValueError("Selected generated-action observations differ from original paired evaluation")
    pairs, primary, mask = {}, [], None
    reference = None
    models = read_json(paired_run / "summary.json")["models"]
    action_reference = ROOT / "output/mini_heading_prior/angle_sensitivity_20260909"
    action_completion = read_json(action_reference / "evaluation_complete.json")
    for seed in [42, 43]:
        for arm in ARMS:
            run, manifest = (zero_run, new) if arm == "prior_zero" else (old_run, old)
            record, mask = primary_record(run, manifest, old_split, seed, arm, rows, mask)
            primary.append(record)
            if arm != "prior_zero":
                path = paired_run / f"seed{seed}_{MODES[arm]}.npz"
                data = archive_at(path)
                same_rows(data, rows)
                if data["epsilon"].shape != (4, 200, 16, 7) or data["angular_jitter"].shape != (4, 200):
                    raise ValueError("Original paired noise dimensions differ")
                if reference is None:
                    reference = data
                for key in ("epsilon", "angular_jitter", "valid", "episode", "frame", "task"):
                    if not np.array_equal(data[key], reference[key]):
                        raise ValueError(f"Original cross-arm/seed pairing differs: {key}")
                model = next(row for row in models if row["seed"] == seed and row["mode"] == MODES[arm])
                checkpoint = old_run / f"seed{seed}" / MODES[arm] / "ema.pt"
                if sha256(checkpoint) != model["checkpoint_sha256"]:
                    raise ValueError("Old checkpoint differs from the saved paired evaluation")
                if arm == "prior_kappa4":
                    action_path = action_reference / f"seed{seed}.npz"
                    if sha256(action_path) != action_completion["checks"][str(seed)]["archive_sha256"]:
                        raise ValueError("Saved legacy action-reference hash differs")
                    action_data = archive_at(action_path)
                    for key in ("epsilon", "angular_jitter", "episode", "frame", "task", "valid"):
                        if not np.array_equal(action_data[key], data[key]):
                            raise ValueError(f"Legacy action reference is not paired: {key}")
                    data["action_pred"] = action_data["predicted__action_pred"].copy()
                pairs[(seed, arm)] = data
        first = archive_at(old_run / f"seed{seed}/condition/eval_00000.npz")
        second = archive_at(zero_run / f"seed{seed}/condition_prior_xy/eval_00000.npz")
        for key in ("heading_mae_deg", "heading_cosine", "valid"):
            if not np.array_equal(first[key], second[key]):
                raise ValueError(f"Initial predictor observations/weights differ: seed{seed}, {key}")
    privileged = read_json(reference_run / "summary.json")
    if privileged["seeds"] != [42, 43] or privileged["windows"] != 200 or privileged["draws"] != 4:
        raise ValueError("Privileged reference dimensions differ")
    for seed in [42,43]:
        reference_npz = archive_at(reference_run / f"seed{seed}.npz")
        for key in ("epsilon", "episode", "frame", "task", "valid"):
            if not np.array_equal(reference_npz[key], pairs[(seed,"iid")][key]):
                raise ValueError(f"Privileged reference uses different observations or Gaussian draws: {key}")
    return {"old": old, "new": new, "pair": pair, "split": old_split, "rows": rows,
            "pairs": pairs, "primary": primary, "code_history": code_history, "comparison_plan": plan,
            "action_reference_files": [action_reference / "evaluation_complete.json", *[action_reference / f"seed{seed}.npz" for seed in [42,43]]],
            "privileged_reference": {"run": str(reference_run), "arm": "predicted_zero", "summary_sha256": sha256(reference_run / "summary.json"),
                                     "means": privileged["seed_means"]["predicted_zero"],
                                     "per_seed": [r for r in privileged["per_seed"] if r["arm"] == "predicted_zero"],
                                     "description": "Old κ=4-trained checkpoint, predicted condition, future-label GT source center, zero jitter at inference only. Privileged diagnostic; not retrained zero-jitter model."}}


@torch.no_grad()
def evaluate_model(policy, cache: dict, rows: np.ndarray, pair: dict, zero: bool,
                   batch_size: int) -> dict:
    """Score original Gaussian arrays directly; never resample or select a draw."""
    policy.eval()
    device = next(policy.parameters()).device
    draws, count = pair["epsilon"].shape[:2]
    arrays = {key: rows[:, column].copy() for key, column in (("episode", 0), ("frame", 1), ("task", 2))}
    arrays["epsilon"] = pair["epsilon"].copy()
    arrays["angular_jitter"] = np.zeros_like(pair["angular_jitter"]) if zero else pair["angular_jitter"].copy()
    arrays["target_action"] = cache["action"].numpy().copy()
    arrays["valid"] = np.empty(count, bool)
    for key in ("prediction_theta", "predicted_confidence", "target_theta", "predicted_heading_error_deg"):
        arrays[key] = np.empty(count, np.float32)
    arrays["prediction_direction"] = np.empty((count, 2), np.float32)
    for key in (*METRICS, *DIAGNOSTICS):
        arrays[key] = np.empty((draws, count), np.float32)
    arrays["source_active"] = np.empty((draws, count), bool)
    for key in ("source", "action_pred"):
        arrays[key] = np.empty((draws, count, 16, 7), np.float32)
    for draw in range(draws):
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = batch_at(cache, np.arange(start, stop), device)
            noise = torch.as_tensor(arrays["epsilon"][draw, start:stop], device=device)
            jitter = torch.as_tensor(arrays["angular_jitter"][draw, start:stop], device=device)
            result = policy.predict_action(batch["obs"], noise=noise, angular_jitter=jitter, return_source=True)
            generated, source, prediction = result["action_pred"], result["source"], result["prediction"]
            target = policy.heading_targets(batch["action"])
            error = (generated[:, :8] - batch["action"][:, :8]).square()
            resultant = generated[:, :16, :2].sum(1)
            direction = resultant / resultant.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            raw_source = policy.normalizer["action"].unnormalize(source)
            source_resultant = raw_source[:, :16, :2].sum(1)
            source_direction = source_resultant / source_resultant.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            values = {"raw_prefix8_mse": error.mean((1, 2)), "raw_prefix8_xyz_mse": error[..., :3].mean((1, 2)),
                      "raw_prefix8_rotation_mse": error[..., 3:6].mean((1, 2)), "raw_prefix8_gripper_mse": error[..., 6].mean(1),
                      "generated_heading_mae_deg": angular_error(direction, target["direction"]),
                      "source_normalized_ms": source.square().mean((1, 2)),
                      "iid_normalized_ms": (noise * policy.prior_noise_scale).square().mean((1, 2)),
                      "source_heading_to_prediction_deg": angular_error(source_direction, prediction["direction"]),
                      "source_active": result["source_active"], "source": source, "action_pred": generated}
            for key, value in values.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite generated metric/tensor: {key}")
                arrays[key][draw, start:stop] = value.cpu().numpy()
            static = {"valid": target["valid"], "prediction_theta": prediction["theta"],
                      "predicted_confidence": prediction["confidence"], "prediction_direction": prediction["direction"],
                      "target_theta": target["theta"], "predicted_heading_error_deg": angular_error(prediction["direction"], target["direction"])}
            for key, value in static.items():
                value = value.cpu().numpy()
                if draw == 0:
                    arrays[key][start:stop] = value
                elif not np.array_equal(arrays[key][start:stop], value):
                    raise ValueError(f"Observation-only prediction/label changed with random draw: {key}")
    if not np.array_equal(arrays["valid"], pair["valid"]):
        raise ValueError("Evaluated heading validity differs from saved paired observations")
    return arrays


def replay_checks(actual: dict, expected: dict) -> dict:
    checks = {}
    keys = (*METRICS, *DIAGNOSTICS, "source", "source_active", "valid", "epsilon", "angular_jitter")
    if "action_pred" in expected:
        keys += ("action_pred",)
    for key in keys:
        equal = bool(np.array_equal(actual[key], expected[key]))
        checks[key] = {"bit_exact": equal, "max_abs_difference": float(np.max(np.abs(actual[key].astype(float) - expected[key].astype(float))))}
        if not equal:
            raise ValueError(f"Backward-compatible old-control replay is not bit-exact: {key}")
    return checks


def averaged_metrics(data: dict) -> dict:
    per_draw = []
    for draw in range(data[METRICS[0]].shape[0]):
        per_draw.append({"draw": draw, **{key: float(data[key][draw, data["valid"] if key == "generated_heading_mae_deg" else slice(None)].astype(np.float64).mean()) for key in METRICS}})
    return {"per_draw": per_draw, "mean_over_draws": {key: float(np.mean([r[key] for r in per_draw])) for key in METRICS}}


def paired_comparisons(archives: dict, seeds: list[int], repetitions: int) -> dict:
    averaged = {pair: {**{"sample_" + key: data[key].astype(np.float64).mean(0) for key in METRICS},
                      **{"sample_" + key: data[key] for key in ("valid", "episode", "task")}}
                for pair, data in archives.items()}
    results = {}
    for control in ARMS[:2]:
        estimates = {}
        for index, key in enumerate(METRICS):
            value = paired_effect(averaged, seeds, "prior_zero", control, key, repetitions, np.random.default_rng(47031 + index))
            effects = list(value["per_seed_effect"].values())
            value["same_sign_across_fixed_seeds"] = bool(all(x <= 0 for x in effects) or all(x >= 0 for x in effects))
            estimates[key] = value
        results["zero_minus_" + control] = {"control": control, "treatment": "prior_zero", "metrics": estimates}
    return results


def interpretations(result: dict) -> list[str]:
    notes = []
    for comparison in result["comparisons"].values():
        control = comparison["control"]
        value = comparison["metrics"]["raw_prefix8_mse"]
        baseline = result["mean_over_training_seeds"][control]["raw_prefix8_mse"]
        reduction = -100. * value["effect_treatment_minus_control"] / baseline
        details = "; ".join(f"seed {seed}: {effect:+.6g}" for seed, effect in value["per_seed_effect"].items())
        low, high = value["ci95_episode_bootstrap"]
        notes.append(f"Retrained zero jitter versus {LABELS[control]}: mean action-MSE reduction {reduction:+.3f}%; paired difference {value['effect_treatment_minus_control']:+.6g}, conditional 95% episode interval [{low:+.6g}, {high:+.6g}]; {details}.")
        if not value["same_sign_across_fixed_seeds"]:
            notes.append("This comparison changes sign between the two training seeds; do not describe its mean as a consistent gain.")
    notes += ["This table compares three observation-only policies after equal training budgets. The future-label GT-source reference is separate and does not represent the new trained model.",
              "Primary one-noise endpoints and this four-draw check measure different fixed noise sets; differences between those estimates should be reported rather than selecting whichever is favorable.",
              "Native flow losses use source-dependent interpolation paths and targets and are not a common action-quality metric. Offline imitation errors do not establish a closed-loop SR improvement."]
    return notes


def plot_comparison(result: dict, out: Path, pdf: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#666666", "#0072B2", "#D55E00")
    records = {(r["seed"], r["arm"]): r for r in result["per_seed"]}
    keys = ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "generated_heading_mae_deg", "raw_prefix8_rotation_mse", "raw_prefix8_gripper_mse", "predictor_mae_full_validation_deg")
    with plt.rc_context({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for axis, key in zip(axes.flat, keys):
            means = []
            for i, arm in enumerate(ARMS):
                values = [records[(seed, arm)][key] if key.startswith("predictor_") else records[(seed, arm)]["mean_over_draws"][key] for seed in result["seeds"]]
                means.append(np.mean(values))
                axis.scatter(np.full(len(values), i) + np.linspace(-.06, .06, len(values)), values, c=colors[i], marker="o", s=30, alpha=.65)
                axis.hlines(means[-1], i - .22, i + .22, color=colors[i], linewidth=3)
            for j, seed in enumerate(result["seeds"]):
                vals = [records[(seed, arm)][key] if key.startswith("predictor_") else records[(seed, arm)]["mean_over_draws"][key] for arm in ARMS]
                axis.plot(np.arange(3) + (-.06 if j == 0 else .06), vals, color="#999999", linestyle="--", alpha=.35, linewidth=.8)
            axis.set_xticks(np.arange(3), ["IID", "XY prior, κ=4", "XY prior, zero"])
            axis.set_xlim(-.5, 2.5)
            axis.set_title(TITLES[key], fontsize=11)
            axis.grid(axis="y", alpha=.18)
        fig.suptitle("Predicted-heading policies: matched 5,000-step training", fontsize=15)
        fig.text(.5, .025, "Bars: means of two fixed seeds; dots: individual seeds. Action metrics average four paired draws on 200 windows.\nPredictor MAE uses the original full 1,000-window validation set. All three policies use observations only; no SR measured.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .95))
        fig.savefig(out / "comparison.png", dpi=200)
        if pdf:
            fig.savefig(out / "comparison.pdf")
        plt.close(fig)


def write_report(result: dict, output: Path) -> None:
    lines = ["# Retrained predicted-heading zero-jitter comparison", "",
             "Three deployable observation-only policies are compared after **5,000 updates per arm and training seed**. All share the same initialization protocol, 4,000 training windows, fixed Min–Max normalizer, shared jointly trained heading head, predicted heading condition, and training/evaluation budgets. "
             "The new arm uses a predicted-heading translation-XY source with zero angular dither during both training and inference. Source gradients remain detached; heading gradients continue through conditioning and auxiliary supervision.", "",
             "## Four-draw matched generated-action evaluation", "",
             "The same 200 held-out observations and four original saved Gaussian draws are reused. Old κ=4 controls receive their original saved angular jitter; the new source uses exact zero jitter. "
             "Each generated action sample is scored separately, then errors are averaged; no best draw is selected and actions are not averaged before scoring. "
             "Action MSE uses the first eight raw actions; generated heading uses the 16-action resultant on valid labels. Predictor MAE comes from the **full 1,000-window final validation**, not the 200-window generation subset.", "",
             "| Seed | Arm | Action MSE | XYZ MSE | Rotation MSE | Gripper MSE | Generated heading MAE (°) | Predictor MAE, full validation (°) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["per_seed"]:
        values = " | ".join(f"{row['mean_over_draws'][key]:.6g}" for key in METRICS)
        lines.append(f"| {row['seed']} | {LABELS[row['arm']]} | {values} | {row['predictor_mae_full_validation_deg']:.6g} |")
    for arm in ARMS:
        values = " | ".join(f"{result['mean_over_training_seeds'][arm][key]:.6g}" for key in METRICS)
        lines.append(f"| Mean | {LABELS[arm]} | {values} | {result['mean_over_training_seeds'][arm]['predictor_mae_full_validation_deg']:.6g} |")
    lines += ["", "## Interpretation", ""]
    for note in result["interpretation"]:
        lines += [note, ""]
    lines += ["## Paired episode-bootstrap comparisons", "", "Effects are retrained zero jitter minus the stated control; negative means lower error.", "",
              "| Control | Metric | Paired effect | 95% conditional episode interval |", "| --- | --- | ---: | --- |"]
    for comparison in result["comparisons"].values():
        for key, estimate in comparison["metrics"].items():
            low, high = estimate["ci95_episode_bootstrap"]
            lines.append(f"| {LABELS[comparison['control']]} | {TITLES[key]} | {estimate['effect_treatment_minus_control']:+.6g} | [{low:+.6g}, {high:+.6g}] |")
    lines += ["", f"{result['bootstrap']['draws']:,} paired bootstrap replicates resample entire held-out episodes within each task. Gaussian draws and training seeds are averaged first; the same episode multiplicities are used in both arms, with original eligible-window task weights fixed. "
              "**Intervals condition on these two trained checkpoints per arm and four saved Gaussian draws. They do not establish robust uncertainty over training seeds or success rates.**", "",
              "## Original single-noise final endpoints", "", "These are the original final-budget training-run evaluations, retained separately from the four-draw check. All values are read from saved final arrays and checked against each training summary.", "",
              "| Seed | Arm | Action MSE | XYZ MSE | Generated heading MAE (°) |", "| --- | --- | ---: | ---: | ---: |"]
    for row in result["primary_single_noise"]:
        values = " | ".join(f"{row['metrics'][key]:.6g}" for key in ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "generated_heading_mae_deg"))
        lines.append(f"| {row['seed']} | {LABELS[row['arm']]} | {values} |")
    for arm in ARMS:
        values = " | ".join(f"{np.mean([r['metrics'][key] for r in result['primary_single_noise'] if r['arm'] == arm]):.6g}" for key in ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "generated_heading_mae_deg"))
        lines.append(f"| Mean | {LABELS[arm]} | {values} |")
    ref = result["privileged_reference"]
    lines += ["", "## Separate privileged reference", "",
              "The prior factorial diagnostic used the **old κ=4-trained checkpoint**, predicted condition, future-label **GT source direction**, and zero jitter only at inference. It is not the newly trained zero-jitter policy and is unavailable at deployment.", "",
              "| Reference only | Action MSE | XYZ MSE | Generated heading MAE (°) |", "| --- | ---: | ---: | ---: |",
              f"| Old model + GT source + zero jitter | {ref['means']['raw_prefix8_mse']:.6g} | {ref['means']['raw_prefix8_xyz_mse']:.6g} | {ref['means']['generated_heading_mae_deg']:.6g} |", "",
              "This reference is deliberately excluded from the trained-arm comparisons and confidence intervals.", "",
              "## Compatibility, reproducibility, and limits", "",
              "Old source hashes differ because explicit zero-jitter configuration was added after the earlier run. This difference is recorded in the manifest instead of being hidden. "
              "Historical snapshots are checked against their hashes; current critical code must match the new training snapshot. Missing old `source_jitter` fields resolve to the original `vonmises` behavior. "
              "Both old control checkpoints in both seeds must reproduce the saved source tensors, all five action metrics, source diagnostics, validity, and random arrays **bit-exactly** under the current code. "
              "Old κ=4 generated action tensors are also checked against the previous predicted-source angle-sensitivity archive; the old IID archive has no generated action tensor reference. "
              "The new run must match all training arguments except output path, selected modes, and source-jitter configuration; full split metadata, normalizer tensors, and initial predictor diagnostics must match.", "",
              "Raw targets are loaded only for scoring and heading-label validity. They are never supplied to source construction or observation conditioning in the three trained arms. "
              "All model weights are frozen for this comparison. This is a short offline training experiment with two fixed seeds; no closed-loop SR is measured.", "",
              "![Matched training comparison](comparison.png)", "",
              "`manifest.json` and `evaluation_complete.json` contain source/checkpoint/input hashes and replay checks. `summary.json` retains per-draw/per-seed metrics and paired effects. NPZ archives preserve actions, raw targets, source tensors, effective jitter, Gaussian samples, predictions, and window identities for independent auditing. The initial executed evaluator remains in `source/`; any report regeneration saves its exact source separately in `report_source/`.", ""]
    (output / "report.md").write_text("\n".join(lines))


def summarize(output: Path, repetitions: int = 2000, pdf: bool = False) -> dict:
    manifest, completion = read_json(output / "manifest.json"), read_json(output / "evaluation_complete.json")
    if not completion.get("complete") or completion["arms"] != list(ARMS):
        raise ValueError("Cross-run evaluation incomplete")
    seeds = manifest["seeds"]
    archives = {}
    for seed in seeds:
        for arm in ARMS:
            path = output / f"seed{seed}_{arm}.npz"
            check = completion["checks"][f"{seed}/{arm}"]
            if sha256(path) != check["archive_sha256"]:
                raise ValueError("Comparison archive integrity failed")
            if arm != "prior_zero" and not all(r["bit_exact"] for r in check["control_replay"].values()):
                raise ValueError("Old control replay failed")
            archives[(seed, arm)] = archive_at(path)
    reference = archives[(seeds[0], ARMS[0])]
    for pair, data in archives.items():
        for key in ("episode", "frame", "task", "valid", "epsilon", "target_action"):
            if not np.array_equal(data[key], reference[key]):
                raise ValueError(f"Comparison archive pairing differs: {pair}, {key}")
        if data[METRICS[0]].shape != (manifest["draws"], manifest["windows"]):
            raise ValueError("Comparison dimensions differ from manifest")
        if pair[1] == "prior_zero" and np.any(data["angular_jitter"] != 0):
            raise ValueError("Retrained zero-jitter evaluation contains angular dither")
        if any(not np.isfinite(data[key]).all() for key in METRICS):
            raise ValueError("Nonfinite saved action metric")
    primary = {(r["seed"], r["arm"]): r for r in manifest["primary_single_noise"]}
    per_seed = [{"seed": seed, "arm": arm, **averaged_metrics(archives[(seed, arm)]),
                 "predictor_mae_full_validation_deg": primary[(seed, arm)]["predictor_mae_full_validation_deg"],
                 "full_validation_windows": primary[(seed, arm)]["validation_windows"],
                 "full_validation_valid_heading_windows": primary[(seed, arm)]["valid_heading_windows"],
                 "source_diagnostics": {key: float(archives[(seed, arm)][key].mean()) for key in (*DIAGNOSTICS, "source_active")}}
                for seed in seeds for arm in ARMS]
    means = {arm: {**{key: float(np.mean([r["mean_over_draws"][key] for r in per_seed if r["arm"] == arm])) for key in METRICS},
                   "predictor_mae_full_validation_deg": float(np.mean([r["predictor_mae_full_validation_deg"] for r in per_seed if r["arm"] == arm]))} for arm in ARMS}
    result = {"seeds": seeds, "draws": manifest["draws"], "windows": manifest["windows"], "arms": list(ARMS),
              "per_seed": per_seed, "mean_over_training_seeds": means, "comparisons": paired_comparisons(archives, seeds, repetitions),
              "primary_single_noise": manifest["primary_single_noise"], "privileged_reference": manifest["privileged_reference"],
              "bootstrap": {"draws": repetitions, "rng_seed_base": 47031, "unit": "whole validation episode within task", "conditions_on": "two fixed trained seeds and four saved latent streams"},
              "evaluation_completion": completion, "manifest_sha256": sha256(output / "manifest.json"),
              "report_source_sha256": sha256(Path(__file__)),
              "limitations": ["No SR evaluation", "Only two training seeds", "Fixed short training and cached windows", "GT-source reference is privileged and separate", "Native flow loss is not a common cross-prior quality metric"]}
    result["interpretation"] = interpretations(result)
    dest = output / "report_source" / Path(__file__).relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(__file__).read_bytes())
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, output)
    plot_comparison(result, output, pdf)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-run-dir", type=Path, default=ROOT / "output/mini_heading_prior/predicted_zero_5000_seed42_43")
    parser.add_argument("--old-run-dir", type=Path, default=ROOT / "output/mini_heading_prior/mini_5000_20260909")
    parser.add_argument("--paired-source-dir", type=Path, default=ROOT / "output/mini_heading_prior/sampling_robustness_20260909")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "output/mini_heading_prior/factorial_20260909")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if min(args.cpu_threads, args.bootstrap_draws) < 1:
        parser.error("CPU threads and bootstrap count must be positive")
    output = args.output_dir.resolve()
    if args.report_only:
        result = summarize(output, args.bootstrap_draws, args.pdf)
        print(json.dumps({"output_dir": str(output), "means": result["mean_over_training_seeds"]}, indent=2))
        return
    old_run, zero_run, paired_run, reference_run = [path.resolve() for path in (args.old_run_dir, args.zero_run_dir, args.paired_source_dir, args.reference_dir)]
    prepared = prepare(old_run, zero_run, paired_run, reference_run)
    output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), ROOT / "scripts/evaluate_mini_heading_prior_sampling.py", ROOT / "scripts/summarize_mini_heading_prior.py", ROOT / "scripts/summarize_mini_shared_heading.py", *[ROOT / name for name in CRITICAL_FILES]]
    input_files = [run / name for run in (old_run, zero_run) for name in ("manifest.json", "summary.json", "split.json", "normalizer.pt")]
    input_files += [paired_run / "manifest.json", paired_run / "summary.json", reference_run / "summary.json", zero_run / "comparison_plan.json", *prepared["action_reference_files"]]
    input_files += [paired_run / f"seed{seed}_{MODES[arm]}.npz" for seed in [42,43] for arm in ARMS[:2]]
    protocol = {"kind": "Matched trained policies: original IID, original predicted-heading κ=4 prior, retrained predicted-heading zero-jitter prior",
                "old_run": str(old_run), "zero_run": str(zero_run), "paired_source_run": str(paired_run),
                "arms": list(ARMS), "seeds": [42,43], "draws": 4, "windows": 200, "batch_size": prepared["pair"]["batch_size"],
                "training_steps_per_arm_seed": 5000, "selected_rows": prepared["rows"].tolist(),
                "heading_source": "predicted from observations in all deployable heading-prior arms; no GT intervention",
                "jitter": "iid and prior_kappa4 use original saved angular_jitter; prior_zero uses exact zeros during training and this evaluation",
                "pairing": "original saved epsilon reused across all arms and seeds; each sample scored before averaging, no best-draw selection",
                "legacy_compatibility": "Historical source hashes intentionally differ after explicit source_jitter support; old missing fields default to vonmises. Both old arms must reproduce saved sources and metrics bit-exactly before results are accepted.",
                "training_argument_exceptions": ["output_dir", "modes", "source_jitter"], "code_history": prepared["code_history"],
                "comparison_plan": prepared["comparison_plan"], "comparison_plan_sha256": sha256(zero_run / "comparison_plan.json"),
                "legacy_action_reference": "Old kappa4 generated actions are additionally checked bit-exactly against the original predicted-source arm from the angle-sensitivity diagnostic; old IID has no saved action tensor reference.",
                "primary_single_noise": prepared["primary"], "privileged_reference": prepared["privileged_reference"],
                "source_sha256": {str(path.relative_to(ROOT)):sha256(path) for path in source_files},
                "input_sha256": {str(path):sha256(path) for path in input_files},
                "device":args.device,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch_version":str(torch.__version__),
                "limitations": ["offline imitation metrics, no SR", "conditional on two trained seeds and four fixed Gaussian draws", "short fixed-cache training", "GT-source reference is separate and unavailable at deployment"]}
    (output / "manifest.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for path in source_files:
        dest = output / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache = load_selected_cache(prepared["old"], prepared["split"], prepared["rows"])
    checks, models = {}, []
    started = time.monotonic()
    # Replay all controls first. No zero-policy comparison is accepted after a failed replay.
    for arm in ARMS:
        for seed in [42,43]:
            zero = arm == "prior_zero"
            run, manifest = (zero_run, prepared["new"]) if zero else (old_run, prepared["old"])
            policy, config = load_policy(run, manifest, prepared["split"], seed, MODES[arm], args.device)
            expected_jitter = "zero" if zero else "vonmises"
            if policy.source_jitter != expected_jitter or config["source_jitter"] != expected_jitter:
                raise ValueError("Restored checkpoint jitter configuration differs")
            pair = prepared["pairs"][(seed,"prior_kappa4" if zero else arm)]
            arrays = evaluate_model(policy, cache, prepared["rows"], pair, zero, protocol["batch_size"])
            replay = None if zero else replay_checks(arrays, pair)
            path = output / f"seed{seed}_{arm}.npz"
            np.savez_compressed(path, **arrays)
            checks[f"{seed}/{arm}"] = {"archive_sha256":sha256(path),"checkpoint_sha256":config["checkpoint_sha256"],
                                       "control_replay":replay,"valid_windows":int(arrays["valid"].sum()),
                                       "source_active_fraction":float(arrays["source_active"].mean()),
                                       "max_abs_angular_jitter":float(np.max(np.abs(arrays["angular_jitter"])))}
            models.append({"arm":arm, **config})
            print(json.dumps({"completed_seed":seed,"arm":arm,"metrics":averaged_metrics(arrays)["mean_over_draws"],"control_exact":replay is not None}),flush=True)
            del policy
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    completion = {"complete":True,"elapsed_seconds":time.monotonic()-started,"arms":list(ARMS),"seeds":[42,43],
                  "draws":4,"windows":200,"checks":checks,"models":models}
    (output / "evaluation_complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    result = summarize(output,args.bootstrap_draws,args.pdf)
    print(json.dumps({"output_dir":str(output),"elapsed_seconds":completion["elapsed_seconds"],"means":result["mean_over_training_seeds"],"interpretation":result["interpretation"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
