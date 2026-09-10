#!/usr/bin/env python3
"""Validate and report the completed single-checkpoint LIBERO-10 assessment.

The reserved 400 episodes are initial-state indices 10..49 from the same final
500-episode run. This script never launches simulations or selects checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import textwrap

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from oat.env_runner.libero_official_eval import file_sha256, make_episode_plan, summarize_results

EXPECTED_SHA256 = "cbf4a130f218b2f87a67931bf066ed26915bc77ea7770ab392d9b9bac6ce50f5"
IDENTITY_KEYS = ("episode_id", "task_index", "task_name", "init_index", "env_seed", "policy_seed")
PILOT_NOTE = ("Both epoch-15 U-Net variants achieved 82/100 (82%) in development pilots. "
              "The Gaussian variant was selected for its full-rank prior; these pilots do not "
              "establish Gaussian superiority.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_plan(manifest):
    require(manifest.get("suite") == "libero_10", "Expected the LIBERO-10 suite")
    require(manifest.get("n_per_task") == 50 and manifest.get("init_start") == 0,
            "Final evaluation must plan 50 initial states per task, indices 0..49")
    require(type(manifest.get("seed")) is int, "Manifest must contain the final evaluation seed")
    plan = manifest["episodes"]
    require(len(plan) == 500, "Final plan must contain exactly 500 episodes")
    names = {}
    for episode in plan:
        index = episode["task_index"]
        require(type(index) is int and 0 <= index < 10, "Invalid task index in plan")
        require(index not in names or names[index] == episode["task_name"], "Inconsistent task index/name mapping")
        names[index] = episode["task_name"]
    require(set(names) == set(range(10)) and len(set(names.values())) == 10, "Expected ten distinct tasks")
    ordered_names = [names[index] for index in range(10)]
    canonical = make_episode_plan(ordered_names, 50, manifest["seed"], 0)
    expected = {episode["episode_id"]: episode for episode in canonical}
    seen = set()
    for episode in plan:
        identity = episode["episode_id"]
        require(identity in expected and identity not in seen, f"Unknown or duplicate planned episode: {identity}")
        seen.add(identity)
        for key in IDENTITY_KEYS:
            require(episode.get(key) == expected[identity][key], f"Planned {key} differs from canonical identifier/seed: {identity}")
        require(re.fullmatch(r"[0-9a-f]{64}", episode.get("official_state_sha256", "")) is not None,
                f"Missing official initial-state hash: {identity}")
    return ordered_names


def validate_results(manifest, results, saved_summary):
    names = validate_plan(manifest)
    require(saved_summary.get("complete") is True, "Final evaluation is incomplete; no report will be written")
    require(saved_summary.get("checkpoint_sha256") == manifest["checkpoint_sha256"], "Summary checkpoint hash mismatch")
    require(saved_summary.get("weights") == manifest.get("weights"), "Summary weight selection mismatch")
    require(len(results) == 500, "Episode log must contain exactly 500 results")
    expected = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    seen = set()
    for result in results:
        identity = result["episode_id"]
        require(identity in expected and identity not in seen, f"Unknown or duplicate result: {identity}")
        seen.add(identity)
        require("error" in result and result["error"] is None, f"Episode has an evaluation error: {identity}")
        require(type(result.get("success")) is bool, f"Episode success must be Boolean: {identity}")
        for key in (*IDENTITY_KEYS, "official_state_sha256"):
            require(result.get(key) == expected[identity].get(key), f"Result {key} mismatch: {identity}")
    require(seen == set(expected), "Some planned episodes have no result")
    full = summarize_results(manifest["episodes"], results)
    require(full["complete"] and full["error_episodes"] == 0, "Recomputed evaluation is incomplete or contains errors")
    for key, value in full.items():
        require(saved_summary.get(key) == value, f"Saved summary disagrees with recomputed {key}")
    reserved_plan = [episode for episode in manifest["episodes"] if 10 <= episode["init_index"] <= 49]
    reserved_results = [result for result in results if 10 <= result["init_index"] <= 49]
    reserved = summarize_results(reserved_plan, reserved_results)
    require(reserved["complete"] and reserved["completed_episodes"] == 400, "Reserved subset must have all 400 episodes")
    return names, full, reserved


def validate_method(config):
    required = {"_target_": "oat.policy.flow_policy_heading_gaussian.HeadingGaussianFlowPolicy",
                "backbone_type": "unet", "heading_mode": "condition", "source_mode": "heading",
                "horizon": 16, "n_action_steps": 8, "n_obs_steps": 2, "num_inference_steps": 10}
    for key, value in required.items():
        require(config.get(key) == value, f"Selected policy does not match reported method: {key}")
    require(config.get("prior_parallel_std", 0) > 0 and config.get("prior_perpendicular_std", 0) > 0,
            "Heading Gaussian must have nonzero variance in both planar directions")
    permitted_obs = {"agentview_rgb", "robot0_eye_in_hand_rgb", "robot0_eef_pos",
                     "robot0_eef_quat", "robot0_gripper_qpos", "task_uid"}
    require(set(config["shape_meta"]["obs"]) == permitted_obs, "Unexpected policy observations; review geometry/input claims")


def build_report(final_dir, expected_sha256=EXPECTED_SHA256):
    final_dir = Path(final_dir).resolve()
    evaluation = final_dir / "eval_500"
    paths = {name: evaluation / name for name in ("manifest.json", "episodes.jsonl", "summary.json")}
    content = {name: path.read_bytes() for name, path in paths.items()}
    manifest = json.loads(content["manifest.json"])
    summary = json.loads(content["summary.json"])
    require(summary.get("complete") is True, "Final evaluation is incomplete; no report will be written")
    results = [json.loads(line) for line in content["episodes.jsonl"].splitlines() if line.strip()]
    names, full, reserved = validate_results(manifest, results, summary)
    validate_method(manifest["policy_config"])
    require(manifest.get("weights") == "ema_model", "Expected the selected EMA policy")
    epoch = manifest.get("training_position", {}).get("epoch")
    require(type(epoch) is int and epoch >= 0, "Missing training epoch in manifest")
    selected, snapshot = final_dir / "policy.ckpt", evaluation / "policy.ckpt"
    require(Path(manifest["source_checkpoint"]).resolve() == selected.resolve(), "Evaluation source is not the selected final policy")
    require(Path(manifest["checkpoint"]).resolve() == snapshot.resolve(), "Evaluation snapshot path mismatch")
    for path in (selected, snapshot):
        require(path.is_file() and file_sha256(path) == expected_sha256, f"Selected/exported checkpoint hash mismatch: {path}")
    require(manifest["checkpoint_sha256"] == expected_sha256, "Manifest differs from the selected export hash")
    input_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    for name, path in paths.items():
        require(file_sha256(path) == input_hashes[name], f"Evaluation records changed while reporting: {name}")
    selection_path = final_dir / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection is not None:
        require(selection.get("checkpoint_sha256") == expected_sha256, "Selection record checkpoint mismatch")
        require(selection.get("final_seed") == manifest["seed"], "Selection/final evaluation seed mismatch")
        require(selection.get("reserved_initial_state_indices") == [10, 49], "Selection record has a different reserved subset")
    config = manifest["policy_config"]
    max_steps, settle_steps = manifest.get("max_episode_steps"), manifest.get("settle_steps")
    require(type(max_steps) is int and max_steps > 0, "Missing positive max_episode_steps in manifest")
    require(type(settle_steps) is int and settle_steps >= 0, "Missing nonnegative settle_steps in manifest")
    vision = config["obs_encoder"]["vision_encoder"]
    image_shapes = {name: meta["shape"] for name, meta in config["shape_meta"]["obs"].items()
                    if meta.get("type") == "rgb"}
    require(image_shapes and vision.get("crop_shape") and vision.get("eval_fixed_crop") is True,
            "Expected documented RGB dimensions and fixed evaluation crop")
    prior = {"heading_mean": config.get("prior_heading_mean", 0.5),
             "parallel_std": config["prior_parallel_std"],
             "perpendicular_std": config["prior_perpendicular_std"],
             "noise_scale": config.get("prior_noise_scale", 1.0),
             "validity_threshold": config.get("min_source_confidence", 0.5),
             "raw_xy_formula": "u = r*s*((sigma_parallel*epsilon_x + mu)*d + sigma_perpendicular*epsilon_y*d_perp)",
             "normalization": "x0_xy = a*u + b; r = sqrt((a_x^(-2) + a_y^(-2))/2)",
             "definitions": "epsilon ~ N(0,I); d=(cos(theta),sin(theta)); d_perp=(-sin(theta),cos(theta)); a,b are checkpoint action-normalizer scale/offset; s is noise_scale",
             "other_channels": "x0_j = s*epsilon_j for j >= 2",
             "fallback": "Use IID s*epsilon when heading/validity is nonfinite or validity falls below the threshold",
             "training": "Heading is predicted from observations; source-heading inputs are detached during flow training"}
    command = [sys.executable, "scripts/eval_heading_policy.py", "--checkpoint", str(selected),
               "--output", str(final_dir / "reproduce_eval_500"), "--suite", manifest["suite"],
               "--n-per-task", "50", "--init-start", "0", "--seed", str(manifest["seed"]),
               "--workers", str(manifest.get("workers", 4)), "--device", "cuda:0",
               "--max-episode-steps", str(max_steps), "--settle-steps", str(settle_steps)]
    reproduction = {"working_directory": str(ROOT_DIR),
                    "command": shlex.join(["env", f"PYTHONPATH={ROOT_DIR}",
                                f"CUDA_VISIBLE_DEVICES={manifest.get('cuda_visible_devices') or '0'}", *command]),
                    "loading_python": "from oat.env_runner.libero_official_eval import load_policy\n"
                        + f"policy, cfg, weights = load_policy({str(selected)!r}, 'cuda:0')",
                    "output_note": "Use a new output directory; the evaluator refuses to overwrite an existing run"}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(selected), "checkpoint_sha256": expected_sha256,
        "evaluation_snapshot": str(snapshot), "training_epoch": epoch,
        "training_position": manifest["training_position"], "weights": manifest["weights"],
        "suite": manifest["suite"], "final_seed": manifest["seed"], "task_names": names,
        "full_500": full, "reserved_400": reserved,
        "evaluation_protocol": {"max_episode_steps": max_steps, "settle_steps": settle_steps,
                                "step_accounting": "Policy-action steps; settling steps occur before this counter starts",
                                "settle_action": [0, 0, 0, 0, 0, 0, -1],
                                "observation_steps": config["n_obs_steps"], "action_horizon": config["horizon"],
                                "executed_action_prefix": config["n_action_steps"]},
        "image_transforms": {"input_shapes_hwc": image_shapes, "crop_shape": vision["crop_shape"],
                             "training": "Random crop", "evaluation": "Fixed center crop",
                             "encoder": vision["_target_"], "resize": None},
        "gaussian_prior": prior, "reproduction": reproduction,
        "reserved_initial_state_indices": [10, 49],
        "reserved_subset_origin": "Selected from the same final 500-episode run, with no reruns or policy changes",
        "development_pilots": PILOT_NOTE, "selection_record": selection,
        "method": {"conditioning": "Image/state features and shared observation-predicted heading",
                   "source": "Full-rank heading-conditioned Gaussian in translation XY",
                   "flow": "U-Net velocity field trained with straight flow matching",
                   "horizon": 16, "executed_actions": 8, "observation_steps": 2,
                   "sampler": "Euler", "inference_steps": 10,
                   "privileged_geometry": False, "same_policy_all_tasks": True},
        "integrity": {"recomputed_summary_matches": True, "all_episode_identifiers_and_seeds_match": True,
                      "unique_initial_states_per_task": 50, "missing_episodes": 0, "error_episodes": 0,
                      "input_sha256": input_hashes},
        "interpretation": "LIBERO-10 verification only; cross-benchmark generalization and Gaussian superiority are not established",
    }


def markdown_report(report):
    full, reserved = report["full_500"], report["reserved_400"]
    def score(summary):
        return f"{summary['mean_success_rate']:.1%} ({summary['successes']}/{summary['completed_episodes']})"
    protocol, images, prior = report["evaluation_protocol"], report["image_transforms"], report["gaussian_prior"]
    lines = [
        f"LIBERO-10 success: **{score(full)}** on all 500 episodes; **{score(reserved)}** on the reserved 400 episodes.", "",
        "The reserved result uses initial-state indices 10–49 from the same final run. All ten tasks use one fixed policy checkpoint. "
        "The full result includes indices 0–9 used in development pilots, evaluated here with the final-run seeds.", "",
        "| Task | Full 500 | Reserved 400 |", "|---|---:|---:|",
    ]
    for task in report["task_names"]:
        def task_score(summary):
            item = summary["per_task"][task]
            return f"{item['success_rate']:.1%} ({item['successes']}/{item['completed']})"
        lines.append(f"| {task.replace('_', ' ')} | {task_score(full)} | {task_score(reserved)} |")
    lines += ["", f"Checkpoint: [policy.ckpt](policy.ckpt), training epoch {report['training_epoch']}, {report['weights']} weights. "
              f"Final evaluation seed: {report['final_seed']}.", "",
              f"SHA-256: `{report['checkpoint_sha256']}`", "",
              "The policy combines image/state features with a shared predicted heading, used for both model conditioning and a full-rank Gaussian noise source. "
              "A U-Net predicts the straight-flow velocity field over 16 actions using two observations; inference uses ten Euler steps and executes eight actions before replanning. "
              "Policy inputs contain no privileged object geometry.", "",
              f"Evaluation caps each episode at **{protocol['max_episode_steps']} policy-action steps**, following **{protocol['settle_steps']} settling steps** "
              "with action `[0, 0, 0, 0, 0, 0, -1]`. Settling is outside the policy-action counter. "
              "These episode limits are part of the reported protocol.", "",
              f"RGB inputs (H×W×C): {images['input_shapes_hwc']}. Training uses random {images['crop_shape'][0]}×{images['crop_shape'][1]} crops; "
              f"evaluation uses a fixed {images['crop_shape'][0]}×{images['crop_shape'][1]} center crop, with no resizing.", "",
              "For independent standard Gaussian noise ε and predicted directions d=(cos θ, sin θ), d⊥=(-sin θ, cos θ), the raw XY source is "
              "`u = r s [(σ∥ εx + μ) d + σ⊥ εy d⊥]`. The normalized source is `x0,xy = a u + b`, where "
              "`r = sqrt((a_x^(-2) + a_y^(-2))/2)` and a,b come from the checkpoint action normalizer. Other channels use `s ε`.",
              f"The selected parameters are μ={prior['heading_mean']:g}, σ∥={prior['parallel_std']:g}, σ⊥={prior['perpendicular_std']:g}, s={prior['noise_scale']:g}. "
              f"The source falls back to IID noise below heading-validity probability {prior['validity_threshold']:g}. "
              "Validity denotes whether a heading is defined, not angular accuracy. Source-heading inputs are detached during training.", "",
              "Load the selected checkpoint in the repository environment:", "", "```python", report['reproduction']['loading_python'], "```", "",
              f"Reproduce the final protocol from `{report['reproduction']['working_directory']}` using a new output directory:", "",
              "```bash", report['reproduction']['command'], "```", "", PILOT_NOTE, "",
              "All 500 episode identifiers, task mappings, initial-state indices, recorded official-state hashes and environment/policy seeds match the plan. "
              "Counts were recomputed from episode records; no episodes are missing or report evaluation errors.", "",
              "This result verifies LIBERO-10 performance. Cross-benchmark generalization has not been measured. "
              "Episode-level Wilson intervals in report.json do not include model-selection or training-seed uncertainty.", ""]
    return "\n".join(lines)


def plot_success(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = report["task_names"]
    figure, axis = plt.subplots(figsize=(15, 12))
    for offset, key, label, color in ((-.18, "full_500", "Full 500", "#2563eb"),
                                      (.18, "reserved_400", "Reserved 400", "#d97706")):
        items = [report[key]["per_task"][name] for name in names]
        bars = axis.barh([index + offset for index in range(len(names))],
                         [100 * item["success_rate"] for item in items], height=.32, label=label, color=color)
        axis.bar_label(bars, labels=[f"{item['successes']}/{item['completed']}" for item in items], padding=3, fontsize=8)
    axis.set_yticks(range(len(names)), [textwrap.fill(name.replace('_', ' '), 56) for name in names], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 110)
    axis.set_xticks(range(0, 101, 20))
    axis.set_xlabel("Success rate (%)")
    axis.set_title(f"LIBERO-10 — one checkpoint, epoch {report['training_epoch']}")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=.2)
    figure.tight_layout()
    temporary = path.with_suffix(".tmp")
    figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=ROOT_DIR / "output/heading_goal/final")
    parser.add_argument("--plot", action="store_true", help="also write success_by_task.png using matplotlib")
    args = parser.parse_args()
    try:
        report = build_report(args.final_dir)
        outputs = {"report.json": json.dumps(report, indent=2, sort_keys=True) + "\n", "RESULTS.md": markdown_report(report)}
        for name, text in outputs.items():
            path = args.final_dir / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text)
            temporary.replace(path)
        if args.plot:
            plot_success(report, args.final_dir / "success_by_task.png")
        print(f"Validated report: {args.final_dir / 'RESULTS.md'}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Report refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
