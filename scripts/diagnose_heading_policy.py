#!/usr/bin/env python3
"""Offline held-out diagnostics for one observation-conditioned heading flow policy.

No parameter, normalizer, or heading statistic is fitted. Future demonstration
actions are used only AFTER inference for scoring. These metrics are diagnostics,
not estimates of simulator success rate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from oat.env_runner.libero_official_eval import (
    array_sha256, file_sha256, load_policy, stable_seed, verify_repository_imports,
)


def select_validation_windows(dataset, validation, windows_per_task, selection_seed, expected_validation_mask=None):
    """Select through scalar replay metadata; never decode/sample image windows."""
    if not 1 <= windows_per_task <= 128:
        raise ValueError("windows_per_task must be between 1 and 128")
    if dataset.replay_buffer is not validation.replay_buffer:
        raise ValueError("validation must share the training dataset's loaded replay buffer")
    replay = dataset.replay_buffer
    ends = np.asarray(replay.episode_ends[:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    task_ids = np.asarray(replay["task_uid"][:]).reshape(int(ends[-1]), -1)
    if task_ids.shape[1] != 1:
        raise ValueError("task_uid must be scalar per replay frame")
    task_ids = task_ids[:, 0]
    if not np.isfinite(task_ids).all() or not np.equal(task_ids, task_ids.astype(np.int64)).all():
        raise ValueError("task_uid must contain finite integer identifiers")
    task_ids = task_ids.astype(np.int64)
    episode_tasks = task_ids[starts]
    if not np.array_equal(task_ids, np.repeat(episode_tasks, ends - starts)):
        raise ValueError("each episode must contain one consistent task_uid")
    indices = np.asarray(validation.seq_sampler.indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("checkpoint dataset split has no validation windows")
    episode_ids = np.searchsorted(ends, indices[:, 0], side="right")
    train_mask = np.asarray(dataset.train_mask, dtype=bool)
    if np.any(train_mask[episode_ids]):
        raise ValueError("validation sampler includes training episodes")
    fixed_holdout = expected_validation_mask
    if fixed_holdout is None:
        fixed_holdout = getattr(dataset, "_validation_episode_mask", None)
    if fixed_holdout is not None and not np.all(np.asarray(fixed_holdout)[episode_ids]):
        raise ValueError("validation sampler contains episodes outside its fixed holdout")
    window_tasks = episode_tasks[episode_ids]
    selected = []
    for task_uid in sorted(set(episode_tasks.tolist())):
        candidates = np.flatnonzero(window_tasks == task_uid)
        if len(candidates) < windows_per_task:
            raise ValueError(f"task_uid={task_uid} has only {len(candidates)} held-out windows; requested {windows_per_task}")
        rng = np.random.default_rng(stable_seed(selection_seed, "validation_selection", int(task_uid)))
        # A permutation gives nested selections when increasing 64 -> 128.
        chosen = rng.permutation(candidates)[:windows_per_task]
        for rank, window_index in enumerate(chosen):
            episode_index = int(episode_ids[window_index])
            row = indices[window_index]
            first_action = int(row[0] + validation.pad_before - row[2])
            selected.append({
                "validation_index": int(window_index), "task_uid": int(task_uid),
                "episode_index": episode_index, "selection_rank_within_task": rank,
                "sequence_indices": row.tolist(),
                "first_action_replay_index": first_action,
                "first_action_episode_index": first_action - int(starts[episode_index]),
                "episode_start": int(starts[episode_index]), "episode_end": int(ends[episode_index]),
            })
    return selected, {
        "episode_ends": ends.tolist(),
        "train_episode_indices": np.flatnonzero(train_mask).tolist(),
        "validation_episode_indices": sorted(set(episode_ids.tolist())),
        "validation_sampler_indices_sha256": array_sha256(indices),
        "task_uid_frames_sha256": array_sha256(task_ids),
        "available_windows_per_task": {str(int(task)): int(np.sum(window_tasks == task)) for task in np.unique(window_tasks)},
        "selection_uses_images": False,
    }


def window_noise_seed(base_seed, window):
    return stable_seed(base_seed, "noise", window["task_uid"], window["episode_index"], window["first_action_episode_index"])


def explicit_noise(policy, windows, base_seed):
    import torch
    samples = []
    jitters = []
    for window in windows:
        seed = window_noise_seed(base_seed, window)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        samples.append(torch.randn(policy.horizon, policy.action_dim, generator=generator))
        if getattr(policy, "source_jitter", "zero") == "vonmises":
            rng = np.random.default_rng(stable_seed(seed, "angular_jitter"))
            jitters.append(float(rng.vonmises(0., policy.source_kappa)))
        else:
            jitters.append(0.)
    return (torch.stack(samples).to(device=policy.device, dtype=policy.dtype),
            torch.tensor(jitters, device=policy.device, dtype=policy.dtype))


def raw_heading(actions, horizon, xy_rms, minimum_confidence):
    import torch
    resultant = actions[:, :horizon, :2].float().sum(1)
    norm = resultant.norm(dim=-1)
    theta = torch.atan2(resultant[:, 1], resultant[:, 0])
    confidence = norm / (float(xy_rms) * math.sqrt(horizon))
    return {"theta": theta, "direction": torch.stack((theta.cos(), theta.sin()), -1),
            "confidence": confidence, "valid": confidence >= minimum_confidence}


def angle_error_deg(a, b):
    import torch
    return torch.remainder(a - b + math.pi, 2 * math.pi).sub(math.pi).abs() * (180 / math.pi)


BINARY_SIGN_COUNT_METRICS = frozenset({
    "prefix_binary_sign_confusion_counts", "prefix_binary_sign_disagreement_count",
    "prefix_binary_sign_sample_count",
})


def parse_binary_action_indices(value):
    """Parse explicit sign-controlled action dimensions, preserving their order."""
    if not value.strip():
        return ()
    try:
        indices = tuple(int(part.strip()) for part in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError("binary action indices must be comma-separated integers") from error
    if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("binary action indices must be unique and nonnegative")
    return indices


def validate_binary_action_indices(indices, action_dim):
    indices = tuple(indices)
    if (any(isinstance(index, bool) or not isinstance(index, (int, np.integer))
            or not 0 <= index < action_dim for index in indices)
            or len(indices) != len(set(indices))):
        raise ValueError("binary action indices must be unique integers within the action dimensions")
    return indices


def binary_sign_metrics(generated, target_actions, indices):
    """Count sign commands; rows=true, columns=predicted, labels=[-1,0,+1].

    Both arrays must already contain only the execution prefix. Exact zero is
    a distinct sign command, not silently coerced into an open/close category.
    Counts include each sampled action separately, including repeated noise draws.
    """
    import torch
    predicted = generated[..., list(indices)].sign().to(torch.int64)
    target = target_actions[..., list(indices)].sign().to(torch.int64)
    disagreement = predicted.ne(target)
    rows = []
    for i in range(generated.shape[0]):
        confusion = []
        for dimension in range(len(indices)):
            codes = 3 * (target[i, :, dimension] + 1) + predicted[i, :, dimension] + 1
            confusion.append(torch.bincount(codes, minlength=9).reshape(3, 3).cpu().tolist())
        rows.append({
            "prefix_binary_sign_disagreement_rate": disagreement[i].float().mean(0).cpu().tolist(),
            "prefix_binary_sign_disagreement_count": disagreement[i].sum(0).cpu().tolist(),
            "prefix_binary_sign_sample_count": [generated.shape[1]] * len(indices),
            "prefix_binary_sign_confusion_counts": confusion,
        })
    return rows


def score_predictions(policy, result, target_actions, binary_action_indices=()):
    """Compute labels/metrics after action generation; no label reaches inference."""
    import torch
    import torch.nn.functional as F
    generated = result["action_pred"].float()
    target_actions = target_actions.float()
    if generated.shape != target_actions.shape or not torch.isfinite(generated).all():
        raise ValueError("generated/target action shapes must match and generated values must be finite")
    binary_action_indices = validate_binary_action_indices(binary_action_indices, generated.shape[-1])
    if binary_action_indices and not torch.isfinite(target_actions).all():
        raise ValueError("binary sign targets must be finite")
    prefix = int(policy.n_action_steps)
    binary_rows = binary_sign_metrics(generated[:, :prefix], target_actions[:, :prefix], binary_action_indices) \
        if binary_action_indices else None
    squared = (generated - target_actions).square()
    target = policy.heading_targets(target_actions)
    gen_target = policy.heading_targets(generated)
    prediction = result["prediction"]
    valid = target["valid"].bool()
    gen_valid = gen_target["valid"].bool()
    confidence = prediction["confidence"].float()
    predicted_valid = confidence >= float(policy.min_source_confidence)
    heading_cosine = (prediction["direction"] * target["direction"]).sum(-1).clamp(-1, 1)
    pred_error = angle_error_deg(prediction["theta"], target["theta"])
    gen_error = angle_error_deg(gen_target["theta"], target["theta"])
    gen_cosine = (gen_target["direction"] * target["direction"]).sum(-1).clamp(-1, 1)
    bce = F.binary_cross_entropy(confidence.clamp(1e-7, 1-1e-7), valid.float(), reduction="none")
    target_prefix = raw_heading(target_actions, prefix, policy.heading_xy_rms, policy.min_target_confidence)
    generated_prefix = raw_heading(generated, prefix, policy.heading_xy_rms, policy.min_target_confidence)
    prefix_error = angle_error_deg(generated_prefix["theta"], target_prefix["theta"])
    rows = []
    for i in range(generated.shape[0]):
        both_valid = bool(valid[i] and gen_valid[i])
        prefix_both_valid = bool(target_prefix["valid"][i] and generated_prefix["valid"][i])
        rows.append({
            "prefix_channel_mse": squared[i, :prefix].mean(0).cpu().tolist(),
            "prefix_mse": float(squared[i, :prefix].mean()),
            "complete_chunk_mse": float(squared[i].mean()),
            "predicted_heading_mae_deg": float(pred_error[i]) if valid[i] else None,
            "predicted_heading_cosine": float(heading_cosine[i]) if valid[i] else None,
            "target_heading_valid_fraction": float(valid[i]),
            "predicted_valid_fraction": float(predicted_valid[i]),
            "predicted_validity_confidence": float(confidence[i]),
            "validity_bce": float(bce[i]),
            "validity_accuracy": float(predicted_valid[i] == valid[i]),
            "validity_true_positive_fraction": float(predicted_valid[i] and valid[i]),
            "validity_false_positive_fraction": float(predicted_valid[i] and not valid[i]),
            "generated_heading_mae_deg_both_valid": float(gen_error[i]) if both_valid else None,
            "generated_heading_cosine_both_valid": float(gen_cosine[i]) if both_valid else None,
            "generated_heading_valid_fraction": float(gen_valid[i]),
            "generated_heading_valid_on_target_valid_fraction": float(gen_valid[i]) if valid[i] else None,
            "generated_prefix_heading_mae_deg_both_valid": float(prefix_error[i]) if prefix_both_valid else None,
            "generated_prefix_heading_valid_fraction": float(generated_prefix["valid"][i]),
            "prior_active_fraction": float(result["source_active"][i]),
        })
        if binary_rows is not None:
            rows[-1].update(binary_rows[i])
    return rows


def summarize_metric_rows(rows):
    if not rows:
        raise ValueError("cannot aggregate empty diagnostics")
    metric_keys = rows[0]["metrics"].keys()
    averages = {}
    counts = {}
    for key in metric_keys:
        values = [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
        counts[key] = len(values)
        aggregate = np.sum if key in BINARY_SIGN_COUNT_METRICS else np.mean
        averages[key] = aggregate(values, axis=0).tolist() if values else None
    return {"metrics": averages, "metric_sample_counts": counts,
            "noise_draw_count": len(rows),
            "unique_window_count": len({row["validation_index"] for row in rows})}


def diagnose_batches(policy, validation, windows, batch_size, noise_seeds, binary_action_indices=()):
    import torch
    from torch.utils.data._utils.collate import default_collate
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    binary_action_indices = validate_binary_action_indices(binary_action_indices, policy.action_dim)
    rows = []
    for start in range(0, len(windows), batch_size):
        this_windows = windows[start:start + batch_size]
        # Decode each selected image window only once, even for multiple seeds.
        batch = default_collate([validation[row["validation_index"]] for row in this_windows])
        obs = {key: value.to(policy.device, dtype=policy.dtype) if torch.is_tensor(value) else value
               for key, value in batch["obs"].items()}
        for base_seed in noise_seeds:
            noise, jitter = explicit_noise(policy, this_windows, base_seed)
            with torch.inference_mode():
                policy.reset()
                # No target actions/labels, episode IDs, or task geometry enter this call.
                result = policy.predict_action(obs, noise=noise, angular_jitter=jitter, return_source=True)
                target = batch["action"].to(policy.device)
                metrics = score_predictions(policy, result, target, binary_action_indices)
            rows.extend({**window, "base_noise_seed": int(base_seed),
                         "window_noise_seed": window_noise_seed(base_seed, window), "metrics": metric}
                        for window, metric in zip(this_windows, metrics))
        print(f"Diagnosed {min(start + batch_size, len(windows))}/{len(windows)} held-out windows", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--windows-per-task", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[31415])
    parser.add_argument("--batch-size", type=int, choices=[32, 64], default=32)
    parser.add_argument("--binary-action-indices", type=parse_binary_action_indices, default=(),
                        help="Optional comma-separated action indices scored by sign over the execution prefix; no action modification")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if not checkpoint.is_file() or output.exists():
        raise ValueError("provide one checkpoint file and a new output JSON path")
    if len(set(args.noise_seeds)) != len(args.noise_seeds):
        raise ValueError("noise seeds must be unique")
    if args.torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    os.chdir(ROOT_DIR)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import hydra
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    checkpoint_hash = file_sha256(checkpoint)
    policy, cfg, weights = load_policy(checkpoint, args.device)
    if file_sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint changed while loading; use a finished immutable checkpoint")
    required = ["heading_targets", "heading_xy_rms", "min_target_confidence", "min_source_confidence"]
    if any(not hasattr(policy, name) for name in required):
        raise TypeError("checkpoint must expose the shared predicted-heading policy diagnostics")
    binary_action_indices = validate_binary_action_indices(args.binary_action_indices, policy.action_dim)
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    validation = dataset.get_validation_dataset()
    from oat.common.seq_sampler import get_val_mask
    expected_holdout = get_val_mask(dataset.replay_buffer.n_episodes,
                                   float(cfg.task.policy.dataset.val_ratio),
                                   int(cfg.task.policy.dataset.seed))
    windows, split_metadata = select_validation_windows(dataset, validation, args.windows_per_task,
                                                       args.selection_seed, expected_holdout)
    rows = diagnose_batches(policy, validation, windows, args.batch_size, args.noise_seeds, binary_action_indices)
    per_task = {str(task): summarize_metric_rows([row for row in rows if row["task_uid"] == task])
                for task in sorted({row["task_uid"] for row in rows})}
    pooled = summarize_metric_rows(rows)
    balanced_metrics = {}
    balanced_task_counts = {}
    for metric in pooled["metrics"]:
        # Actual counts belong to per-task/pooled reports; averaging them would
        # turn confusion counts into misleading fractional pseudo-counts.
        if metric in BINARY_SIGN_COUNT_METRICS:
            continue
        task_values = [value["metrics"][metric] for value in per_task.values() if value["metrics"][metric] is not None]
        balanced_metrics[metric] = np.mean(task_values, axis=0).tolist() if task_values else None
        balanced_task_counts[metric] = len(task_values)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash, "weights": weights,
        "policy_target": cfg.policy._target_,
        "policy_config": OmegaConf.to_container(cfg.policy, resolve=True),
        "script_sha256": file_sha256(__file__), "actual_oat_module_paths": verify_repository_imports(ROOT_DIR),
        "dataset_config": OmegaConf.to_container(cfg.task.policy.dataset, resolve=True),
        "split": split_metadata, "windows_per_task": args.windows_per_task,
        "selection_seed": args.selection_seed, "noise_seeds": args.noise_seeds,
        "batch_size": args.batch_size, "device": args.device,
        "action_horizon": int(policy.horizon), "execution_prefix": int(policy.n_action_steps),
        "heading_horizon": int(policy.heading_horizon), "heading_xy_rms_from_checkpoint": float(policy.heading_xy_rms),
        "heading_target_min_confidence": float(policy.min_target_confidence),
        "predicted_validity_threshold": float(policy.min_source_confidence),
        "action_channel_indices": list(range(policy.action_dim)),
        "fitting": "none; all policy weights, normalizers, and heading RMS loaded exclusively from checkpoint",
        "inference_inputs": "observation window, explicit independent Gaussian noise and optional angular jitter only",
        "validity_semantics": "Probability that the target heading is defined, not probability the predicted angle is accurate.",
        "generated_heading_semantics": "Angle errors require both target and generated headings to be valid; inspect accompanying validity coverage.",
        "normalization": "All MSE metrics use unnormalized raw actions.",
        "interpretation": "Held-out imitation diagnostics; not simulator success-rate estimates. Validation metrics may guide model selection.",
        "windows": windows, "per_task": per_task,
        "aggregate_task_balanced": balanced_metrics, "aggregate_metric_task_counts": balanced_task_counts,
        "aggregate_pooled": pooled,
        "window_metrics": rows,
    }
    if binary_action_indices:
        report.update({
            "binary_action_indices": list(binary_action_indices),
            "binary_action_sign_labels": [-1, 0, 1],
            "binary_action_sign_confusion_axes": "configured action dimension, true sign, predicted sign",
            "binary_action_sign_semantics": (
                "Sign disagreement over execution-prefix actions, with exact zero as its own command; "
                "arrays follow binary_action_indices order. Counts sum all scored action positions and noise draws. "
                "Task-balanced aggregates contain rates; per-task and pooled aggregates also contain actual counts. "
                "Scoring does not modify generated actions or apply an environment-specific gripper convention."
            ),
        })
    if file_sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint file changed during diagnostics; report withheld")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "checkpoint_sha256": checkpoint_hash,
                      "aggregate_task_balanced": balanced_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
