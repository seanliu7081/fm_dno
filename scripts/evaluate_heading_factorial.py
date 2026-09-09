"""Fixed-checkpoint 2x2: predicted/GT heading condition x original/zero jitter.

All four cells use the GT source center when the label is valid. Future labels
are diagnostic-only inputs; no training, rollout, or production policy changes.
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

from scripts.evaluate_heading_angle_sensitivity import (
    METRICS, EXTRA_METRICS, ROOT, angular_error, batch_at, load_policy,
    load_selected_cache, metrics_for, read_json, sample_from_condition, sha256,
)

ARMS = ("predicted_jitter", "predicted_zero", "gt_jitter", "gt_zero")
FACTORS = {
    "predicted_jitter": {"condition": "predicted", "jitter": "original"},
    "predicted_zero": {"condition": "predicted", "jitter": "zero"},
    "gt_jitter": {"condition": "gt", "jitter": "original"},
    "gt_zero": {"condition": "gt", "jitter": "zero"},
}


def eligible_targets(prediction, target):
    return (target["valid"] & torch.isfinite(target["theta"])
            & torch.isfinite(target["direction"]).all(-1)
            & torch.isfinite(prediction["theta"]))


def make_conditions(cond, prediction, target):
    """Replace only the explicit cos/sin channels, retaining features and validity."""
    if cond.ndim != 3 or cond.shape[-1] < 3:
        raise ValueError("Expected observation tokens ending with direction and validity")
    eligible = eligible_targets(prediction, target)
    if target["direction"].shape != (cond.shape[0], 2):
        raise ValueError("Target direction must have shape (batch, 2)")
    gt = cond.clone()
    gt[..., -3:-1] = torch.where(eligible[:, None, None],
                                target["direction"][:, None, :].to(cond), cond[..., -3:-1])
    return {"predicted": cond, "gt": gt}


@torch.no_grad()
def evaluate_batch(policy, batch, noise, angular_jitter):
    cond, prediction = policy.obs_encoder.encode_with_heading(batch["obs"])
    target = policy.heading_targets(batch["action"])
    eligible = eligible_targets(prediction, target)
    conditions = make_conditions(cond, prediction, target)
    originals = {key: value.clone() for key, value in conditions.items()}
    theta = torch.where(eligible, target["theta"], prediction["theta"])
    jitter = torch.as_tensor(angular_jitter, device=cond.device, dtype=torch.float32)
    if jitter.shape != prediction["theta"].shape or not bool(torch.isfinite(jitter).all()):
        raise ValueError("Angular jitter must be a finite vector with one value per observation")
    # Undefined targets keep all original inputs, even in the nominal zero arm.
    zero = torch.where(eligible, torch.zeros_like(jitter), jitter)
    results = {}
    for arm, factors in FACTORS.items():
        effective_jitter = jitter if factors["jitter"] == "original" else zero
        result = sample_from_condition(policy, conditions[factors["condition"]], prediction,
                                       noise, effective_jitter, theta)
        result.update(source_theta=theta, effective_angular_jitter=effective_jitter)
        results[arm] = result
    for key in conditions:
        if not torch.equal(conditions[key], originals[key]):
            raise ValueError("Condition mutated during sampling")
    if not torch.equal(conditions["gt"][..., :-3], cond[..., :-3]):
        raise ValueError("Observation features changed")
    if not torch.equal(conditions["gt"][..., -1], cond[..., -1]):
        raise ValueError("Predicted validity condition changed")
    for left, right in (("predicted_jitter", "gt_jitter"), ("predicted_zero", "gt_zero")):
        if not torch.equal(results[left]["source"], results[right]["source"]):
            raise ValueError("Changing condition changed the source")
    for arm in ARMS:
        if not torch.equal(results[arm]["source_active"], results[ARMS[0]]["source_active"]):
            raise ValueError("Original source gate changed between factorial cells")
        for key in ("source", "action_pred"):
            if not torch.equal(results[arm][key][~eligible], results[ARMS[0]][key][~eligible]):
                raise ValueError("Undefined-target fallback changed")
    return {"conditions": conditions, "prediction": prediction, "target": target,
            "eligible": eligible, "arms": results}


@torch.no_grad()
def evaluate_model(policy, cache, rows, reference, batch_size):
    draws, count = reference["angular_jitter"].shape
    device = next(policy.parameters()).device
    arrays = {key: rows[:, col].copy() for key, col in (("episode", 0), ("frame", 1), ("task", 2))}
    arrays.update(epsilon=reference["epsilon"].copy(), angular_jitter=reference["angular_jitter"].copy())
    arrays["valid"] = np.empty(count, dtype=bool)
    for key in ("prediction_theta", "predicted_confidence", "target_theta", "predicted_heading_error_deg"):
        arrays[key] = np.empty(count, dtype=np.float32)
    for key in ("conditioning_predicted", "conditioning_gt"):
        arrays[key] = np.empty((count, policy.n_obs_steps, policy.obs_encoder.output_feature_dim()), np.float32)
    for key in ("source_active", "intervention_applied"):
        arrays[key] = np.empty((draws, count), bool)
    for arm in ARMS:
        for key in (*METRICS, *EXTRA_METRICS, "source_theta", "effective_angular_jitter"):
            arrays[f"{arm}__{key}"] = np.empty((draws, count), np.float32)
        for key in ("source", "action_pred"):
            arrays[f"{arm}__{key}"] = np.empty((draws, count, 16, 7), np.float32)
    for draw in range(draws):
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = batch_at(cache, np.arange(start, stop), device)
            noise = torch.as_tensor(reference["epsilon"][draw, start:stop], device=device)
            jitter = torch.as_tensor(reference["angular_jitter"][draw, start:stop], device=device)
            result = evaluate_batch(policy, batch, noise, jitter)
            pred, target = result["prediction"], result["target"]
            values = {"valid": target["valid"], "prediction_theta": pred["theta"],
                      "predicted_confidence": pred["confidence"], "target_theta": target["theta"],
                      "predicted_heading_error_deg": angular_error(pred["direction"], target["direction"]),
                      **{f"conditioning_{key}": value for key, value in result["conditions"].items()}}
            for key, value in values.items():
                value = value.cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite saved input: {key}")
                if draw == 0:
                    arrays[key][start:stop] = value
                elif not np.array_equal(arrays[key][start:stop], value):
                    raise ValueError(f"Fixed observation or target changed across draws: {key}")
            active = result["arms"][ARMS[0]]["source_active"]
            arrays["source_active"][draw, start:stop] = active.cpu().numpy()
            # This mask describes eligibility for the GT interventions; condition
            # can change even if the original source gate was inactive.
            arrays["intervention_applied"][draw, start:stop] = result["eligible"].cpu().numpy()
            for arm in ARMS:
                arm_result = result["arms"][arm]
                values = metrics_for(policy, batch, arm_result, target, arm_result["source_theta"])
                values.update({key: arm_result[key] for key in
                               ("source", "action_pred", "source_theta", "effective_angular_jitter")})
                for key, value in values.items():
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Nonfinite result: {arm}, {key}")
                    arrays[f"{arm}__{key}"][draw, start:stop] = value.cpu().numpy()
    replay = {}
    for key in (*METRICS, *EXTRA_METRICS, "source_theta", "source", "action_pred"):
        actual, expected = arrays[f"predicted_jitter__{key}"], reference[f"oracle__{key}"]
        replay[key] = {"bit_exact": bool(np.array_equal(actual, expected)),
                       "max_abs_difference": float(np.max(np.abs(actual.astype(float) - expected.astype(float))))}
        if not replay[key]["bit_exact"]:
            raise ValueError(f"Existing GT-prior cell did not reproduce the prior experiment: {key}")
    for key, ref_key in (("conditioning_predicted", "conditioning"), ("valid", "valid"),
                         ("source_active", "source_active"), ("prediction_theta", "prediction_theta"),
                         ("target_theta", "target_theta"), ("predicted_confidence", "predicted_confidence")):
        if not np.array_equal(arrays[key], reference[ref_key]):
            raise ValueError(f"Original input or gate differs from saved evaluation: {key}")
    # All undefined-label results also replay the deployable predicted baseline.
    invalid = ~arrays["valid"]
    for arm in ARMS:
        for key in ("source", "action_pred", *METRICS):
            if not np.array_equal(arrays[f"{arm}__{key}"][:, invalid], reference[f"predicted__{key}"][:, invalid]):
                raise ValueError(f"Invalid-label fallback differs from original baseline: {arm}, {key}")
    return arrays, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "output/mini_heading_prior/angle_sensitivity_20260909")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    ref_dir = args.reference_dir.resolve()
    ref_manifest = read_json(ref_dir / "manifest.json")
    ref_complete = read_json(ref_dir / "evaluation_complete.json")
    if not ref_complete["complete"]:
        raise ValueError("The reference evaluation is incomplete")
    seeds, draws, count = ref_manifest["seeds"], ref_manifest["draws"], ref_manifest["windows"]
    if seeds != [42, 43] or draws != 4 or count != 200:
        raise ValueError("This matched diagnostic requires two original checkpoints and 4x200 paired samples")
    primary = Path(ref_manifest["primary_run"])
    manifest, split = read_json(primary / "manifest.json"), read_json(primary / "split.json")
    rows = np.asarray(ref_manifest["selected_rows"], dtype=np.int64)
    for path, digest in ref_manifest["input_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Original experiment inputs changed: {path}")
    source_files = [Path(__file__), *[ROOT / path for path in ref_manifest["source_sha256"]]]
    for path, digest in ref_manifest["source_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Reference inference implementation changed: {path}")
    references, models = {}, []
    for seed in seeds:
        path = ref_dir / f"seed{seed}.npz"
        if sha256(path) != ref_complete["checks"][str(seed)]["archive_sha256"]:
            raise ValueError("Reference arrays failed their recorded hash")
        with np.load(path, allow_pickle=False) as archive:
            references[seed] = {key: archive[key] for key in archive.files}
        for key, col in (("episode", 0), ("frame", 1), ("task", 2)):
            if not np.array_equal(references[seed][key], rows[:, col]):
                raise ValueError("Reference sample identities differ")
        if references[seed]["epsilon"].shape != (draws, count, 16, 7):
            raise ValueError("Reference noise dimensions differ")
        checkpoint = primary / f"seed{seed}/condition_prior_xy/ema.pt"
        digest = sha256(checkpoint)
        if digest != ref_complete["checks"][str(seed)]["checkpoint_sha256"]:
            raise ValueError("Reference checkpoint changed")
        models.append({"seed": seed, "mode": "condition_prior_xy", "checkpoint": str(checkpoint), "checkpoint_sha256": digest})
    for key in ("epsilon", "angular_jitter", "episode", "frame", "task", "valid", "target_theta"):
        if not np.array_equal(references[seeds[0]][key], references[seeds[1]][key]):
            raise ValueError(f"Cross-checkpoint pairing differs: {key}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "kind": "offline fixed-model GT-source 2x2 factorial; future labels; no retraining or SR",
        "reference_run": str(ref_dir), "primary_run": str(primary), "seeds": seeds,
        "draws": draws, "windows": count, "arms": list(ARMS), "factors": FACTORS,
        "checkpoint_mode": "condition_prior_xy", "checkpoint_step": manifest["args"]["steps"],
        "batch_size": ref_manifest["batch_size"], "source_kappa": ref_manifest["source_kappa"],
        "heading_target": "raw XY resultant of original 16 demonstration actions",
        "condition": "GT replaces only explicit cos/sin channels; all shared observation features and predicted validity stay fixed",
        "source": "GT source center in all four cells on eligible targets; original predicted gate unchanged",
        "jitter": "original exact saved VonMises(0,4) samples or explicit zeros; no resampling",
        "invalid_label_handling": "original predicted source center, original jitter, and original condition in every cell",
        "intervention_applied": "eligible finite GT label; explicit condition can change even if source gate is inactive",
        "control": "predicted_jitter must reproduce reference oracle source, actions, and metrics bit for bit",
        "selected_rows": rows.tolist(), "models": models,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "input_sha256": {str(path): sha256(path) for path in [ref_dir / "manifest.json", ref_dir / "evaluation_complete.json",
                           *[ref_dir / f"seed{seed}.npz" for seed in seeds], primary / "manifest.json", primary / "split.json"]},
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_version": str(torch.__version__),
        "limitations": ["GT future labels are unavailable at deployment; all cells are diagnostic",
                        "both GT condition and zero jitter change the training input distribution",
                        "changing the explicit heading condition does not replace heading information in shared features",
                        "source alignment is not a constraint on generated action direction",
                        "factorial contrasts measure interventions on fixed checkpoints, not additive causes of angular MAE",
                        "only two fixed checkpoints and four fixed noise streams; no success-rate measurement"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for path in source_files:
        dest = args.output_dir / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache = load_selected_cache(manifest, split, rows)
    started = time.monotonic()
    checks = {}
    for seed in seeds:
        policy, config = load_policy(primary, manifest, split, seed, "condition_prior_xy", args.device)
        arrays, replay = evaluate_model(policy, cache, rows, references[seed], ref_manifest["batch_size"])
        path = args.output_dir / f"seed{seed}.npz"
        np.savez_compressed(path, **arrays)
        checks[str(seed)] = {"control_replay": replay, "valid_windows": int(arrays["valid"].sum()),
                             "invalid_windows": int((~arrays["valid"]).sum()),
                             "source_active_samples": int(arrays["source_active"].sum()),
                             "intervention_samples": int(arrays["intervention_applied"].sum()),
                             "archive_sha256": sha256(path), "checkpoint_sha256": config["checkpoint_sha256"]}
        print(json.dumps({"seed": seed, "completed": True, "checks": checks[str(seed)]}), flush=True)
        del policy
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    completion = {"complete": True, "elapsed_seconds": time.monotonic() - started, "checks": checks,
                  "arms": list(ARMS), "seeds": seeds, "draws": draws, "windows": count}
    (args.output_dir / "evaluation_complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
