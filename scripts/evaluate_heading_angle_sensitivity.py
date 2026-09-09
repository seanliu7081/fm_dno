"""Offline source-angle intervention on fixed shared-heading flow checkpoints.

The predicted heading condition is encoded once and held fixed. Only the angle
used to construct the XY source changes. Oracle angles use future demonstration
labels: this is a sensitivity diagnostic, not a deployable policy or SR test.
"""
from __future__ import annotations

import argparse
import json
import math
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

ARMS = ("predicted", "oracle", "oracle_15", "oracle_30", "oracle_60")
ANGLE_ERRORS_DEG = {"predicted": None, "oracle": 0., "oracle_15": 15.,
                    "oracle_30": 30., "oracle_60": 60.}
SIGN_SEED = 2_400_000_000
EXTRA_METRICS = ("source_heading_error_deg", "source_angle_error_deg")


def wrapped_error_degrees(a, b):
    difference = a - b
    return torch.atan2(difference.sin(), difference.cos()).abs() * (180. / math.pi)


def source_angles(prediction, target, signs):
    """Only valid, finite GT headings replace the original source-angle input."""
    predicted = prediction["theta"]
    if signs.shape != predicted.shape:
        raise ValueError("signs must have the same shape as prediction theta")
    if not bool(((signs == -1) | (signs == 1)).all()):
        raise ValueError("signs must contain only -1 or +1")
    eligible = target["valid"] & torch.isfinite(target["theta"]) & torch.isfinite(predicted)
    result = {"predicted": predicted}
    for arm in ARMS[1:]:
        angle = target["theta"] + signs * math.radians(ANGLE_ERRORS_DEG[arm])
        result[arm] = torch.where(eligible, angle, predicted)
    return result


@torch.no_grad()
def sample_from_condition(policy, cond, prediction, noise, angular_jitter, source_theta):
    """Reuse one unchanged condition tensor; override only the source theta field."""
    if source_theta.shape != prediction["theta"].shape:
        raise ValueError("source_theta must have the same shape as prediction theta")
    source_prediction = dict(prediction)
    source_prediction["theta"] = source_theta
    source, active = policy._transform_source(
        policy.prior_noise_scale * noise, source_prediction, angular_jitter,
    )
    x = source
    batch_size = cond.shape[0]
    dt = 1. / policy.num_inference_steps
    for i in range(policy.num_inference_steps):
        t = torch.full((batch_size,), i * dt, device=cond.device, dtype=cond.dtype)
        x = x + dt * policy.model(x, policy._scale_t(t), cond)
    action_pred = policy.normalizer["action"].unnormalize(x)
    return {"action_pred": action_pred, "action": action_pred[:, :policy.n_action_steps],
            "source": source, "source_active": active}


@torch.no_grad()
def evaluate_batch(policy, batch, noise, angular_jitter, signs):
    cond, prediction = policy.obs_encoder.encode_with_heading(batch["obs"])
    target = policy.heading_targets(batch["action"])
    angles = source_angles(prediction, target, signs)
    before = cond.clone()
    results = {arm: sample_from_condition(policy, cond, prediction, noise, angular_jitter, angles[arm])
               for arm in ARMS}
    if not torch.equal(before, cond):
        raise ValueError("Condition was mutated during the source-angle intervention")
    for arm in ARMS[1:]:
        if not torch.equal(results[arm]["source_active"], results["predicted"]["source_active"]):
            raise ValueError("Source activation gate changed between arms")
        inactive = ~target["valid"] | ~results["predicted"]["source_active"]
        for key in ("source", "action_pred"):
            if not torch.equal(results[arm][key][inactive], results["predicted"][key][inactive]):
                raise ValueError(f"Invalid-label or gated-off fallback changed: {arm}, {key}")
    return {"condition": cond, "prediction": prediction, "target": target,
            "source_angles": angles, "arms": results}


def metrics_for(policy, batch, result, target, angle):
    generated, source = result["action_pred"], result["source"]
    error = (generated[:, :8] - batch["action"][:, :8]).square()
    resultant = generated[:, :16, :2].sum(1)
    direction = resultant / resultant.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    raw_source = policy.normalizer["action"].unnormalize(source)
    source_resultant = raw_source[:, :16, :2].sum(1)
    source_theta = torch.atan2(source_resultant[:, 1], source_resultant[:, 0])
    return {
        "raw_prefix8_mse": error.mean((1, 2)),
        "raw_prefix8_xyz_mse": error[..., :3].mean((1, 2)),
        "raw_prefix8_rotation_mse": error[..., 3:6].mean((1, 2)),
        "raw_prefix8_gripper_mse": error[..., 6].mean(1),
        "generated_heading_mae_deg": angular_error(direction, target["direction"]),
        "source_heading_error_deg": wrapped_error_degrees(source_theta, target["theta"]),
        "source_angle_error_deg": wrapped_error_degrees(angle, target["theta"]),
    }


def validate_pair_archive(pair, rows, draws):
    for key, column in (("episode", 0), ("frame", 1), ("task", 2)):
        if not np.array_equal(pair[key], rows[:, column]):
            raise ValueError(f"Saved sampling identities differ: {key}")
    if pair["epsilon"].shape != (draws, len(rows), 16, 7):
        raise ValueError("Use exactly the declared saved noise draws and sample subset")
    if pair["angular_jitter"].shape != (draws, len(rows)):
        raise ValueError("Angular jitter dimensions differ")


@torch.no_grad()
def evaluate_model(policy, cache, rows, pair, signs, batch_size):
    draws, count = signs.shape
    device = next(policy.parameters()).device
    arrays = {key: rows[:, col].copy() for key, col in (("episode", 0), ("frame", 1), ("task", 2))}
    arrays.update(epsilon=pair["epsilon"].copy(), angular_jitter=pair["angular_jitter"].copy(),
                  signed_error_sign=signs.copy())
    for key in ("valid",):
        arrays[key] = np.empty(count, dtype=bool)
    for key in ("prediction_theta", "predicted_confidence", "target_theta", "predicted_heading_error_deg"):
        arrays[key] = np.empty(count, dtype=np.float32)
    arrays["conditioning"] = np.empty((count, policy.n_obs_steps, policy.obs_encoder.output_feature_dim()), dtype=np.float32)
    arrays["source_active"] = np.empty((draws, count), dtype=bool)
    arrays["intervention_applied"] = np.empty((draws, count), dtype=bool)
    for arm in ARMS:
        for metric in (*METRICS, *EXTRA_METRICS, "source_theta"):
            arrays[f"{arm}__{metric}"] = np.empty((draws, count), dtype=np.float32)
        for key in ("source", "action_pred"):
            arrays[f"{arm}__{key}"] = np.empty((draws, count, 16, 7), dtype=np.float32)
    for draw in range(draws):
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = batch_at(cache, np.arange(start, stop), device)
            noise = torch.as_tensor(pair["epsilon"][draw, start:stop], device=device)
            jitter = torch.as_tensor(pair["angular_jitter"][draw, start:stop], device=device)
            sign = torch.as_tensor(signs[draw, start:stop], device=device, dtype=torch.float32)
            result = evaluate_batch(policy, batch, noise, jitter, sign)
            pred, target = result["prediction"], result["target"]
            values = {
                "valid": target["valid"], "prediction_theta": pred["theta"],
                "predicted_confidence": pred["confidence"], "target_theta": target["theta"],
                "predicted_heading_error_deg": angular_error(pred["direction"], target["direction"]),
                "conditioning": result["condition"],
            }
            for key, value in values.items():
                value = value.cpu().numpy()
                if draw == 0:
                    arrays[key][start:stop] = value
                elif not np.array_equal(arrays[key][start:stop], value):
                    raise ValueError(f"Fixed observation condition/target changed across draws: {key}")
            active = result["arms"]["predicted"]["source_active"]
            arrays["source_active"][draw, start:stop] = active.cpu().numpy()
            arrays["intervention_applied"][draw, start:stop] = (active & target["valid"]).cpu().numpy()
            for arm in ARMS:
                arm_result = result["arms"][arm]
                values = metrics_for(policy, batch, arm_result, target, result["source_angles"][arm])
                values.update(source=arm_result["source"], action_pred=arm_result["action_pred"],
                              source_theta=result["source_angles"][arm])
                for key, value in values.items():
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Nonfinite value: {arm}, {key}")
                    arrays[f"{arm}__{key}"][draw, start:stop] = value.cpu().numpy()
    checks = {}
    for key in (*METRICS, "source"):
        actual, expected = arrays[f"predicted__{key}"], pair[key]
        checks[key] = {"bit_exact":bool(np.array_equal(actual, expected)),
                       "max_abs_difference":float(np.max(np.abs(actual.astype(float)-expected.astype(float))))}
        if not checks[key]["bit_exact"]:
            raise ValueError(f"Predicted-source control did not exactly reproduce saved evaluation: {key}")
    for key in ("valid", "source_active"):
        if not np.array_equal(arrays[key], pair[key]):
            raise ValueError(f"Original gate or target validity changed: {key}")
    return arrays, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "output/mini_heading_prior/mini_5000_20260909")
    parser.add_argument("--paired-source-dir", type=Path, default=ROOT / "output/mini_heading_prior/sampling_robustness_20260909")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    primary = args.run_dir.resolve()
    paired = args.paired_source_dir.resolve()
    manifest, split = read_json(primary / "manifest.json"), read_json(primary / "split.json")
    primary_summary, pair_manifest, pair_summary = (read_json(primary / "summary.json"),
        read_json(paired / "manifest.json"), read_json(paired / "summary.json"))
    seeds, draws = list(map(int, pair_manifest["seeds"])), int(pair_manifest["draws"])
    if seeds != [42, 43] or draws != 4 or pair_manifest["windows"] != 200:
        raise ValueError("This authorized diagnostic uses exactly seeds42/43 and four draws on200windows")
    if pair_manifest["primary_manifest_sha256"] != sha256(primary / "manifest.json"):
        raise ValueError("Paired samples belong to a different primary experiment")
    if pair_manifest["primary_summary_sha256"] != sha256(primary / "summary.json"):
        raise ValueError("Primary summary identity differs")
    trained = {(int(r["seed"]), r["mode"]) for r in primary_summary["final"]}
    if any((seed, "condition_prior_xy") not in trained for seed in seeds):
        raise ValueError("Missing completed XY-prior checkpoints")
    rows_all = np.asarray(split["validation_windows"], dtype=np.int64)
    rows = rows_all[sample_indices_for(rows_all, manifest["args"]["sample_per_task"])]
    if not np.array_equal(rows, np.asarray(pair_manifest["selected_rows"], dtype=np.int64)):
        raise ValueError("Selected observation windows differ from paired source experiment")
    source_files = [Path(__file__), ROOT / "scripts/evaluate_mini_heading_prior_sampling.py",
                    ROOT / "scripts/mini_heading_prior.py", ROOT / "scripts/mini_shared_heading.py",
                    ROOT / "oat/policy/flow_policy_shared_heading.py"]
    for path in source_files[1:]:
        key = str(path.relative_to(ROOT))
        if sha256(path) != pair_manifest["source_sha256"][key]:
            raise ValueError(f"Inference source changed since paired samples were generated: {key}")
    signs = np.random.default_rng(SIGN_SEED).choice(np.array([-1, 1], dtype=np.int8), size=(draws, len(rows)))
    pairs, checkpoint_records = {}, []
    for seed in seeds:
        with np.load(paired / f"seed{seed}_condition_prior_xy.npz", allow_pickle=False) as archive:
            pairs[seed] = {key:archive[key] for key in archive.files}
        validate_pair_archive(pairs[seed], rows, draws)
        config = next(r for r in pair_summary["models"] if r["seed"] == seed and r["mode"] == "condition_prior_xy")
        path = primary / f"seed{seed}" / "condition_prior_xy" / "ema.pt"
        if sha256(path) != config["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after paired sample evaluation")
        checkpoint_records.append(config)
    for key in ("epsilon", "angular_jitter", "valid", "episode", "frame", "task"):
        if not np.array_equal(pairs[seeds[0]][key], pairs[seeds[1]][key]):
            raise ValueError(f"Cross-seed pairing differs: {key}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "kind":"offline fixed-model source-angle sensitivity; future-label intervention; no rollout/SR",
        "primary_run":str(primary), "paired_source_run":str(paired), "seeds":seeds,
        "draws":draws,"windows":len(rows),"arms":list(ARMS),"angle_errors_deg":ANGLE_ERRORS_DEG,
        "checkpoint_mode":"condition_prior_xy","checkpoint_step":manifest["args"]["steps"],
        "batch_size":pair_manifest["batch_size"],"source_kappa":manifest["args"]["source_kappa"],
        "source_gate":"original predicted validity and numerical source gate unchanged in every arm",
        "heading_target":"raw XY resultant of original16 demonstration actions; validity threshold unchanged",
        "condition":"original predicted direction+validity and shared observation features, encoded once per batch and held fixed across allarms",
        "invalid_label_handling":"retain original predicted source angle and report those windows separately",
        "noise":"exact saved epsilon and VonMises angular_jitter reused in every arm and checkpoint",
        "signed_error":{"seed":SIGN_SEED,"distribution":"Rademacher ±1 perdraw/window, same signs across allerrorlevels and trainingseeds"},
        "selected_rows":rows.tolist(),"models":checkpoint_records,
        "source_sha256":{str(path.relative_to(ROOT)):sha256(path) for path in source_files},
        "input_sha256":{str(path):sha256(path) for path in [primary/"manifest.json",primary/"split.json",primary/"summary.json",paired/"manifest.json",paired/"summary.json",*[paired/f"seed{seed}_condition_prior_xy.npz" for seed in seeds]]},
        "device":args.device,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch_version":str(torch.__version__),
        "limitations":["oracle source angles use future demonstration labels and are unavailable to a deployable policy",
                       "models were trained with predicted source angles; the intervention changes their source distribution",
                       "predicted heading condition deliberately stays unchanged, including any error in that condition",
                       "not a strict performance upper bound or a causal test of retraining with a better predictor",
                       "only two fixed checkpoints, four fixed noise streams and200heldoutwindows",
                       "offline imitation errors need not track closed-loop success"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(protocol,indent=2)+"\n")
    for path in source_files:
        dest=args.output_dir / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes())
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    cache=load_selected_cache(manifest,split,rows)
    started=time.monotonic()
    checks={}
    for seed in seeds:
        policy,config=load_policy(primary,manifest,split,seed,"condition_prior_xy",args.device)
        arrays,replay=evaluate_model(policy,cache,rows,pairs[seed],signs,pair_manifest["batch_size"])
        path=args.output_dir / f"seed{seed}.npz"
        np.savez_compressed(path,**arrays)
        checks[str(seed)]={"predicted_control_replay":replay,"valid_windows":int(arrays["valid"].sum()),
                           "invalid_windows":int((~arrays["valid"]).sum()),
                           "source_active_samples":int(arrays["source_active"].sum()),
                           "intervention_samples":int(arrays["intervention_applied"].sum()),
                           "archive_sha256":sha256(path),"checkpoint_sha256":config["checkpoint_sha256"]}
        print(json.dumps({"seed":seed,"completed":True,"checks":checks[str(seed)]}),flush=True)
        del policy
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    completion={"complete":True,"elapsed_seconds":time.monotonic()-started,"checks":checks,
                "arms":list(ARMS),"seeds":seeds,"draws":draws,"windows":len(rows)}
    (args.output_dir / "evaluation_complete.json").write_text(json.dumps(completion,indent=2)+"\n")
    print(json.dumps(completion),flush=True)


if __name__ == "__main__":
    main()
