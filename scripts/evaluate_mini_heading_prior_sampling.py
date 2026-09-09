"""Secondary sampling robustness check for completed heading-prior mini runs.

The fixed-endpoint metrics in the training run remain primary. This evaluation
averages additional predeclared latent draws on exactly the same held-out windows;
it never retrains models, chooses a best draw, or changes the primary artifacts.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import zarr

from oat.model.common.normalizer import LinearNormalizer
from scripts.mini_heading_prior import (
    BLOCKS, MODES, ROOT, angular_error, angular_jitter, batch_at,
    build_policy, sample_indices_for, set_variant,
)
from scripts.summarize_mini_heading_prior import LABELS, paired_effect
from scripts.summarize_mini_shared_heading import validate_rows


METRICS = (
    "raw_prefix8_mse", "raw_prefix8_xyz_mse", "raw_prefix8_rotation_mse",
    "raw_prefix8_gripper_mse", "generated_heading_mae_deg",
)
GAUSSIAN_SEED_BASE = 1_900_000
JITTER_SEED_BASE = 2_300_000_000
DRAW_SEED_STRIDE = 10_000


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_selected_cache(manifest, split, selected_rows):
    """Read only selected observation/action frames; never load a training cache."""
    validate_rows(selected_rows, split)
    dataset = Path(manifest["args"]["dataset"])
    root = zarr.open(str(dataset if dataset.is_absolute() else ROOT / dataset), mode="r")
    ends = np.asarray(split["episode_ends"], dtype=np.int64)
    if not np.array_equal(np.asarray(root["meta"]["episode_ends"][:]), ends):
        raise ValueError("Dataset episode boundaries differ from the saved experiment")
    starts = np.r_[0, ends[:-1]]
    episodes, frames = selected_rows[:, 0], selected_rows[:, 1]
    obs_idx = np.maximum(frames[:, None] + np.array([-1, 0]), starts[episodes, None])
    action_idx = np.minimum(frames[:, None] + np.arange(16), ends[episodes, None] - 1)
    unique_obs, inverse_obs = np.unique(obs_idx, return_inverse=True)
    unique_action, inverse_action = np.unique(action_idx, return_inverse=True)
    data = root["data"]
    observations = {}
    for key, meta in manifest["shape_meta"]["obs"].items():
        values = np.asarray(data[key].oindex[unique_obs])
        selected = values[inverse_obs].reshape(len(selected_rows), 2, *data[key].shape[1:])
        tensor = torch.from_numpy(np.ascontiguousarray(selected))
        observations[key] = tensor if meta["type"] == "rgb" else tensor.float()
    actions = np.asarray(data["action"].oindex[unique_action])
    actions = actions[inverse_action].reshape(len(selected_rows), 16, -1)
    return {"obs": observations, "action": torch.from_numpy(np.ascontiguousarray(actions)).float()}


def load_policy(run_dir, manifest, split, seed, mode, device):
    """Strictly restore tensors plus the non-tensor source configuration."""
    path = run_dir / f"seed{seed}" / mode / "ema.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    args = SimpleNamespace(**manifest["args"])
    expected = {
        "seed": seed, "mode": mode, "step": args.steps, "heading_mode": "condition",
        "source_mode": "iid" if mode == "condition" else "heading",
        "source_kappa": args.source_kappa, "min_source_confidence": args.min_source_confidence,
        "shape_meta": manifest["shape_meta"],
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError(f"Checkpoint metadata differs from run manifest: {path}, {key}")
    source_jitter = getattr(args, "source_jitter", "vonmises")
    if payload.get("source_jitter", "vonmises") != source_jitter:
        raise ValueError(f"Checkpoint jitter mode differs from run manifest: {path}")
    if tuple(map(tuple, payload["source_vector_blocks"])) != BLOCKS[mode]:
        raise ValueError(f"Checkpoint source blocks differ: {path}")
    if payload["manifest"]["args"] != manifest["args"]:
        raise ValueError(f"Checkpoint training arguments differ: {path}")
    if payload["heading_xy_rms"] != split["heading_xy_rms"]:
        raise ValueError(f"Checkpoint heading validity scale differs: {path}")
    normalizer_state = torch.load(run_dir / "normalizer.pt", map_location="cpu", weights_only=True)
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(normalizer_state)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = build_policy(manifest["shape_meta"], args, normalizer, payload["heading_xy_rms"])
    set_variant(policy, mode, args)
    policy.load_state_dict(payload["state_dict"], strict=True)
    for key, value in normalizer_state.items():
        torch.testing.assert_close(policy.normalizer.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(policy.heading_xy_rms, torch.tensor(split["heading_xy_rms"]), rtol=0, atol=0)
    if (policy.horizon, policy.n_obs_steps, policy.n_action_steps, policy.num_inference_steps) != (16, 2, 8, 10):
        raise ValueError("Unexpected inference dimensions or Euler budget")
    config = {key: payload[key] for key in expected if key != "shape_meta"}
    config.update(source_jitter=source_jitter, source_vector_blocks=payload["source_vector_blocks"],
                  checkpoint=str(path), checkpoint_sha256=sha256(path),
                  heading_xy_rms=payload["heading_xy_rms"],
                  prior_noise_scale=policy.prior_noise_scale, num_inference_steps=policy.num_inference_steps)
    return policy.to(device).eval().requires_grad_(False), config


@torch.no_grad()
def evaluate_model(policy, cache, rows, draws=4, batch_size=32):
    """Draw-major arrays retain pairing across every arm and training seed."""
    policy.eval()
    device = next(policy.parameters()).device
    count = len(rows)
    arrays = {key: np.empty((draws, count), dtype=np.float32) for key in METRICS}
    arrays.update({key: np.empty((draws, count), dtype=np.float32) for key in
                   ("source_normalized_ms", "iid_normalized_ms", "source_heading_to_prediction_deg")})
    arrays["source_active"] = np.empty((draws, count), dtype=bool)
    arrays["source"] = np.empty((draws, count, 16, 7), dtype=np.float32)
    arrays["epsilon"] = np.empty_like(arrays["source"])
    arrays["angular_jitter"] = np.empty((draws, count), dtype=np.float32)
    arrays["valid"] = np.empty(count, dtype=bool)
    for name, column in (("episode", 0), ("frame", 1), ("task", 2)):
        arrays[name] = rows[:, column].copy()
    for draw in range(draws):
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = batch_at(cache, np.arange(start, stop), device)
            gaussian_seed = GAUSSIAN_SEED_BASE + draw * DRAW_SEED_STRIDE + start
            generator = torch.Generator(device=device).manual_seed(gaussian_seed)
            noise = torch.randn(batch["action"].shape, device=device,
                                dtype=batch["action"].dtype, generator=generator)
            jitter = angular_jitter(JITTER_SEED_BASE + draw * DRAW_SEED_STRIDE + start,
                                    stop - start, device, policy.source_kappa, policy.source_jitter)
            result = policy.predict_action(batch["obs"], noise=noise,
                                           angular_jitter=jitter, return_source=True)
            generated, source, prediction = result["action_pred"], result["source"], result["prediction"]
            if not torch.isfinite(generated).all() or not torch.isfinite(source).all():
                raise ValueError("Nonfinite generated actions or source")
            target = policy.heading_targets(batch["action"])
            error = (generated[:, :8] - batch["action"][:, :8]).square()
            resultant = generated[:, :16, :2].sum(1)
            direction = resultant / resultant.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            raw_source = policy.normalizer["action"].unnormalize(source)
            source_resultant = raw_source[:, :16, :2].sum(1)
            source_direction = source_resultant / source_resultant.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            values = {
                "raw_prefix8_mse": error.mean((1, 2)),
                "raw_prefix8_xyz_mse": error[..., :3].mean((1, 2)),
                "raw_prefix8_rotation_mse": error[..., 3:6].mean((1, 2)),
                "raw_prefix8_gripper_mse": error[..., 6].mean(1),
                "generated_heading_mae_deg": angular_error(direction, target["direction"]),
                "source_normalized_ms": source.square().mean((1, 2)),
                "iid_normalized_ms": (noise * policy.prior_noise_scale).square().mean((1, 2)),
                "source_heading_to_prediction_deg": angular_error(source_direction, prediction["direction"]),
                "source_active": result["source_active"], "source": source,
                "epsilon": noise, "angular_jitter": jitter,
            }
            for key, value in values.items():
                arrays[key][draw, start:stop] = value.cpu().numpy()
            valid = target["valid"].cpu().numpy()
            if draw == 0:
                arrays["valid"][start:stop] = valid
            elif not np.array_equal(arrays["valid"][start:stop], valid):
                raise ValueError("Target validity changed between sampling streams")
    if not arrays["valid"].any():
        raise ValueError("No valid target headings in selected observations")
    return arrays


def summarize_arrays(arrays):
    per_draw = []
    for draw in range(arrays[METRICS[0]].shape[0]):
        record = {"draw": draw}
        for key in METRICS:
            valid = arrays["valid"] if key == "generated_heading_mae_deg" else slice(None)
            record[key] = float(arrays[key][draw, valid].mean())
        per_draw.append(record)
    means = {key: float(np.mean([row[key] for row in per_draw])) for key in METRICS}
    diagnostics = {key: float(arrays[key].mean()) for key in
                   ("source_active", "source_normalized_ms", "iid_normalized_ms", "source_heading_to_prediction_deg")}
    return {"mean_over_draws": means, "per_draw": per_draw, "source_diagnostics": diagnostics}


def comparisons(archives, seeds, modes, bootstrap_draws):
    """Reuse paired episode resampling after averaging the fixed four streams."""
    averaged = {}
    for key, arrays in archives.items():
        averaged[key] = {"sample_" + metric: arrays[metric].mean(axis=0) for metric in METRICS}
        averaged[key].update({"sample_" + name: arrays[name] for name in ("valid", "episode", "task")})
    result = {}
    for mode in modes:
        if mode == "condition":
            continue
        result[mode] = {}
        for i, key in enumerate(METRICS):
            estimate = paired_effect(averaged, seeds, mode, "condition", key,
                                     bootstrap_draws, np.random.default_rng(37031 + i))
            effects = list(estimate["per_seed_effect"].values())
            estimate["sign_consistent_across_training_seeds"] = bool(
                all(value <= 0 for value in effects) or all(value >= 0 for value in effects))
            estimate["interpretation"] = "Negative treatment-minus-IID means lower error; interval conditions on fixed training seeds and latent streams."
            result[mode][key] = estimate
    return result


def write_report(summary, output_dir):
    lines = ["# Secondary sampling robustness check", "",
             "The original fixed-endpoint measurements remain primary. This check averages "
             f"{summary['draws']} additional latent streams on the same {summary['windows']} validation windows. "
             "It uses final EMA weights and does not retrain or select a best draw.", "",
             "| Training seed | Arm | Prefix-8 MSE | XYZ MSE | Rotation MSE | Gripper MSE | Generated heading MAE (°) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for record in summary["per_seed"]:
        values = " | ".join(f"{record['mean_over_draws'][key]:.6g}" for key in METRICS)
        lines.append(f"| {record['seed']} | {LABELS[record['mode']]} | {values} |")
    for mode, record in summary["mean_over_training_seeds"].items():
        values = " | ".join(f"{record[key]:.6g}" for key in METRICS)
        lines.append(f"| Mean | {LABELS[mode]} | {values} |")
    lines += ["", "Effects below are treatment minus IID; negative values mean lower error.", ""]
    for mode, estimates in summary["comparisons"].items():
        for key in ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "generated_heading_mae_deg"):
            estimate = estimates[key]
            per_seed = ", ".join(f"seed {seed}: {value:+.6g}" for seed, value in estimate["per_seed_effect"].items())
            consistency = "same sign" if estimate["sign_consistent_across_training_seeds"] else "different signs"
            lines.append(f"- {LABELS[mode]}, {key}: mean {estimate['effect_treatment_minus_control']:+.6g}; {per_seed} ({consistency}).")
    lines += ["", "All errors average independent one-sample predictions; actions are not averaged before scoring. "
              "NPZ files retain every draw and window, identities, target-validity masks, source diagnostics, and Gaussian/source snapshots. "
              "Per-draw values and paired episode-bootstrap intervals are in summary.json. "
              "Intervals resample held-out episodes within task, conditional on the fixed training seeds and these latent streams. "
              "This check does not measure success rate or establish robustness over training seeds.", ""]
    (output_dir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--modes", choices=MODES, nargs="+")
    cli = parser.parse_args()
    if min(cli.draws, cli.cpu_threads, cli.bootstrap_draws, cli.batch_size or 1) < 1:
        parser.error("draws, threads, batch size, and bootstrap count must be positive")
    run_dir = cli.run_dir.resolve()
    if not (run_dir / "summary.json").is_file():
        raise ValueError("The primary training run must finish before this secondary evaluation")
    manifest, split = read_json(run_dir / "manifest.json"), read_json(run_dir / "split.json")
    primary_summary = read_json(run_dir / "summary.json")
    args = manifest["args"]
    seeds, modes = cli.seeds or args["seeds"], cli.modes or args["modes"]
    if "condition" not in modes or len(set(modes)) != len(modes) or len(set(seeds)) != len(seeds):
        raise ValueError("Choose unique seeds/arms and include the IID condition control")
    if not set(seeds).issubset(args["seeds"]) or not set(modes).issubset(args["modes"]):
        raise ValueError("Requested models are not part of the primary run")
    completed = {(int(r["seed"]), r["mode"], int(r["step"])) for r in primary_summary["final"]}
    if any((seed, mode, args["steps"]) not in completed for seed in seeds for mode in modes):
        raise ValueError("Primary summary lacks a selected final checkpoint")
    validation_rows = np.asarray(split["validation_windows"], dtype=np.int64)
    rows = validation_rows[sample_indices_for(validation_rows, args["sample_per_task"])]
    validate_rows(rows, split)
    batch_size = cli.batch_size or args["batch_size"]
    if len(rows) >= DRAW_SEED_STRIDE:
        raise ValueError("Selected windows exceed the predeclared independent-stream seed stride")
    primary_valid = {}
    for seed in seeds:
        for mode in modes:
            primary = run_dir / f"seed{seed}" / mode / f"eval_{args['steps']:05d}.npz"
            with np.load(primary, allow_pickle=False) as archive:
                primary_valid[(seed, mode)] = archive["sample_valid"].astype(bool).copy()
                for name, column in (("episode", 0), ("frame", 1), ("task", 2)):
                    if not np.array_equal(archive["sample_" + name], rows[:, column]):
                        raise ValueError(f"Selected windows differ from primary evaluation: {primary}")
    source_files = [Path(__file__), ROOT / "scripts/mini_heading_prior.py",
                    ROOT / "scripts/mini_shared_heading.py", ROOT / "oat/policy/flow_policy_shared_heading.py",
                    ROOT / "scripts/summarize_mini_heading_prior.py",
                    ROOT / "scripts/summarize_mini_shared_heading.py"]
    for path in source_files[1:4]:
        key = str(path.relative_to(ROOT))
        if sha256(path) != manifest["source_sha256"][key]:
            raise ValueError(f"Inference implementation differs from primary source snapshot: {key}")
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "secondary; primary fixed-endpoint results are unchanged", "primary_run": str(run_dir),
        "primary_manifest_sha256": sha256(run_dir / "manifest.json"),
        "primary_summary_sha256": sha256(run_dir / "summary.json"),
        "normalizer_sha256": sha256(run_dir / "normalizer.pt"),
        "seeds": seeds, "modes": modes, "checkpoint_step": args["steps"], "draws": cli.draws,
        "selected_rows": rows.tolist(), "windows": len(rows), "batch_size": batch_size,
        "gaussian_seeds": "1900000 + draw_index * 10000 + batch_start (local torch Generator)",
        "angular_seeds": "2300000000 + draw_index * 10000 + batch_start (local NumPy VonMises)",
        "pairing": "Identical epsilon and angular jitter for every arm and training seed; independent additional streams",
        "aggregation": "Mean of per-window errors over all draws, then fixed training seeds; no best-draw selection",
        "bootstrap": {"draws": cli.bootstrap_draws, "conditional_on": "fixed training seeds and latent draws", "unit": "episodes within task"},
        "device": cli.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": str(torch.__version__), "primary_protocol": manifest,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
    }
    (cli.output_dir / "manifest.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for path in source_files:
        target = cli.output_dir / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    torch.set_num_threads(cli.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache = load_selected_cache(manifest, split, rows)
    records, archives, model_configs = [], {}, []
    started = time.monotonic()
    for seed in seeds:
        for mode in modes:
            policy, config = load_policy(run_dir, manifest, split, seed, mode, cli.device)
            arrays = evaluate_model(policy, cache, rows, cli.draws, batch_size)
            if not np.array_equal(arrays["valid"], primary_valid[(seed, mode)]):
                raise ValueError(f"Target validity differs from primary evaluation: seed {seed}, {mode}")
            if archives:
                reference = next(iter(archives.values()))
                for key in ("episode", "frame", "task", "valid", "epsilon", "angular_jitter"):
                    if not np.array_equal(reference[key], arrays[key]):
                        raise ValueError(f"Pairing audit failed for seed {seed}, {mode}, {key}")
            archives[(seed, mode)] = arrays
            np.savez_compressed(cli.output_dir / f"seed{seed}_{mode}.npz", **arrays)
            record = {"seed": seed, "mode": mode, **summarize_arrays(arrays)}
            records.append(record)
            model_configs.append(config)
            print(json.dumps(record), flush=True)
            del policy
            if str(cli.device).startswith("cuda"):
                torch.cuda.empty_cache()
    summary = {
        "status": "secondary; primary fixed-endpoint results are unchanged", "draws": cli.draws,
        "windows": len(rows), "seeds": seeds, "per_seed": records, "models": model_configs,
        "mean_over_training_seeds": {mode: {key: float(np.mean([
            row["mean_over_draws"][key] for row in records if row["mode"] == mode])) for key in METRICS} for mode in modes},
        "comparisons": comparisons(archives, seeds, modes, cli.bootstrap_draws),
        "elapsed_seconds": time.monotonic() - started,
    }
    (cli.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, cli.output_dir)
    print(json.dumps({"output_dir": str(cli.output_dir), "elapsed_seconds": summary["elapsed_seconds"],
                      "mean_over_training_seeds": summary["mean_over_training_seeds"]}), flush=True)


if __name__ == "__main__":
    main()
