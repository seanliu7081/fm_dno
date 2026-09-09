"""Matched shared-condition flow experiment with two directional source variants.

All arms use the same Min-Max normalizer, shared trainable heading head, data,
weights and torch RNG. Explicit Von Mises angles use a separate NumPy stream.
Generated raw actions are the primary comparison; native flow losses have
source-dependent targets and are only diagnostics. This does not measure SR.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import OmegaConf
import torch

from oat.model.diffusion.ema_model import EMAModel
from scripts.mini_shared_heading import (
    ROOT, angular_error, batch_at, build_policy, load_data, optimizer_for, seed_all,
)

MODES = ("condition", "condition_prior_xy", "condition_prior_both")
BLOCKS = {"condition": ((0, 1),), "condition_prior_xy": ((0, 1),),
          "condition_prior_both": ((0, 1), (3, 4))}


def angular_jitter(seed, batch_size, device, kappa, mode="vonmises"):
    """Independent explicit source randomness, never advancing global RNG."""
    if mode == "zero":
        return torch.zeros(batch_size, device=device, dtype=torch.float32)
    if mode != "vonmises":
        raise ValueError("source jitter mode must be 'vonmises' or 'zero'")
    values = np.random.default_rng(seed).vonmises(0., kappa, size=batch_size)
    return torch.as_tensor(values, device=device, dtype=torch.float32)


def set_variant(policy, mode, args):
    policy.set_heading_mode("condition")
    policy.set_source_mode("iid" if mode == "condition" else "heading")
    policy.set_source_vector_blocks(BLOCKS[mode])
    policy.source_kappa = args.source_kappa
    policy.set_source_jitter(getattr(args, "source_jitter", "vonmises"))
    policy.min_source_confidence = args.min_source_confidence


def sample_indices_for(rows, per_task):
    selected = []
    rng = np.random.default_rng(20260910)
    for task in np.unique(rows[:, 2]):
        episodes = np.unique(rows[rows[:, 2] == task, 0])
        available_by_episode = [np.flatnonzero((rows[:, 2] == task) & (rows[:, 0] == episode))
                                for episode in episodes]
        counts = [min(per_task // len(episodes) + (j < per_task % len(episodes)), len(available))
                  for j, available in enumerate(available_by_episode)]
        remaining = per_task - sum(counts)
        while remaining:
            changed = False
            for j, available in enumerate(available_by_episode):
                if counts[j] < len(available) and remaining:
                    counts[j] += 1
                    remaining -= 1
                    changed = True
            if not changed:
                raise ValueError(f"Not enough validation windows to sample task {task}")
        for count, available in zip(counts, available_by_episode):
            selected.extend(rng.choice(available, count, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


@torch.no_grad()
def evaluate(policy, cache, rows, args, step, mode, seed, output_dir):
    policy.eval()
    device = next(policy.parameters()).device
    all_metrics = {k: [] for k in (
        "flow_mse", "heading_mae_deg", "heading_cosine", "valid", "validity_brier",
        "source_active", "source_normalized_ms", "iid_normalized_ms",
        "source_heading_to_prediction_deg", "source_to_target_mse",
    )}
    sample_metrics = {k: [] for k in (
        "raw_prefix8_mse", "raw_prefix8_xyz_mse", "raw_prefix8_rotation_mse",
        "raw_prefix8_gripper_mse", "generated_heading_mae_deg", "valid", "episode", "task", "frame",
    )}
    sample_indices = sample_indices_for(rows, args.sample_per_task)
    for start in range(0, len(rows), args.batch_size):
        indices = np.arange(start, min(start + args.batch_size, len(rows)))
        batch = batch_at(cache, indices, device)
        seed_all(800_000 + start)
        noise = torch.randn_like(batch["action"])
        t = torch.rand(len(indices), device=device)
        jitter = angular_jitter(1_200_000_000 + start, len(indices), device, args.source_kappa, getattr(args, "source_jitter", "vonmises"))
        result = policy.loss_components(batch, noise=noise, t=t, angular_jitter=jitter)
        pred, target, source = result["prediction"], result["target"], result["source"]
        values = {
            "flow_mse": result["flow_mse_per_sample"],
            "heading_mae_deg": angular_error(pred["direction"], target["direction"]),
            "heading_cosine": (pred["direction"] * target["direction"]).sum(-1),
            "valid": target["valid"],
            "validity_brier": (pred["confidence"] - target["valid"].float()) ** 2,
            "source_active": result["source_active"],
            "source_normalized_ms": source.square().mean((1, 2)),
            "iid_normalized_ms": (noise * policy.prior_noise_scale).square().mean((1, 2)),
            "source_to_target_mse": (source - policy.normalizer["action"].normalize(batch["action"])).square().mean((1, 2)),
        }
        raw_source = policy.normalizer["action"].unnormalize(source)
        resultant = raw_source[:, :policy.heading_horizon, :2].sum(1)
        direction = resultant / resultant.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        values["source_heading_to_prediction_deg"] = angular_error(direction, pred["direction"])
        for key, value in values.items():
            all_metrics[key].extend(value.cpu().tolist())
    for start in range(0, len(sample_indices), args.batch_size):
        indices = sample_indices[start:start + args.batch_size]
        batch = batch_at(cache, indices, device)
        seed_all(900_000 + start)
        jitter = angular_jitter(1_300_000_000 + start, len(indices), device, args.source_kappa, getattr(args, "source_jitter", "vonmises"))
        generated = policy.predict_action(batch["obs"], angular_jitter=jitter)["action_pred"]
        error = (generated[:, :8] - batch["action"][:, :8]) ** 2
        target = policy.heading_targets(batch["action"])
        resultant = generated[..., :2].sum(1)
        direction = resultant / resultant.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        values = {
            "raw_prefix8_mse": error.mean((1, 2)),
            "raw_prefix8_xyz_mse": error[..., :3].mean((1, 2)),
            "raw_prefix8_rotation_mse": error[..., 3:6].mean((1, 2)),
            "raw_prefix8_gripper_mse": error[..., 6].mean(1),
            "generated_heading_mae_deg": angular_error(direction, target["direction"]),
            "valid": target["valid"],
        }
        for key, value in values.items():
            sample_metrics[key].extend(value.cpu().tolist())
        for key, col in (("episode", 0), ("frame", 1), ("task", 2)):
            sample_metrics[key].extend(rows[indices, col].tolist())
    arrays = {k: np.asarray(v) for k, v in all_metrics.items()}
    sampled = {k: np.asarray(v) for k, v in sample_metrics.items()}
    valid = arrays["valid"].astype(bool)
    record = {
        "seed": seed, "mode": mode, "step": step, "weights": "ema",
        "validation_samples": len(rows), "sampling_samples": len(sample_indices),
        "flow_mse": float(arrays["flow_mse"].mean()),
        "heading_mae_deg": float(arrays["heading_mae_deg"][valid].mean()),
        "heading_cosine": float(arrays["heading_cosine"][valid].mean()),
        "valid_fraction": float(valid.mean()), "validity_brier": float(arrays["validity_brier"].mean()),
        "generated_heading_mae_deg": float(sampled["generated_heading_mae_deg"][sampled["valid"].astype(bool)].mean()),
        "per_task": {},
    }
    for key in ("source_active", "source_normalized_ms", "iid_normalized_ms", "source_heading_to_prediction_deg", "source_to_target_mse"):
        record[key] = float(arrays[key].mean())
    for key in ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "raw_prefix8_rotation_mse", "raw_prefix8_gripper_mse"):
        record[key] = float(sampled[key].mean())
    for task in np.unique(rows[:, 2]):
        selected = rows[:, 2] == task
        record["per_task"][str(task)] = {
            "flow_mse": float(arrays["flow_mse"][selected].mean()),
            "heading_mae_deg": float(arrays["heading_mae_deg"][selected & valid].mean()),
        }
    np.savez_compressed(output_dir / f"eval_{step:05d}.npz", **arrays,
                        episode=rows[:, 0], frame=rows[:, 1], task=rows[:, 2],
                        **{f"sample_{k}": v for k, v in sampled.items()})
    print(json.dumps({k: v for k, v in record.items() if k != "per_task"}), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=ROOT / "data/libero/libero10_N500.zarr")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--train-per-task", type=int, default=400)
    p.add_argument("--val-per-task", type=int, default=100)
    p.add_argument("--sample-per-task", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--data-seed", type=int, default=20260909)
    p.add_argument("--aux-weight", type=float, default=.1)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--policy-lr", type=float, default=5e-5)
    p.add_argument("--source-kappa", type=float, default=4.)
    p.add_argument("--source-jitter", choices=("vonmises", "zero"), default="vonmises",
                   help="Angular dither during both training and inference; zero still uses the predicted source direction.")
    p.add_argument("--min-source-confidence", type=float, default=.5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = p.parse_args()
    if min(args.steps, args.eval_every, args.train_per_task, args.val_per_task,
           args.sample_per_task, args.batch_size) < 1:
        p.error("budgets and sample counts must be positive")
    if args.sample_per_task > args.val_per_task:
        p.error("sample-per-task cannot exceed val-per-task")
    if not np.isfinite(args.source_kappa) or args.source_kappa < 0:
        p.error("source-kappa must be finite and nonnegative")
    if not 0 <= args.min_source_confidence <= 1:
        p.error("min-source-confidence must be in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    shape_meta = OmegaConf.to_container(OmegaConf.load(ROOT / "oat/config/task/policy/libero/libero10.yaml")["shape_meta"], resolve=True)
    cache, rows, normalizer, xy_rms, split_info = load_data(args, shape_meta)
    (args.output_dir / "split.json").write_text(json.dumps(split_info, indent=2) + "\n")
    torch.save(normalizer.state_dict(), args.output_dir / "normalizer.pt")
    source_files = [Path(__file__), ROOT / "scripts/mini_shared_heading.py",
                    ROOT / "oat/policy/flow_policy_shared_heading.py",
                    ROOT / "tests/test_shared_heading_policy.py"]
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        "device": str(args.device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": str(torch.__version__), "shape_meta": shape_meta,
        "initialization": "from scratch; identical deep-copied weights per seed; extra condition columns zero",
        "normalization": "ordinary per-dimension minmax, 450 training episodes only; fixed RGB bounds",
        "heading_target": "raw translation XY resultant over16 actions; fixed train-only XY RMS validity scaling",
        "heading_training": "shared head jointly trained via condition + 0.1 cosine + 0.1 validity BCE; prediction detached ONLY for source construction",
        "sources": {mode: {"source_mode": "iid" if mode == "condition" else "heading", "vector_blocks": BLOCKS[mode]} for mode in args.modes},
        "source_transform": "z=epsilon*prior_scale; N(R_phi N^-1(z)); phi=pred_theta+delta-theta(raw_source_translation_XY); delta=0 for zero mode, otherwise VonMises(0,kappa); identity below confidence threshold",
        "source_randomness": "zero mode returns exact zeros; vonmises mode uses explicit private numpy Generator jitter at train seed 1100000000+seed*100000+step, validation 1200000000+batch_start, sampling 1300000000+batch_start; no global RNG advance",
        "evaluation": "fixed center crops; EMA; same base epsilon, time and jitter; ten Euler steps; final endpoint fixed in advance",
        "primary_metrics": ["raw_prefix8_mse", "raw_prefix8_xyz_mse", "generated_heading_mae_deg"],
        "flow_loss_caveat": "Native flow loss uses different sources/velocity targets across arms and is not directly comparable as action quality.",
        "limitations": ["short offline training", "no rollout or success rate", "only two training seeds", "single fixed kappa and auxiliary weight", "one generated sample per observation", "both-block conjugation changes normalized rotation-block means/covariances; not original SO2-normalized F", "detached source gradient convention; not a test of gradients through the source"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source_file in source_files:
        snapshot = args.output_dir / "source" / source_file.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source_file.read_bytes())
    records = []
    started = time.monotonic()
    for seed in args.seeds:
        seed_all(seed)
        initial = build_policy(shape_meta, args, normalizer, xy_rms)
        rng = np.random.default_rng(seed + 10_000)
        order = np.concatenate([rng.permutation(len(rows["train"])) for _ in range(
            (args.steps * args.batch_size + len(rows["train"]) - 1) // len(rows["train"]))])
        for mode in args.modes:
            run_dir = args.output_dir / f"seed{seed}" / mode
            run_dir.mkdir(parents=True)
            policy = copy.deepcopy(initial)
            set_variant(policy, mode, args)
            policy.to(args.device)
            ema_policy = copy.deepcopy(policy)
            ema = EMAModel(ema_policy, inv_gamma=1., power=.75, min_value=0., max_value=.9999)
            optimizer, core, head = optimizer_for(policy, args)
            records.append(evaluate(ema_policy, cache["validation"], rows["validation"], args, 0, mode, seed, run_dir))
            run_started = time.monotonic()
            running = []
            for step in range(1, args.steps + 1):
                seed_all(seed * 100_000 + step)
                indices = order[(step-1)*args.batch_size:step*args.batch_size]
                batch = batch_at(cache["train"], indices, args.device)
                jitter = angular_jitter(1_100_000_000 + seed*100_000 + step, len(indices), args.device, args.source_kappa, getattr(args, "source_jitter", "vonmises"))
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                warmup = min(step / 100., 1.)
                for group in optimizer.param_groups:
                    group.setdefault("base_lr", group["lr"])
                    group["lr"] = group["base_lr"] * warmup
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
                    losses = policy.loss_components(batch, angular_jitter=jitter)
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError(f"Nonfinite loss seed={seed} mode={mode} step={step}")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(core, 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head, 1., error_if_nonfinite=True)
                optimizer.step()
                ema.step(policy)
                running.append([float(losses[k].detach()) for k in ("flow_loss", "heading_loss", "validity_loss")])
                if step % 50 == 0:
                    print(json.dumps({"progress": mode, "seed": seed, "step": step,
                                      "train_mean": np.mean(running, axis=0).tolist(),
                                      "seconds": time.monotonic()-run_started}), flush=True)
                    running.clear()
                if step % args.eval_every == 0 or step == args.steps:
                    records.append(evaluate(ema_policy, cache["validation"], rows["validation"], args, step, mode, seed, run_dir))
                    (args.output_dir / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
            torch.save({"state_dict": ema_policy.state_dict(), "mode": mode,
                        "heading_mode": "condition", "source_mode": policy.source_mode,
                        "source_vector_blocks": BLOCKS[mode], "source_kappa": args.source_kappa,
                        "source_jitter": policy.source_jitter,
                        "min_source_confidence": args.min_source_confidence,
                        "seed": seed, "step": args.steps, "shape_meta": shape_meta,
                        "heading_xy_rms": xy_rms, "manifest": manifest}, run_dir / "ema.pt")
            del policy, ema_policy, ema, optimizer, core, head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del initial
    summary = {"elapsed_seconds": time.monotonic()-started,
               "final": [r for r in records if r["step"] == args.steps]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
