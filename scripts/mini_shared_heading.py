"""Matched, offline shared-heading ablation on a small LIBERO image/state sample.

This is an early-learning diagnostic, not a closed-loop success-rate benchmark.
Each arm starts with identical parameters and sees identical batches, crops,
dropout, flow noise and time draws. The baseline's heading readout is detached
from its encoder; separate gradient clipping prevents that readout changing the
baseline update. No standalone reference checkpoint is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

from oat.common.seq_sampler import get_val_mask
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.diffusion.ema_model import EMAModel
from oat.policy.flow_policy_shared_heading import SharedHeadingFlowPolicy


ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "auxiliary", "condition")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_windows(ends, tasks, mask, per_task, seed):
    """Equal task counts, near-equal episode counts, unique times per episode."""
    rng = np.random.default_rng(seed)
    starts = np.r_[0, ends[:-1]]
    rows = []
    for task in sorted(set(tasks.tolist())):
        episodes = np.flatnonzero(mask & (tasks == task))
        if not len(episodes):
            raise ValueError(f"No episodes for task {task}")
        episodes = rng.permutation(episodes)
        for j, ep in enumerate(episodes):
            count = per_task // len(episodes) + (j < per_task % len(episodes))
            if count > ends[ep] - starts[ep]:
                raise ValueError("Requested more unique windows than episode frames")
            for frame in rng.choice(np.arange(starts[ep], ends[ep]), size=count, replace=False):
                rows.append((int(ep), int(frame), int(task)))
    return np.asarray(rows, dtype=np.int64)


def load_data(args, shape_meta):
    root = zarr.open(str(args.dataset), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    numeric_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "state"]
    numeric = {k: np.asarray(data[k][:]) for k in ["action", *numeric_keys]}
    tasks = numeric["task_uid"][starts, 0]
    val_mask = get_val_mask(len(ends), .1, args.split_seed)
    train_mask = ~val_mask
    rows = {
        "train": select_windows(ends, tasks, train_mask, args.train_per_task, args.data_seed),
        "validation": select_windows(ends, tasks, val_mask, args.val_per_task, args.data_seed + 1),
    }
    step_mask = np.repeat(train_mask, np.diff(np.r_[0, ends]))
    normalizer = LinearNormalizer()
    normalizer.fit({k: a[step_mask] for k, a in numeric.items()}, mode="limits")
    xy_rms = float(np.sqrt(np.mean(numeric["action"][step_mask, :2].astype(np.float64) ** 2)))
    rgb_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]
    for k in rgb_keys:
        normalizer[k] = SingleFieldLinearNormalizer.create_fit(
            np.array([[0., 0., 0.], [255., 255., 255.]], dtype=np.float32), mode="limits")
    cache = {}
    for split, selected in rows.items():
        ep, t = selected[:, 0], selected[:, 1]
        obs_idx = np.maximum(t[:, None] + np.array([-1, 0]), starts[ep, None])
        action_idx = np.minimum(t[:, None] + np.arange(16), ends[ep, None] - 1)
        obs = {k: torch.from_numpy(np.ascontiguousarray(numeric[k][obs_idx])).float()
               for k in numeric_keys}
        unique, inverse = np.unique(obs_idx, return_inverse=True)
        for k in rgb_keys:
            print(f"Loading {split} {k}: {len(unique)} unique frames", flush=True)
            images = data[k].oindex[unique]
            obs[k] = torch.from_numpy(np.ascontiguousarray(
                images[inverse].reshape(len(selected), 2, *data[k].shape[1:])))
        cache[split] = {"obs": obs, "action": torch.from_numpy(
            np.ascontiguousarray(numeric["action"][action_idx])).float()}
    split_info = {
        "train_episodes": np.flatnonzero(train_mask).tolist(),
        "validation_episodes": np.flatnonzero(val_mask).tolist(),
        "episode_ends": ends.tolist(), "episode_tasks": tasks.tolist(),
        "normalizer_fit": "all 450 training episodes only; RGB fixed [0,255]",
        "heading_xy_rms": xy_rms,
        "train_windows": rows["train"].tolist(),
        "validation_windows": rows["validation"].tolist(),
    }
    return cache, rows, normalizer, xy_rms, split_info


def batch_at(cache, indices, device):
    return {"obs": {k: v[indices].to(device=device, dtype=torch.float32)
                    for k, v in cache["obs"].items()},
            "action": cache["action"][indices].to(device)}


def build_policy(shape_meta, args, normalizer, xy_rms):
    encoder = hydra.utils.instantiate({
        "_target_": "oat.perception.fused_obs_encoder.FusedObservationEncoder",
        "_recursive_": False, "shape_meta": shape_meta,
        "vision_encoder": {
            "_target_": "oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder",
            "crop_shape": [76, 76], "eval_fixed_crop": True},
        "state_encoder": {
            "_target_": "oat.perception.state_encoder.ProjectionStateEncoder", "out_dim": None},
    })
    policy = SharedHeadingFlowPolicy(
        shape_meta=shape_meta, obs_encoder=encoder, horizon=16, n_action_steps=8,
        n_obs_steps=2, embed_dim=256, n_layers=4, n_heads=4, dropout=.1,
        num_inference_steps=10, heading_mode="baseline", heading_hidden_dim=128,
        heading_xy_rms=xy_rms, heading_loss_weight=args.aux_weight,
        validity_loss_weight=args.aux_weight)
    policy.set_normalizer(normalizer)
    return policy


def optimizer_for(policy, args):
    head = list(policy.obs_encoder.head.parameters())
    head_ids = {id(p) for p in head}
    optimizer = policy.get_optimizer(args.policy_lr, args.encoder_lr, 0., (.9, .95))
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in head_ids]
    optimizer.add_param_group({"params": head, "lr": args.head_lr, "weight_decay": 0.})
    core = [p for p in policy.parameters() if p.requires_grad and id(p) not in head_ids]
    return optimizer, core, head


def angular_error(direction, target):
    cross = direction[:, 0] * target[:, 1] - direction[:, 1] * target[:, 0]
    dot = (direction * target).sum(-1)
    return torch.atan2(cross, dot).abs() * (180. / np.pi)


@torch.no_grad()
def evaluate(policy, cache, rows, args, step, mode, seed, output_dir):
    policy.eval()
    device = next(policy.parameters()).device
    all_metrics = {k: [] for k in ("flow_mse", "heading_mae_deg", "heading_cosine", "valid", "validity_brier")}
    sample_metrics = {k: [] for k in ("raw_prefix8_mse", "raw_prefix8_xyz_mse", "generated_heading_mae_deg", "valid", "episode", "task")}
    # Sampling metrics use an equal-size, fixed subset from every task.
    sample_indices = []
    selection_rng = np.random.default_rng(20260910)
    for task in np.unique(rows[:, 2]):
        episodes = np.unique(rows[rows[:, 2] == task, 0])
        for j, episode in enumerate(episodes):
            count = args.sample_per_task // len(episodes) + (j < args.sample_per_task % len(episodes))
            available = np.flatnonzero((rows[:, 2] == task) & (rows[:, 0] == episode))
            sample_indices.extend(selection_rng.choice(available, count, replace=False).tolist())
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    for start in range(0, len(rows), args.batch_size):
        indices = np.arange(start, min(start + args.batch_size, len(rows)))
        batch = batch_at(cache, indices, device)
        seed_all(800_000 + start)
        noise = torch.randn_like(batch["action"])
        t = torch.rand(len(indices), device=device)
        result = policy.loss_components(batch, noise=noise, t=t)
        pred, target = result["prediction"], result["target"]
        all_metrics["flow_mse"].extend(result["flow_mse_per_sample"].cpu().tolist())
        all_metrics["heading_mae_deg"].extend(angular_error(pred["direction"], target["direction"]).cpu().tolist())
        all_metrics["heading_cosine"].extend((pred["direction"]*target["direction"]).sum(-1).cpu().tolist())
        all_metrics["valid"].extend(target["valid"].cpu().tolist())
        all_metrics["validity_brier"].extend(((pred["confidence"]-target["valid"].float())**2).cpu().tolist())
    for start in range(0, len(sample_indices), args.batch_size):
        indices = sample_indices[start:start + args.batch_size]
        batch = batch_at(cache, indices, device)
        seed_all(900_000 + start)
        generated = policy.predict_action(batch["obs"])["action_pred"]
        error = (generated[:, :8] - batch["action"][:, :8]) ** 2
        target = policy.heading_targets(batch["action"])
        resultant = generated[..., :2].sum(1)
        direction = resultant / resultant.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sample_metrics["raw_prefix8_mse"].extend(error.mean((1, 2)).cpu().tolist())
        sample_metrics["raw_prefix8_xyz_mse"].extend(error[..., :3].mean((1, 2)).cpu().tolist())
        sample_metrics["generated_heading_mae_deg"].extend(angular_error(direction, target["direction"]).cpu().tolist())
        sample_metrics["valid"].extend(target["valid"].cpu().tolist())
        sample_metrics["episode"].extend(rows[indices, 0].tolist())
        sample_metrics["task"].extend(rows[indices, 2].tolist())
    arrays = {k: np.asarray(v) for k, v in all_metrics.items()}
    sampled = {k: np.asarray(v) for k, v in sample_metrics.items()}
    valid = arrays["valid"].astype(bool)
    record = {"seed": seed, "mode": mode, "step": step, "weights": "ema",
              "validation_samples": len(rows), "sampling_samples": len(sample_indices),
              "flow_mse": float(arrays["flow_mse"].mean()),
              "heading_mae_deg": float(arrays["heading_mae_deg"][valid].mean()),
              "heading_cosine": float(arrays["heading_cosine"][valid].mean()),
              "valid_fraction": float(valid.mean()),
              "validity_brier": float(arrays["validity_brier"].mean()),
              "raw_prefix8_mse": float(sampled["raw_prefix8_mse"].mean()),
              "raw_prefix8_xyz_mse": float(sampled["raw_prefix8_xyz_mse"].mean()),
              "generated_heading_mae_deg": float(sampled["generated_heading_mae_deg"][sampled["valid"].astype(bool)].mean()),
              "per_task": {}}
    for task in np.unique(rows[:, 2]):
        selected = rows[:, 2] == task
        record["per_task"][str(task)] = {
            "flow_mse": float(arrays["flow_mse"][selected].mean()),
            "heading_mae_deg": float(arrays["heading_mae_deg"][selected & valid].mean())}
    np.savez_compressed(output_dir / f"eval_{step:05d}.npz", **arrays,
                        episode=rows[:, 0], task=rows[:, 2],
                        **{f"sample_{k}": v for k, v in sampled.items()})
    print(json.dumps({k: v for k, v in record.items() if k != "per_task"}), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=ROOT / "data/libero/libero10_N500.zarr")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=200)
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
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = p.parse_args()
    if min(args.steps, args.eval_every, args.train_per_task, args.val_per_task,
           args.sample_per_task, args.batch_size) < 1:
        p.error("budgets and sample counts must be positive")
    if args.sample_per_task > args.val_per_task:
        p.error("sample-per-task cannot exceed val-per-task")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    shape_meta = OmegaConf.to_container(OmegaConf.load(ROOT / "oat/config/task/policy/libero/libero10.yaml")["shape_meta"], resolve=True)
    cache, rows, normalizer, xy_rms, split_info = load_data(args, shape_meta)
    (args.output_dir / "split.json").write_text(json.dumps(split_info, indent=2) + "\n")
    torch.save(normalizer.state_dict(), args.output_dir / "normalizer.pt")
    source_files = [Path(__file__), ROOT / "oat/policy/flow_policy_shared_heading.py"]
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        "device": str(args.device), "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": str(torch.__version__), "shape_meta": shape_meta,
        "initialization": "from scratch; identical deep-copied weights per seed; extra condition columns zero",
        "normalization": "ordinary per-dimension minmax, 450 training episodes only; fixed RGB bounds",
        "heading_target": "raw XY resultant over16 actions; confidence scaled by fixed train-only XY RMS",
        "probe": "baseline diagnostic head uses detached features and separate gradient clipping",
        "evaluation": "fixed center crops; EMA; common fixed time/noise and Euler latent samples",
        "limitations": ["short offline training", "no environment rollout or success rate", "only two training seeds by default", "single preset auxiliary weight", "one action sample per observation; imitation error is not success"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
            policy.set_heading_mode(mode)
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
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                warmup = min(step / 100., 1.)
                for group in optimizer.param_groups:
                    group.setdefault("base_lr", group["lr"])
                    group["lr"] = group["base_lr"] * warmup
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
                    losses = policy.loss_components(batch)
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
