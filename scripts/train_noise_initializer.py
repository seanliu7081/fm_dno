"""Distill successful DNO searches into an E/F noise initializer (not RL).

Example:
    python scripts/train_noise_initializer.py --data output/dno/teacher.pt \
        --output-dir output/dno/initializer --successful-only --device cpu

An archive must contain cond, base_noise, optimized_noise, context_features,
episode_id and meta.source_policy/meta.base_checkpoint_sha256. Optional accepted
filters proxy-improving teacher searches; --successful-only additionally requires
actual episode success. All splitting and normalization respect episode boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from oat.dno.noise_initializer import (
    CONTEXT_DIM,
    NoiseInitializer,
    save_noise_initializer,
    validate_identity,
)


def split_by_episode(
    episode_ids: Sequence[Any] | torch.Tensor,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[Any], list[Any]]:
    """Return row indices and episode IDs, never splitting chunks of an episode."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one.")
    ids = episode_ids.tolist() if isinstance(episode_ids, torch.Tensor) else list(episode_ids)
    if any(not isinstance(item, (str, int)) or isinstance(item, bool) for item in ids):
        raise ValueError("episode_id entries must be integer or string episode identifiers.")
    unique = sorted(set(ids), key=lambda item: (type(item).__name__, str(item)))
    if len(unique) < 2:
        raise ValueError("At least two retained episodes are required for disjoint train/validation sets.")
    random.Random(seed).shuffle(unique)
    n_val = min(len(unique) - 1, max(1, round(len(unique) * val_ratio)))
    val_episodes, train_episodes = unique[:n_val], unique[n_val:]
    val_set = set(val_episodes)
    train_rows = torch.tensor([i for i, episode in enumerate(ids) if episode not in val_set], dtype=torch.long)
    val_rows = torch.tensor([i for i, episode in enumerate(ids) if episode in val_set], dtype=torch.long)
    return train_rows, val_rows, train_episodes, val_episodes


def load_archive(path: str | Path, *, include_rejected=False, successful_only=False):
    archive = torch.load(path, map_location="cpu", weights_only=True)
    required = ("cond", "base_noise", "optimized_noise", "context_features", "episode_id", "meta")
    missing = [key for key in required if key not in archive]
    if missing:
        raise ValueError(f"Teacher archive is missing keys: {missing}.")
    meta = archive["meta"]
    # Permit an explicitly nested identity without requiring it in new exports.
    identity_source = meta.get("identity", meta)
    identity = {key: identity_source[key] for key in ("source_policy", "base_checkpoint_sha256") if key in identity_source}
    if "reference_identity" in identity_source:
        identity["reference_identity"] = identity_source["reference_identity"]
    identity = validate_identity(identity)
    tensors = {}
    for key in required[:4]:
        if not isinstance(archive[key], torch.Tensor):
            raise ValueError(f"{key} must be a tensor.")
        tensors[key] = archive[key].float()
        if not bool(torch.isfinite(tensors[key]).all()):
            raise ValueError(f"{key} contains nonfinite values.")
    cond, base, optimized, context = (tensors[key] for key in required[:4])
    if cond.ndim != 3 or base.ndim != 3 or optimized.shape != base.shape:
        raise ValueError("Expected cond[N,To,D] and matching base_noise/optimized_noise[N,H,A].")
    count = base.shape[0]
    if count == 0 or cond.shape[0] != count or context.shape != (count, CONTEXT_DIM):
        raise ValueError("Archive batch lengths differ, context_features is not [N,9], or archive is empty.")
    episode_ids = archive["episode_id"]
    episode_ids = episode_ids.tolist() if isinstance(episode_ids, torch.Tensor) else list(episode_ids)
    if len(episode_ids) != count:
        raise ValueError("episode_id must have one entry per teacher example.")
    keep = torch.ones(count, dtype=torch.bool)

    def bool_rows(key):
        value = torch.as_tensor(archive[key])
        if tuple(value.shape) != (count,):
            raise ValueError(f"{key} must have shape [N].")
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(f"{key} must contain boolean values.")
        return value.bool()

    if not include_rejected and "accepted" in archive:
        keep &= bool_rows("accepted")
    if successful_only:
        if "episode_success" not in archive:
            raise ValueError("--successful-only requires actual episode_success labels in the archive.")
        keep &= bool_rows("episode_success")
    selected = keep.nonzero(as_tuple=True)[0]
    if selected.numel() == 0:
        raise ValueError("No teacher examples remain after filtering.")
    return (
        {key: value[selected] for key, value in tensors.items()},
        [episode_ids[i] for i in selected.tolist()],
        identity,
        {"original_rows": count, "retained_rows": selected.numel(), "archive_meta": meta},
    )


def make_loader(tensors, rows, batch_size, *, shuffle=False, seed=42):
    dataset = TensorDataset(*(tensors[key][rows] for key in ("cond", "base_noise", "optimized_noise", "context_features")))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    squared_error = baseline_error = 0.0
    count = 0
    for cond, base, target, context in loader:
        cond, base, target, context = (item.to(device) for item in (cond, base, target, context))
        prediction = model(cond, base, {"context_features": context})
        squared_error += F.mse_loss(prediction, target, reduction="sum").item()
        baseline_error += F.mse_loss(base, target, reduction="sum").item()
        count += target.numel()
    return {"mse": squared_error / count, "baseline_mse": baseline_error / count}


def train(args):
    if args.epochs < 0 or args.batch_size <= 0 or args.lr <= 0 or args.cpu_threads <= 0:
        raise ValueError("epochs must be nonnegative; batch-size/lr/cpu-threads must be positive.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    tensors, episode_ids, identity, archive_summary = load_archive(
        args.data, include_rejected=args.include_rejected, successful_only=args.successful_only,
    )
    train_rows, val_rows, train_episodes, val_episodes = split_by_episode(
        episode_ids, val_ratio=args.val_ratio, seed=args.seed,
    )
    cond, base = tensors["cond"], tensors["base_noise"]
    model = NoiseInitializer(
        cond_dim=cond.shape[-1], n_obs_steps=cond.shape[1], horizon=base.shape[1], action_dim=base.shape[2],
        hidden_dims=args.hidden_dims, max_residual_rms=args.max_residual_rms,
        cond_std_floor=args.cond_std_floor, displacement_std_floor=args.displacement_std_floor,
        normalized_input_clip=args.normalized_input_clip,
    )
    model.fit_input_normalizer(
        cond[train_rows], base[train_rows], {"context_features": tensors["context_features"][train_rows]},
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(tensors, train_rows, args.batch_size, shuffle=True, seed=args.seed)
    train_eval_loader = make_loader(tensors, train_rows, args.batch_size)
    val_loader = make_loader(tensors, val_rows, args.batch_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "method": "dno_teacher_distillation",
        "data": str(Path(args.data).resolve()),
        "identity": identity,
        "structure": model.structure(),
        "train_episodes": train_episodes,
        "validation_episodes": val_episodes,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "seed": args.seed,
        "include_rejected": args.include_rejected,
        "successful_only": args.successful_only,
        **archive_summary,
    }
    with (output_dir / "run.json").open("w") as handle:
        json.dump(run_meta, handle, indent=2)

    def metrics(epoch):
        train_metrics = evaluate(model, train_eval_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        return {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_baseline_mse": train_metrics["baseline_mse"],
            "val_mse": val_metrics["mse"],
            "val_baseline_mse": val_metrics["baseline_mse"],
        }

    initial = metrics(-1)
    best_val = initial["val_mse"]
    best_epoch = -1
    checkpoint_meta = {"train_episodes": train_episodes, "validation_episodes": val_episodes, "seed": args.seed}
    # Preserve the identity baseline when every learned update worsens validation.
    save_noise_initializer(output_dir / "best.pt", model, identity=identity, training={**checkpoint_meta, **initial})
    last_metrics = initial
    with (output_dir / "metrics.jsonl").open("w") as log:
        log.write(json.dumps(initial) + "\n")
        log.flush()
        print(json.dumps(initial), flush=True)
        for epoch in range(args.epochs):
            model.train()
            for cond_batch, base_batch, target_batch, context_batch in train_loader:
                cond_batch, base_batch, target_batch, context_batch = (
                    item.to(device) for item in (cond_batch, base_batch, target_batch, context_batch)
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = model(cond_batch, base_batch, {"context_features": context_batch})
                loss = F.mse_loss(prediction, target_batch)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite distillation loss; refusing to save this update.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            last_metrics = metrics(epoch)
            log.write(json.dumps(last_metrics) + "\n")
            log.flush()
            print(json.dumps(last_metrics), flush=True)
            if last_metrics["val_mse"] < best_val:
                best_val = last_metrics["val_mse"]
                best_epoch = epoch
                save_noise_initializer(
                    output_dir / "best.pt", model, identity=identity,
                    training={**checkpoint_meta, **last_metrics},
                )
            save_noise_initializer(
                output_dir / "last.pt", model, identity=identity,
                training={**checkpoint_meta, **last_metrics},
            )
            if args.patience > 0 and epoch - best_epoch >= args.patience:
                break
    if args.epochs == 0:
        save_noise_initializer(output_dir / "last.pt", model, identity=identity, training={**checkpoint_meta, **initial})
    summary = {"best_epoch": best_epoch, "best_val_mse": best_val, "val_baseline_mse": initial["val_baseline_mse"], "last_epoch": last_metrics["epoch"]}
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"output_dir": str(output_dir), **summary}), flush=True)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--data", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--epochs", type=int, default=50)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--lr", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    result.add_argument("--max-residual-rms", type=float, default=0.5)
    result.add_argument("--cond-std-floor", type=float, default=0.1)
    result.add_argument("--displacement-std-floor", type=float, default=0.05,
                        help="Minimum fitted standard deviation for goal-minus-eef position, in meters.")
    result.add_argument("--normalized-input-clip", type=float, default=10.0,
                        help="Clipping for fitted condition/displacement blocks; source noise remains raw.")
    result.add_argument("--val-ratio", type=float, default=0.2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--patience", type=int, default=10, help="Epochs without validation improvement; 0 disables early stopping.")
    result.add_argument("--cpu-threads", type=int, default=2)
    result.add_argument("--include-rejected", action="store_true", help="Include searches rejected by the teacher proxy objective.")
    result.add_argument("--successful-only", action="store_true", help="Use only examples from episodes marked successful by the environment.")
    return result


if __name__ == "__main__":
    train(parser().parse_args())
