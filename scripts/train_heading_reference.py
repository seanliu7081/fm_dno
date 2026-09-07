"""Train an observation-only reference heading with an episode-disjoint validation set.

Run from the repository root:
    python scripts/train_heading_reference.py --config-name=train_heading_reference
    python scripts/train_heading_reference.py reference_mode=state

Checkpoints contain the reference encoder, heading head, normalization and construction
configuration. Future actions supervise the target; they never enter the predictor.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from oat.common.seq_sampler import SequenceSampler, get_val_mask
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.perception.heading_predictor import ObservationHeadingPredictor
from oat.symmetry.so2_chunk import SO2ChunkSpec, wrap_angle


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def build_datasets(cfg: DictConfig, shape_meta: dict):
    dataset_cfg = OmegaConf.to_container(cfg.task.policy.dataset, resolve=True)
    dataset_cfg["obs_keys"] = list(shape_meta["obs"])
    dataset_cfg["seed"] = int(cfg.split_seed)
    val_ratio = float(dataset_cfg.get("val_ratio", 0.0))
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("Heading training requires 0 < task.policy.dataset.val_ratio < 1.")
    dataset = hydra.utils.instantiate(dataset_cfg)
    if dataset.n_obs_steps != int(cfg.n_obs_steps) or dataset.n_action_steps != int(cfg.horizon):
        raise ValueError("Dataset observation/action windows must match n_obs_steps and horizon.")
    val_mask = get_val_mask(dataset.replay_buffer.n_episodes, val_ratio, int(cfg.split_seed))
    if not dataset.train_mask.any() or not val_mask.any():
        raise ValueError("Heading training requires at least one train and one validation episode.")
    if np.any(dataset.train_mask & val_mask):
        raise RuntimeError("Train and validation episodes overlap.")
    # ZarrDataset's complement-based validation would include episodes excluded by
    # max_train_episodes. Keep the original held-out split even when subsampling train.
    validation = copy.copy(dataset)
    validation.train_mask = val_mask.copy()
    validation.seq_sampler = SequenceSampler(
        replay_buffer=dataset.replay_buffer,
        sequence_length=dataset.seq_len,
        pad_before=dataset.pad_before,
        pad_after=dataset.pad_after,
        episode_mask=val_mask,
    )
    return dataset, validation, dataset_cfg


def fit_train_normalizer(dataset, shape_meta: dict) -> LinearNormalizer:
    """Fit action/state statistics using train episodes; use a fixed RGB input range."""
    ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
    step_mask = np.repeat(dataset.train_mask, np.diff(np.r_[0, ends]))
    numeric_data = {"action": np.asarray(dataset.replay_buffer[dataset.action_key])[step_mask]}
    for key in dataset.numeric_obs_keys:
        if shape_meta["obs"][key]["type"] != "rgb":
            numeric_data[key] = np.asarray(dataset.replay_buffer[key])[step_mask]
    normalizer = LinearNormalizer()
    normalizer.fit(numeric_data, last_n_dims=1, mode="limits")
    # RobomimicRgbEncoder expects channel-last pixels in [0, 255]. This avoids both
    # validation leakage and materializing a float copy of every training image.
    for key, attr in shape_meta["obs"].items():
        if attr["type"] == "rgb":
            channels = int(attr["shape"][-1])
            bounds = np.stack([np.zeros(channels), np.full(channels, 255)]).astype(np.float32)
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(bounds, mode="limits")
    return normalizer


def build_loader(dataset, cfg: DictConfig, *, training: bool, seed: int):
    kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
    if int(kwargs.get("num_workers", 0)) == 0:
        kwargs["persistent_workers"] = False
    return DataLoader(
        dataset, **kwargs, shuffle=training, drop_last=False,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
    )


def angular_metrics(error: torch.Tensor, mask: torch.Tensor) -> dict:
    selected = error[mask]
    if not selected.numel():
        return {"count": 0, "mae_deg": None, "mean_cosine": None, "residual_r1": None}
    cosine = selected.cos().mean().item()
    sine = selected.sin().mean().item()
    return {
        "count": int(selected.numel()),
        "mae_deg": selected.abs().mean().item() * 180.0 / math.pi,
        "mean_cosine": cosine,
        "residual_r1": math.hypot(cosine, sine),
    }


@torch.no_grad()
def evaluate(model, loader, spec, device, cfg: DictConfig) -> dict:
    model.eval()
    errors, confidences, validities, stationary = [], [], [], []
    heading_sum = validity_sum = 0.0
    valid_total = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if cfg.training.max_val_batches is not None and batch_idx >= cfg.training.max_val_batches:
            break
        batch = to_device(batch, device)
        prediction = model(batch["obs"])
        losses = model.loss(batch, spec, prediction=prediction)
        target = model.targets(batch["action"], spec)
        batch_size = batch["action"].shape[0]
        total += batch_size
        valid_count = int(target["valid"].sum())
        valid_total += valid_count
        heading_sum += float(losses["heading_loss"]) * valid_count
        validity_sum += float(losses["validity_loss"]) * batch_size
        errors.append(wrap_angle(prediction["theta"] - target["theta"]).cpu())
        confidences.append(prediction["confidence"].cpu())
        validities.append(target["valid"].bool().cpu())
        pos = batch["obs"].get(str(cfg.evaluation.motion_key))
        if pos is not None:
            displacement = pos[:, -1, :2] - pos[:, 0, :2]
            stationary.append((displacement.norm(dim=-1) < cfg.evaluation.min_motion).cpu())
    if total == 0:
        raise ValueError("Validation loader yielded no samples.")
    error = torch.cat(errors)
    confidence = torch.cat(confidences)
    valid = torch.cat(validities)
    accepted = confidence >= float(cfg.evaluation.confidence_threshold)
    result = {"heading_loss": heading_sum / max(valid_total, 1),
              "validity_loss": validity_sum / total}
    result["loss"] = result["heading_loss"] + result["validity_loss"]
    result.update({
        "samples": total,
        "valid_fraction": valid.float().mean().item(),
        "accepted_fraction": accepted.float().mean().item(),
        "validity_brier": ((confidence - valid.float()) ** 2).mean().item(),
        "validity_accuracy": (accepted == valid).float().mean().item(),
        "valid_heading": angular_metrics(error, valid),
        "accepted_valid_heading": angular_metrics(error, accepted & valid),
    })
    if stationary:
        is_stationary = torch.cat(stationary)
        result["stationary_valid_heading"] = angular_metrics(error, valid & is_stationary)
        result["moving_valid_heading"] = angular_metrics(error, valid & ~is_stationary)
    bins = []
    for idx in range(10):
        selected = (confidence >= idx / 10.0) & (
            confidence <= 1.0 if idx == 9 else confidence < (idx + 1) / 10.0
        )
        count = int(selected.sum())
        bins.append({
            "lower": idx / 10.0, "upper": (idx + 1) / 10.0, "count": count,
            "mean_probability": float(confidence[selected].mean()) if count else None,
            "valid_fraction": float(valid[selected].float().mean()) if count else None,
        })
    result["validity_calibration"] = bins
    return result


@hydra.main(version_base=None, config_path="../oat/config", config_name="train_heading_reference")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    if cfg.reference_mode not in ("image_state", "state"):
        raise ValueError("reference_mode must be image_state or state.")
    if int(cfg.training.num_epochs) < 1:
        raise ValueError("training.num_epochs must be positive.")
    if not 0.0 <= float(cfg.evaluation.confidence_threshold) <= 1.0:
        raise ValueError("evaluation.confidence_threshold must be in [0, 1].")
    for name in ("max_train_batches", "max_val_batches"):
        if cfg.training[name] is not None and int(cfg.training[name]) < 1:
            raise ValueError(f"training.{name} must be positive or null.")
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_name = str(cfg.training.device)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    if cfg.reference_mode == "state":
        shape_meta["obs"] = {
            key: attr for key, attr in shape_meta["obs"].items() if attr["type"] == "state"
        }
    if not shape_meta["obs"]:
        raise ValueError("The selected reference_mode contains no observations.")
    dataset, validation, dataset_cfg = build_datasets(cfg, shape_meta)
    train_loader = build_loader(dataset, cfg, training=True, seed=seed)
    val_loader = build_loader(validation, cfg, training=False, seed=seed)
    spec = SO2ChunkSpec(**OmegaConf.to_container(cfg.action_spec, resolve=True))
    model_config = OmegaConf.to_container(cfg.reference, resolve=True)
    model_config["obs_encoder"]["shape_meta"] = shape_meta
    if cfg.reference_mode == "state":
        model_config["obs_encoder"]["vision_encoder"] = None
    model: ObservationHeadingPredictor = hydra.utils.instantiate(model_config)
    model.set_normalizer(fit_train_normalizer(dataset, shape_meta), spec,
                         vector_mode=str(cfg.normalizer_vector_mode))
    model.to(device)
    encoder_parameters = [p for p in model.obs_encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in model.obs_encoder.parameters()}
    head_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = [{"params": head_parameters, "lr": float(cfg.optimizer.head_lr)}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": float(cfg.optimizer.encoder_lr)})
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg.optimizer.betas),
                                  weight_decay=float(cfg.optimizer.weight_decay))
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"config": OmegaConf.to_container(cfg),
                                   "resolved_reference": model_config}), output_dir / "config.yaml")
    metadata = {
        "reference_mode": str(cfg.reference_mode), "seed": seed,
        "split_seed": int(cfg.split_seed), "dataset": dataset_cfg,
        "train_episode_ids": np.flatnonzero(dataset.train_mask).tolist(),
        "validation_episode_ids": np.flatnonzero(validation.train_mask).tolist(),
        "normalizer_fit": "train_episodes_only", "rgb_range": [0, 255],
        "evaluation": OmegaConf.to_container(cfg.evaluation, resolve=True),
    }
    (output_dir / "split.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Reference mode={cfg.reference_mode}, device={device}; "
          f"train={len(dataset)} samples/{int(dataset.train_mask.sum())} episodes, "
          f"validation={len(validation)} samples/{int(validation.train_mask.sum())} episodes", flush=True)
    best_loss = math.inf
    history = []
    for epoch in range(int(cfg.training.num_epochs)):
        model.train()
        train_sums = {key: 0.0 for key in ("loss", "heading_loss", "validity_loss")}
        samples = 0
        for batch_idx, batch in enumerate(train_loader):
            if cfg.training.max_train_batches is not None and batch_idx >= cfg.training.max_train_batches:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(batch, spec)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Non-finite heading loss at epoch {epoch}, batch {batch_idx}.")
            losses["loss"].backward()
            if cfg.training.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.max_grad_norm))
            optimizer.step()
            batch_size = batch["action"].shape[0]
            samples += batch_size
            for key in train_sums:
                train_sums[key] += float(losses[key].detach()) * batch_size
        if not samples:
            raise ValueError("Training loader yielded no samples.")
        val_metrics = evaluate(model, val_loader, spec, device, cfg)
        if not math.isfinite(val_metrics["loss"]):
            raise FloatingPointError("Non-finite validation loss.")
        record = {"epoch": epoch, "train": {k: v / samples for k, v in train_sums.items()},
                  "validation": val_metrics}
        history.append(record)
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        checkpoint_meta = {**metadata, "epoch": epoch, "metrics": record}
        model.save_checkpoint(checkpoint_dir / "last.pt", action_spec=spec, horizon=int(cfg.horizon),
                              model_config=model_config, metadata=checkpoint_meta)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            model.save_checkpoint(checkpoint_dir / "best.pt", action_spec=spec, horizon=int(cfg.horizon),
                                  model_config=model_config, metadata=checkpoint_meta)
        print(json.dumps({"epoch": epoch, "train_loss": record["train"]["loss"],
                          "val_loss": val_metrics["loss"],
                          **val_metrics["valid_heading"]}, allow_nan=False), flush=True)
    print(f"Reference checkpoints: {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()
