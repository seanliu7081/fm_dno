"""Exact, distributed action reconstruction diagnostics in dataset action units.

Each rank evaluates a disjoint stride of dataset windows. This module deliberately
does not use Accelerate's prepared loaders or a DistributedSampler, both of which
can pad shards by repeating examples. The caller gathers rank statistics after
handling any rank-local errors, and must provide an unwrapped policy in eval mode
(normally EMA) and preserve training RNG state around the evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from oat.common.pytorch_util import dict_apply, maybe_to_device


class _IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def action_mse_shard_indices(dataset_size: int, rank: int, world_size: int) -> range:
    """Visit every window once globally, including uneven and empty shards."""
    if dataset_size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid dataset size or distributed rank/world size")
    return range(rank, dataset_size, world_size)


def _names_by_uid(task_names):
    if task_names is None:
        return None
    if isinstance(task_names, Mapping):
        names = {int(uid): str(name) for uid, name in task_names.items()}
    elif isinstance(task_names, Sequence) and not isinstance(task_names, (str, bytes)):
        names = {uid: str(name) for uid, name in enumerate(task_names)}
    else:
        raise TypeError("task_names must be a UID-to-name mapping or sequence")
    if not names or len(set(names.values())) != len(names):
        raise ValueError("task_names must contain nonempty, unique task names")
    if any(not name or "/" in name for name in names.values()):
        raise ValueError("task names must be nonempty and must not contain '/'")
    return names


def _window_noise(indices, horizon, action_dim, seed):
    # A separate CPU generator per global window keeps the draw independent of
    # policy, batch size, GPU identity, rank assignment, and the training RNG.
    return torch.stack([
        torch.randn(
            horizon, action_dim,
            generator=torch.Generator(device="cpu").manual_seed(
                (int(seed) + int(index)) % (2**63 - 1)),
            dtype=torch.float32,
        )
        for index in indices
    ])


@torch.inference_mode()
def evaluate_action_mse(
    policy,
    dataset,
    device,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    task_names=None,
):
    """Return one rank's additive, JSON-safe full-window action MSE statistics.

    Targets are the dataset's unnormalized seven delta-controller channels.
    ``action_pred`` must cover the entire target window, even if the policy's
    deployable ``action`` output executes only a shorter prefix. Padded boundary
    actions are included, matching the dataset's training-window convention.
    No held-out observations/actions are used to fit or change normalization.

    There are no collectives here: callers can gather errors safely before
    merging. Evaluation never retains DataLoader workers between calls.
    """
    names = _names_by_uid(task_names)
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if policy.training:
        raise ValueError("Action MSE requires a policy already in eval mode")
    if int(policy.action_dim) != 7:
        raise ValueError("Action MSE requires all seven delta-controller channels")
    size = len(dataset)
    indices = action_mse_shard_indices(size, rank, world_size)
    device = torch.device(device)
    loader = DataLoader(
        _IndexedDataset(dataset), sampler=indices, batch_size=batch_size,
        num_workers=num_workers, drop_last=False, persistent_workers=False,
        pin_memory=device.type == "cuda",
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    result = {
        "rank": rank, "world_size": world_size, "dataset_size": size,
        "seed": int(seed), "horizon": int(policy.horizon), "action_dim": 7,
        "window_count": 0, "dataset_index_sum": 0, "tasks": {},
    }
    for window_indices, batch in loader:
        target = batch["action"]
        batch_size_actual = target.shape[0]
        if tuple(target.shape) != (batch_size_actual, int(policy.horizon), 7):
            raise ValueError("Targets must cover the policy's full action horizon and seven channels")
        uid_values = batch["obs"]["task_uid"].reshape(batch_size_actual, -1)
        if (not torch.isfinite(uid_values).all()
                or not torch.equal(uid_values, uid_values[:, :1].expand_as(uid_values))
                or not torch.equal(uid_values, uid_values.round())):
            raise ValueError("Every observation window must have one finite integer task_uid")
        uids = uid_values[:, 0].to(dtype=torch.int64)
        if names is not None and any(int(uid) not in names for uid in uids):
            raise ValueError("Dataset contains a task_uid missing from task_names")

        obs = dict_apply(batch["obs"], lambda value: maybe_to_device(value, device))
        noise = _window_noise(window_indices.tolist(), int(policy.horizon), 7, seed).to(device)
        prediction = policy.predict_action(obs, noise=noise)["action_pred"]
        if tuple(prediction.shape) != tuple(target.shape):
            raise ValueError("action_pred must have the same full-window shape as dataset actions")
        # Accumulate in float64 after subtracting in float64 so that long uneven
        # shards retain precision and use exactly the same source action units.
        error = prediction.to(dtype=torch.float64) - target.to(device=device, dtype=torch.float64)
        if not torch.isfinite(error).all():
            raise ValueError("Action predictions and targets must be finite")
        channel_sse = error.square().sum(dim=1).cpu()
        for uid in uids.unique().tolist():
            selected = channel_sse[uids == uid]
            windows = len(selected)
            task = result["tasks"].setdefault(str(uid), {
                "windows": 0, "elements": 0, "sse": 0.,
                "translation_sse": 0., "rotation_sse": 0., "gripper_sse": 0.,
            })
            task["windows"] += windows
            task["elements"] += windows * int(policy.horizon) * 7
            task["sse"] += selected.sum().item()
            task["translation_sse"] += selected[:, :3].sum().item()
            task["rotation_sse"] += selected[:, 3:6].sum().item()
            task["gripper_sse"] += selected[:, 6].sum().item()
        result["window_count"] += batch_size_actual
        result["dataset_index_sum"] += sum(window_indices.tolist())
    return result


def merge_action_mse_results(results, task_names=None):
    """Validate exact rank coverage, then emit macro and micro W&B metrics."""
    names = _names_by_uid(task_names)
    if not results:
        raise ValueError("Action MSE requires rank results")
    first = results[0]
    size, world_size = first["dataset_size"], first["world_size"]
    if size < 1 or len(results) != world_size:
        raise ValueError("Action MSE requires a nonempty dataset and every rank")
    if {result["rank"] for result in results} != set(range(world_size)):
        raise ValueError("Action MSE rank results must be complete and unique")
    tasks = {}
    for result in results:
        for key in ("world_size", "dataset_size", "seed", "horizon", "action_dim"):
            if result[key] != first[key]:
                raise ValueError(f"Inconsistent action MSE rank metadata: {key}")
        indices = action_mse_shard_indices(size, result["rank"], world_size)
        expected_sum = (len(indices) * (indices[0] + indices[-1]) // 2) if indices else 0
        if result["window_count"] != len(indices) or result["dataset_index_sum"] != expected_sum:
            raise ValueError("Action MSE shard did not evaluate its exact dataset indices")
        if sum(task["windows"] for task in result["tasks"].values()) != len(indices):
            raise ValueError("Action MSE task window counts do not match the rank's shard")
        for uid, task in result["tasks"].items():
            if (task["windows"] < 1 or int(task["windows"]) != task["windows"]
                    or task["elements"] != task["windows"] * first["horizon"] * 7):
                raise ValueError("Invalid action MSE task counts")
            for key in ("sse", "translation_sse", "rotation_sse", "gripper_sse"):
                if not math.isfinite(task[key]) or task[key] < 0:
                    raise ValueError("Invalid action MSE sum of squared errors")
            if not math.isclose(task["sse"], sum(task[key] for key in (
                    "translation_sse", "rotation_sse", "gripper_sse")), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Action MSE channel sums disagree with full action SSE")
            total = tasks.setdefault(int(uid), {key: 0 for key in task})
            for key, value in task.items():
                total[key] += value
    if names is not None and set(tasks) != set(names):
        raise ValueError("Action MSE must evaluate every expected task exactly once across shards")
    if names is None:
        names = {uid: f"task_{uid}" for uid in tasks}
    log = {}
    per_task_mse = []
    for uid, task in sorted(tasks.items()):
        prefix = f"val/{names[uid]}"
        mse = task["sse"] / task["elements"]
        per_task_mse.append(mse)
        log.update({f"{prefix}/action_mse": mse,
                    f"{prefix}/action_mse_sse": task["sse"],
                    f"{prefix}/action_mse_elements": task["elements"],
                    f"{prefix}/action_mse_windows": task["windows"]})
        per_channel_count = task["elements"] // 7
        for component, channels in (("translation", 3), ("rotation", 3), ("gripper", 1)):
            log[f"{prefix}/action_mse_{component}"] = task[f"{component}_sse"] / (per_channel_count * channels)
    macro_mse = sum(per_task_mse) / len(per_task_mse)
    total_elements = sum(task["elements"] for task in tasks.values())
    log.update({
        "val/action_mse": macro_mse,
        "val/action_mse_micro": sum(task["sse"] for task in tasks.values()) / total_elements,
        "val/action_mse_windows": size,
        "val/action_mse_elements": total_elements,
        "val/action_mse_tasks": len(tasks),
        "test_reconst_mse": macro_mse,
    })
    return log
