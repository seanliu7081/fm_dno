"""Raw full-horizon action error, exact sharding, and reproducible inference."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from oat.common.action_mse import (
    action_mse_shard_indices, evaluate_action_mse, merge_action_mse_results,
)


class ActionDataset(Dataset):
    def __init__(self, uids=(0, 0, 0, 1, 1)):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        target = torch.full((16, 7), 10. * (index + 1))
        return {"obs": {"task_uid": torch.tensor([[self.uids[index]], [self.uids[index]]]),
                        "target": target, "index": torch.tensor(index)}, "action": target}


class KnownErrorPolicy(nn.Module):
    horizon = 16
    action_dim = 7

    def __init__(self, use_noise=False):
        super().__init__()
        self.eval()
        self.seen = []
        self.noises = {}
        self.use_noise = use_noise

    def predict_action(self, obs, noise):
        for index, source in zip(obs["index"].tolist(), noise):
            self.seen.append(index)
            self.noises[index] = source.clone()
        # All error lies outside the shorter executable prefix, proving the
        # evaluator uses full raw action_pred rather than truncated action.
        error = torch.zeros_like(obs["target"])
        error[:, 8:, :] = 1. + obs["task_uid"][:, :1].to(dtype=torch.float32)
        if self.use_noise:
            error += noise
        return {"action": obs["target"][:, :8], "action_pred": obs["target"] + error}


def evaluate(policy=None, dataset=None, **kwargs):
    return evaluate_action_mse(policy or KnownErrorPolicy(), dataset or ActionDataset(), "cpu", **kwargs)


def test_known_full_horizon_raw_action_mse_and_macro_weighting():
    # Task 0: 3 windows, SSE 3 * 8 * 7. Task 1: 2 windows, SSE 2 * 8 * 7 * 4.
    result = evaluate(batch_size=2)
    log = merge_action_mse_results([result], ["stack", "coffee"])
    assert log["val/stack/action_mse"] == .5
    assert log["val/coffee/action_mse"] == 2.
    assert log["val/action_mse"] == 1.25
    assert log["test_reconst_mse"] == 1.25
    assert log["val/action_mse_micro"] == 1.1
    assert log["val/action_mse_windows"] == 5
    assert log["val/action_mse_elements"] == 5 * 16 * 7
    assert log["val/stack/action_mse_sse"] == 3 * 8 * 7
    for component in ("translation", "rotation", "gripper"):
        assert log[f"val/coffee/action_mse_{component}"] == 2.


@pytest.mark.parametrize("world_size", [2, 3, 7])
def test_uneven_shards_and_empty_ranks_evaluate_every_window_once(world_size):
    policies = [KnownErrorPolicy() for _ in range(world_size)]
    shards = [evaluate(policy, rank=rank, world_size=world_size, batch_size=2)
              for rank, policy in enumerate(policies)]
    assert sorted(index for policy in policies for index in policy.seen) == list(range(5))
    expected = merge_action_mse_results([evaluate()])
    assert merge_action_mse_results(shards) == expected
    for rank in range(world_size):
        assert policies[rank].seen == list(action_mse_shard_indices(5, rank, world_size))


def test_noise_matches_across_policy_batch_size_and_rank_and_preserves_torch_rng():
    torch.manual_seed(51)
    state = torch.get_rng_state()
    full = KnownErrorPolicy(use_noise=True)
    full_result = evaluate(full, batch_size=4, seed=81)
    assert torch.equal(torch.get_rng_state(), state)
    shards = []
    all_noise = {}
    for rank in range(2):
        policy = KnownErrorPolicy(use_noise=True)
        shards.append(evaluate(policy, rank=rank, world_size=2, batch_size=1, seed=81))
        all_noise.update(policy.noises)
    for index in range(5):
        torch.testing.assert_close(full.noises[index], all_noise[index], rtol=0, atol=0)
    assert not torch.equal(full.noises[0], full.noises[1])
    assert merge_action_mse_results(shards)["val/action_mse"] == pytest.approx(
        merge_action_mse_results([full_result])["val/action_mse"], rel=1e-12)


def test_loader_workers_follow_the_same_exact_shard():
    full = merge_action_mse_results([evaluate(num_workers=0)])
    assert merge_action_mse_results([evaluate(num_workers=2)]) == full


@pytest.mark.parametrize("field,value", [
    ("window_count", 6), ("dataset_index_sum", 11), ("world_size", 2),
])
def test_merge_rejects_incomplete_or_duplicate_window_counts(field, value):
    result = evaluate()
    result[field] = value
    with pytest.raises(ValueError):
        merge_action_mse_results([result])


def test_merge_rejects_missing_tasks_duplicate_ranks_and_nonfinite_errors():
    result = evaluate()
    with pytest.raises(ValueError, match="every expected task"):
        merge_action_mse_results([result], ["one", "two", "three"])
    duplicate = evaluate(rank=0, world_size=2)
    with pytest.raises(ValueError, match="complete and unique"):
        merge_action_mse_results([duplicate, copy.deepcopy(duplicate)])
    result["tasks"]["0"]["sse"] = float("nan")
    with pytest.raises(ValueError, match="sum of squared errors"):
        merge_action_mse_results([result])


def test_rejects_short_action_predictions_and_training_mode():
    policy = KnownErrorPolicy()
    original = policy.predict_action
    policy.predict_action = lambda obs, noise: {"action_pred": original(obs, noise)["action"]}
    with pytest.raises(ValueError, match="full-window shape"):
        evaluate(policy)
    policy.train()
    with pytest.raises(ValueError, match="eval mode"):
        evaluate(policy)


def test_rejects_mixed_task_ids_and_unknown_tasks():
    class MixedTaskDataset(ActionDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["obs"]["task_uid"][0, 0] = 99
            return sample

    with pytest.raises(ValueError, match="one finite integer"):
        evaluate(dataset=MixedTaskDataset())
    with pytest.raises(ValueError, match="missing from task_names"):
        evaluate(task_names=["stack"])
