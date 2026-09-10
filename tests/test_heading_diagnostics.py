"""Offline diagnostics must use fixed held-out windows and keep labels out of inference."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.diagnose_heading_policy import (
    diagnose_batches, explicit_noise, raw_heading, score_predictions,
    select_validation_windows, summarize_metric_rows,
)


class Replay(dict):
    def __init__(self):
        super().__init__(task_uid=np.repeat([4, 4, 7, 7], 4)[:, None])
        self.episode_ends = np.array([4, 8, 12, 16])


def split_fixture():
    from oat.common.seq_sampler import create_indices
    replay = Replay()
    heldout = np.array([False, True, False, True])
    indices = create_indices(replay.episode_ends, 4, heldout, 1, 2)
    train = SimpleNamespace(replay_buffer=replay, train_mask=~heldout, _validation_episode_mask=heldout)
    validation = SimpleNamespace(replay_buffer=replay, pad_before=1, seq_sampler=SimpleNamespace(indices=indices))
    return train, validation


def test_selection_is_task_balanced_nested_and_excludes_training_episodes():
    train, validation = split_fixture()
    small, metadata = select_validation_windows(train, validation, 2, 42)
    large, _ = select_validation_windows(train, validation, 3, 42)
    for task in [4, 7]:
        assert [r for r in small if r["task_uid"] == task] == [r for r in large if r["task_uid"] == task][:2]
    assert {row["episode_index"] for row in small} == {1, 3}
    assert metadata["train_episode_indices"] == [0, 2]
    assert metadata["validation_episode_indices"] == [1, 3]
    assert all(0 <= row["first_action_episode_index"] < 4 for row in small)
    train.train_mask[1] = True
    with pytest.raises(ValueError, match="training episodes"):
        select_validation_windows(train, validation, 2, 42)


class FakePolicy:
    device = torch.device("cpu")
    dtype = torch.float32
    horizon = 4
    action_dim = 7
    n_action_steps = 2
    heading_horizon = 4
    heading_xy_rms = 1.
    min_target_confidence = .05
    min_source_confidence = .5
    source_jitter = "zero"

    def __init__(self):
        self.generated = []

    def reset(self):
        pass

    def heading_targets(self, actions):
        return raw_heading(actions, 4, 1., .05)

    def predict_action(self, obs, noise, angular_jitter, return_source):
        assert set(obs) == {"state"}
        assert torch.is_inference_mode_enabled()
        assert return_source
        self.generated.extend(noise.clone().unbind(0))
        batch = noise.shape[0]
        return {"action_pred": noise, "source_active": torch.ones(batch, dtype=torch.bool),
                "prediction": {"theta": torch.zeros(batch), "direction": torch.tensor([[1., 0.]]).repeat(batch, 1),
                               "confidence": torch.full((batch,), .9)}}


class Validation:
    def __init__(self, target_sign=1.):
        self.accesses = []
        self.target_sign = target_sign

    def __getitem__(self, index):
        self.accesses.append(index)
        actions = torch.zeros(4, 7)
        actions[:, 0] = self.target_sign
        return {"obs": {"state": torch.tensor([[float(index)], [float(index)]])}, "action": actions}


def windows():
    return [{"validation_index": i, "task_uid": 4, "episode_index": 1, "first_action_episode_index": i} for i in range(3)]


def test_explicit_noise_does_not_depend_on_batch_partition_or_global_rng():
    policy = FakePolicy()
    expected, _ = explicit_noise(policy, windows(), 123)
    torch.randn(100)
    first, _ = explicit_noise(policy, windows()[:1], 123)
    rest, _ = explicit_noise(policy, windows()[1:], 123)
    torch.testing.assert_close(expected, torch.cat((first, rest)))


def test_images_are_sampled_once_and_future_labels_cannot_change_inference():
    policy = FakePolicy()
    validation = Validation(1.)
    rows = diagnose_batches(policy, validation, windows(), 2, [101, 102])
    assert validation.accesses == [0, 1, 2]
    assert len(rows) == 6
    changed_policy = FakePolicy()
    changed_rows = diagnose_batches(changed_policy, Validation(-1.), windows(), 2, [101, 102])
    for original, changed in zip(policy.generated, changed_policy.generated):
        torch.testing.assert_close(original, changed)
    assert rows[0]["metrics"]["predicted_heading_mae_deg"] == 0.
    assert changed_rows[0]["metrics"]["predicted_heading_mae_deg"] == pytest.approx(180.)
    summary = summarize_metric_rows(rows)
    assert summary["unique_window_count"] == 3
    assert summary["noise_draw_count"] == 6


def test_generated_heading_reports_collapse_as_missing_angle_with_zero_coverage():
    policy = FakePolicy()
    target = torch.zeros(1, 4, 7)
    target[:, :, 0] = 1.
    with torch.inference_mode():
        result = policy.predict_action({"state": torch.zeros(1, 2, 1)}, torch.zeros_like(target), torch.zeros(1), True)
        metric = score_predictions(policy, result, target)[0]
    assert metric["prefix_channel_mse"] == [1., 0., 0., 0., 0., 0., 0.]
    assert metric["generated_heading_mae_deg_both_valid"] is None
    assert metric["generated_heading_valid_on_target_valid_fraction"] == 0.
    assert metric["predicted_heading_mae_deg"] == 0.


def sign_result(policy, actions):
    with torch.inference_mode():
        return policy.predict_action({"state": torch.zeros(len(actions), 2, 1)},
                                     actions, torch.zeros(len(actions)), True)


def test_large_binary_action_mse_can_have_perfect_sign_commands():
    policy = FakePolicy()
    target = torch.ones(1, 4, 7)
    generated = target.clone()
    target[0, :, 6] = torch.tensor([-1., 1., -1., 1.])
    # Prefix commands agree despite MSE=4. Suffix commands disagree and must not
    # contaminate metrics for the two actions the policy actually executes.
    generated[0, :, 6] = torch.tensor([-3., 3., 1., -1.])
    before = generated.clone()
    result = sign_result(policy, generated)
    default = score_predictions(policy, result, target)[0]
    metric = score_predictions(policy, result, target, binary_action_indices=(6,))[0]
    assert metric['prefix_channel_mse'][6] == 4.
    assert metric['prefix_binary_sign_disagreement_rate'] == [0.]
    assert metric['prefix_binary_sign_disagreement_count'] == [0]
    assert metric['prefix_binary_sign_sample_count'] == [2]
    assert metric['prefix_binary_sign_confusion_counts'] == [[[1, 0, 0], [0, 0, 0], [0, 0, 1]]]
    assert {key: metric[key] for key in default} == default
    assert not any('binary_sign' in key for key in default)
    torch.testing.assert_close(generated, before, rtol=0, atol=0)


def test_sign_confusion_preserves_zero_and_configured_dimension_order():
    policy = FakePolicy()
    policy.n_action_steps = 4
    target = torch.ones(1, 4, 7)
    generated = target.clone()
    target[0, :, 2] = torch.tensor([-1., 0., 1., 1.])
    generated[0, :, 2] = torch.tensor([-2., 1., 0., -1.])
    generated[0, :, 6] = -1.
    metric = score_predictions(policy, sign_result(policy, generated), target, (6, 2))[0]
    assert metric['prefix_binary_sign_disagreement_rate'] == [1., .75]
    assert metric['prefix_binary_sign_disagreement_count'] == [4, 3]
    assert metric['prefix_binary_sign_sample_count'] == [4, 4]
    assert metric['prefix_binary_sign_confusion_counts'] == [
        [[0, 0, 0], [0, 0, 0], [4, 0, 0]],
        [[1, 0, 0], [0, 0, 1], [1, 1, 0]],
    ]
    rows = [{'metrics': metric, 'validation_index': i} for i in range(2)]
    pooled = summarize_metric_rows(rows)['metrics']
    assert pooled['prefix_binary_sign_disagreement_rate'] == [1., .75]
    assert pooled['prefix_binary_sign_disagreement_count'] == [8, 6]
    assert pooled['prefix_binary_sign_sample_count'] == [8, 8]
    np.testing.assert_array_equal(pooled['prefix_binary_sign_confusion_counts'],
                                  2 * np.asarray(metric['prefix_binary_sign_confusion_counts']))


def test_optional_sign_scoring_does_not_change_noise_or_inference():
    ordinary, scored = FakePolicy(), FakePolicy()
    default = diagnose_batches(ordinary, Validation(), windows(), 2, [101])
    additional = diagnose_batches(scored, Validation(), windows(), 2, [101], binary_action_indices=(6,))
    for original, current in zip(ordinary.generated, scored.generated):
        torch.testing.assert_close(original, current, rtol=0, atol=0)
    for previous, current in zip(default, additional):
        assert {key: current['metrics'][key] for key in previous['metrics']} == previous['metrics']
        assert current['metrics']['prefix_binary_sign_sample_count'] == [2]


@pytest.mark.parametrize('value, expected', [('', ()), ('6', (6,)), (' 6, 2 ', (6, 2))])
def test_binary_action_cli_parser(value, expected):
    from scripts.diagnose_heading_policy import parse_binary_action_indices
    assert parse_binary_action_indices(value) == expected


@pytest.mark.parametrize('value', ['-1', '6,6', '1,,2', 'a', '1.5'])
def test_binary_action_cli_rejects_invalid_indices(value):
    import argparse
    from scripts.diagnose_heading_policy import parse_binary_action_indices
    with pytest.raises(argparse.ArgumentTypeError):
        parse_binary_action_indices(value)


@pytest.mark.parametrize('indices', [(7,), (-1,), (True,), (6, 6), (1.5,)])
def test_binary_dimensions_are_validated_before_inference(indices):
    policy = FakePolicy()
    with pytest.raises(ValueError, match='binary action indices'):
        diagnose_batches(policy, Validation(), windows(), 2, [101], binary_action_indices=indices)
    assert not policy.generated
