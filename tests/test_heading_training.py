import math

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import zarr

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.heading_predictor import ObservationHeadingPredictor
from oat.perception.state_encoder import ProjectionStateEncoder
from oat.symmetry.so2_chunk import SO2ChunkSpec
from scripts.train_heading_reference import build_datasets, evaluate, fit_train_normalizer


def test_episode_split_excludes_subsampled_train_episodes_and_fits_train_only(tmp_path):
    shape_meta = {"obs": {"state": {"type": "state", "shape": [3]}}}
    path = tmp_path / "episodes.zarr"
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    data.create_dataset("state", data=np.arange(36, dtype=np.float32).reshape(12, 3))
    data.create_dataset("action", data=np.arange(84, dtype=np.float32).reshape(12, 7))
    root.create_group("meta").create_dataset("episode_ends", data=np.array([3, 6, 9, 12]))
    cfg = OmegaConf.create({
        "n_obs_steps": 2, "horizon": 2, "split_seed": 42,
        "task": {"policy": {"dataset": {
            "_target_": "oat.dataset.zarr_dataset.ZarrDataset",
            "zarr_path": str(path), "obs_keys": ["state", "missing_rgb"],
            "n_obs_steps": 2, "n_action_steps": 2, "val_ratio": 0.25,
            "max_train_episodes": 1,
        }}},
    })
    train, validation, _ = build_datasets(cfg, shape_meta)
    assert train.train_mask.sum() == 1
    assert validation.train_mask.sum() == 1
    assert not (train.train_mask & validation.train_mask).any()
    assert len(train) == len(validation) == 3
    assert train.obs_keys == ["state"]
    assert "missing_rgb" not in train.replay_buffer.keys()
    selected_episode = int(np.flatnonzero(train.train_mask)[0])
    selected = np.arange(36, dtype=np.float32).reshape(12, 3)[
        selected_episode * 3:(selected_episode + 1) * 3
    ]
    normalizer = fit_train_normalizer(train, shape_meta)
    stats = normalizer["state"].get_input_stats()
    np.testing.assert_allclose(stats["min"].detach().numpy(), selected.min(axis=0))
    np.testing.assert_allclose(stats["max"].detach().numpy(), selected.max(axis=0))


def test_validation_loss_and_metrics_do_not_depend_on_batch_partition():
    spec = SO2ChunkSpec.translation_only()
    shape_meta = {"obs": {"robot0_eef_pos": {"type": "state", "shape": [3]}}}
    model = ObservationHeadingPredictor(ProjectionStateEncoder(shape_meta), hidden_dim=4)
    observations = {"robot0_eef_pos": torch.zeros(4, 2, 3)}
    actions = torch.zeros(4, 2, 7)
    actions[2, :, 0] = 1
    actions[3, :, 0] = -1
    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions.flatten(0, 1),
                    "robot0_eef_pos": observations["robot0_eef_pos"].flatten(0, 1)})
    model.set_normalizer(normalizer, spec)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
    cfg = OmegaConf.create({
        "training": {"max_val_batches": None},
        "evaluation": {"motion_key": "robot0_eef_pos", "min_motion": 1e-4,
                       "confidence_threshold": 0.5},
    })
    together = evaluate(model, [{"obs": observations, "action": actions}], spec, "cpu", cfg)
    batches = [{"obs": {key: value[start:start + 2] for key, value in observations.items()},
                "action": actions[start:start + 2]} for start in (0, 2)]
    apart = evaluate(model, batches, spec, "cpu", cfg)
    assert together["loss"] == pytest.approx(apart["loss"])
    assert apart["heading_loss"] == pytest.approx(1.0)
    assert apart["validity_loss"] == pytest.approx(math.log(2))
    assert apart["validity_brier"] == pytest.approx(0.25)
    assert apart["valid_heading"]["count"] == 2
    assert apart["valid_heading"]["mae_deg"] == pytest.approx(90.0)
    assert apart["stationary_valid_heading"]["count"] == 2
    assert apart["moving_valid_heading"]["mae_deg"] is None
