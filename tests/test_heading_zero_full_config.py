"""Configuration contract for the full-data zero-jitter heading policy."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'oat/config')


def config(name='train_flowpolicy_heading_zero', overrides=None):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=name, overrides=overrides or [])


def test_zero_heading_full_config_matches_mini_mechanism():
    cfg = config()
    assert cfg.policy._target_ == 'oat.policy.flow_policy_heading_zero.HeadingZeroFlowPolicy'
    assert (cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps) == (16, 8, 2)
    assert cfg.policy.heading_mode == 'condition'
    assert cfg.policy.source_mode == 'heading'
    assert cfg.policy.source_jitter == 'zero'
    assert OmegaConf.to_container(cfg.policy.source_vector_blocks) == [[0, 1]]
    assert cfg.policy.obs_encoder.vision_encoder.eval_fixed_crop
    assert cfg.optimizer.policy_lr == 5e-5
    assert cfg.optimizer.obs_enc_lr == 1e-5
    assert cfg.optimizer.heading_lr == 1e-3
    assert cfg.policy.heading_loss_weight == cfg.policy.validity_loss_weight == .1
    assert cfg.training.max_grad_norm == 1.
    assert cfg.training.max_train_steps is None
    assert cfg.training.max_val_steps is None
    assert cfg.training.max_reconst_steps is None
    assert cfg.training.num_epochs == 5001
    assert cfg.training.lr_scheduler == 'constant_with_warmup'
    assert cfg.training.lr_warmup_steps == 100
    assert cfg.checkpoint.topk.monitor_key == 'test_reconst_mse'
    assert cfg.checkpoint.topk.mode == 'min'
    assert cfg.training.checkpoint_every == cfg.training.sample_every
    assert 'reference_checkpoint' not in cfg.policy


def test_changing_training_seed_keeps_full_dataset_holdout_fixed():
    cfg = config(overrides=['seed=43'])
    assert cfg.training.seed == 43
    assert cfg.task.policy.dataset.seed == 42
    assert cfg.task.policy.dataset._target_ == 'oat.dataset.heading_zarr_dataset.HeadingZarrDataset'
    assert cfg.task.policy.dataset.val_ratio == .1
    assert cfg.task.policy.dataset.zarr_path == 'data/libero/libero10_N500.zarr'
    assert set(cfg.task.policy.dataset.rgb_keys) == {'agentview_rgb', 'robot0_eye_in_hand_rgb'}


def test_legacy_flow_configuration_is_unchanged():
    cfg = config('train_flowpolicy')
    assert cfg.policy._target_ == 'oat.policy.flow_policy.FlowPolicy'
    assert cfg.task.policy.dataset._target_ == 'oat.dataset.zarr_dataset.ZarrDataset'
    assert 'heading_lr' not in cfg.optimizer
    assert 'source_jitter' not in cfg.policy
    assert cfg.checkpoint.topk.monitor_key == 'mean_success_rate'
