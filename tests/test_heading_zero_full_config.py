"""Public heading-flow configurations: exactly two priors and three backbones."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'oat/config'
BACKBONES = {'transfomer': 'transformer', 'unet': 'unet', 'starvlaDiT': 'starvla_dit'}
NAMES = {f'train_flowpolicy_heading{prior}_{backbone}'
         for prior in ('zero', 'gaussian') for backbone in BACKBONES}


def config(name, overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name, overrides=overrides or [])


def test_only_six_standalone_heading_training_configs_are_exposed():
    assert {p.stem for p in CONFIG_DIR.glob('train_flowpolicy_heading*.yaml')} == NAMES
    for name in NAMES:
        raw = OmegaConf.load(CONFIG_DIR / (name + '.yaml'))
        assert OmegaConf.to_container(raw.defaults) == [{'task/policy': 'libero/libero10'}, '_self_']
        assert raw.name == name


@pytest.mark.parametrize('prior', ['zero', 'gaussian'])
@pytest.mark.parametrize('suffix,backbone', BACKBONES.items())
def test_heading_config_has_matched_policy_and_training_settings(prior, suffix, backbone):
    cfg = config(f'train_flowpolicy_heading{prior}_{suffix}')
    target = 'HeadingZeroFlowPolicy' if prior == 'zero' else 'HeadingGaussianFlowPolicy'
    assert cfg.policy._target_ == f'oat.policy.flow_policy_heading_{prior}.{target}'
    assert cfg.policy.backbone_type == backbone
    assert (cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps) == (16, 8, 2)
    assert cfg.policy.heading_mode == 'condition'
    assert cfg.policy.source_mode == 'heading'
    assert cfg.policy.source_jitter == 'zero'
    assert OmegaConf.to_container(cfg.policy.source_vector_blocks) == [[0, 1]]
    assert cfg.policy.obs_encoder.vision_encoder.eval_fixed_crop
    assert list(cfg.policy.obs_encoder.vision_encoder.crop_shape) == [116, 116]
    assert cfg.optimizer.policy_lr == cfg.optimizer.obs_enc_lr == cfg.optimizer.heading_lr == 1e-4
    assert cfg.policy.heading_loss_weight == cfg.policy.validity_loss_weight == .1
    assert cfg.training.max_grad_norm == 1.
    assert cfg.training.max_train_steps is None
    assert cfg.training.max_val_steps is None
    assert cfg.training.max_reconst_steps is None
    assert cfg.training.num_epochs == 61
    assert cfg.training.lr_scheduler == 'constant_with_warmup'
    assert cfg.training.lr_warmup_steps == 500
    assert cfg.checkpoint.topk.monitor_key == 'test_reconst_mse'
    assert cfg.checkpoint.topk.mode == 'min'
    assert cfg.training.checkpoint_every == cfg.training.sample_every == 5
    assert cfg.dataloader.batch_size == 64
    assert 'reference_checkpoint' not in cfg.policy
    if prior == 'gaussian':
        assert (cfg.policy.prior_heading_mean, cfg.policy.prior_parallel_std,
                cfg.policy.prior_perpendicular_std) == (.5, 1., .5)
    if backbone == 'starvla_dit':
        assert cfg.policy.backbone_kwargs.num_register_tokens == 32
    if backbone == 'unet':
        assert list(cfg.policy.backbone_kwargs.down_dims) == [128, 256, 512]


@pytest.mark.parametrize('name', sorted(NAMES))
def test_changing_training_seed_keeps_dataset_holdout_fixed(name):
    cfg = config(name, ['seed=43'])
    assert cfg.training.seed == 43
    assert cfg.task.policy.dataset.seed == 42
    assert cfg.task.policy.dataset._target_ == 'oat.dataset.heading_zarr_dataset.HeadingZarrDataset'
    assert cfg.task.policy.dataset.val_ratio == .1
    assert cfg.task.policy.dataset.zarr_path == 'data/libero/libero10_N500.zarr'
    assert set(cfg.task.policy.dataset.rgb_keys) == {'agentview_rgb', 'robot0_eye_in_hand_rgb'}


def test_plain_flow_configuration_is_unchanged():
    cfg = config('train_flowpolicy')
    assert cfg.policy._target_ == 'oat.policy.flow_policy.FlowPolicy'
    assert cfg.task.policy.dataset._target_ == 'oat.dataset.zarr_dataset.ZarrDataset'
    assert 'heading_lr' not in cfg.optimizer
    assert 'source_jitter' not in cfg.policy
    assert cfg.checkpoint.topk.monitor_key == 'mean_success_rate'
