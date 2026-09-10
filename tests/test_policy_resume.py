"""Training continuation must preserve EMA, optimizer, scheduler, and progress."""

import copy
import random
from types import SimpleNamespace

import dill
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from oat.model.diffusion.ema_model import EMAModel
from oat.policy.base_policy import BasePolicy
from oat.workspace.train_policy import TrainPolicyWorkspace


class TinyResumePolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, value):
        # Exercise the restored Python, NumPy, and CPU torch streams as well as
        # optimizer/EMA schedules, while explicitly keeping the batches fixed.
        perturbation = .01 * torch.randn_like(value[:, :2])
        target = value[:, :2] + perturbation + .01 * (random.random() + np.random.rand())
        return (self.projection(value) - target).square().mean()

    def get_optimizer(self, lr=.03):
        return torch.optim.AdamW(self.parameters(), lr=lr)


CPU_ACCELERATOR = SimpleNamespace(device=torch.device('cpu'), num_processes=1, process_index=0)


def configuration(scheduler='constant_with_warmup', accumulation=1, use_ema=True):
    return OmegaConf.create({
        '_target_': 'oat.workspace.train_policy.TrainPolicyWorkspace',
        'policy': {'_target_': 'test_policy_resume.TinyResumePolicy'},
        'optimizer': {'lr': .03},
        'training': {
            'use_ema': use_ema, 'gradient_accumulate_every': accumulation,
            'lr_scheduler': scheduler, 'lr_warmup_steps': 4, 'num_epochs': 10,
        },
        'ema': {'_target_': 'oat.model.diffusion.ema_model.EMAModel',
                'power': .75, 'inv_gamma': 1., 'max_value': .9999},
    })


def make_workspace(cfg, directory):
    return TrainPolicyWorkspace(cfg, output_dir=str(directory), lazy_instantiation=False)


def update(workspace, ema, scheduler, count):
    inputs = torch.linspace(-1., 1., 12).reshape(4, 3)
    recorded = []
    accumulation = workspace.cfg.training.gradient_accumulate_every
    for _ in range(count):
        for _ in range(accumulation):
            (workspace.model(inputs) / accumulation).backward()
            workspace.global_step += 1
        recorded.append([group['lr'] for group in workspace.optimizer.param_groups])
        workspace.optimizer.step()
        workspace.optimizer.zero_grad(set_to_none=True)
        workspace.optimizer_step += 1
        scheduler.step()
        if ema is not None:
            ema.step(workspace.model)
    return recorded


def assert_parameters_equal(expected, actual):
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('scheduler_name', ['constant_with_warmup', 'cosine'])
@pytest.mark.parametrize('accumulation', [1, 3])
def test_workspace_resume_matches_uninterrupted_updates_and_rng(
    tmp_path, scheduler_name, accumulation,
):
    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)
    cfg = configuration(scheduler_name, accumulation)
    reference = make_workspace(cfg, tmp_path)
    ema, scheduler = reference._create_training_dynamics(cfg, 24, reference.ema_model)
    update(reference, ema, scheduler, 6)
    reference.epoch = 2
    saved_epoch, saved_global_step = reference.epoch, reference.global_step
    reference._capture_training_state(ema, scheduler, CPU_ACCELERATOR)
    checkpoint = tmp_path / 'resume.ckpt'
    reference.save_checkpoint(path=checkpoint, use_thread=False)
    # Capture must not rewrite the epoch number stored with old metric names.
    assert reference.epoch == saved_epoch
    assert reference.global_step == saved_global_step
    expected_lrs = update(reference, ema, scheduler, 7)

    restored = make_workspace(cfg, tmp_path)
    payload = restored.load_checkpoint(checkpoint)
    assert restored.epoch == saved_epoch  # inference loading alone changes no progress
    restored._resume_training_progress(payload, legacy_epoch_complete=True)
    assert restored.epoch == saved_epoch + 1
    assert restored.global_step == saved_global_step + 1
    assert restored.optimizer_step == 6
    ema_before = copy.deepcopy(restored.ema_model.state_dict())
    restored_ema, restored_scheduler = restored._create_training_dynamics(cfg, 24, restored.ema_model)
    assert restored_ema.optimization_step == 6 and restored_ema.decay > 0
    for key, value in ema_before.items():
        torch.testing.assert_close(restored.ema_model.state_dict()[key], value, rtol=0, atol=0)
    restored._restore_training_rng(CPU_ACCELERATOR)
    actual_lrs = update(restored, restored_ema, restored_scheduler, 7)
    assert actual_lrs == expected_lrs
    assert_parameters_equal(reference.model, restored.model)
    assert_parameters_equal(reference.ema_model, restored.ema_model)
    assert restored_ema.state_dict() == ema.state_dict()
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_legacy_resume_uses_adam_counter_and_retains_ema(tmp_path):
    cfg = configuration(accumulation=3)
    original = make_workspace(cfg, tmp_path)
    ema, scheduler = original._create_training_dynamics(cfg, 24, original.ema_model)
    update(original, ema, scheduler, 8)
    original.epoch, original.global_step = 2, 71  # deliberately unlike Adam's actual count
    checkpoint = tmp_path / 'legacy.ckpt'
    original.save_checkpoint(path=checkpoint, use_thread=False)
    payload = torch.load(checkpoint, pickle_module=dill)
    payload['pickles'] = {key: value for key, value in payload['pickles'].items()
                          if key in ('epoch', 'global_step', '_output_dir')}
    restored = make_workspace(cfg, tmp_path)
    restored.load_payload(payload)
    with pytest.warns(RuntimeWarning, match='Legacy training checkpoint'):
        restored._resume_training_progress(payload, legacy_epoch_complete=True)
    assert restored.optimizer_step == 8 and restored.epoch == 3 and restored.global_step == 72
    restored_ema, restored_scheduler = restored._create_training_dynamics(cfg, 24, restored.ema_model)
    assert restored_ema.optimization_step == 8
    before = copy.deepcopy(restored.ema_model.state_dict())
    update(restored, restored_ema, restored_scheduler, 1)
    decay = restored_ema.decay
    assert decay > 0
    for key, online in restored.model.state_dict().items():
        expected = before[key] * decay + online * (1 - decay)
        torch.testing.assert_close(restored.ema_model.state_dict()[key], expected)
        if online.numel():
            assert not torch.equal(restored.ema_model.state_dict()[key], online)


@pytest.mark.parametrize('known_workspace', [True, False])
def test_unmarked_legacy_epoch_only_advances_for_known_training_checkpoint(tmp_path, known_workspace):
    cfg = configuration(accumulation=3)
    workspace = make_workspace(cfg, tmp_path)
    workspace.epoch, workspace.global_step = 2, 8
    saved_cfg = copy.deepcopy(cfg)
    if not known_workspace:
        saved_cfg._target_ = 'unrelated.Workspace'
    payload = {'cfg': saved_cfg, 'pickles': {'epoch': b'', 'global_step': b''},
               'state_dicts': {'optimizer': {}}}
    with pytest.warns(RuntimeWarning):
        workspace._resume_training_progress(payload, legacy_epoch_complete=True)
    assert workspace.epoch == (3 if known_workspace else 2)
    assert workspace.optimizer_step == 3  # fallback from logging count / accumulation


def test_new_manual_checkpoint_without_completion_marker_does_not_skip_epoch(tmp_path):
    cfg = configuration()
    workspace = make_workspace(cfg, tmp_path)
    workspace.epoch, workspace.global_step = 3, 20
    path = tmp_path / 'manual.ckpt'
    workspace.save_checkpoint(path=path, use_thread=False)
    restored = make_workspace(cfg, tmp_path)
    payload = restored.load_checkpoint(path)
    restored._resume_training_progress(payload, legacy_epoch_complete=True)
    assert restored.epoch == 3 and restored.global_step == 20


def test_ema_schedule_state_roundtrip_does_not_replace_model_weights():
    model = nn.Linear(2, 1)
    original = EMAModel(copy.deepcopy(model), power=.6, inv_gamma=2., max_value=.97)
    for _ in range(7):
        with torch.no_grad():
            model.weight.add_(.1)
        original.step(model)
    resumed = EMAModel(copy.deepcopy(original.averaged_model))
    resumed.load_state_dict(original.state_dict())
    with torch.no_grad():
        model.weight.add_(.1)
    original.step(model)
    resumed.step(model)
    assert original.state_dict() == resumed.state_dict()
    assert_parameters_equal(original.averaged_model, resumed.averaged_model)


def test_resume_without_ema_restores_scheduler_and_progress(tmp_path):
    cfg = configuration(use_ema=False)
    workspace = make_workspace(cfg, tmp_path)
    assert workspace.ema_model is None
    ema, scheduler = workspace._create_training_dynamics(cfg, 24)
    assert ema is None
    update(workspace, ema, scheduler, 5)
    workspace._capture_training_state(ema, scheduler, CPU_ACCELERATOR)
    path = tmp_path / 'no_ema.ckpt'
    workspace.save_checkpoint(path=path, use_thread=False)
    restored = make_workspace(cfg, tmp_path)
    payload = restored.load_checkpoint(path)
    restored._resume_training_progress(payload, legacy_epoch_complete=True)
    restored_ema, restored_scheduler = restored._create_training_dynamics(cfg, 24)
    assert restored_ema is None and restored.epoch == 1
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored.optimizer.param_groups[0]['lr'] == workspace.optimizer.param_groups[0]['lr']
