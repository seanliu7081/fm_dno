"""Resume the real workspace and W&B tracker on CPU, with every upload mocked."""
from __future__ import annotations

import copy
from functools import partial
import json
from types import SimpleNamespace
from unittest.mock import Mock

from accelerate import Accelerator
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from oat.dataset.base_dataset import BaseDataset
from oat.model.common.normalizer import LinearNormalizer
from oat.policy.base_policy import BasePolicy
from oat.workspace import train_policy as training


class OnlineResumeDataset(BaseDataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            'obs': {'state': torch.tensor([[index / 4., 1.]]),
                    'task_uid': torch.tensor([[index % 2]])},
            'action': torch.full((2, 7), index / 10.),
        }

    def get_normalizer(self):
        # Deliberately differs from the checkpoint to prove restoration wins.
        normalizer = LinearNormalizer()
        normalizer.fit({'action': torch.stack([torch.full((7,), -100.), torch.full((7,), 100.)])})
        return normalizer


class OnlineResumePolicy(BasePolicy):
    horizon = 2
    action_dim = 7

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 7)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict, noise=None):
        normalized = self.projection(obs_dict['state']).expand(-1, self.horizon, -1)
        action = self.normalizer['action'].unnormalize(normalized)
        return {'action': action[:, :1], 'action_pred': action}

    def forward(self, batch):
        prediction = self.predict_action(batch['obs'])['action_pred']
        return (prediction - batch['action']).square().mean()


class FakeConfig(dict):
    def update(self, values=None, allow_val_change=False, **kwargs):
        super().update(values or {}, **kwargs)


class FakeRun:
    """Model W&B's monotonic history while never creating an actual run."""
    def __init__(self, remote_next_step):
        self.config = FakeConfig()
        self.remote_next_step = remote_next_step
        self.accepted = []
        self.dropped = []
        self.finished = False

    @property
    def step(self):
        return self.remote_next_step

    def log(self, values, step=None, **kwargs):
        entry = {'step': step, 'values': copy.deepcopy(values)}
        if step < self.remote_next_step:
            self.dropped.append(entry)
            return
        self.accepted.append(entry)
        self.remote_next_step = step + 1

    def finish(self):
        self.finished = True


def configuration():
    return OmegaConf.create({
        '_target_': 'oat.workspace.train_policy.TrainPolicyWorkspace',
        'policy': {'_target_': 'test_mimicgen_online_resume.OnlineResumePolicy'},
        'optimizer': {'lr': .01},
        'task': {'policy': {'lazy_eval': True,
                           'dataset': {'_target_': 'test_mimicgen_online_resume.OnlineResumeDataset'}}},
        'dataloader': {'batch_size': 2, 'num_workers': 0, 'shuffle': False},
        'val_dataloader': {'batch_size': 2, 'num_workers': 0, 'shuffle': False},
        'training': {
            'seed': 42, 'allow_bf16': False, 'gradient_accumulate_every': 1,
            'use_ema': True, 'resume': True, 'num_epochs': 10,
            'rollout_every': 50, 'rollout_epoch_offset': 1,
            'checkpoint_every': 1, 'max_grad_norm': 1., 'max_train_steps': None,
            'max_val_steps': None, 'max_reconst_steps': None, 'val_every': 50,
            'sample_every': 50, 'tqdm_interval_sec': 60.,
            'lr_scheduler': 'constant_with_warmup', 'lr_warmup_steps': 4,
        },
        'ema': {'_target_': 'oat.model.diffusion.ema_model.EMAModel'},
        'logging': {'project': 'fm_dno_mimicgen', 'mode': 'offline', 'id': None, 'resume': True},
        'checkpoint': {'topk': {'monitor_key': 'val_action_mse', 'k': 0},
                       'save_last_ckpt': True, 'save_last_snapshot': False},
    })


def make_checkpoint(directory):
    workspace = training.TrainPolicyWorkspace(configuration(), output_dir=str(directory), lazy_instantiation=False)
    normalizer = LinearNormalizer()
    normalizer.fit({'action': torch.stack([torch.full((7,), -2.), torch.full((7,), 2.)])})
    for model in (workspace.model, workspace.ema_model):
        model.set_normalizer(normalizer)
    ema, scheduler = workspace._create_training_dynamics(workspace.cfg, 2, workspace.ema_model)
    batch = next(iter(DataLoader(OnlineResumeDataset(), batch_size=2)))
    workspace.model(batch).backward()
    workspace.optimizer.step()
    workspace.optimizer.zero_grad(set_to_none=True)
    workspace.optimizer_step = 1
    scheduler.step()
    ema.step(workspace.model)
    workspace.epoch, workspace.global_step = 5, 23
    workspace._capture_training_state(ema, scheduler, SimpleNamespace(device=torch.device('cpu'), num_processes=1))
    workspace.save_checkpoint(use_thread=False)
    return workspace


@pytest.mark.parametrize('run_id', ['1hg9rqh5', 'cc5kl9wl'])
def test_online_resume_preserves_state_and_logs_first_epoch_mse_without_uploads(tmp_path, monkeypatch, run_id):
    source = make_checkpoint(tmp_path)
    source_normalizer = copy.deepcopy(source.model.normalizer.state_dict())
    source_scheduler = copy.deepcopy(source.lr_scheduler_state)
    manifest = tmp_path / 'mse_manifest.json'
    manifest.write_text(json.dumps({'tasks': [{'task_uid': 0, 'task_name': 'first'},
                                             {'task_uid': 1, 'task_name': 'second'}]}))
    cfg = configuration()
    cfg.training.num_epochs = 7  # Resume epoch 6, complete exactly one new epoch.
    cfg.logging.update({'mode': 'online', 'entity': 'andyliu7081-northeastern-university',
                        'id': run_id, 'resume': 'must'})
    cfg.action_mse = {
        'enabled': True, 'evaluate_first_epoch': True, 'every': 50,
        'manifest_path': str(manifest), 'batch_size': 2, 'num_workers': 0,
        'dataset': {'_target_': 'test_mimicgen_online_resume.OnlineResumeDataset'},
    }

    # Saved state wins over fresh fitting, while new config remains active.
    restored = training.TrainPolicyWorkspace(cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    for model in (restored.model, restored.ema_model):
        model.set_normalizer(OnlineResumeDataset().get_normalizer())
    payload = restored.load_checkpoint()
    restored._resume_training_progress(payload, legacy_epoch_complete=True)
    assert (restored.epoch, restored.global_step, restored.optimizer_step) == (6, 24, 1)
    assert restored.cfg.logging.id == run_id and restored.cfg.logging.resume == 'must'
    assert restored.cfg.action_mse.enabled and 'action_mse' not in payload['cfg']
    assert restored.ema_state == source.ema_state
    assert restored.lr_scheduler_state == source_scheduler
    for key, value in source_normalizer.items():
        torch.testing.assert_close(restored.model.normalizer.state_dict()[key], value, rtol=0, atol=0)
    for index, state in source.optimizer.state_dict()['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(restored.optimizer.state_dict()['state'][index][key], value, rtol=0, atol=0)

    # Cloud history may be ahead after offline sync and partial-epoch replay.
    # Its transport index must advance without changing actual training steps.
    fake_run = FakeRun(remote_next_step=100)
    init = Mock(return_value=fake_run)
    monkeypatch.setenv('WANDB_MODE', 'online')
    monkeypatch.setattr(wandb, 'init', init)
    monkeypatch.setattr(wandb, 'config', fake_run.config)
    monkeypatch.setattr(training, 'Accelerator', partial(Accelerator, cpu=True))
    resumed = training.TrainPolicyWorkspace(cfg, output_dir=str(tmp_path))
    resumed.run()

    init.assert_called_once()
    assert init.call_args.kwargs['id'] == run_id
    assert init.call_args.kwargs['resume'] == 'must'
    assert init.call_args.kwargs['mode'] == 'online'
    assert init.call_args.kwargs['project'] == 'fm_dno_mimicgen'
    assert init.call_args.kwargs['entity'] == 'andyliu7081-northeastern-university'
    assert fake_run.config['logging']['id'] == run_id
    assert fake_run.config['action_mse']['enabled']
    assert fake_run.finished
    assert fake_run.dropped == []
    assert fake_run.config['wandb_step_offset'] == 76
    mse_logs = [entry for entry in fake_run.accepted if 'val/action_mse' in entry['values']]
    assert len(mse_logs) == 1
    assert mse_logs[0]['step'] == 101
    assert mse_logs[0]['values']['action_mse/completed_epochs'] == 7
    assert mse_logs[0]['values']['val/action_mse_windows'] == 4
    assert mse_logs[0]['values']['val/action_mse'] >= 0
    assert (resumed.epoch, resumed.global_step, resumed.optimizer_step) == (7, 26, 3)
    assert resumed.lr_scheduler_state['last_epoch'] == source_scheduler['last_epoch'] + 2
    assert resumed.ema_state['optimization_step'] == source.ema_state['optimization_step'] + 2
    for model in (resumed.model, resumed.ema_model):
        for key, value in source_normalizer.items():
            torch.testing.assert_close(model.normalizer.state_dict()[key], value, rtol=0, atol=0)
