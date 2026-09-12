"""Metric correctness and a bounded, opt-in actual two-GPU workspace run."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from oat.dataset.base_dataset import BaseDataset
from oat.policy.base_policy import BasePolicy
from oat.workspace.train_policy import (
    TrainPolicyWorkspace, _merge_rollout_logs, _preserve_training_rng,
    _run_distributed_rollout,
)


def test_reduces_counts_for_uneven_rollout_shards():
    logs = [
        {'_eval_counts': {'mean_success_rate': [2, 3]}, 'rank0/artifact': 'zero'},
        {'_eval_counts': {'mean_success_rate': [0, 2]}, 'rank1/artifact': 'one'},
    ]
    original = copy.deepcopy(logs)
    merged = _merge_rollout_logs(logs)
    assert merged == {
        'mean_success_rate': .4, 'mean_success_rate/successes': 2,
        'mean_success_rate/episodes': 5, 'rank0/artifact': 'zero', 'rank1/artifact': 'one',
    }
    assert logs == original


@pytest.mark.parametrize('logs', [
    [{}], [{'_eval_counts': {}}],
    [{'_eval_counts': {'rate': [1, 0]}}],
    [{'_eval_counts': {'rate': [float('nan'), 2]}}],
    [{'_eval_counts': {'rate': [1.5, 2]}}],
    [{'_eval_counts': {'rate': [0, 0]}}],
    [{'_eval_counts': {'rate': [1, 2]}}, {'_eval_counts': {'different': [1, 2]}}],
    [{'_eval_counts': {'rate': [1, 2]}, 'video': 'a'},
     {'_eval_counts': {'rate': [1, 2]}, 'video': 'b'}],
    [{'_eval_counts': {'rate': [1, 2]}, 'rate': .5}],
])
def test_rejects_incomplete_or_invalid_distributed_metrics(logs):
    with pytest.raises(ValueError):
        _merge_rollout_logs(logs)


@pytest.mark.parametrize('failure', [False, True])
def test_rollout_restores_all_cpu_training_rng_streams(failure):
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    try:
        with _preserve_training_rng(torch.device('cpu')):
            random.seed(500)
            np.random.seed(500)
            torch.manual_seed(500)
            if failure:
                raise RuntimeError('simulator failure')
    except RuntimeError:
        pass
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_remote_rollout_error_is_reported_on_every_rank(monkeypatch):
    runner = SimpleNamespace(run=lambda *args, **kwargs: {'_eval_counts': {'rate': [1, 2]}})
    accelerator = SimpleNamespace(device=torch.device('cpu'), num_processes=2)

    def gather(results, local):
        results[:] = [local, {'log': None, 'error': 'RuntimeError: simulator failure'}]

    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    with pytest.raises(RuntimeError, match='rank 1: RuntimeError: simulator failure'):
        _run_distributed_rollout(runner, object(), accelerator, epoch=49, global_step=10)


class TinyDataset(BaseDataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {'obs': {'state': torch.tensor([[float(index), 1.]])},
                'action': torch.tensor([[[float(index) * .1]]]).reshape(1, 1)}

    def get_normalizer(self):
        return None


class TinyPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.seen = []
        self.backward_calls = 0
        self.linear.weight.register_hook(self._record_backward)

    def _record_backward(self, gradient):
        self.backward_calls += 1
        return gradient

    def set_normalizer(self, normalizer):
        pass

    def forward(self, batch):
        self.seen.extend(batch['obs']['state'][:, 0, 0].detach().cpu().tolist())
        return (self.linear(batch['obs']['state']) - batch['action']).square().mean()

    def predict_action(self, obs_dict):
        action = self.linear(obs_dict['state'])
        return {'action': action, 'action_pred': action}


class TinyRunner:
    def __init__(self, output_dir, rank, world_size, device):
        self.output_dir = Path(output_dir)
        self.rank, self.world_size, self.device = rank, world_size, torch.device(device)
        self.calls = []

    def run(self, policy, epoch, global_step):
        assert self.world_size == 2
        assert not isinstance(policy, torch.nn.parallel.DistributedDataParallel)
        assert policy.device == self.device == torch.device('cuda', self.rank)
        # Uneven inference counts exercise the unwrapped model on both GPUs.
        with torch.inference_mode():
            for _ in range(self.rank + 1):
                action = policy.predict_action({'state': torch.ones(2, 1, 2, device=self.device)})['action']
        torch.cuda.synchronize(self.device)
        self.calls.append({'epoch': epoch, 'device': str(action.device), 'finite': bool(action.isfinite().all())})
        (self.output_dir / f'eval-rank{self.rank}.json').write_text(json.dumps(self.calls))
        return {'_eval_counts': {'mean_success_rate': [self.rank + 1, self.rank + 2]}}

    def close(self):
        pass


def _ddp_worker(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    cfg = OmegaConf.create({
        'policy': {'_target_': 'test_distributed_policy_training.TinyPolicy'},
        'optimizer': {'lr': .01},
        'task': {'policy': {
            'lazy_eval': False, 'distributed_eval': True,
            'dataset': {'_target_': 'test_distributed_policy_training.TinyDataset'},
            'env_runner': {'_target_': 'test_distributed_policy_training.TinyRunner'},
        }},
        'dataloader': {'batch_size': 2, 'num_workers': 0, 'shuffle': False},
        'val_dataloader': {'batch_size': 2, 'num_workers': 0, 'shuffle': False},
        'training': {
            'seed': 42, 'allow_bf16': False, 'expected_num_processes': 2,
            'gradient_accumulate_every': 1, 'use_ema': True, 'resume': False,
            'num_epochs': 2, 'rollout_every': 1, 'rollout_epoch_offset': 1,
            'checkpoint_every': 1, 'max_grad_norm': 1., 'max_train_steps': None,
            'max_val_steps': None, 'max_reconst_steps': None, 'val_every': 1,
            'sample_every': 1, 'tqdm_interval_sec': 60., 'lr_scheduler': 'constant',
            'lr_warmup_steps': 0,
        },
        'ema': {'_target_': 'oat.model.diffusion.ema_model.EMAModel'},
        'logging': {'project': 'fm_dno_ddp_test', 'mode': 'disabled'},
        'checkpoint': {
            'topk': {'monitor_key': 'mean_success_rate', 'k': 0},
            'save_last_ckpt': True, 'save_last_snapshot': False,
            'save_rollout_ckpt': True, 'save_final_ckpt': True,
        },
    })
    workspace = TrainPolicyWorkspace(cfg, output_dir=str(output_dir))
    workspace.run()
    model = workspace.model.module
    rank = int(os.environ['RANK'])
    report = {'device': str(model.device), 'backward_calls': model.backward_calls,
              'seen': model.seen, 'weight': model.linear.weight.detach().cpu().tolist(),
              'optimizer_step': workspace.optimizer_step}
    (output_dir / f'train-rank{rank}.json').write_text(json.dumps(report))


@pytest.mark.skipif(not os.environ.get('FM_DNO_DDP_TEST_GPUS'),
                    reason='Set FM_DNO_DDP_TEST_GPUS to two free GPU indices')
def test_actual_two_gpu_workspace_training_and_inline_eval(tmp_path):
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ['FM_DNO_DDP_TEST_GPUS'],
                   'OMP_NUM_THREADS': '1', 'WANDB_MODE': 'disabled'}
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=2', str(Path(__file__).resolve()), '--worker', str(tmp_path)]
    result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = [json.loads((tmp_path / f'train-rank{rank}.json').read_text()) for rank in range(2)]
    assert [report['device'] for report in reports] == ['cuda:0', 'cuda:1']
    assert [report['backward_calls'] for report in reports] == [4, 4]
    assert [report['optimizer_step'] for report in reports] == [4, 4]
    assert set(reports[0]['seen']).isdisjoint(reports[1]['seen'])
    assert set(reports[0]['seen']) | set(reports[1]['seen']) == set(range(8))
    assert reports[0]['weight'] == reports[1]['weight']
    for rank in range(2):
        calls = json.loads((tmp_path / f'eval-rank{rank}.json').read_text())
        assert [call['epoch'] for call in calls] == [0, 1]
        assert all(call['device'] == f'cuda:{rank}' and call['finite'] for call in calls)
    logs = [json.loads(line) for line in (tmp_path / 'logs.json').read_text().splitlines()]
    evaluations = [row for row in logs if 'mean_success_rate' in row]
    assert len(evaluations) == 2
    assert all(row['mean_success_rate'] == .6 and row['mean_success_rate/episodes'] == 5
               for row in evaluations)
    assert [row['completed_epochs'] for row in evaluations] == [1, 2]
    assert not any('val_loss' in row or 'test_reconst_mse' in row for row in logs)
    assert (tmp_path / 'checkpoints' / 'epoch-0002.ckpt').is_file()


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--worker':
    _ddp_worker(sys.argv[2])
