from types import SimpleNamespace
import json

from omegaconf import OmegaConf
import pytest
import torch

from oat.workspace.train_policy import _action_mse_due, _run_action_mse


def test_action_mse_uses_completed_epoch_schedule_and_initial_measurement():
    cfg = {'enabled': True, 'every': 50, 'evaluate_first_epoch': True}
    assert _action_mse_due(44, cfg, first_epoch=44)
    assert not _action_mse_due(45, cfg, first_epoch=44)
    assert _action_mse_due(49, cfg, first_epoch=44)
    assert _action_mse_due(99, cfg, first_epoch=44)
    assert _action_mse_due(599, cfg, first_epoch=44)
    assert not _action_mse_due(0, {'enabled': False}, first_epoch=0)
    with pytest.raises(ValueError):
        _action_mse_due(0, {'every': 0}, first_epoch=0)


def test_action_mse_propagates_rank_failure_before_merging(monkeypatch, tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'tasks': [{'task_uid': 0, 'task_name': 'task'}]}))
    import oat.common.action_mse as mse
    monkeypatch.setattr(mse, 'evaluate_action_mse', lambda *args, **kwargs: {})
    def gather(results, local):
        results[:] = [local, {'result': None, 'error': 'invalid prediction'}]
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    accelerator = SimpleNamespace(device=torch.device('cpu'), process_index=0, num_processes=2)
    with pytest.raises(RuntimeError, match='rank 1: invalid prediction'):
        _run_action_mse(None, None, accelerator, OmegaConf.create({'manifest_path': str(manifest)}), completed_epochs=50)


def test_missing_manifest_is_gathered_as_error(monkeypatch, tmp_path):
    gathered = []
    def gather(results, local):
        gathered.append(local)
        results[:] = [local, local]
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    accelerator = SimpleNamespace(device=torch.device('cpu'), process_index=0, num_processes=2)
    with pytest.raises(RuntimeError, match='FileNotFoundError'):
        _run_action_mse(None, None, accelerator, OmegaConf.create({'manifest_path': str(tmp_path / 'absent')}), completed_epochs=50)
    assert len(gathered) == 1
