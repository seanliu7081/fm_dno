import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    'heading_watcher', Path(__file__).resolve().parents[1] / 'scripts/watch_heading_evaluations.py')
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def test_prefers_exact_checkpoint_and_reports_later_epoch_honestly(tmp_path):
    exact = {'path': str(tmp_path/'ep-0005_sr-0.1.ckpt'), 'epoch': 5}
    latest = {'path': str(tmp_path/'latest.ckpt'), 'epoch': 17}
    assert watcher.choose_checkpoint([latest, exact], 5) == exact
    assert watcher.choose_checkpoint([exact, latest], 15) == latest
    assert watcher.choose_checkpoint([exact, latest], 30) is None


def test_metadata_only_loads_after_stable_poll_and_once_per_change(tmp_path, monkeypatch):
    path = tmp_path/'latest.ckpt'
    path.write_bytes(b'complete')
    calls = []
    monkeypatch.setattr(watcher, 'read_metadata', lambda p: calls.append(p) or {'epoch': 5})
    cache = watcher.MetadataCache()
    assert cache.inspect(path) is None
    assert not calls
    assert cache.inspect(path)['epoch'] == 5
    assert cache.inspect(path)['epoch'] == 5
    assert len(calls) == 1
    path.write_bytes(b'new completed checkpoint')
    assert cache.inspect(path) is None
    assert cache.inspect(path)['epoch'] == 5
    assert len(calls) == 2


def test_partial_checkpoint_waits_for_changed_file(tmp_path, monkeypatch):
    path = tmp_path/'latest.ckpt'
    path.write_bytes(b'partial')
    calls = []
    def read(path):
        calls.append(path)
        raise EOFError('still writing')
    monkeypatch.setattr(watcher, 'read_metadata', read)
    cache = watcher.MetadataCache()
    assert cache.inspect(path) is None
    assert cache.inspect(path) is None
    assert cache.inspect(path) is None
    assert len(calls) == 1
    path.write_bytes(b'finished checkpoint')
    monkeypatch.setattr(watcher, 'read_metadata', lambda p: {'epoch': 15})
    assert cache.inspect(path) is None
    assert cache.inspect(path)['epoch'] == 15


def test_snapshot_rejects_source_change_and_keeps_no_partial(tmp_path, monkeypatch):
    source, snapshot = tmp_path/'latest.ckpt', tmp_path/'snapshot.ckpt'
    source.write_bytes(b'complete checkpoint')
    candidate = {'path': str(source), 'signature': watcher.signature(source), 'epoch': 5}
    original_hash = watcher.sha256
    def mutate(path):
        path.write_bytes(b'changed while copying')
        return original_hash(path)
    monkeypatch.setattr(watcher, 'sha256', mutate)
    with pytest.raises(RuntimeError, match='changed'):
        watcher.make_snapshot(candidate, snapshot)
    assert not snapshot.exists() and not snapshot.with_suffix('.partial').exists()


def test_snapshot_validates_before_commit(tmp_path, monkeypatch):
    source, snapshot = tmp_path/'latest.ckpt', tmp_path/'snapshot.ckpt'
    source.write_bytes(b'complete checkpoint')
    candidate = {'path': str(source), 'signature': watcher.signature(source), 'epoch': 5}
    monkeypatch.setattr(watcher, 'read_metadata', lambda p: {'epoch': 5, 'weights': 'ema_model'})
    result = watcher.make_snapshot(candidate, snapshot)
    assert result['sha256'] == watcher.sha256(source)
    assert snapshot.read_bytes() == source.read_bytes()
    assert snapshot.stat().st_mode & 0o222 == 0


def test_failed_or_mismatched_metrics_never_count_as_completed(tmp_path):
    args = SimpleNamespace(n_per_task=10, init_start=0, seed=42, suite='libero_10')
    stage = {'actual_epoch': 5, 'checkpoint_sha256': 'hash'}
    manifest = {'episodes': list(range(100)), 'training_position': {'epoch': 5},
                'checkpoint_sha256': 'hash', 'n_per_task': 10, 'init_start': 0,
                'seed': 42, 'suite': 'libero_10'}
    summary = {'complete': True, 'mean_success_rate': .75, 'expected_episodes': 100,
               'completed_episodes': 100, 'checkpoint_sha256': 'hash'}
    watcher.write_json(tmp_path/'manifest.json', manifest)
    watcher.write_json(tmp_path/'summary.json', summary)
    assert watcher.completed_attempt(tmp_path, stage, args)
    for field, value in [('complete', False), ('mean_success_rate', None), ('completed_episodes', 99),
                         ('checkpoint_sha256', 'other')]:
        watcher.write_json(tmp_path/'summary.json', dict(summary, **{field: value}))
        assert not watcher.completed_attempt(tmp_path, stage, args)


def test_real_cpu_checkpoint_metadata_snapshot(tmp_path):
    import dill
    import torch
    from omegaconf import OmegaConf
    source, snapshot = tmp_path/'latest.ckpt', tmp_path/'snapshot.ckpt'
    payload = {'cfg': OmegaConf.create({'training': {'use_ema': True}}),
               'state_dicts': {'ema_model': {'test': torch.ones(2)}},
               'pickles': {'epoch': dill.dumps(15)}}
    torch.save(payload, source, pickle_module=dill)
    metadata = watcher.read_metadata(source)
    assert metadata == {'epoch': 15, 'weights': 'ema_model'}
    result = watcher.make_snapshot({'path': str(source), 'signature': watcher.signature(source), **metadata}, snapshot)
    assert result['epoch'] == 15 and result['sha256'] == watcher.sha256(source)


def test_timeout_cleanup_terminates_descendants_and_kills_survivors(monkeypatch):
    import psutil
    calls = []
    class Child:
        def terminate(self):
            calls.append('child terminate')
        def kill(self):
            calls.append('child kill')
    child = Child()
    class Parent:
        pid = 12345
        def poll(self):
            return None
        def terminate(self):
            calls.append('parent terminate')
        def wait(self, timeout=None):
            calls.append('parent wait')
            return 0
    monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(children=lambda recursive: [child]))
    monkeypatch.setattr(psutil, 'wait_procs', lambda children, timeout: ([], children) if timeout == 10 else (children, []))
    watcher.terminate_evaluator(Parent())
    assert calls == ['child terminate', 'parent terminate', 'child kill', 'parent wait']
