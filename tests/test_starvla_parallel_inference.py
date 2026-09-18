"""Replica concurrency, request identity, failure recovery and shutdown."""
import threading
import numpy as np
import pytest
from oat.starvla_heading.server import BatchingPredictor


def test_replicas_execute_concurrently_and_keep_request_rng():
    barrier = threading.Barrier(3)
    called = []
    lock = threading.Lock()

    def replica(index):
        def predict(payloads):
            with lock:
                called.append(index)
            barrier.wait(timeout=3)
            return [{"id": p["id"], "noise": np.random.default_rng(p["seed"]).normal(size=3).tolist()}
                    for p in payloads]
        return predict

    predictor = BatchingPredictor(replica(0), max_batch_size=1, max_wait_ms=0,
                                 replica_predictors=[replica(1), replica(2)])
    try:
        futures = [predictor.submit({"id": i, "seed": 42}) for i in range(3)]
        results = [f.result(timeout=5) for f in futures]
        assert sorted(called) == [0, 1, 2]
        assert [r["id"] for r in results] == [0, 1, 2]
        assert results[0]["noise"] == results[1]["noise"] == results[2]["noise"]
    finally:
        predictor.close()
    assert not any(t.is_alive() for t in predictor.threads)


def test_replica_exception_does_not_lose_later_requests():
    def predict(payloads):
        if payloads[0].get("fail"):
            raise ValueError("bad request")
        return [p["id"] for p in payloads]

    predictor = BatchingPredictor(predict, max_batch_size=1, max_wait_ms=0,
                                 replica_predictors=[predict])
    try:
        with pytest.raises(ValueError, match="bad request"):
            predictor.submit({"fail": True}).result(timeout=3)
        assert predictor.submit({"id": 7}).result(timeout=3) == 7
    finally:
        predictor.close()


def test_close_drains_queued_work_before_stopping_all_replicas():
    gate = threading.Event()

    def predict(payloads):
        assert gate.wait(timeout=3)
        return [p["id"] for p in payloads]

    predictor = BatchingPredictor(predict, max_batch_size=4, max_wait_ms=10,
                                 replica_predictors=[predict, predict])
    futures = [predictor.submit({"id": i}) for i in range(17)]
    gate.set()
    predictor.close()
    assert [f.result(timeout=3) for f in futures] == list(range(17))
    assert not any(t.is_alive() for t in predictor.threads)
    with pytest.raises(RuntimeError, match="shutting down"):
        predictor.submit({"id": 18})


def test_replica_launch_preserves_checkpoint_port_and_legacy_arguments(tmp_path):
    from scripts.run_starvla_experiment import server_command, evaluation_command
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    before = server_command(checkpoint)
    after = server_command(checkpoint, devices=['cuda:0', 'cuda:2', 'cuda:3'])
    assert after == before + ['--devices', 'cuda:0', 'cuda:2', 'cuda:3']
    command = evaluation_command('libero', tmp_path, 'http://127.0.0.1:18080',
                                 tmp_path / 'manifest.json', workers=20, render_gpu=1)
    assert command[command.index('--workers') + 1] == '20'
    assert command[command.index('--render-gpu-device-id') + 1] == '1'
    assert command[command.index('--trials') + 1] == '50'


class FakeProcessModel:
    def __init__(self, checkpoint, device):
        self.metadata = {'checkpoint': checkpoint}
        self.parameter_count = 1

    def __call__(self, payloads):
        import os
        if payloads[0].get('fail'):
            raise ValueError('bad process request')
        return [{'id': p['id'], 'pid': os.getpid(),
                 'noise': np.random.default_rng(p['seed']).normal(size=3).tolist()}
                for p in payloads]


class FailedProcessModel:
    def __init__(self, checkpoint, device):
        raise ValueError('checkpoint load failed')


def test_process_replicas_use_distinct_processes_and_preserve_seeded_outputs():
    from oat.starvla_heading.server import ProcessModelReplica
    replicas = [ProcessModelReplica('checkpoint', f'cuda:{i}', FakeProcessModel) for i in range(2)]
    try:
        for replica in replicas:
            replica.wait_ready(timeout=20)
        a, b = [r([{'id': 7, 'seed': 42}])[0] for r in replicas]
        assert a['pid'] != b['pid']
        assert a['noise'] == b['noise']
        assert replicas[0].metadata == replicas[1].metadata
        with pytest.raises(RuntimeError, match='bad process request'):
            replicas[0]([{'fail': True}])
        assert replicas[0]([{'id': 8, 'seed': 42}])[0]['id'] == 8
    finally:
        for replica in replicas:
            replica.close()
    assert all(not r.process.is_alive() for r in replicas)


def test_process_replica_surfaces_startup_failure():
    from oat.starvla_heading.server import ProcessModelReplica
    replica = ProcessModelReplica('checkpoint', 'cuda:0', FailedProcessModel)
    try:
        with pytest.raises(RuntimeError, match='checkpoint load failed'):
            replica.wait_ready(timeout=20)
    finally:
        replica.close()
    assert not replica.process.is_alive()
