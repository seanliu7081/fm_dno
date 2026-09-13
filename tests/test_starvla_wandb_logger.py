"""Offline contracts for safely replaying an append-only training log to W&B."""
import hashlib
import json
import math

import pytest

from oat.starvla_heading.wandb_logger import event_metrics, pending_events, read_events


def encoded(event):
    return (json.dumps(event, allow_nan=False) + "\n").encode("utf-8")


def test_read_events_defers_partial_row_then_accepts_completed_append(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = {"event": "training", "step": 10, "action_loss": 1.25}
    second = {"event": "validation", "step": 10, "normalized_action_mse": 0.5}
    prefix = encoded(first)
    second_bytes = encoded(second)
    path.write_bytes(prefix + second_bytes[:20])

    events, fingerprint = read_events(path)
    assert events == [first]
    assert fingerprint["byte_count"] == len(prefix)
    assert fingerprint["sha256"] == hashlib.sha256(prefix).hexdigest()

    with path.open("ab") as log:
        log.write(second_bytes[20:])
    events, completed_fingerprint = read_events(path, previous=fingerprint)
    assert events == [first, second]
    assert completed_fingerprint["byte_count"] == len(prefix + second_bytes)
    assert completed_fingerprint["sha256"] == hashlib.sha256(prefix + second_bytes).hexdigest()


def test_read_events_partial_first_row_does_not_advance_fingerprint(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"event": "training", "step": 10')
    events, fingerprint = read_events(path)
    assert events == []
    assert fingerprint["byte_count"] == 0
    assert fingerprint["sha256"] == hashlib.sha256(b"").hexdigest()


def test_read_events_rejects_mutation_in_previously_seen_prefix(tmp_path):
    path = tmp_path / "metrics.jsonl"
    original = encoded({"event": "training", "step": 10, "action_loss": 1.0})
    path.write_bytes(original)
    _, fingerprint = read_events(path)
    # Same-length replacement defeats checks based only on the file size.
    changed = original.replace(b'1.0', b'9.0')
    assert len(changed) == len(original)
    path.write_bytes(changed)
    with pytest.raises(ValueError):
        read_events(path, previous=fingerprint)


def test_read_events_rejects_truncation_of_previously_seen_prefix(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = encoded({"event": "training", "step": 10})
    second = encoded({"event": "checkpoint", "step": 10})
    path.write_bytes(first + second)
    _, fingerprint = read_events(path)
    path.write_bytes(first)
    with pytest.raises(ValueError):
        read_events(path, previous=fingerprint)


def test_pending_events_preserves_shared_and_revisited_optimizer_steps():
    events = [
        {"event": "training", "step": 10},
        {"event": "validation", "step": 10},
        {"event": "checkpoint", "step": 10},
        {"event": "training", "step": 20},
        {"event": "started", "step": 10},
        {"event": "training", "step": 20},
    ]
    pending = list(pending_events(events, next_step=0))
    assert pending == list(enumerate(events))
    rows = [event_metrics(event, index) for index, event in pending]
    assert [row["source/event_index"] for row in rows] == list(range(6))
    assert [row["optimizer_step"] for row in rows] == [10, 10, 10, 20, 10, 20]
    assert [row["event_type"] for row in rows] == [event["event"] for event in events]


def test_pending_events_skips_only_acknowledged_source_rows():
    events = [{"event": event, "step": 10} for event in ("training", "validation", "checkpoint")]
    assert list(pending_events(events, next_step=1)) == [(1, events[1]), (2, events[2])]
    assert list(pending_events(events, next_step=3)) == []
    with pytest.raises(ValueError):
        list(pending_events(events, next_step=4))


def test_training_metric_flattening_preserves_losses_rates_and_rank_peaks():
    event = {
        "event": "training", "step": 500, "time_unix": 1234.5, "epoch": 0,
        "elapsed_seconds": 456.7, "action_loss": 0.3, "flow_loss": 0.2,
        "heading_error_deg": 35.0, "source_active_fraction": 1.0,
        "learning_rates": {"vision_decay": 5e-6, "qwen_decay": 1e-5, "expert_decay": 1e-4},
        "gpu_memory": [
            {"rank": 0, "max_allocated_gib": 15.5, "max_reserved_gib": 22.1},
            {"rank": 5, "max_allocated_gib": 15.6, "max_reserved_gib": 22.2},
        ],
    }
    row = event_metrics(event, 54)
    assert row["source/event_index"] == 54
    assert row["optimizer_step"] == 500
    assert row["event_type"] == "training"
    for key in ("epoch", "elapsed_seconds", "action_loss", "flow_loss", "heading_error_deg", "source_active_fraction"):
        assert row[f"train/{key}"] == event[key]
    for name, value in event["learning_rates"].items():
        assert row[f"lr/{name}"] == value
    assert row["gpu/rank_0/peak_allocated_gib"] == 15.5
    assert row["gpu/rank_0/peak_reserved_gib"] == 22.1
    assert row["gpu/rank_5/peak_allocated_gib"] == 15.6
    assert row["gpu/rank_5/peak_reserved_gib"] == 22.2
    assert not ({"train/step", "train/time_unix", "train/gpu_memory", "train/learning_rates"} & row.keys())
    assert all(not isinstance(value, (dict, list)) for value in row.values())


def test_validation_metrics_have_original_libero_namespace():
    row = event_metrics({"event": "validation", "step": 1000, "benchmark": "libero",
                         "normalized_action_mse": 0.125, "sample_count": 600, "best": True}, 103)
    assert row["optimizer_step"] == 1000
    assert row["validation/libero/normalized_action_mse"] == 0.125
    assert row["validation/libero/sample_count"] == 600
    assert not any(key.startswith("train/") for key in row)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("location", ["training", "learning_rate", "gpu_peak", "validation"])
def test_nonfinite_scalar_metrics_are_rejected(value, location):
    event = {"event": "training", "step": 10, "action_loss": 0.25}
    if location == "training":
        event["action_loss"] = value
    elif location == "learning_rate":
        event["learning_rates"] = {"expert_decay": value}
    elif location == "gpu_peak":
        event["gpu_memory"] = [{"rank": 0, "max_allocated_gib": value, "max_reserved_gib": 22.0}]
    else:
        event = {"event": "validation", "step": 10, "benchmark": "libero", "normalized_action_mse": value}
    with pytest.raises(ValueError):
        event_metrics(event, 1)


@pytest.mark.parametrize("invalid", [-1, math.nan, math.inf, -math.inf, 1.0, 0.5, True, False])
@pytest.mark.parametrize("field", ["optimizer_step", "source_index"])
def test_source_and_optimizer_indices_are_nonnegative_integers(invalid, field):
    event = {"event": "training", "step": 10, "action_loss": 0.25}
    index = 1
    if field == "optimizer_step":
        event["step"] = invalid
    else:
        index = invalid
    with pytest.raises(ValueError):
        event_metrics(event, index)


from oat.starvla_heading import wandb_logger as logger


def periodic_event(step=10000, benchmark="libero", seed=42, rate=0.5, digest="a" * 64, time_unix=100.0):
    return {"event": "evaluation_verified", "seed": seed, "step": step, "benchmark": benchmark,
            "success_rate": rate, "checkpoint_sha256": digest,
            "episodes": {"libero": 2000, "libero_plus": 10030}[benchmark], "time_unix": time_unix}


def make_sources(tmp_path, training=None, periodic=None):
    directory = tmp_path / "seed42"
    directory.mkdir(exist_ok=True)
    training = training if training is not None else [
        {"event": "training", "step": 10000, "time_unix": 10.0, "action_loss": 0.2},
        {"event": "validation", "step": 10000, "normalized_action_mse": 0.1},
        {"event": "training", "step": 10100, "time_unix": 20.0, "action_loss": 0.15},
    ]
    (directory / "metrics.jsonl").write_bytes(b"".join(encoded(e) for e in training))
    config = {"training": {"max_steps": 30000}, "seed": 42}
    (directory / "run_config.json").write_text(json.dumps(config))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    identity = {"entity": "test-entity", "project": "test-project", "run_id": f"svhg42-{digest[:12]}",
                "config_sha256": digest}
    if periodic is not None:
        path = tmp_path / logger.PERIODIC_STREAM
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"".join(encoded(e) for e in periodic))
    return directory, identity, training


class FakeRun:
    def __init__(self, remote_step, ledger_path):
        self.step = remote_step
        self.url = "https://wandb.example/test-run"
        self.summary = {}
        self.logged = []
        self.finished = []
        self.ledger_path = ledger_path
        self.definitions = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, metrics, *, step, commit):
        # This checks the actual upload boundary, not just an eventual file save.
        state = json.loads(self.ledger_path.read_text())
        assert metrics == logger.ledger_metrics(state["rows"][step], step)
        assert commit is True
        self.logged.append((step, metrics))
        self.step = step + 1

    def finish(self, *, exit_code):
        self.finished.append(exit_code)


class FakeWandb:
    def __init__(self, root, remote_step=0):
        self.run = FakeRun(remote_step, root / "seed42/wandb_history_ledger.json")
        self.kwargs = None

    @staticmethod
    def Settings(**kwargs):
        return kwargs

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def mirror_for(root, remote_step=0):
    wandb = FakeWandb(root, remote_step)
    mirror = logger.SeedMirror(root, 42, "test-entity", "test-project", "existing-group", wandb)
    return mirror, wandb


def test_legacy_migration_preserves_all_training_indices_before_periodic_events(tmp_path):
    directory, identity, training = make_sources(tmp_path, periodic=[periodic_event()])
    _, fingerprint = read_events(directory / "metrics.jsonl")
    logger.write_json(directory / "wandb_online.json", {**identity, "source_fingerprint": fingerprint,
                                                      "last_queued_source_event_index": 99999})
    mirror, wandb = mirror_for(tmp_path, remote_step=2)
    assert [row["event"] for row in mirror.ledger.rows[:3]] == training
    assert [row["source_index"] for row in mirror.ledger.rows[:3]] == [0, 1, 2]
    assert mirror.ledger.rows[3]["stream"] == logger.PERIODIC_STREAM
    mirror.poll(True)
    assert [step for step, _ in wandb.run.logged] == [2, 3]
    assert wandb.kwargs["id"] == identity["run_id"]
    assert wandb.kwargs["config"] == json.loads((directory / "run_config.json").read_text())
    assert wandb.kwargs["resume"] == "allow"
    assert wandb.run.logged[-1][1]["optimizer_step"] == 10000
    assert wandb.run.logged[-1][1]["evaluation/periodic/libero/success_rate"] == 0.5
    assert wandb.run.logged[-1][1]["source/event_index"] == 0
    assert wandb.run.logged[-1][1]["source/ledger_index"] == 3


def test_restart_uses_remote_ack_and_keeps_merged_history_order(tmp_path):
    directory, identity, _ = make_sources(tmp_path, periodic=[periodic_event()])
    first, first_wandb = mirror_for(tmp_path, remote_step=2)
    first.poll(True)
    original_rows = list(first.ledger.rows)
    # Locally all four rows were queued; only the first three are remote-acked.
    with (directory / "metrics.jsonl").open("ab") as handle:
        handle.write(encoded({"event": "training", "step": 10200, "time_unix": 30.0, "action_loss": 0.12}))
    with (tmp_path / logger.PERIODIC_STREAM).open("ab") as handle:
        handle.write(encoded(periodic_event(time_unix=200.0)))  # Harmless coordinator retry.
        handle.write(encoded(periodic_event(benchmark="libero_plus")))
    restarted, remote = mirror_for(tmp_path, remote_step=3)
    assert restarted.ledger.rows[:4] == original_rows
    restarted.poll(True)
    assert [step for step, _ in remote.run.logged] == [3, 4, 5]
    assert [row["event_type"] for _, row in remote.run.logged] == [
        "periodic_evaluation_verified", "training", "periodic_evaluation_verified"]
    assert [row["optimizer_step"] for _, row in remote.run.logged] == [10000, 10200, 10000]
    assert remote.run.logged[-1][1]["source/event_index"] == 2
    assert len(restarted.ledger.rows) == 6
    restarted.poll(True)
    assert len(remote.run.logged) == 3  # No duplicates within a live writer either.
    final_restart, final_remote = mirror_for(tmp_path, remote_step=6)
    final_restart.poll(True)
    assert final_remote.run.logged == []
    metadata = json.loads((directory / "wandb_online.json").read_text())
    assert metadata["last_queued_source_event_index"] == 3
    assert metadata["last_queued_history_event_index"] == 5


@pytest.mark.parametrize("field,value", [("success_rate", 0.8), ("checkpoint_sha256", "b" * 64)])
def test_conflicting_periodic_retry_is_rejected_before_any_new_upload(tmp_path, field, value):
    directory, _, _ = make_sources(tmp_path, periodic=[periodic_event()])
    mirror, remote = mirror_for(tmp_path)
    mirror.poll(True)
    previously_logged = list(remote.run.logged)
    conflict = periodic_event()
    conflict[field] = value
    with (tmp_path / logger.PERIODIC_STREAM).open("ab") as handle:
        handle.write(encoded(conflict))
    with pytest.raises(ValueError, match="Conflicting periodic"):
        mirror.poll(True)
    assert remote.run.logged == previously_logged


def test_periodic_other_seed_and_control_events_do_not_add_history_rows(tmp_path):
    events = [{"event": "started", "time_unix": 1.0}, periodic_event(seed=43), periodic_event()]
    directory, identity, training = make_sources(tmp_path, periodic=events)
    ledger = logger.EventLedger(directory, tmp_path, 42, identity)
    assert len(ledger.rows) == len(training) + 1
    assert ledger.rows[-1]["source_index"] == 2
    assert ledger.state["periodic_count"] == 3


@pytest.mark.parametrize("stream", [logger.TRAINING_STREAM, logger.PERIODIC_STREAM])
@pytest.mark.parametrize("change", ["rewrite", "truncate", "remove"])
def test_merged_ledger_preserves_both_raw_source_prefix_audits(tmp_path, stream, change):
    directory, identity, _ = make_sources(tmp_path, periodic=[periodic_event()])
    ledger = logger.EventLedger(directory, tmp_path, 42, identity)
    path = directory / stream if stream == logger.TRAINING_STREAM else tmp_path / stream
    if change == "rewrite":
        data = path.read_bytes()
        path.write_bytes(data.replace(b'10000', b'90000'))
    elif change == "truncate":
        path.write_bytes(b"")
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        ledger.refresh()


def test_missing_ledger_after_migration_is_not_rebuilt_with_new_order(tmp_path):
    directory, identity, _ = make_sources(tmp_path, periodic=[periodic_event()])
    mirror, _ = mirror_for(tmp_path)
    mirror.poll(True)
    mirror.ledger.path.unlink()
    with pytest.raises(ValueError, match="ledger is missing"):
        mirror_for(tmp_path)


@pytest.mark.parametrize("stream", [logger.TRAINING_STREAM, logger.PERIODIC_STREAM])
def test_missing_record_in_persisted_ledger_is_detected(tmp_path, stream):
    directory, identity, _ = make_sources(tmp_path, periodic=[periodic_event()])
    ledger = logger.EventLedger(directory, tmp_path, 42, identity)
    state = json.loads(ledger.path.read_text())
    index = next(i for i, row in enumerate(state["rows"]) if row["stream"] == stream)
    del state["rows"][index]
    logger.write_json(ledger.path, state)
    with pytest.raises(ValueError, match="omitted"):
        logger.EventLedger(directory, tmp_path, 42, identity)


def write_periodic_schedule(root, completed=True):
    directory = root / "periodic_evaluation"
    directory.mkdir(exist_ok=True)
    logger.write_json(directory / "config.json", {"protocol_version": 1, "seeds": [42, 43, 44],
                      "steps": [10000, 20000, 30000], "benchmarks": {"libero": 2000, "libero_plus": 10030}})
    (directory / "seed42").mkdir(exist_ok=True)
    logger.write_json(directory / "seed42/status.json", {"status": "completed" if completed else "running"})


def all_periodic_events():
    return [periodic_event(step=step, benchmark=benchmark)
            for step in (10000, 20000, 30000) for benchmark in ("libero", "libero_plus")]


def test_original_completion_waits_for_all_verified_periodic_results(tmp_path):
    directory, _, _ = make_sources(tmp_path, periodic=all_periodic_events()[:-1])
    write_periodic_schedule(tmp_path)  # Status alone must not race ahead of log delivery.
    logger.write_json(directory / "experiment_seed_status.json", {"status": "completed"})
    mirror, remote = mirror_for(tmp_path)
    metadata = mirror.poll(True)
    assert metadata["experiment_stage"] == "periodic_evaluation"
    assert not mirror.finished and remote.run.finished == []
    assert remote.run.summary["evaluation/periodic/verified_jobs"] == 5
    with (tmp_path / logger.PERIODIC_STREAM).open("ab") as handle:
        handle.write(encoded(all_periodic_events()[-1]))
    metadata = mirror.poll(True)
    assert metadata["experiment_stage"] == "completed"
    assert mirror.finished and remote.run.finished == [0]
    assert remote.run.summary["evaluation/periodic/verified_jobs"] == 6
    assert remote.run.logged[-1][1]["evaluation/periodic/libero_plus/episodes"] == 10030


def test_all_periodic_results_wait_for_both_completion_statuses(tmp_path):
    directory, _, _ = make_sources(tmp_path, periodic=all_periodic_events())
    write_periodic_schedule(tmp_path, completed=False)
    logger.write_json(directory / "status.json", {"status": "completed", "step": 30000})
    mirror, remote = mirror_for(tmp_path)
    assert mirror.poll(True)["experiment_stage"] == "evaluation"
    assert not mirror.finished
    logger.write_json(directory / "experiment_seed_status.json", {"status": "completed"})
    assert mirror.poll(True)["experiment_stage"] == "periodic_evaluation"
    assert not mirror.finished
    logger.write_json(tmp_path / "periodic_evaluation/seed42/status.json", {"status": "completed"})
    assert mirror.poll(True)["experiment_stage"] == "completed"
    assert remote.run.finished == [0]


def test_legacy_experiment_can_complete_without_periodic_schedule(tmp_path):
    directory, _, _ = make_sources(tmp_path)
    logger.write_json(directory / "experiment_seed_status.json", {"status": "completed"})
    mirror, remote = mirror_for(tmp_path)
    assert mirror.poll(True)["experiment_stage"] == "completed"
    assert remote.run.finished == [0]


def test_periodic_summary_is_separate_and_uses_highest_checkpoint_step(tmp_path):
    directory, _, _ = make_sources(tmp_path, periodic=[periodic_event(step=20000, rate=0.7),
                                                      periodic_event(step=10000, rate=0.5)])
    final = {"event": "evaluation_verified", "seed": 42, "benchmark": "libero", "success_rate": 0.8}
    (tmp_path / "experiment_events.jsonl").write_bytes(encoded(final))
    mirror, remote = mirror_for(tmp_path)
    mirror.poll(True)
    summary = remote.run.summary
    assert summary["evaluation/libero/success_rate"] == 0.8
    assert summary["evaluation/periodic/libero/latest_success_rate"] == 0.7
    assert summary["evaluation/periodic/libero/latest_checkpoint_step"] == 20000
    assert [row["optimizer_step"] for _, row in remote.run.logged[-2:]] == [20000, 10000]


def test_persistence_failure_prevents_remote_initialization(tmp_path, monkeypatch):
    make_sources(tmp_path, periodic=[periodic_event()])
    remote = FakeWandb(tmp_path)
    def fail_write(*args, **kwargs):
        raise OSError("disk is full")
    monkeypatch.setattr(logger, "write_json", fail_write)
    with pytest.raises(OSError, match="disk is full"):
        logger.SeedMirror(tmp_path, 42, "test-entity", "test-project", "existing-group", remote)
    assert remote.kwargs is None and remote.run.logged == []


def test_unacknowledged_durable_events_survive_failure_before_metadata_save(tmp_path, monkeypatch):
    directory, _, _ = make_sources(tmp_path, periodic=[periodic_event()])
    mirror, remote = mirror_for(tmp_path, remote_step=3)
    original_write = logger.write_json
    def fail_metadata(path, value):
        if path.name == "wandb_online.json":
            raise OSError("metadata write interrupted")
        original_write(path, value)
    monkeypatch.setattr(logger, "write_json", fail_metadata)
    with pytest.raises(OSError, match="interrupted"):
        mirror.poll(True)
    assert len(remote.run.logged) == 1
    monkeypatch.setattr(logger, "write_json", original_write)
    resumed, new_remote = mirror_for(tmp_path, remote_step=3)
    resumed.poll(True)
    assert new_remote.run.logged == remote.run.logged


@pytest.mark.parametrize("field,value", [("success_rate", math.nan), ("success_rate", math.inf),
    ("success_rate", -0.1), ("success_rate", 1.1), ("success_rate", True), ("episodes", 1999),
    ("episodes", True), ("checkpoint_sha256", "bad"), ("step", -1), ("step", True), ("step", 0), ("step", 11000),
    ("time_unix", math.nan), ("time_unix", None)])
def test_invalid_verified_periodic_metrics_are_rejected(field, value):
    event = periodic_event()
    event[field] = value
    with pytest.raises(ValueError):
        logger.periodic_metrics(event, 0)


def test_periodic_status_and_errors_are_visible_and_clear_after_recovery(tmp_path):
    directory, _, _ = make_sources(tmp_path, periodic=[])
    write_periodic_schedule(tmp_path, completed=False)
    coordinator = tmp_path / "periodic_evaluation/status.json"
    seed_status = tmp_path / "periodic_evaluation/seed42/status.json"
    logger.write_json(coordinator, {"status": "failed", "error": "evaluation server exited"})
    logger.write_json(seed_status, {"status": "failed", "error": "seed job failed"})
    mirror, remote = mirror_for(tmp_path)
    mirror.poll(True)
    summary = remote.run.summary
    assert summary["evaluation/periodic/interval_steps"] == 10000
    assert summary["evaluation/periodic/coordinator_status"] == "failed"
    assert summary["evaluation/periodic/coordinator_error"] == "evaluation server exited"
    assert summary["evaluation/periodic/seed_status"] == "failed"
    assert summary["evaluation/periodic/seed_error"] == "seed job failed"
    logger.write_json(coordinator, {"status": "running"})
    logger.write_json(seed_status, {"status": "pending"})
    mirror.poll(True)
    assert summary["evaluation/periodic/coordinator_status"] == "running"
    assert summary["evaluation/periodic/seed_status"] == "pending"
    assert summary["evaluation/periodic/coordinator_error"] is None
    assert summary["evaluation/periodic/seed_error"] is None
