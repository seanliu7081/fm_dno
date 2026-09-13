"""Milestone capture, pruning races, resume and verified periodic success rates."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from oat.starvla_heading.data import manifest_sha256
from oat.starvla_heading.evaluation import smoke_metadata

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


periodic = load_script("periodic_evaluation_under_test", ROOT / "scripts/run_starvla_periodic_eval.py")
artifacts = load_script("periodic_evaluation_test_artifacts", ROOT / "tests/test_starvla_experiment.py")


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "oat/config/starvla_heading_gaussian.yaml").read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def metric(seed_output, event):
    seed_output.mkdir(parents=True, exist_ok=True)
    with (seed_output / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def prepared_checkpoint(tmp_path, config, seed=42, step=10000, committed=True):
    seed_output = tmp_path / "training" / f"seed{seed}"
    expected = copy.deepcopy(config)
    expected.update(seed=seed, output_dir=str(seed_output.resolve()))
    exported = {**expected, "statistics": {"training_only": True}}
    source = seed_output / "exports" / f"step_{step:08d}"
    source.mkdir(parents=True)
    (source / "weights.pt").write_bytes(f"small checkpoint seed={seed} step={step}".encode())
    write_json(source / "config.json", exported)
    write_json(seed_output / "run_config.json", exported)
    manifest = {"version": 1, "dataset_origin": "original_libero",
                "tasks": [{"suite": suite, "task": f"task{i}", "lang": f"Move item {i}."}
                          for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
                          for i in range(10)]}
    manifest["sha256"] = manifest_sha256(manifest)
    write_json(source / "dataset_manifest.json", manifest)
    metadata = {**smoke_metadata(), "training_benchmark": "libero", "selection_benchmark": "libero",
                "seed": seed, "step": step, "world_size": 6, "full_finetuning": True,
                "selection_metric": "normalized_action_mse", "validation_normalized_action_mse": .125,
                "dataset_manifest_sha256": manifest["sha256"],
                "checkpoint_sha256": artifacts.runner.file_hash(source / "weights.pt"),
                "config_sha256": artifacts.runner.file_hash(source / "config.json")}
    write_json(source / "metadata.json", metadata)
    if committed:
        metric(seed_output, {"event": "checkpoint", "step": step, "optimizer_state_saved": True})
    job = tmp_path / "periodic_evaluation" / f"seed{seed}" / f"step_{step:08d}"
    return SimpleNamespace(seed_output=seed_output, expected=expected, source=source,
                           job=job, destination=job / "checkpoint", metadata=metadata, seed=seed, step=step)


def capture(fixture):
    return periodic.capture_checkpoint(fixture.seed_output, fixture.destination,
                                       fixture.seed, fixture.step, fixture.expected)


def test_checkpoint_events_require_completed_lines_and_committed_optimizer(tmp_path):
    assert periodic.read_committed_steps(tmp_path) == set()
    metric(tmp_path, {"event": "training", "step": 10000})
    metric(tmp_path, {"event": "validation", "step": 10000})
    metric(tmp_path, {"event": "checkpoint", "step": 10000, "optimizer_state_saved": False})
    metric(tmp_path, {"event": "checkpoint", "step": 1000, "optimizer_state_saved": True})
    partial = json.dumps({"event": "checkpoint", "step": 10000, "optimizer_state_saved": True})
    with (tmp_path / "metrics.jsonl").open("a") as stream:
        stream.write(partial)
    assert periodic.read_committed_steps(tmp_path) == {1000}
    with (tmp_path / "metrics.jsonl").open("a") as stream:
        stream.write("\n")
    assert periodic.read_committed_steps(tmp_path) == {1000, 10000}
    metric(tmp_path, {"event": "checkpoint", "step": 10000, "optimizer_state_saved": True})
    assert periodic.read_committed_steps(tmp_path) == {1000, 10000}


def test_complete_corrupt_metric_line_is_not_silently_skipped(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"event":"checkpoint","step":\n')
    with pytest.raises((ValueError, RuntimeError)):
        periodic.read_committed_steps(tmp_path)


def test_visible_export_is_not_captured_before_optimizer_commit_event(tmp_path, config):
    f = prepared_checkpoint(tmp_path, config, committed=False)
    assert capture(f) is None
    assert not f.destination.exists()
    metric(f.seed_output, {"event": "checkpoint", "step": f.step, "optimizer_state_saved": False})
    assert capture(f) is None
    metric(f.seed_output, {"event": "checkpoint", "step": f.step, "optimizer_state_saved": True})
    assert capture(f)["metadata"]["step"] == f.step


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("step", [10000, 20000, 30000])
def test_capture_preserves_every_exact_seed_milestone_against_trainer_pruning(tmp_path, config, seed, step):
    f = prepared_checkpoint(tmp_path, config, seed, step)
    info = capture(f)
    assert info["checkpoint"] == str(f.destination.resolve())
    assert info["metadata"] == f.metadata
    for name in ("weights.pt", "metadata.json", "config.json", "dataset_manifest.json"):
        assert (f.source / name).stat().st_ino == (f.destination / name).stat().st_ino
    shutil.rmtree(f.source)
    assert artifacts.runner.file_hash(f.destination / "weights.pt") == f.metadata["checkpoint_sha256"]
    assert capture(f) == info  # a restart reuses the captured export even after source pruning


@pytest.mark.parametrize("target_event", [False, True])
def test_missing_or_pruned_milestone_fails_instead_of_substituting_later_checkpoint(tmp_path, config, target_event):
    f = prepared_checkpoint(tmp_path, config, committed=target_event)
    shutil.rmtree(f.source)
    metric(f.seed_output, {"event": "checkpoint", "step": 11000, "optimizer_state_saved": True})
    later = f.seed_output / "exports/step_00011000"
    later.mkdir()
    (later / "weights.pt").write_bytes(b"later checkpoint must not be selected")
    with pytest.raises(RuntimeError):
        capture(f)
    assert not f.destination.exists()


@pytest.mark.parametrize("field,value", [("seed", 43), ("step", 20000), ("world_size", 5),
                                          ("full_finetuning", False), ("variant", "heading_zero"),
                                          ("dataset_manifest_sha256", "f" * 64)])
def test_capture_rejects_mismatched_checkpoint_identity(tmp_path, config, field, value):
    f = prepared_checkpoint(tmp_path, config)
    changed = {**f.metadata, field: value}
    write_json(f.source / "metadata.json", changed)
    with pytest.raises((ValueError, RuntimeError)):
        capture(f)
    assert not f.destination.exists()


@pytest.mark.parametrize("name", ["weights.pt", "config.json", "dataset_manifest.json"])
def test_capture_rejects_modified_export_files(tmp_path, config, name):
    f = prepared_checkpoint(tmp_path, config)
    with (f.source / name).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises((ValueError, RuntimeError)):
        capture(f)
    assert not f.destination.exists()


def test_capture_rejects_changed_run_configuration_even_when_export_digest_is_recomputed(tmp_path, config):
    f = prepared_checkpoint(tmp_path, config)
    changed = json.loads((f.source / "config.json").read_text())
    changed["training"]["max_steps"] = 1000
    write_json(f.source / "config.json", changed)
    changed_metadata = {**f.metadata, "config_sha256": artifacts.runner.file_hash(f.source / "config.json")}
    write_json(f.source / "metadata.json", changed_metadata)
    with pytest.raises((ValueError, RuntimeError)):
        capture(f)


def test_snapshot_cannot_live_inside_the_trainers_pruned_export_directory(tmp_path, config):
    f = prepared_checkpoint(tmp_path, config)
    f.destination = f.seed_output / "exports" / "periodic" / "checkpoint"
    with pytest.raises((ValueError, RuntimeError)):
        capture(f)
    assert not f.destination.exists()


def test_source_pruned_mid_capture_never_publishes_partial_checkpoint(tmp_path, config, monkeypatch):
    f = prepared_checkpoint(tmp_path, config)
    real_link = os.link
    linked = []

    def prune_after_first_link(source, destination, *args, **kwargs):
        real_link(source, destination, *args, **kwargs)
        linked.append(Path(destination))
        if len(linked) == 1:
            shutil.rmtree(f.source)

    monkeypatch.setattr(periodic.os, "link", prune_after_first_link)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        capture(f)
    assert linked
    assert not f.destination.exists()


def verified_event(**changes):
    return {"event": "evaluation_verified", "seed": 42, "step": 10000, "benchmark": "libero",
            "checkpoint_sha256": "a" * 64, "success_rate": .5, "episodes": 2000, "time_unix": 1,
            **changes}


def test_verified_events_are_idempotent_across_resume_and_preserve_step_identity(tmp_path):
    path = tmp_path / "periodic_events.jsonl"
    assert periodic.record_verified_event(path, verified_event())
    assert not periodic.record_verified_event(path, verified_event(time_unix=2))
    assert periodic.record_verified_event(path, verified_event(step=20000))
    assert periodic.record_verified_event(path, verified_event(seed=43))
    assert periodic.record_verified_event(path, verified_event(benchmark="libero_plus", episodes=10030))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(events) == 4
    assert {(e["seed"], e["step"], e["benchmark"]) for e in events} == {
        (42, 10000, "libero"), (42, 20000, "libero"), (43, 10000, "libero"), (42, 10000, "libero_plus")}


@pytest.mark.parametrize("changes", [{"checkpoint_sha256": "b" * 64}, {"success_rate": .75}, {"episodes": 1999}])
def test_conflicting_verified_event_cannot_overwrite_success_history(tmp_path, changes):
    path = tmp_path / "periodic_events.jsonl"
    periodic.record_verified_event(path, verified_event())
    before = path.read_bytes()
    with pytest.raises(ValueError):
        periodic.record_verified_event(path, verified_event(**changes))
    assert path.read_bytes() == before



@pytest.fixture
def captured_job(tmp_path, config):
    f = prepared_checkpoint(tmp_path, config)
    info = capture(f)
    reports = {benchmark: artifacts.official_evaluation(f.job / benchmark, benchmark, info)
               for benchmark in ("libero", "libero_plus")}
    receipt = {"status": "completed", "checkpoint_sha256": info["metadata"]["checkpoint_sha256"],
               "metadata": info["metadata"], "server_stopped": True}
    write_json(f.job / "verification.json", receipt)
    return f, info, reports, receipt


def test_release_only_snapshot_weights_after_both_complete_benchmarks_and_server_stop(captured_job):
    f, info, reports, receipt = captured_job
    assert reports["libero"]["overall"]["expected"] == 2000
    assert reports["libero_plus"]["overall"]["expected"] == 10030
    assert reports["libero"]["official_benchmark_complete"]
    assert reports["libero_plus"]["official_benchmark_complete"]
    periodic.release_snapshot_weights(f.job, info, reports)
    assert not (f.destination / "weights.pt").exists()
    assert (f.source / "weights.pt").is_file()
    assert artifacts.runner.file_hash(f.source / "weights.pt") == f.metadata["checkpoint_sha256"]
    for name in ("metadata.json", "config.json", "dataset_manifest.json"):
        assert (f.destination / name).is_file()
    assert (f.job / "libero_plus/results.jsonl").is_file()
    saved = json.loads((f.job / "verification.json").read_text())
    assert saved["weights_released_at_unix"] > 0
    periodic.release_snapshot_weights(f.job, info, reports)
    assert json.loads((f.job / "verification.json").read_text()) == saved


@pytest.mark.parametrize("change", [{"server_stopped": False}, {"status": "running"},
                                   {"checkpoint_sha256": "f" * 64}])
def test_release_refuses_unstopped_server_or_unverified_checkpoint(captured_job, change):
    f, info, reports, receipt = captured_job
    write_json(f.job / "verification.json", {**receipt, **change})
    with pytest.raises((ValueError, RuntimeError)):
        periodic.release_snapshot_weights(f.job, info, reports)
    assert (f.destination / "weights.pt").is_file()


def test_release_recomputes_full_coverage_instead_of_trusting_completed_summary(captured_job):
    f, info, reports, receipt = captured_job
    artifacts.official_evaluation(f.job / "libero_plus", "libero_plus", info, missing_last=True)
    with pytest.raises((ValueError, RuntimeError)):
        periodic.release_snapshot_weights(f.job, info, reports)
    assert (f.destination / "weights.pt").is_file()
    assert "weights_released_at_unix" not in json.loads((f.job / "verification.json").read_text())


def test_release_refuses_reports_from_another_checkpoint_or_changed_scores(captured_job):
    f, info, reports, receipt = captured_job
    changed = copy.deepcopy(reports)
    changed["libero"]["official_success_rate"] = .25
    with pytest.raises((ValueError, RuntimeError)):
        periodic.release_snapshot_weights(f.job, info, changed)
    assert (f.destination / "weights.pt").is_file()


def test_release_rejects_symlink_escape_without_removing_original_weights(captured_job):
    f, info, reports, receipt = captured_job
    shutil.rmtree(f.destination)
    f.destination.symlink_to(f.source, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError)):
        periodic.release_snapshot_weights(f.job, info, reports)
    assert (f.source / "weights.pt").is_file()



def test_resume_repairs_only_torn_owned_event_tail_preserving_complete_history(tmp_path):
    path = tmp_path / "events.jsonl"
    initial = {"event": "checkpoint_captured", "seed": 42, "step": 10000}
    path.write_text(json.dumps(initial) + "\n")
    periodic.record_verified_event(path, verified_event())
    complete_prefix = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"event":"evaluation_verified","seed":43')
    next_event = verified_event(seed=43)
    assert periodic.record_verified_event(path, next_event)
    assert path.read_bytes().startswith(complete_prefix)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [initial, verified_event(), next_event]
    assert not periodic.record_verified_event(path, {**next_event, "time_unix": 99})
    assert len(path.read_text().splitlines()) == 3


@pytest.mark.parametrize("changes", [{"episodes": 1999}, {"benchmark": "libero_plus", "episodes": 10029},
                                     {"step": 9999}, {"success_rate": float("nan")},
                                     {"success_rate": 1.1}, {"checkpoint_sha256": "not-a-sha"}])
def test_unverified_partial_or_invalid_rate_events_are_never_published(tmp_path, changes):
    path = tmp_path / "events.jsonl"
    with pytest.raises(ValueError):
        periodic.record_verified_event(path, verified_event(**changes))
    assert not path.exists()


def coordinator(tmp_path, config):
    args = SimpleNamespace(output_root=tmp_path / "training", server_port=18087,
                           server_device="cuda:6", render_gpu_device_id=7, eval_workers=4,
                           poll_seconds=.000001, server_startup_timeout=10)
    result = periodic.PeriodicExperiment(args, {"config": config})
    result.output.mkdir(parents=True, exist_ok=True)
    return result


def test_all_seeds_keep_capturing_new_milestones_while_evaluation_child_is_running(tmp_path, config, monkeypatch):
    prepared_checkpoint(tmp_path, config, 42, 5000)  # not a periodic milestone
    prepared_checkpoint(tmp_path, config, 42, 10000)
    experiment = coordinator(tmp_path, config)
    state = {"running": False, "polls": 0, "stopped": False}
    links = []
    real_link = os.link

    def link_during_live_child(source, destination, *args, **kwargs):
        assert state["running"], "a delayed capture was postponed until evaluation exited"
        links.append(Path(destination))
        return real_link(source, destination, *args, **kwargs)

    def poll():
        state["polls"] += 1
        state["running"] = state["polls"] <= 2
        if state["polls"] == 2:
            for seed in (42, 43, 44):
                for step in (10000, 20000, 30000):
                    if (seed, step) != (42, 10000):
                        prepared_checkpoint(tmp_path, config, seed, step)
        return None if state["running"] else 0

    def stop():
        state["stopped"] = True

    child = SimpleNamespace(process=SimpleNamespace(poll=poll, returncode=0), stop=stop)
    monkeypatch.setattr(experiment, "start", lambda *args: child)
    monkeypatch.setattr(periodic.os, "link", link_during_live_child)
    monkeypatch.setattr(periodic.time, "sleep", lambda seconds: None)
    experiment.run_command(["fake-evaluator"], "evaluation", tmp_path / "eval.log")
    expected = {(seed, step) for seed in (42, 43, 44) for step in (10000, 20000, 30000)}
    assert set(experiment.infos) == expected
    assert len(links) == 9 * 4
    assert state["stopped"]
    for seed, step in expected:
        snapshot = experiment.job_directory(seed, step) / "checkpoint"
        assert (snapshot / "weights.pt").is_file()
        shutil.rmtree(tmp_path / "training" / f"seed{seed}" / "exports" / f"step_{step:08d}")

    # Current process keeps verified snapshots in memory without rereading large
    # weight files on every poll; a fresh coordinator validates them on resume.
    with monkeypatch.context() as patch:
        patch.setattr(periodic, "file_hash", lambda path: pytest.fail("cached snapshot was rehashed on every poll"))
        experiment.capture_due(force=True)
    resumed = coordinator(tmp_path, config)
    resumed.capture_due(force=True)
    assert resumed.infos == experiment.infos
    assert set(resumed.infos) == expected


def test_capture_continues_during_server_startup_without_network(tmp_path, config, monkeypatch):
    import io
    experiment = coordinator(tmp_path, config)
    observed = []
    info = {"metadata": {"checkpoint_sha256": "a" * 64}}
    server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), log_path=tmp_path / "server.log")

    def response(*args, **kwargs):
        if len(observed) == 1:
            raise periodic.URLError("server is still loading")
        return io.StringIO(json.dumps(info["metadata"]))

    monkeypatch.setattr(experiment, "capture_due", lambda: observed.append("capture"))
    monkeypatch.setattr(periodic, "urlopen", response)
    monkeypatch.setattr(periodic.time, "sleep", lambda seconds: None)
    experiment.wait_for_server(server, info)
    assert observed == ["capture", "capture"]


def test_capture_failure_during_active_child_stops_that_owned_child(tmp_path, config, monkeypatch):
    experiment = coordinator(tmp_path, config)
    stopped = []
    child = SimpleNamespace(process=SimpleNamespace(poll=lambda: None, returncode=None),
                            stop=lambda: stopped.append(True))
    monkeypatch.setattr(experiment, "start", lambda *args: child)

    def missed_milestone():
        raise RuntimeError("Required committed export was pruned")

    monkeypatch.setattr(experiment, "capture_due", missed_milestone)
    with pytest.raises(RuntimeError, match="was pruned"):
        experiment.run_command(["fake-evaluator"], "evaluation", tmp_path / "eval.log")
    assert stopped == [True]


def test_archived_completed_job_resumes_after_weights_release_without_republishing_rates(captured_job, config):
    f, info, reports, receipt = captured_job
    periodic.release_snapshot_weights(f.job, info, reports)
    released_receipt = (f.job / "verification.json").read_bytes()
    shutil.rmtree(f.source)
    resumed_info = capture(f)
    assert resumed_info == info
    assert not (f.destination / "weights.pt").exists()
    experiment = coordinator(f.seed_output.parent.parent, config)
    experiment.output = f.job.parents[1]
    experiment.job_directory = lambda seed, step: f.job
    experiment.accept_completed_job(f.seed, f.step, resumed_info)
    experiment.accept_completed_job(f.seed, f.step, resumed_info)
    assert experiment.reports == {(f.seed, f.step): reports}
    events = [json.loads(line) for line in (experiment.output / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2
    assert {event["benchmark"] for event in events} == {"libero", "libero_plus"}
    assert all(event["step"] == 10000 for event in events)
    assert (f.job / "verification.json").read_bytes() == released_receipt
    assert not (f.destination / "weights.pt").exists()
    assert (f.job / "libero_plus/results.jsonl").is_file()


def test_summary_requires_all_nine_seed_milestone_pairs_and_keeps_seed_uncertainty_separate(tmp_path, config):
    experiment = coordinator(tmp_path, config)

    def report(rate):
        return {"official_success_rate": rate,
                "breakdowns": {kind: {"task": {"success_rate": rate}}
                               for kind in ("suite", "category", "difficulty_level")}}

    for seed in (42, 43, 44):
        for step in (10000, 20000, 30000):
            if (seed, step) != (44, 30000):
                experiment.reports[seed, step] = {
                    "libero": report(.4 + .1 * (seed - 42)), "libero_plus": report(.2 + .1 * (seed - 42))}
    experiment.update_summary()
    summary = json.loads((experiment.output / "summary.json").read_text())
    assert not summary["complete"]
    assert summary["steps"] == [10000, 20000, 30000]
    assert summary["seeds"] == [42, 43, 44]
    assert summary["by_step"]["10000"]["benchmarks"]["libero"]["mean_success_rate"] == pytest.approx(.5)
    assert summary["by_step"]["10000"]["benchmarks"]["libero"]["std_across_training_seeds"] == pytest.approx(.1)
    status = json.loads((experiment.output / "seed44/status.json").read_text())
    assert status["completed_steps"] == [10000, 20000] and status["status"] == "pending"
    experiment.reports[44, 30000] = {"libero": report(.6), "libero_plus": report(.4)}
    experiment.update_summary()
    assert json.loads((experiment.output / "summary.json").read_text())["complete"]
