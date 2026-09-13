"""Experiment completion, checkpoint identity, coverage and retention contracts."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from oat.starvla_heading.data import manifest_sha256
from oat.starvla_heading.evaluation import audit_prompt_mapping, json_hash, load_instruction_catalog, smoke_metadata, summarize_results

spec = importlib.util.spec_from_file_location("experiment_runner", Path(__file__).resolve().parents[1] / "scripts/run_starvla_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "oat/config/starvla_heading_gaussian.yaml").read_text())


def completed_run(output, config):
    expected = runner.seed_configuration(config, 42, output)
    saved = {**expected, "statistics": {"training_only": True}}
    checkpoint = output / "exports/step_00001000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.pt").write_bytes(b"frozen selected checkpoint")
    runner.write_json(checkpoint / "config.json", saved)
    runner.write_json(output / "run_config.json", saved)
    manifest = {"version": 1, "dataset_origin": "original_libero",
                "tasks": [{"suite": suite, "task": f"task{i}", "lang": f"Move item {i}."}
                          for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
                          for i in range(10)]}
    manifest["sha256"] = manifest_sha256(manifest)
    runner.write_json(checkpoint / "dataset_manifest.json", manifest)
    metadata = {**smoke_metadata(), "checkpoint_sha256": runner.file_hash(checkpoint / "weights.pt"),
                "config_sha256": runner.file_hash(checkpoint / "config.json"),
                "training_benchmark": "libero", "selection_benchmark": "libero",
                "validation_normalized_action_mse": .125, "seed": 42, "world_size": 6,
                "full_finetuning": True, "step": 1000, "dataset_manifest_sha256": manifest["sha256"]}
    runner.write_json(checkpoint / "metadata.json", metadata)
    status = {"status": "completed", "step": 30000, "best_validation_mse": .125}
    runner.write_json(output / "status.json", status)
    (output / "best").symlink_to("exports/step_00001000")
    return expected, {"checkpoint": str(checkpoint), "metadata": metadata, "training_status": status}


def official_evaluation(output, benchmark, info, successes=True, missing_last=False):
    output.mkdir(parents=True, exist_ok=True)
    counts = dict(zip(("libero_spatial", "libero_object", "libero_goal", "libero_10"),
                      (10, 10, 10, 10) if benchmark == "libero" else (2402, 2518, 2591, 2519)))
    trials = 50 if benchmark == "libero" else 1
    plan = [{"episode_id": f"{suite}/{task}/init-{trial}", "suite": suite, "task_index": task,
             "init_index": trial, "category": "Original LIBERO" if benchmark == "libero" else "Camera Viewpoints",
             "difficulty_level": "original" if benchmark == "libero" else 1}
            for suite, count in counts.items() for task in range(count) for trial in range(trials)]
    catalog = load_instruction_catalog(Path(info["checkpoint"]) / "dataset_manifest.json")
    for entry in plan:
        task = entry["task_index"]
        original = f"task{task % 10}"
        language = f"Move item {task % 10}."
        entry.update(task_name=original if benchmark == "libero" else original + f"_view_{task}",
                     language=language, canonical_task_name=original,
                     prompt_source="original_training_manifest", prompt_sha256=json_hash(language),
                     language_bddl_sha256=None, source_task_language_sha256=json_hash(language))
    manifest = {"settings": {"benchmark": benchmark, "smoke": False, "policy": info["metadata"],
                             "instruction_catalog": catalog}, "prompt_audit": audit_prompt_mapping(plan, catalog),
                "trials": trials, "scope": {"all_available_selected": True, "all_four_suites": True,
                "available_episodes": len(plan), "selected_episodes": len(plan), "suite_counts": counts}, "plan": plan}
    manifest_hash = json_hash(manifest)
    results = [{**entry, "success": successes, "error": None, "manifest_sha256": manifest_hash} for entry in plan]
    if missing_last:
        results.pop()
    summary = summarize_results(plan, results, official_scope=True)
    runner.write_json(output / "manifest.json", manifest)
    runner.write_json(output / "summary.json", summary)
    (output / "results.jsonl").write_text("".join(json.dumps(result) + "\n" for result in results))
    return summary


def test_fixed_budget_and_gaussian_only_scope(config):
    runner.validate_experiment(config, [42, 43, 44])
    for path, value in [(('variant',), 'heading_zero'), (('training', 'max_steps'), 100),
                        (('training', 'full_finetuning'), False), (('training', 'world_size'), 4),
                        (('training', 'gradient_accumulation_steps'), 0), (('validation', 'benchmark'), 'libero_plus')]:
        broken = copy.deepcopy(config)
        target = broken
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            runner.validate_experiment(broken, [42])
    with pytest.raises(ValueError, match='distinct'):
        runner.validate_experiment(config, [42, 42])


def test_training_command_resumes_only_committed_state_and_never_shortens_budget(tmp_path):
    command = runner.training_command('/config.yaml', 42, tmp_path)
    assert '--resume' not in command and '--max-steps' not in command and '--skip-resume-save' not in command
    assert command[command.index('--seed') + 1] == '42'
    (tmp_path / 'run_config.json').write_text('{}')
    with pytest.raises(RuntimeError, match='refusing to restart'):
        runner.training_command('/config.yaml', 42, tmp_path)
    (tmp_path / 'resume').mkdir()
    (tmp_path / 'resume/commit.json').write_text('{}')
    assert '--resume' in runner.training_command('/config.yaml', 42, tmp_path)
    (tmp_path / 'resume/INCOMPLETE').touch()
    with pytest.raises(RuntimeError, match='INCOMPLETE'):
        runner.training_command('/config.yaml', 42, tmp_path)


def test_commands_use_separate_simulators_and_immutable_selected_export(tmp_path):
    selected = tmp_path / 'exports/step_00001000'
    selected.mkdir(parents=True)
    (tmp_path / 'best').symlink_to(selected)
    server = runner.server_command(tmp_path / 'best')
    assert server[server.index('--checkpoint') + 1] == str(selected)
    assert server[server.index('--device') + 1] == 'cuda:0'
    for benchmark, python, config, trials in [
        ('libero', '/venv/fm_dno/bin/python', '/workspace/libero_config/original', '50'),
        ('libero_plus', '/workspace/venvs/libero-plus/bin/python', '/workspace/libero_config/plus', '1')]:
        command = runner.evaluation_command(benchmark, tmp_path / benchmark, 'http://127.0.0.1:18080', selected / 'dataset_manifest.json')
        assert command[0] == python
        assert command[command.index('--libero-config-path') + 1] == config
        assert command[command.index('--trials') + 1] == trials
        assert command[command.index('--training-manifest') + 1] == str(selected / 'dataset_manifest.json')
        assert command[command.index('--render-gpu-device-id') + 1] == '1'
        assert '--limit' not in command and '--smoke' not in command


def test_completed_training_requires_status_configuration_selection_and_weight_integrity(tmp_path, config):
    expected, info = completed_run(tmp_path, config)
    assert runner.verify_completed_training(tmp_path, expected) == info
    wrong = copy.deepcopy(expected)
    wrong['seed'] = 43
    with pytest.raises(RuntimeError, match='configuration differs'):
        runner.verify_completed_training(tmp_path, wrong)
    runner.write_json(tmp_path / 'status.json', {**info['training_status'], 'step': 20})
    with pytest.raises(RuntimeError, match='fixed training budget'):
        runner.verify_completed_training(tmp_path, expected)
    runner.write_json(tmp_path / 'status.json', info['training_status'])
    (Path(info['checkpoint']) / 'weights.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='weights changed'):
        runner.verify_completed_training(tmp_path, expected)
    runner.write_json(tmp_path / 'status.json', {'status': 'interrupted', 'step': 20})
    assert runner.verify_completed_training(tmp_path, expected) is None


def test_evaluation_completion_recomputed_from_all_official_episodes(tmp_path, config):
    _, info = completed_run(tmp_path / 'seed42', config)
    output = tmp_path / 'eval'
    assert runner.verify_evaluation(output, 'libero', info) is None
    official_evaluation(output, 'libero', info, missing_last=True)
    assert runner.verify_evaluation(output, 'libero', info) is None
    summary = official_evaluation(output, 'libero', info)
    assert runner.verify_evaluation(output, 'libero', info) == summary
    runner.write_json(output / 'summary.json', {**summary, 'official_success_rate': .3})
    with pytest.raises(RuntimeError, match='summary differs'):
        runner.verify_evaluation(output, 'libero', info)
    official_evaluation(output, 'libero', info)
    manifest = runner.read_json(output / 'manifest.json')
    manifest['scope']['all_available_selected'] = False
    runner.write_json(output / 'manifest.json', manifest)
    with pytest.raises(RuntimeError, match='Partial evaluation'):
        runner.verify_evaluation(output, 'libero', info)


def test_mean_and_sample_std_measure_training_seed_variance_only():
    def summary(rate):
        return {'official_success_rate': rate, 'breakdowns': {
            kind: {'name': {'success_rate': rate}} for kind in ('suite', 'category', 'difficulty_level')}}
    reports = {42: {'libero': summary(.8), 'libero_plus': summary(.4)},
               43: {'libero': summary(.9), 'libero_plus': summary(.6)}}
    result = runner.aggregate_results([42, 43, 44], reports)
    assert not result['complete']
    assert result['completed_training_seeds'] == [42, 43]
    assert result['benchmarks']['libero']['mean_success_rate'] == pytest.approx(.85)
    assert result['benchmarks']['libero_plus']['std_across_training_seeds'] == pytest.approx(.2 / 2**.5)
    assert runner.aggregate_results([42], {42: reports[42]})['benchmarks']['libero']['std_across_training_seeds'] is None


def test_retention_requires_real_coverage_then_preserves_selected_policy_and_evidence(tmp_path, config):
    _, info = completed_run(tmp_path, config)
    resume = tmp_path / 'resume/state'
    resume.mkdir(parents=True)
    (resume / 'optimizer.pt').write_bytes(b'generated optimizer state')
    latest = tmp_path / 'exports/step_00030000'
    latest.mkdir()
    (latest / 'weights.pt').write_bytes(b'unselected last checkpoint')
    (tmp_path / 'latest').symlink_to(latest)
    with pytest.raises(RuntimeError, match='reload verification'):
        runner.prune_completed_optimizer(tmp_path, info, {})
    runner.write_json(tmp_path / 'evaluation/checkpoint_verified.json', {**info,
                      'verification': 'owned_inference_server_loaded_checkpoint'})
    with pytest.raises(RuntimeError, match='both full benchmark'):
        runner.prune_completed_optimizer(tmp_path, info, {})
    assert resume.exists() and latest.exists()
    reports = {benchmark: official_evaluation(tmp_path / 'evaluation' / benchmark, benchmark, info)
               for benchmark in ('libero', 'libero_plus')}
    runner.prune_completed_optimizer(tmp_path, info, reports)
    assert not (tmp_path / 'resume').exists() and not latest.exists()
    assert (tmp_path / 'best/weights.pt').is_file()
    assert (tmp_path / 'evaluation/libero_plus/results.jsonl').is_file()
    assert (tmp_path / 'run_config.json').is_file()
    retention = runner.read_json(tmp_path / 'retention.json')
    assert len(retention['removed']) == 2
    runner.prune_completed_optimizer(tmp_path, info, reports)
    assert runner.read_json(tmp_path / 'retention.json') == retention


def test_process_cleanup_is_idempotent_and_does_not_revisit_reused_pid(monkeypatch, tmp_path):
    child = runner.OwnedProcess.__new__(runner.OwnedProcess)
    child.kind, child.stopped = 'evaluation', False
    child.worker_groups = set()
    child.process = SimpleNamespace(pid=999999, poll=lambda: 0)
    child.log = (tmp_path / 'log').open('w')
    signals = []
    child._group_signal = lambda sig: signals.append(sig)
    def missing_group(pid, sig):
        raise ProcessLookupError()
    monkeypatch.setattr(runner.os, 'killpg', missing_group)
    child.stop()
    first = list(signals)
    child.stop()
    assert signals == first and child.stopped and child.log.closed


def test_torchrun_workers_in_separate_groups_are_owned_without_signaling_loader_children(tmp_path):
    def proc(pid, ppid, pgrp, argv):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / 'stat').write_text(f'{pid} (python worker) S {ppid} {pgrp} 1 0 0')
        (directory / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in argv))
    proc(201, 100, 201, ['python', 'scripts/train_starvla_heading.py', '--seed', '42'])
    proc(202, 100, 202, ['python', 'scripts/train_starvla_heading.py', '--seed', '42'])
    proc(301, 201, 201, ['python', 'scripts/train_starvla_heading.py'])  # inherited loader argv
    proc(401, 99, 401, ['python', 'scripts/train_starvla_heading.py'])  # unrelated training
    child = runner.OwnedProcess.__new__(runner.OwnedProcess)
    child.kind, child.worker_groups = 'training', set()
    child.process = SimpleNamespace(pid=100)
    assert sorted(child.track_training_workers(tmp_path)) == [201, 202]
    assert child.worker_groups == {201, 202}


def test_sequential_evaluation_defaults_and_overrides_fit_six_cards(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    default_server = runner.server_command(checkpoint)
    custom_server = runner.server_command(checkpoint, device='cuda:4')
    assert default_server[default_server.index('--device') + 1] == 'cuda:0'
    assert custom_server[custom_server.index('--device') + 1] == 'cuda:4'
    evaluation = runner.evaluation_command('libero_plus', tmp_path / 'eval', 'http://localhost:18080', checkpoint / 'dataset_manifest.json', workers=2, render_gpu=5)
    assert evaluation[evaluation.index('--workers') + 1] == '2'
    assert evaluation[evaluation.index('--render-gpu-device-id') + 1] == '5'


def test_measured_microbatch_changes_preserve_fixed_effective_batch(config):
    for microbatch, accumulation in ((1, 10), (2, 5), (5, 2), (10, 1)):
        measured = copy.deepcopy(config)
        measured['training'].update(micro_batch_size=microbatch, gradient_accumulation_steps=accumulation)
        runner.validate_experiment(measured, [42, 43, 44])
    for microbatch, accumulation in ((0, 10), (5, 1), (1, 9), (1.5, 10)):
        broken = copy.deepcopy(config)
        broken['training'].update(micro_batch_size=microbatch, gradient_accumulation_steps=accumulation)
        with pytest.raises(ValueError, match='effective batch60'):
            runner.validate_experiment(broken, [42])


def test_experiment_transition_verifies_training_before_server_and_resumes_only_missing_evaluation(tmp_path, config):
    args = SimpleNamespace(output_root=tmp_path, config=tmp_path / 'config.yaml', seeds=[42],
                           server_port=0, server_startup_timeout=1, server_device='cuda:0',
                           render_gpu_device_id=1, eval_workers=4, prune_completed_optimizer=False)
    transitions = []

    class ArtifactProcessExperiment(runner.Experiment):
        # Replace only external execution; all checkpoint/coverage/resume
        # verification operates on complete real on-disk protocol artifacts.
        def run_command(self, command, kind, log, env=None):
            transitions.append((kind, tuple(command)))
            if kind == 'training':
                assert env['STARVLA_TRAIN_GPUS'] == '0,1,2,3,4,5'
                completed_run(tmp_path / 'seed42', config)
            else:
                benchmark = command[command.index('--benchmark') + 1]
                info = runner.verify_completed_training(tmp_path / 'seed42',
                        runner.seed_configuration(config, 42, tmp_path / 'seed42'))
                official_evaluation(tmp_path / 'seed42/evaluation' / benchmark, benchmark, info)

        def start(self, command, kind, log, env=None):
            assert kind == 'server'
            assert runner.read_json(tmp_path / 'seed42/status.json')['step'] == 30000
            transitions.append((kind, tuple(command)))
            return SimpleNamespace(stop=lambda: transitions.append(('server_stopped', ())))

        def wait_for_server(self, server, info):
            transitions.append(('checkpoint_reloaded', (info['metadata']['checkpoint_sha256'],)))

    experiment = ArtifactProcessExperiment(args, config)
    experiment.run_seed(42)
    assert [kind for kind, _ in transitions] == [
        'training', 'server', 'checkpoint_reloaded', 'evaluation', 'evaluation', 'server_stopped']
    assert runner.read_json(tmp_path / 'experiment_summary.json')['complete']
    count = len(transitions)
    experiment.run_seed(42)
    assert len(transitions) == count  # Completed training and both evaluations are verified then reused.
    info = runner.verify_completed_training(tmp_path / 'seed42',
            runner.seed_configuration(config, 42, tmp_path / 'seed42'))
    official_evaluation(tmp_path / 'seed42/evaluation/libero_plus', 'libero_plus', info, missing_last=True)
    experiment.run_seed(42)
    assert [kind for kind, _ in transitions[count:]] == ['server', 'checkpoint_reloaded', 'evaluation', 'server_stopped']
    resumed = transitions[count + 2][1]
    assert resumed[resumed.index('--benchmark') + 1] == 'libero_plus'



def test_coordinator_rejects_legacy_filename_prompt_protocol_even_with_complete_results(tmp_path, config):
    _, info = completed_run(tmp_path / "seed42", config)
    output = tmp_path / "evaluation"
    official_evaluation(output, "libero_plus", info)
    manifest = runner.read_json(output / "manifest.json")
    del manifest["settings"]["instruction_catalog"]
    del manifest["prompt_audit"]
    runner.write_json(output / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="obsolete instruction protocol"):
        runner.verify_evaluation(output, "libero_plus", info)


def test_coordinator_rejects_modified_canonical_instructions_even_if_rehashed(tmp_path, config):
    _, info = completed_run(tmp_path / "seed42", config)
    output = tmp_path / "evaluation"
    official_evaluation(output, "libero", info)
    manifest = runner.read_json(output / "manifest.json")
    entry = manifest["plan"][0]
    entry["language"] += " view 0 0 initstate 0 noise 40"
    entry["prompt_sha256"] = json_hash(entry["language"])
    runner.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="changed the canonical original"):
        runner.verify_evaluation(output, "libero", info)
