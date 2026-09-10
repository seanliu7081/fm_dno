"""Synthetic assessment integrity tests; no checkpoint deserialization or GPUs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'heading_comparison', Path(__file__).resolve().parents[1] / 'scripts/report_heading_checkpoint_comparison.py')
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)
from scripts.report_heading_goal import make_episode_plan, summarize_results


@pytest.fixture
def assessments(tmp_path):
    before, after = tmp_path/'final', tmp_path/'epoch20_latest'
    data = []
    hashes = []
    for directory, epoch in ((before, 15), (after, 20)):
        evaluation = directory/'eval_500'
        evaluation.mkdir(parents=True)
        payload = f'Synthetic epoch {epoch} export'.encode()
        checksum = hashlib.sha256(payload).hexdigest()
        hashes.append(checksum)
        (directory/'policy.ckpt').write_bytes(payload)
        (evaluation/'policy.ckpt').write_bytes(payload)
        plan = make_episode_plan([f'task_{i}' for i in range(10)], 50, 20260911)
        for episode in plan:
            episode['official_state_sha256'] = hashlib.sha256(episode['episode_id'].encode()).hexdigest()
        # Epoch 20 gains ten cases per task and loses five: net +50/500.
        results = [{**item, 'success': item['init_index'] < 30 if epoch == 15 else 5 <= item['init_index'] < 40,
                    'error': None} for item in plan]
        summary = summarize_results(plan, results)
        summary.update(checkpoint_sha256=checksum, weights='ema_model')
        manifest = {
            'suite': 'libero_10', 'n_per_task': 50, 'init_start': 0, 'seed': 20260911,
            'max_episode_steps': 550, 'settle_steps': 10, 'protocol_version': 1,
            'software_versions': {'torch': 'test'}, 'source_sha256': {'evaluator.py': 'a'*64},
            'episodes': plan, 'checkpoint_sha256': checksum, 'weights': 'ema_model',
            'training_position': {'epoch': epoch, 'global_step': epoch*2000},
            'source_checkpoint': str(directory/'policy.ckpt'), 'checkpoint': str(evaluation/'policy.ckpt'),
            'policy_config': {
                '_target_': 'oat.policy.flow_policy_heading_gaussian.HeadingGaussianFlowPolicy',
                'backbone_type': 'unet', 'heading_mode': 'condition', 'source_mode': 'heading',
                'horizon': 16, 'n_action_steps': 8, 'n_obs_steps': 2, 'num_inference_steps': 10,
                'prior_parallel_std': 1., 'prior_perpendicular_std': .5,
                'obs_encoder': {'vision_encoder': {'crop_shape': [116, 116], 'eval_fixed_crop': True}},
                'shape_meta': {'obs': {name: {} for name in ('agentview_rgb', 'robot0_eye_in_hand_rgb',
                    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'task_uid')}},
            },
        }
        data.append({'directory': directory, 'manifest': manifest, 'summary': summary, 'results': results})
    manifest = data[1]['manifest']
    exact = {'exactly_equal': True, 'compared_tensor_count': 4, 'left_state_entry_count': 4,
             'right_state_entry_count': 4, 'left_only_keys': [], 'right_only_keys': [],
             'different_entries': [], 'nonfinite_tensor_keys': [], 'max_absolute_tensor_difference': 0.}
    audit = {
        'checkpoints': [{'path': f'/training/{name}', 'file_sha256': character*64, 'use_ema': True,
                         'training_position': manifest['training_position'], 'selected_state': 'ema_model'}
                        for name, character in (('ep-0020.ckpt', 'b'), ('latest.ckpt', 'c'))],
        'ema_comparison': copy.deepcopy(exact), 'resolved_policy_configs_equal': True,
        'resolved_inference_configs_equal': True, 'training_positions_equal': True,
        'one_full_evaluation_covers_both': True,
        'resolved_policy_config': copy.deepcopy(manifest['policy_config']),
        'evaluation_protocol_to_share': {key: manifest[key] for key in reporter.PROTOCOL_KEYS},
        'exported_checkpoint': {'path': str(after/'policy.ckpt'), 'file_sha256': hashes[1],
                                'training_position': manifest['training_position'], 'selected_state': 'ema_model'},
        'exported_resolved_policy_config_equal': True, 'exported_ema_comparison': copy.deepcopy(exact),
    }
    def save():
        for item in data:
            evaluation = item['directory']/'eval_500'
            (evaluation/'manifest.json').write_text(json.dumps(item['manifest']))
            (evaluation/'summary.json').write_text(json.dumps(item['summary']))
            (evaluation/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in item['results']))
        (after/'checkpoint_equivalence.json').write_text(json.dumps(audit))
    save()
    return before, after, hashes, data, audit, save


def build(assessments):
    before, after, hashes, *_ = assessments
    return reporter.build_comparison(before, after, *hashes)


def test_counts_deltas_and_pairs_use_matching_complete_episodes(assessments):
    report = build(assessments)
    assert report['epoch15']['full_500']['successes'] == 300
    assert report['epoch20_and_latest']['full_500']['successes'] == 350
    assert report['delta_epoch20_minus_epoch15']['full_500']['percentage_points'] == pytest.approx(10)
    assert report['paired_outcomes']['full_500'] == {
        'both_succeed': 250, 'both_fail': 100, 'epoch15_only_succeeds': 50, 'epoch20_only_succeeds': 100}
    assert report['epoch15']['previously_reserved_400']['successes'] == 200
    assert report['epoch20_and_latest']['previously_reserved_400']['successes'] == 300
    assert sum(report['paired_outcomes']['previously_reserved_400'].values()) == 400
    markdown = reporter.markdown_report(report)
    for text in ('70.0% (350/500)', '60.0% (300/500)', '+10.0 percentage points',
                 'not a fresh holdout', 'not two independent replications',
                 '550 policy-action steps', '10 settling steps', assessments[2][1]):
        assert text in markdown


def test_accepts_different_record_order(assessments):
    _, _, _, data, _, save = assessments
    data[1]['manifest']['episodes'].reverse()
    data[1]['results'].reverse()
    save()
    assert build(assessments)['integrity']['same_task_state_and_seed_assignments']


@pytest.mark.parametrize('problem', ['incomplete', 'error', 'duplicate', 'missing', 'forged_summary'])
def test_rejects_invalid_assessment(assessments, problem):
    _, _, _, data, _, save = assessments
    item = data[1]
    if problem == 'incomplete':
        item['summary']['complete'] = False
    elif problem == 'error':
        item['results'][0]['error'] = 'simulator failure'
    elif problem == 'duplicate':
        item['results'][-1] = copy.deepcopy(item['results'][0])
    elif problem == 'missing':
        item['results'].pop()
    else:
        item['summary']['successes'] += 1
    save()
    with pytest.raises(ValueError):
        build(assessments)


def test_rejects_changed_state_even_when_each_run_is_internally_consistent(assessments):
    _, _, _, data, _, save = assessments
    data[1]['manifest']['episodes'][0]['official_state_sha256'] = 'f'*64
    data[1]['results'][0]['official_state_sha256'] = 'f'*64
    save()
    with pytest.raises(ValueError, match='state/seed assignments'):
        build(assessments)


@pytest.mark.parametrize('field', ['env_seed', 'policy_seed'])
def test_rejects_changed_seeds(assessments, field):
    _, _, _, data, _, save = assessments
    data[1]['manifest']['episodes'][0][field] += 1
    data[1]['results'][0][field] += 1
    save()
    with pytest.raises(ValueError, match='canonical'):
        build(assessments)


@pytest.mark.parametrize('field', ['max_episode_steps', 'settle_steps', 'source_sha256', 'software_versions', 'policy_config'])
def test_rejects_protocol_or_implementation_difference(assessments, field):
    _, _, _, data, _, save = assessments
    manifest = data[1]['manifest']
    if field == 'policy_config':
        manifest[field]['obs_encoder']['vision_encoder']['crop_shape'] = [100, 100]
    else:
        manifest[field] = {'changed': True} if isinstance(manifest[field], dict) else manifest[field]+1
    save()
    with pytest.raises(ValueError, match='configuration differs'):
        build(assessments)


@pytest.mark.parametrize('file', ['policy.ckpt', 'eval_500/policy.ckpt'])
def test_rejects_checkpoint_replacement(assessments, file):
    _, after, *_ = assessments
    (after/file).write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='Checkpoint hash mismatch'):
        build(assessments)


@pytest.mark.parametrize('problem', ['not_equal', 'export_hash', 'export_tensor', 'uncompared_entries', 'nonfinite', 'wrong_config'])
def test_rejects_incomplete_or_inconsistent_equivalence_evidence(assessments, problem):
    _, _, _, _, audit, save = assessments
    if problem == 'not_equal':
        audit['resolved_inference_configs_equal'] = False
    elif problem == 'export_hash':
        audit['exported_checkpoint']['file_sha256'] = 'f'*64
    elif problem == 'export_tensor':
        audit['exported_ema_comparison']['different_entries'] = ['layer.weight']
    elif problem == 'uncompared_entries':
        audit['ema_comparison']['compared_tensor_count'] -= 1
    elif problem == 'nonfinite':
        audit['ema_comparison']['nonfinite_tensor_keys'] = ['layer.weight']
    else:
        audit['resolved_policy_config']['num_inference_steps'] = 11
    save()
    with pytest.raises(ValueError):
        build(assessments)


def test_cli_refuses_incomplete_without_writing_outputs(assessments, monkeypatch):
    before, after, _, data, _, save = assessments
    data[0]['summary']['complete'] = False
    save()
    monkeypatch.setattr('sys.argv', ['report', '--baseline-dir', str(before), '--comparison-dir', str(after)])
    assert reporter.main() == 1
    assert not (after/'comparison.json').exists()
    assert not (after/'RESULTS.md').exists()
    assert not (before/'RESULTS.md').exists()
