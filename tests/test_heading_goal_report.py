"""Integrity checks on synthetic completed assessments; no policies or GPUs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'heading_report', Path(__file__).resolve().parents[1] / 'scripts/report_heading_goal.py')
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


@pytest.fixture
def assessment(tmp_path):
    evaluation = tmp_path/'eval_500'
    evaluation.mkdir()
    checkpoint_bytes = b'one fixed synthetic checkpoint for integrity tests'
    checksum = hashlib.sha256(checkpoint_bytes).hexdigest()
    for path in (tmp_path/'policy.ckpt', evaluation/'policy.ckpt'):
        path.write_bytes(checkpoint_bytes)
    plan = reporter.make_episode_plan([f'task_{index}' for index in range(10)], 50, 20260911)
    for episode in plan:
        episode['official_state_sha256'] = hashlib.sha256(episode['episode_id'].encode()).hexdigest()
    results = [{**episode, 'success': episode['init_index'] < 40, 'error': None} for episode in plan]
    summary = reporter.summarize_results(plan, results)
    summary.update(checkpoint_sha256=checksum, weights='ema_model')
    manifest = {
        'suite': 'libero_10', 'n_per_task': 50, 'init_start': 0, 'seed': 20260911,
        'episodes': plan, 'checkpoint_sha256': checksum, 'weights': 'ema_model',
        'max_episode_steps': 550, 'settle_steps': 10, 'workers': 8, 'cuda_visible_devices': '0',
        'training_position': {'epoch': 15, 'global_step': 123},
        'source_checkpoint': str(tmp_path/'policy.ckpt'), 'checkpoint': str(evaluation/'policy.ckpt'),
        'policy_config': {
            '_target_': 'oat.policy.flow_policy_heading_gaussian.HeadingGaussianFlowPolicy',
            'backbone_type': 'unet', 'heading_mode': 'condition', 'source_mode': 'heading',
            'horizon': 16, 'n_action_steps': 8, 'n_obs_steps': 2, 'num_inference_steps': 10,
            'prior_parallel_std': 1., 'prior_perpendicular_std': .5,
            'obs_encoder': {'vision_encoder': {'_target_': 'oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder',
                            'crop_shape': [116, 116], 'eval_fixed_crop': True}},
            'shape_meta': {'obs': {name: {} for name in ('agentview_rgb', 'robot0_eye_in_hand_rgb',
                'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'task_uid')}},
        },
    }
    for key in ('agentview_rgb', 'robot0_eye_in_hand_rgb'):
        manifest['policy_config']['shape_meta']['obs'][key] = {'type': 'rgb', 'shape': [128, 128, 3]}
    def save():
        (evaluation/'manifest.json').write_text(json.dumps(manifest))
        (evaluation/'summary.json').write_text(json.dumps(summary))
        (evaluation/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in results))
    save()
    return tmp_path, checksum, manifest, results, summary, save


def test_full_and_reserved_counts_come_from_same_complete_run(assessment):
    root, checksum, _, _, _, _ = assessment
    report = reporter.build_report(root, checksum)
    assert report['full_500']['successes'] == 400
    assert report['reserved_400']['successes'] == 300
    assert report['reserved_400']['completed_episodes'] == 400
    assert report['full_500']['mean_success_rate'] == pytest.approx(.8)
    assert report['reserved_400']['mean_success_rate'] == pytest.approx(.75)
    assert all(task['completed'] == 40 for task in report['reserved_400']['per_task'].values())
    markdown = reporter.markdown_report(report)
    assert '80.0% (400/500)' in markdown and '75.0% (300/400)' in markdown
    assert checksum in markdown and 'do not establish Gaussian superiority' in markdown
    assert '550 policy-action steps' in markdown and '10 settling steps' in markdown
    assert '--max-episode-steps 550 --settle-steps 10' in report['reproduction']['command']
    assert report['evaluation_protocol']['action_horizon'] == 16
    assert report['evaluation_protocol']['observation_steps'] == 2
    assert report['image_transforms']['crop_shape'] == [116, 116]
    assert report['gaussian_prior']['perpendicular_std'] == .5
    assert 'load_policy(' in markdown and 'fixed 116×116 center crop' in markdown


@pytest.mark.parametrize('field', ['env_seed', 'policy_seed', 'init_index', 'task_index', 'official_state_sha256'])
def test_rejects_result_identity_changes_even_when_success_counts_match(assessment, field):
    root, checksum, _, results, _, save = assessment
    results[3][field] = '0'*64 if field == 'official_state_sha256' else results[3][field] + 1
    save()
    with pytest.raises(ValueError, match='mismatch'):
        reporter.build_report(root, checksum)


def test_rejects_manifest_seed_tampering(assessment):
    root, checksum, manifest, _, _, save = assessment
    manifest['episodes'][0]['policy_seed'] += 1
    save()
    with pytest.raises(ValueError, match='canonical'):
        reporter.build_report(root, checksum)


@pytest.mark.parametrize('problem', ['incomplete', 'missing', 'duplicate', 'error', 'forged_summary'])
def test_refuses_incomplete_or_forged_assessments(assessment, problem):
    root, checksum, _, results, summary, save = assessment
    if problem == 'incomplete':
        summary['complete'] = False
    elif problem == 'missing':
        results.pop()
    elif problem == 'duplicate':
        results[-1] = copy.deepcopy(results[0])
    elif problem == 'error':
        results[0]['error'] = 'simulator failed'
    else:
        summary['successes'] += 1
    save()
    with pytest.raises(ValueError):
        reporter.build_report(root, checksum)


@pytest.mark.parametrize('checkpoint', ['policy.ckpt', 'eval_500/policy.ckpt'])
def test_rejects_changed_selected_or_evaluated_checkpoint(assessment, checkpoint):
    root, checksum, _, _, _, _ = assessment
    (root/checkpoint).write_bytes(b'wrong checkpoint')
    with pytest.raises(ValueError, match='checkpoint hash mismatch'):
        reporter.build_report(root, checksum)


def test_default_checksum_is_pinned_to_selected_export(assessment):
    root, _, _, _, _, _ = assessment
    with pytest.raises(ValueError, match='checkpoint hash mismatch'):
        reporter.build_report(root)


def test_refuses_missing_official_initial_state(assessment):
    root, checksum, manifest, _, _, save = assessment
    manifest['episodes'][-1]['init_index'] = 50
    save()
    with pytest.raises(ValueError, match='canonical'):
        reporter.build_report(root, checksum)


@pytest.mark.parametrize('field', ['max_episode_steps', 'settle_steps'])
def test_refuses_missing_episode_horizon_metadata(assessment, field):
    root, checksum, manifest, _, _, save = assessment
    del manifest[field]
    save()
    with pytest.raises(ValueError, match=field):
        reporter.build_report(root, checksum)
