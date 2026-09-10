"""Reject incomplete or confounded backbone comparisons without using a GPU."""
import copy
import hashlib
import json

import pytest
import yaml

from scripts import report_heading_backbone_comparison as reporter
from scripts.report_heading_goal import make_episode_plan, summarize_results


@pytest.fixture
def comparison(tmp_path):
    root = tmp_path / 'backbone_500'
    entries, hashes = [], []
    policy = {'_target_': 'oat.policy.flow_policy_heading_zero.HeadingZeroFlowPolicy',
              'heading_mode': 'condition', 'source_mode': 'heading', 'source_jitter': 'zero',
              'source_vector_blocks': [[0, 1]], 'horizon': 16, 'n_obs_steps': 2,
              'n_action_steps': 8, 'num_inference_steps': 10}
    for i, name in enumerate(reporter.RUNS):
        directory = root / name
        evaluation = directory / 'eval_500'
        evaluation.mkdir(parents=True)
        data = name.encode()
        checksum = hashlib.sha256(data).hexdigest()
        hashes.append(checksum)
        (directory / 'policy.ckpt').write_bytes(data)
        (evaluation / 'policy.ckpt').write_bytes(data)
        config = copy.deepcopy(policy)
        if i:
            config.update(backbone_type='unet', backbone_kwargs={
                'down_dims': [128, 256, 512], 'diffusion_step_embed_dim': 128, 'cond_predict_scale': True})
        plan = make_episode_plan([f'task_{j}' for j in range(10)], 50, 20260911)
        for item in plan:
            item['official_state_sha256'] = hashlib.sha256(item['episode_id'].encode()).hexdigest()
        rows = [{**item, 'success': item['init_index'] < 30 if not i else 5 <= item['init_index'] < 40,
                 'error': None} for item in plan]
        summary = summarize_results(plan, rows)
        summary.update(checkpoint_sha256=checksum, weights='ema_model')
        manifest = {'episodes': plan, 'suite': 'libero_10', 'n_per_task': 50, 'init_start': 0,
                    'seed': 20260911, 'max_episode_steps': 550, 'settle_steps': 10,
                    'protocol_version': 1, 'software_versions': {'torch': 'test'},
                    'source_sha256': {'evaluator.py': 'a' * 64}, 'weights': 'ema_model',
                    'checkpoint_sha256': checksum, 'source_checkpoint': str(directory / 'policy.ckpt'),
                    'checkpoint': str(evaluation / 'policy.ckpt'), 'policy_config': config,
                    'training_position': {'epoch': 15, 'global_step': 31023}}
        training = {key: {} for key in reporter.TRAINING}
        training.update(seed=42, horizon=16, n_action_steps=8, n_obs_steps=2,
                        policy=copy.deepcopy(config), optimizer={'policy_lr': 1e-4})
        training_path = tmp_path / name / '.hydra/config.yaml'
        training_path.parent.mkdir(parents=True)
        entries.append({'evaluation': evaluation, 'manifest': manifest, 'summary': summary,
                        'rows': rows, 'training': training, 'training_path': training_path})

    def save():
        for entry in entries:
            for filename, data in [('manifest.json', entry['manifest']), ('summary.json', entry['summary'])]:
                (entry['evaluation'] / filename).write_text(json.dumps(data))
            (entry['evaluation'] / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in entry['rows']))
            entry['training_path'].write_text(yaml.safe_dump(entry['training']))
    save()
    return root, hashes, entries, save


def build(comparison):
    root, hashes, _, _ = comparison
    return reporter.build_report(root, expected_hashes=hashes)


def test_complete_matched_comparison_recomputes_counts_and_paired_outcomes(comparison):
    report = build(comparison)
    assert report['models']['transformer_wide']['full_500']['successes'] == 300
    assert report['models']['unet_wide']['full_500']['successes'] == 350
    assert report['unet_minus_transformer'] == {'successes': 50, 'percentage_points': 10.0}
    assert report['paired_outcomes'] == {'both_succeed': 250, 'both_fail': 100,
                                        'transformer_only_succeeds': 50, 'unet_only_succeeds': 100}
    text = reporter.markdown(report)
    assert '60.0% (300/500)' in text and '70.0% (350/500)' in text
    assert 'not a capacity-matched comparison' in text


@pytest.mark.parametrize('issue', ['incomplete', 'missing', 'duplicate', 'error', 'forged_summary',
                                   'state_mismatch', 'seed_mismatch', 'wrong_epoch', 'checkpoint_tamper'])
def test_refuses_invalid_episode_or_checkpoint_evidence(comparison, issue):
    root, _, entries, save = comparison
    item = entries[1]
    if issue == 'incomplete': item['summary']['complete'] = False
    elif issue == 'missing': item['rows'].pop()
    elif issue == 'duplicate': item['rows'][1] = copy.deepcopy(item['rows'][0])
    elif issue == 'error': item['rows'][0]['error'] = 'failed rollout'
    elif issue == 'forged_summary': item['summary']['successes'] += 1
    elif issue == 'state_mismatch': item['rows'][0]['official_state_sha256'] = 'b' * 64
    elif issue == 'seed_mismatch': item['manifest']['seed'] += 1
    elif issue == 'wrong_epoch': item['manifest']['training_position']['epoch'] = 20
    elif issue == 'checkpoint_tamper': (root / 'unet_wide/policy.ckpt').write_bytes(b'changed')
    save()
    with pytest.raises(ValueError): build(comparison)


@pytest.mark.parametrize('issue', ['crop', 'prior', 'training_lr', 'horizon', 'settling', 'code', 'software'])
def test_refuses_changes_beyond_backbone(comparison, issue):
    _, _, entries, save = comparison
    item = entries[1]
    if issue == 'crop': item['manifest']['policy_config']['crop_shape'] = [76, 76]
    elif issue == 'prior': item['manifest']['policy_config']['source_jitter'] = 'vonmises'
    elif issue == 'training_lr': item['training']['optimizer']['policy_lr'] = 1e-3
    elif issue == 'horizon': item['manifest']['max_episode_steps'] = 600
    elif issue == 'settling': item['manifest']['settle_steps'] = 5
    elif issue == 'code': item['manifest']['source_sha256']['evaluator.py'] = 'c' * 64
    elif issue == 'software': item['manifest']['software_versions']['torch'] = 'changed'
    save()
    with pytest.raises(ValueError): build(comparison)


def test_rejects_gaussian_prior_even_when_both_policies_match(comparison):
    _, _, entries, save = comparison
    for item in entries:
        item['manifest']['policy_config']['_target_'] = 'oat.policy.flow_policy_heading_gaussian.HeadingGaussianFlowPolicy'
    save()
    with pytest.raises(ValueError, match='exact-heading'): build(comparison)
