"""Synthetic integrity tests for the three-way matched-backbone report."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import report_heading_starvla_comparison as reporter
from scripts.report_heading_goal import make_episode_plan, summarize_results


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    baseline, starvla = tmp_path/'backbone_500', tmp_path/'starvla_dit_500'
    entries, hashes = [], []
    policy = {'_target_': 'oat.policy.flow_policy_heading_zero.HeadingZeroFlowPolicy',
              'heading_mode': 'condition', 'source_mode': 'heading', 'source_jitter': 'zero',
              'source_vector_blocks': [[0, 1]], 'horizon': 16, 'n_obs_steps': 2,
              'n_action_steps': 8, 'num_inference_steps': 10}
    for index, name in enumerate(reporter.RUNS):
        directory = baseline/name if index < 2 else starvla
        evaluation = directory/'eval_500'
        evaluation.mkdir(parents=True)
        config = copy.deepcopy(policy)
        if index == 1:
            config.update(backbone_type='unet', backbone_kwargs={'down_dims': [128, 256, 512],
                          'diffusion_step_embed_dim': 128, 'cond_predict_scale': True})
        elif index == 2:
            config.update(backbone_type='starvla_dit', backbone_kwargs={'num_register_tokens': 32})
        metadata = {'policy_config': config, 'training_position': dict(reporter.POSITION),
                    'weights': 'ema_model', 'training_seed': 42, 'export_metadata': None}
        source = tmp_path/name/'source.ckpt'
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps(metadata))
        provenance = {'format': 'oat_inference_policy_v1', 'inference_only': True, 'weights': 'ema_model',
                      'source_checkpoint': str(source), 'source_checkpoint_sha256': reporter.file_sha256(source)}
        exported = {**metadata, 'export_metadata': copy.deepcopy(provenance)}
        payload = json.dumps(exported).encode()
        checksum = hashlib.sha256(payload).hexdigest()
        hashes.append(checksum)
        for path in (directory/'policy.ckpt', evaluation/'policy.ckpt'):
            path.write_bytes(payload)
        provenance.update(checkpoint=str(directory/'policy.ckpt'), checkpoint_sha256=checksum,
                          bytes=len(payload), exact_state_dict_verified=True)
        plan = make_episode_plan([f'task_{i}' for i in range(10)], 50, 20260911)
        for row in plan:
            row['official_state_sha256'] = hashlib.sha256(row['episode_id'].encode()).hexdigest()
        results = [{**row, 'success': row['init_index'] < 30 + 5*index, 'error': None} for row in plan]
        summary = summarize_results(plan, results)
        summary.update(checkpoint_sha256=checksum, weights='ema_model')
        manifest = {'episodes': plan, 'suite': 'libero_10', 'n_per_task': 50, 'init_start': 0,
                    'seed': 20260911, 'max_episode_steps': 550, 'settle_steps': 10,
                    'protocol_version': 1, 'software_versions': {'torch': 'test'},
                    'source_sha256': {'evaluator.py': 'a'*64}, 'weights': 'ema_model',
                    'checkpoint_sha256': checksum, 'source_checkpoint': str(directory/'policy.ckpt'),
                    'checkpoint': str(evaluation/'policy.ckpt'), 'policy_config': config,
                    'training_position': dict(reporter.POSITION)}
        training = {key: {} for key in reporter.TRAINING}
        training.update(seed=42, policy=copy.deepcopy(config), optimizer={'policy_lr': 1e-4},
                        horizon=16, n_obs_steps=2, n_action_steps=8)
        training_path = tmp_path/name/'.hydra/config.yaml'
        training_path.parent.mkdir()
        entries.append({'directory': directory, 'evaluation': evaluation, 'manifest': manifest,
                        'summary': summary, 'rows': results, 'training': training,
                        'training_path': training_path, 'provenance': provenance, 'source': source})
    def save():
        for item in entries:
            for name, value in [('manifest.json', item['manifest']), ('summary.json', item['summary'])]:
                (item['evaluation']/name).write_text(json.dumps(value))
            (item['evaluation']/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in item['rows']))
            (item['directory']/'policy.ckpt.json').write_text(json.dumps(item['provenance']))
            item['training_path'].write_text(yaml.safe_dump(item['training']))
    monkeypatch.setattr(reporter, 'checkpoint_metadata', lambda path: json.loads(Path(path).read_text()))
    save()
    return baseline, starvla, tmp_path, hashes, entries, save


def build(comparison):
    baseline, starvla, training, hashes, *_ = comparison
    return reporter.build_report(baseline, starvla, training, hashes[:2])


def test_three_way_counts_deltas_and_provenance(comparison):
    report = build(comparison)
    assert [report['models'][name]['full_500']['successes'] for name in reporter.RUNS] == [300, 350, 400]
    assert report['starvla_minus_baseline']['transformer_wide']['percentage_points'] == 20
    assert report['starvla_minus_baseline']['unet_wide']['percentage_points'] == 10
    assert report['paired_outcomes']['unet_wide'] == {'both_succeed': 350, 'both_fail': 100,
                                                    'baseline_only_succeeds': 0, 'starvla_only_succeeds': 50}
    assert report['optimizer_updates'] == 31024
    assert report['export_provenance']['starvla_dit_wide']['source_and_export_metadata_verified']
    markdown = reporter.markdown(report)
    for text in ('StarVLA-DiT: 80.0% (400/500)', 'Transformer: 60.0% (300/500)',
                 'U-Net: 70.0% (350/500)', '+20.0 percentage points', '32 register tokens',
                 '550 policy-action steps', '10 settling steps', 'not a fresh holdout', comparison[3][2]):
        assert text in markdown


@pytest.mark.parametrize('issue', ['incomplete', 'error', 'missing', 'duplicate', 'forged_summary', 'wrong_epoch', 'wrong_updates'])
def test_rejects_incomplete_or_invalid_assessments(comparison, issue):
    *_, entries, save = comparison
    item = entries[2]
    if issue == 'incomplete': item['summary']['complete'] = False
    elif issue == 'error': item['rows'][0]['error'] = 'rollout failed'
    elif issue == 'missing': item['rows'].pop()
    elif issue == 'duplicate': item['rows'][-1] = copy.deepcopy(item['rows'][0])
    elif issue == 'forged_summary': item['summary']['successes'] += 1
    elif issue == 'wrong_epoch': item['manifest']['training_position']['epoch'] = 20
    else: item['manifest']['training_position']['global_step'] += 1
    save()
    with pytest.raises(ValueError): build(comparison)


@pytest.mark.parametrize('field', ['official_state_sha256', 'env_seed', 'policy_seed'])
def test_rejects_changed_episode_assignments(comparison, field):
    *_, entries, save = comparison
    for row in (entries[2]['rows'][0], entries[2]['manifest']['episodes'][0]):
        row[field] = 'b'*64 if field == 'official_state_sha256' else row[field]+1
    save()
    with pytest.raises(ValueError): build(comparison)


@pytest.mark.parametrize('issue', ['prior', 'registers', 'crop', 'horizon', 'settling', 'code', 'software', 'lr', 'seed', 'saved_policy'])
def test_rejects_unmatched_recipe_or_protocol(comparison, issue):
    *_, entries, save = comparison
    item = entries[2]
    if issue == 'prior': item['manifest']['policy_config']['source_jitter'] = 'vonmises'
    elif issue == 'registers': item['manifest']['policy_config']['backbone_kwargs']['num_register_tokens'] = 16
    elif issue == 'crop': item['manifest']['policy_config']['crop_shape'] = [76, 76]
    elif issue == 'horizon': item['manifest']['max_episode_steps'] = 600
    elif issue == 'settling': item['manifest']['settle_steps'] = 5
    elif issue == 'code': item['manifest']['source_sha256']['evaluator.py'] = 'b'*64
    elif issue == 'software': item['manifest']['software_versions']['torch'] = 'changed'
    elif issue == 'lr': item['training']['optimizer']['policy_lr'] = 1e-3
    elif issue == 'seed': item['training']['seed'] = 43
    else:
        # Matching edited recipes still must agree with the actual evaluation policy.
        for entry in entries: entry['training']['policy']['horizon'] = 17
    save()
    with pytest.raises(ValueError): build(comparison)


@pytest.mark.parametrize('issue', ['source_hash', 'export_hash', 'snapshot_hash', 'sidecar_path', 'unverified_tensors', 'byte_count', 'embedded_metadata'])
def test_rejects_broken_export_chain(comparison, issue):
    *_, entries, save = comparison
    item = entries[2]
    if issue == 'source_hash': item['source'].write_bytes(b'changed source')
    elif issue == 'export_hash': (item['directory']/'policy.ckpt').write_bytes(b'changed export')
    elif issue == 'snapshot_hash': (item['evaluation']/'policy.ckpt').write_bytes(b'changed snapshot')
    elif issue == 'sidecar_path': item['provenance']['checkpoint'] = '/wrong/policy.ckpt'
    elif issue == 'unverified_tensors': item['provenance']['exact_state_dict_verified'] = False
    elif issue == 'byte_count': item['provenance']['bytes'] += 1
    else:
        # Sidecar alone cannot redirect the source to another byte-identical file.
        alternate = item['source'].with_name('alternate.ckpt')
        alternate.write_bytes(item['source'].read_bytes())
        item['provenance']['source_checkpoint'] = str(alternate)
    save()
    with pytest.raises(ValueError): build(comparison)


def test_cli_does_not_write_partial_or_original_reports(comparison, monkeypatch):
    baseline, starvla, training, _, entries, save = comparison
    entries[2]['summary']['complete'] = False
    save()
    monkeypatch.setattr('sys.argv', ['report', '--baseline-root', str(baseline), '--starvla-dir', str(starvla),
                                    '--training-root', str(training)])
    assert reporter.main() == 1
    assert not (starvla/'comparison.json').exists() and not (starvla/'RESULTS.md').exists()
    assert not (baseline/'RESULTS.md').exists()


def test_real_cpu_checkpoint_metadata_without_policy_construction(tmp_path):
    import dill
    import torch
    from omegaconf import OmegaConf
    path = tmp_path/'minimal.ckpt'
    torch.save({'cfg': OmegaConf.create({'seed': 42, 'training': {'use_ema': True}, 'policy': {'horizon': 16}}),
                'state_dicts': {'ema_model': {'weight': torch.ones(1)}},
                'pickles': {key: dill.dumps(value) for key, value in reporter.POSITION.items()}}, path, pickle_module=dill)
    metadata = reporter.checkpoint_metadata(path)
    assert metadata['training_position'] == reporter.POSITION
    assert metadata['policy_config'] == {'horizon': 16}
    assert metadata['weights'] == 'ema_model' and metadata['training_seed'] == 42
