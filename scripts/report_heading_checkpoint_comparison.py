#!/usr/bin/env python3
"""Validate paired epoch-15 versus epoch-20/latest LIBERO-10 assessments.

Requires both complete 500-episode runs and checkpoint_equivalence.json. Never
launches rollouts or modifies the original final assessment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from scripts.report_heading_goal import (
    EXPECTED_SHA256, IDENTITY_KEYS, file_sha256, require, validate_method, validate_results,
)

EPOCH20_SHA256 = '4e759448ead056c04162e77f426487ba0a4b5fd3f7beb47d0d41d2eddab2efc8'
PROTOCOL_KEYS = ('suite', 'n_per_task', 'init_start', 'seed', 'max_episode_steps', 'settle_steps')


def load_assessment(directory, expected_hash, expected_epoch):
    directory = Path(directory).resolve()
    evaluation = directory / 'eval_500'
    paths = {name: evaluation / name for name in ('manifest.json', 'summary.json', 'episodes.jsonl')}
    content = {name: path.read_bytes() for name, path in paths.items()}
    manifest, summary = [json.loads(content[name]) for name in ('manifest.json', 'summary.json')]
    results = [json.loads(line) for line in content['episodes.jsonl'].splitlines() if line.strip()]
    names, full, subset = validate_results(manifest, results, summary)
    validate_method(manifest['policy_config'])
    require(manifest.get('weights') == 'ema_model', 'Comparison requires EMA weights')
    require(manifest.get('training_position', {}).get('epoch') == expected_epoch,
            f'Expected training epoch {expected_epoch}')
    for key in ('max_episode_steps', 'settle_steps'):
        require(type(manifest.get(key)) is int and manifest[key] >= (1 if key == 'max_episode_steps' else 0),
                f'Missing or invalid {key}')
    selected, snapshot = directory / 'policy.ckpt', evaluation / 'policy.ckpt'
    require(Path(manifest['source_checkpoint']).resolve() == selected, 'Selected checkpoint path mismatch')
    require(Path(manifest['checkpoint']).resolve() == snapshot, 'Evaluation snapshot path mismatch')
    require(manifest['checkpoint_sha256'] == expected_hash, 'Manifest checkpoint hash mismatch')
    for path in (selected, snapshot):
        require(file_sha256(path) == expected_hash, f'Checkpoint hash mismatch: {path}')
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    for name, path in paths.items():
        require(file_sha256(path) == hashes[name], f'Evaluation records changed while reporting: {path}')
    return {
        'manifest': manifest, 'results': results, 'task_names': names,
        'checkpoint': str(selected), 'checkpoint_sha256': expected_hash,
        'training_position': manifest['training_position'], 'weights': manifest['weights'],
        'evaluation': str(evaluation), 'input_sha256': hashes,
        'full_500': full, 'previously_reserved_400': subset,
    }


def validate_matching_protocol(before, after):
    left, right = before['manifest'], after['manifest']
    for key in (*PROTOCOL_KEYS, 'policy_config', 'protocol_version', 'software_versions', 'source_sha256'):
        require(key in left and key in right and left[key] == right[key], f'Evaluation configuration differs or is missing: {key}')
    require(isinstance(left['source_sha256'], dict) and left['source_sha256'], 'Missing evaluation source hashes')
    def identities(manifest):
        return {episode['episode_id']: {key: episode[key] for key in (*IDENTITY_KEYS, 'official_state_sha256')}
                for episode in manifest['episodes']}
    require(identities(left) == identities(right), 'Evaluations use different task/state/seed assignments')


def validate_tensor_comparison(comparison, label):
    require(comparison.get('exactly_equal') is True, f'{label}: EMA weights are not exactly equal')
    count = comparison.get('compared_tensor_count')
    require(type(count) is int and count > 0, f'{label}: no tensors were compared')
    require(comparison.get('left_state_entry_count') == count and comparison.get('right_state_entry_count') == count,
            f'{label}: not every state entry was compared')
    for key in ('left_only_keys', 'right_only_keys', 'different_entries', 'nonfinite_tensor_keys'):
        require(comparison.get(key) == [], f'{label}: {key} must be empty')
    require(comparison.get('max_absolute_tensor_difference') == 0, f'{label}: tensor differences are nonzero')


def validate_equivalence(path, assessment):
    content = path.read_bytes()
    audit = json.loads(content)
    manifest = assessment['manifest']
    for key in ('resolved_policy_configs_equal', 'resolved_inference_configs_equal',
                'training_positions_equal', 'one_full_evaluation_covers_both'):
        require(audit.get(key) is True, f'Checkpoint equivalence audit failed: {key}')
    checkpoints = audit['checkpoints']
    require(len(checkpoints) == 2, 'Equivalence audit must identify epoch-20 and latest checkpoints')
    require(any(Path(item['path']).name == 'latest.ckpt' for item in checkpoints), 'Audit does not identify latest.ckpt')
    require(len({item['path'] for item in checkpoints}) == 2, 'Audit checkpoint paths are duplicated')
    for item in checkpoints:
        require(item.get('training_position') == manifest['training_position'], 'Audit training position differs from evaluated checkpoint')
        require(item.get('selected_state') == 'ema_model' and item.get('use_ema') is True, 'Audit did not select EMA weights')
        require(re.fullmatch(r'[0-9a-f]{64}', item.get('file_sha256', '')) is not None, 'Missing audited serialization hash')
    validate_tensor_comparison(audit['ema_comparison'], 'Epoch-20/latest audit')
    require(audit.get('resolved_policy_config') == manifest['policy_config'], 'Audited policy configuration differs from evaluation')
    require(audit.get('evaluation_protocol_to_share') == {key: manifest[key] for key in PROTOCOL_KEYS},
            'Audited evaluation protocol differs from evaluation')
    exported = audit['exported_checkpoint']
    require(Path(exported['path']).resolve() == Path(assessment['checkpoint']), 'Audited export path mismatch')
    require(exported.get('file_sha256') == assessment['checkpoint_sha256'], 'Audited export hash mismatch')
    require(exported.get('training_position') == manifest['training_position'], 'Audited export training position mismatch')
    require(exported.get('selected_state') == 'ema_model', 'Audited export did not select EMA weights')
    require(audit.get('exported_resolved_policy_config_equal') is True, 'Exported policy configuration was not verified')
    validate_tensor_comparison(audit['exported_ema_comparison'], 'Exported EMA audit')
    checksum = hashlib.sha256(content).hexdigest()
    require(file_sha256(path) == checksum, 'Equivalence audit changed while reporting')
    return {'path': str(path), 'sha256': checksum, 'evidence': audit}


def paired_counts(before, after, initial_state_start=0):
    left = {row['episode_id']: row for row in before if row['init_index'] >= initial_state_start}
    right = {row['episode_id']: row for row in after if row['init_index'] >= initial_state_start}
    require(left.keys() == right.keys(), 'Paired episodes differ')
    counts = {'both_succeed': 0, 'both_fail': 0, 'epoch15_only_succeeds': 0, 'epoch20_only_succeeds': 0}
    for identity, row in left.items():
        a, b = row['success'], right[identity]['success']
        key = ('both_succeed' if a else 'both_fail') if a == b else ('epoch15_only_succeeds' if a else 'epoch20_only_succeeds')
        counts[key] += 1
    return counts


def build_comparison(baseline_dir, comparison_dir, baseline_hash=EXPECTED_SHA256, comparison_hash=EPOCH20_SHA256):
    before = load_assessment(baseline_dir, baseline_hash, 15)
    after = load_assessment(comparison_dir, comparison_hash, 20)
    validate_matching_protocol(before, after)
    audit = validate_equivalence(Path(comparison_dir).resolve() / 'checkpoint_equivalence.json', after)
    deltas = {}
    for key in ('full_500', 'previously_reserved_400'):
        a, b = before[key], after[key]
        deltas[key] = {
            'successes': b['successes'] - a['successes'],
            'percentage_points': 100 * (b['mean_success_rate'] - a['mean_success_rate']),
            'per_task': {task: {'successes': b['per_task'][task]['successes'] - a['per_task'][task]['successes'],
                               'percentage_points': 100 * (b['per_task'][task]['success_rate'] - a['per_task'][task]['success_rate'])}
                         for task in before['task_names']},
        }
    manifest = after['manifest']
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'epoch15': {key: value for key, value in before.items() if key not in ('manifest', 'results')},
        'epoch20_and_latest': {key: value for key, value in after.items() if key not in ('manifest', 'results')},
        'delta_epoch20_minus_epoch15': deltas,
        'paired_outcomes': {key: paired_counts(before['results'], after['results'], start)
                            for key, start in (('full_500', 0), ('previously_reserved_400', 10))},
        'checkpoint_equivalence': audit,
        'evaluation_protocol': {key: manifest[key] for key in PROTOCOL_KEYS},
        'policy_config': manifest['policy_config'],
        'previously_reserved_initial_state_indices': [10, 49],
        'subset_interpretation': 'The 400 episodes at indices 10–49 were previously reserved for the epoch-15 assessment. '
                                 'They are now also used in this later-checkpoint comparison; this is not a fresh holdout.',
        'integrity': {'same_task_state_and_seed_assignments': True, 'same_policy_configuration': True,
                      'same_evaluation_source_hashes_and_software': True, 'recomputed_summaries_match': True,
                      'error_episodes': 0, 'missing_episodes': 0, 'unique_initial_states_per_task': 50},
    }


def markdown_report(report):
    before, after = report['epoch15'], report['epoch20_and_latest']
    def score(summary):
        return f"{summary['mean_success_rate']:.1%} ({summary['successes']}/{summary['completed_episodes']})"
    delta = report['delta_epoch20_minus_epoch15']
    lines = [
        f"Epoch 20 / latest: **{score(after['full_500'])}** on LIBERO-10, versus **{score(before['full_500'])}** at epoch 15 "
        f"(**{delta['full_500']['percentage_points']:+.1f} percentage points**, {delta['full_500']['successes']:+d} successes).", '',
        f"Previously reserved 400-episode subset: **{score(after['previously_reserved_400'])}**, versus "
        f"**{score(before['previously_reserved_400'])}** at epoch 15 "
        f"({delta['previously_reserved_400']['percentage_points']:+.1f} percentage points).", '',
        report['subset_interpretation'], '',
        '| Task | Epoch 15 (50/task) | Epoch 20 / latest (50/task) | Change |', '|---|---:|---:|---:|',
    ]
    for task in before['task_names']:
        def task_score(assessment):
            item = assessment['full_500']['per_task'][task]
            return f"{item['success_rate']:.1%} ({item['successes']}/{item['completed']})"
        lines.append(f"| {task.replace('_', ' ')} | {task_score(before)} | {task_score(after)} | "
                     f"{delta['full_500']['per_task'][task]['percentage_points']:+.1f} pp |")
    protocol, config = report['evaluation_protocol'], report['policy_config']
    pair = report['paired_outcomes']['full_500']
    evidence = report['checkpoint_equivalence']['evidence']
    hashes_differ = len({item['file_sha256'] for item in evidence['checkpoints']}) > 1
    lines += ['', f"Epoch-20 checkpoint: [policy.ckpt](policy.ckpt), EMA weights, training position `{after['training_position']}`.", '',
              f"SHA-256: `{after['checkpoint_sha256']}`", '',
              f"Epoch-15 checkpoint: [{Path(before['checkpoint']).name}]({before['checkpoint']}), SHA-256 `{before['checkpoint_sha256']}`.", '',
              'The [checkpoint audit](checkpoint_equivalence.json) verifies that the epoch-20 and latest files contain exactly equal EMA tensors and resolved configurations, '
              'and verifies the exported policy against those tensors. '
              + ('Their serialized-file hashes differ. ' if hashes_differ else '')
              + 'One evaluation covers both checkpoint names; it is not two independent replications.', '',
              f"Both assessments use the same 500 task/initial-state/environment-seed/policy-seed assignments (seed {protocol['seed']}), "
              f"with **{protocol['max_episode_steps']} policy-action steps** after **{protocol['settle_steps']} settling steps**. "
              f"The policy uses {config['n_obs_steps']} observations, predicts {config['horizon']} actions, executes {config['n_action_steps']}, "
              f"and uses {config['num_inference_steps']} Euler steps. Image transforms, complete policy configuration, evaluator source hashes, "
              'and recorded software versions match. Each assessment uses one fixed checkpoint for all tasks.', '',
              f"Paired outcomes: both succeed {pair['both_succeed']}; both fail {pair['both_fail']}; "
              f"epoch 15 alone succeeds {pair['epoch15_only_succeeds']}; epoch 20 alone succeeds {pair['epoch20_only_succeeds']}.", '',
              'Counts were recomputed from complete episode records with zero errors. The JSON report also includes per-task subset results and paired subset outcomes. '
              'This is a descriptive checkpoint comparison on LIBERO-10; it does not establish cross-benchmark generalization or a fresh holdout result.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT_DIR / 'output/heading_goal/final')
    parser.add_argument('--comparison-dir', type=Path, default=ROOT_DIR / 'output/heading_goal/epoch20_latest')
    args = parser.parse_args()
    try:
        require(args.baseline_dir.resolve() != args.comparison_dir.resolve(), 'Comparison output must differ from the original final directory')
        report = build_comparison(args.baseline_dir, args.comparison_dir)
        for name, content in {'comparison.json': json.dumps(report, indent=2, sort_keys=True) + '\n',
                              'RESULTS.md': markdown_report(report)}.items():
            path = args.comparison_dir / name
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(content)
            temporary.replace(path)
        print(f"Validated comparison: {args.comparison_dir / 'RESULTS.md'}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Comparison refused: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
