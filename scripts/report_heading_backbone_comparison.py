#!/usr/bin/env python3
"""Validate the matched epoch-15, exact-heading-prior backbone comparison."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_heading_goal import IDENTITY_KEYS, file_sha256, require, validate_results

RUNS = ('transformer_wide', 'unet_wide')
HASHES = (
    'e466bab9001f5078ccd8957416995a7562fb7332e85658a886365192d94870bb',
    '9c6950d847fc810f4d1fab0040a70ec79f3bbe8e8615018a3915d54bfbf8ef6f',
)
PROTOCOL = ('suite', 'n_per_task', 'init_start', 'seed', 'max_episode_steps',
            'settle_steps', 'protocol_version', 'software_versions', 'source_sha256')
TRAINING = ('seed', 'training', 'optimizer', 'dataloader', 'val_dataloader', 'ema',
            'task', 'horizon', 'n_action_steps', 'n_obs_steps')


def validate_backbones(left, right):
    left, right = copy.deepcopy(left), copy.deepcopy(right)
    require(left.pop('backbone_type', 'transformer') == 'transformer', 'Expected Transformer baseline')
    require(right.pop('backbone_type', None) == 'unet', 'Expected U-Net comparison')
    require(not left.pop('backbone_kwargs', None), 'Unexpected Transformer backbone options')
    options = right.pop('backbone_kwargs', None)
    require(options == {'down_dims': [128, 256, 512], 'diffusion_step_embed_dim': 128,
                        'cond_predict_scale': True}, 'Unexpected U-Net configuration')
    require(left == right, 'Policy settings differ beyond the backbone')
    required = {'_target_': 'oat.policy.flow_policy_heading_zero.HeadingZeroFlowPolicy',
                'heading_mode': 'condition', 'source_mode': 'heading', 'source_jitter': 'zero',
                'source_vector_blocks': [[0, 1]]}
    for key, value in required.items():
        require(left.get(key) == value, f'Expected shared exact-heading policy: {key}')
    return left, options


def read_run(directory, expected_hash):
    directory = Path(directory).resolve()
    evaluation = directory / 'eval_500'
    names = ('manifest.json', 'summary.json', 'episodes.jsonl')
    contents = {name: (evaluation / name).read_bytes() for name in names}
    manifest = json.loads(contents['manifest.json'])
    summary = json.loads(contents['summary.json'])
    rows = [json.loads(line) for line in contents['episodes.jsonl'].splitlines() if line.strip()]
    tasks, full, subset = validate_results(manifest, rows, summary)
    require(manifest.get('training_position') == {'epoch': 15, 'global_step': 31023},
            'Expected matched epoch 15 / 31,024 updates')
    require(manifest.get('weights') == 'ema_model', 'Expected EMA policy')
    require(manifest['checkpoint_sha256'] == expected_hash, 'Manifest checkpoint hash mismatch')
    for key, path in (('source_checkpoint', directory / 'policy.ckpt'),
                      ('checkpoint', evaluation / 'policy.ckpt')):
        require(Path(manifest[key]).resolve() == path, 'Checkpoint path mismatch')
        require(file_sha256(path) == expected_hash, 'Checkpoint bytes differ from expected export')
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    for name in names:
        require(file_sha256(evaluation / name) == hashes[name], 'Evaluation records changed while reporting')
    return {'manifest': manifest, 'rows': rows, 'task_names': tasks, 'full_500': full,
            'indices_10_49': subset, 'checkpoint': str(directory / 'policy.ckpt'),
            'checkpoint_sha256': expected_hash, 'input_sha256': hashes,
            'video_errors': sum(bool(row.get('video_error')) for row in rows)}


def build_report(root, training_root=None, expected_hashes=HASHES):
    root = Path(root).resolve()
    training_root = Path(training_root).resolve() if training_root else root.parent
    runs = [read_run(root / name, checksum) for name, checksum in zip(RUNS, expected_hashes)]
    left, right = [run['manifest'] for run in runs]
    common_policy, unet = validate_backbones(left['policy_config'], right['policy_config'])
    for key in PROTOCOL:
        require(key in left and key in right and left[key] == right[key], f'Protocol mismatch: {key}')
    require(left['max_episode_steps'] == 550 and left['settle_steps'] == 10
            and left['seed'] == 20260911, 'Unexpected planned evaluation limits or seed')
    require(left['source_sha256'] and left['software_versions'], 'Missing code/software provenance')
    identities = lambda m: {e['episode_id']: tuple(e[k] for k in (*IDENTITY_KEYS, 'official_state_sha256'))
                            for e in m['episodes']}
    require(identities(left) == identities(right), 'Episode task/state/seed assignments differ')
    configs, config_evidence = [], []
    for name in RUNS:
        path = training_root / name / '.hydra/config.yaml'
        raw = path.read_bytes()
        configs.append(yaml.safe_load(raw))
        config_evidence.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
    for key in TRAINING:
        require(key in configs[0] and key in configs[1] and configs[0][key] == configs[1][key],
                f'Training recipe differs: {key}')
    validate_backbones(configs[0]['policy'], configs[1]['policy'])
    pairs = {row['episode_id']: row for row in runs[1]['rows']}
    paired = {'both_succeed': 0, 'both_fail': 0, 'transformer_only_succeeds': 0, 'unet_only_succeeds': 0}
    for row in runs[0]['rows']:
        a, b = row['success'], pairs[row['episode_id']]['success']
        key = ('both_succeed' if a else 'both_fail') if a == b else ('transformer_only_succeeds' if a else 'unet_only_succeeds')
        paired[key] += 1
    delta = runs[1]['full_500']['successes'] - runs[0]['full_500']['successes']
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'models': {name: {k: v for k, v in run.items() if k not in ('manifest', 'rows')}
                       for name, run in zip(RUNS, runs)},
            'unet_minus_transformer': {'successes': delta, 'percentage_points': delta / 5.0},
            'paired_outcomes': paired, 'common_policy_config': common_policy,
            'unet_backbone_options': unet, 'training_configs': config_evidence,
            'common_training_recipe': {key: configs[0][key] for key in TRAINING},
            'evaluation_protocol': {key: left[key] for key in PROTOCOL},
            'training_position': left['training_position'],
            'integrity': {'only_backbone_configuration_differs': True, 'training_recipes_match': True,
                          'episode_assignments_match': True, 'error_episodes': 0,
                          'missing_episodes': 0, 'episodes_per_policy': 500, 'states_per_task': 50},
            'limitations': 'One training seed and different backbone parameter counts; this is not a capacity-matched comparison. '
                           'Initial-state indices 0–9 were used in development pilots; indices 10–49 are not a newly reserved evaluation set. '
                           'All models retain the exact-heading source; this comparison does not test the Gaussian-prior change.'}


def markdown(report):
    models = report['models']
    score = lambda s: f"{s['mean_success_rate']:.1%} ({s['successes']}/{s['completed_episodes']})"
    delta = report['unet_minus_transformer']
    lines = [f"**Transformer: {score(models[RUNS[0]]['full_500'])}. U-Net: {score(models[RUNS[1]]['full_500'])}.**",
             f"U-Net minus Transformer: **{delta['percentage_points']:+.1f} percentage points** ({delta['successes']:+d} successes).", '',
             'Both use epoch-15 EMA weights (16 completed epochs, 31,024 optimizer updates) and HeadingZeroFlowPolicy with the exact-heading XY source.', '',
             '| Task | Transformer | U-Net | Change |', '|---|---:|---:|---:|']
    for task in models[RUNS[0]]['task_names']:
        a, b = [models[name]['full_500']['per_task'][task] for name in RUNS]
        lines.append(f"| {task.replace('_', ' ')} | {a['successes']}/50 ({a['success_rate']:.0%}) | "
                     f"{b['successes']}/50 ({b['success_rate']:.0%}) | {(b['successes']-a['successes'])*2:+d} pp |")
    lines += ['', 'The saved configurations differ only in velocity-backbone type and its options. '
              'Both use the same jointly trained ResNet18/heading architecture, 116px crops, heading condition, exact-heading source, '
              'dataset split and seed, batch size, optimizer, learning-rate schedule, EMA, observation/action horizons and ten Euler steps.', '',
              'Evaluation uses identical official state indices 0–49 per task and paired environment/policy-noise seeds derived from 20260911, '
              'with 550 policy-action steps after ten open-gripper settling steps. Observations are refreshed after settling. '
              'This is the protocol of our earlier 500-episode runs; it is not the separately named Official matched five-step/3000–3499 protocol.', '',
              f"Paired outcomes: {report['paired_outcomes']}.", '', report['limitations'], '',
              'Each result is recomputed from 500 unique records with zero evaluation errors. Code hashes and software versions match. '
              'The JSON report contains the complete protocol, per-task counts, training configuration hashes and indices 10–49 subset counts.', '']
    for name in RUNS:
        model = models[name]
        lines += [f"[{name} checkpoint]({name}/policy.ckpt): `{model['checkpoint_sha256']}`", '']
    return '\n'.join(lines)


def plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks = report['models'][RUNS[0]]['task_names']
    fig, ax = plt.subplots(figsize=(15, 12))
    for name, offset, label, color in zip(RUNS, [-.18, .18], ['Transformer', 'U-Net'], ['#64748b', '#2563eb']):
        values = [report['models'][name]['full_500']['per_task'][task] for task in tasks]
        bars = ax.barh([i + offset for i in range(10)], [100*v['success_rate'] for v in values],
                       height=.32, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v['successes']}/50" for v in values], padding=3, fontsize=8)
    ax.set_yticks(range(10), [textwrap.fill(t.replace('_', ' '), 55) for t in tasks], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('Success rate (%)')
    ax.set_title('LIBERO-10: Transformer vs U-Net — exact-heading prior, epoch 15, 500 episodes each')
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'output/heading_goal/backbone_500')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    try:
        report = build_report(args.root)
        for name, text in {'comparison.json': json.dumps(report, indent=2) + '\n', 'RESULTS.md': markdown(report)}.items():
            path = args.root / name
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(text)
            temporary.replace(path)
        if args.plot:
            plot(report, args.root / 'success_by_task.png')
        print(f"Validated comparison: {args.root / 'RESULTS.md'}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f'Comparison refused: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
