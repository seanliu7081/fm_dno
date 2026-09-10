#!/usr/bin/env python3
"""Report matched epoch-15 Transformer, U-Net and StarVLA-DiT assessments.

Only completed 500-episode runs are accepted. Source/export/snapshot provenance
is checked on CPU; this script does not run rollouts or alter earlier reports.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import textwrap

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_heading_backbone_comparison import (
    HASHES, IDENTITY_KEYS, PROTOCOL, TRAINING, file_sha256, read_run, require, validate_backbones,
)

RUNS = ('transformer_wide', 'unet_wide', 'starvla_dit_wide')
LABELS = ('Transformer', 'U-Net', 'StarVLA-DiT')
POSITION = {'epoch': 15, 'global_step': 31023}


def checkpoint_metadata(path):
    """Inspect serialized metadata without instantiating a policy or reading GPUs."""
    import dill
    from omegaconf import OmegaConf
    import torch
    from oat.common.hydra_util import register_new_resolvers
    register_new_resolvers()
    payload = torch.load(path, map_location='cpu', pickle_module=dill, mmap=True)
    cfg = payload['cfg']
    selected = 'ema_model' if cfg.training.get('use_ema', False) else 'model'
    require(selected in payload['state_dicts'], f'Missing selected weights: {path}')
    return {'policy_config': OmegaConf.to_container(cfg.policy, resolve=True),
            'training_position': {key: dill.loads(payload['pickles'][key]) for key in POSITION},
            'weights': selected, 'export_metadata': payload.get('export_metadata'),
            'training_seed': int(cfg.seed)}


def read_provenance(directory):
    path = Path(directory).resolve() / 'policy.ckpt.json'
    raw = path.read_bytes()
    evidence = json.loads(raw)
    for key, expected in {'format': 'oat_inference_policy_v1', 'inference_only': True,
                          'weights': 'ema_model', 'exact_state_dict_verified': True}.items():
        require(evidence.get(key) == expected, f'Invalid export provenance: {key}')
    for key in ('checkpoint_sha256', 'source_checkpoint_sha256'):
        require(re.fullmatch(r'[0-9a-f]{64}', evidence.get(key, '')) is not None, f'Missing export hash: {key}')
    require(Path(evidence['checkpoint']).resolve() == path.parent / 'policy.ckpt', 'Export provenance path mismatch')
    return evidence, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify_export_chain(run, provenance, sidecar):
    source = Path(provenance['source_checkpoint']).resolve(strict=True)
    export = Path(run['checkpoint'])
    require(source != export, 'Training source cannot be the exported checkpoint')
    require(export.stat().st_size == provenance['bytes'], 'Export byte count mismatch')
    require(file_sha256(source) == provenance['source_checkpoint_sha256'], 'Training source checkpoint hash mismatch')
    source_meta, export_meta = checkpoint_metadata(source), checkpoint_metadata(export)
    expected = {'policy_config': run['manifest']['policy_config'], 'training_position': POSITION,
                'weights': 'ema_model', 'training_seed': 42}
    for label, metadata in (('source', source_meta), ('export', export_meta)):
        for key, value in expected.items():
            require(metadata.get(key) == value, f'{label} checkpoint metadata mismatch: {key}')
    expected_embedded = {key: provenance[key] for key in ('format', 'inference_only', 'source_checkpoint',
                                                        'source_checkpoint_sha256', 'weights')}
    require(export_meta['export_metadata'] == expected_embedded, 'Embedded export provenance differs from sidecar')
    require(file_sha256(source) == provenance['source_checkpoint_sha256'], 'Source checkpoint changed while reporting')
    require(file_sha256(export) == provenance['checkpoint_sha256'], 'Export changed while reporting')
    require(file_sha256(sidecar['path']) == sidecar['sha256'], 'Export sidecar changed while reporting')
    return {**sidecar, 'record': provenance, 'source_and_export_metadata_verified': True,
            'tensor_equality_evidence': 'exact_state_dict_verified from exporter; reporter verifies hashes and metadata'}


def validate_three_backbones(configs):
    common, unet = validate_backbones(configs[0], configs[1])
    starvla = copy.deepcopy(configs[2])
    require(starvla.pop('backbone_type', None) == 'starvla_dit', 'Expected StarVLA-DiT backbone')
    options = starvla.pop('backbone_kwargs', None)
    require(options == {'num_register_tokens': 32}, 'Expected 32 StarVLA-DiT register tokens')
    require(starvla == common, 'StarVLA policy settings differ beyond backbone options')
    return common, {'transformer_wide': {}, 'unet_wide': unet, 'starvla_dit_wide': options}


def paired_outcomes(left, right):
    paired = {row['episode_id']: row for row in right}
    counts = {'both_succeed': 0, 'both_fail': 0, 'baseline_only_succeeds': 0, 'starvla_only_succeeds': 0}
    for row in left:
        a, b = row['success'], paired[row['episode_id']]['success']
        key = ('both_succeed' if a else 'both_fail') if a == b else ('baseline_only_succeeds' if a else 'starvla_only_succeeds')
        counts[key] += 1
    return counts


def build_report(baseline_root, starvla_dir, training_root=None, expected_baseline_hashes=HASHES):
    baseline_root, starvla_dir = Path(baseline_root).resolve(), Path(starvla_dir).resolve()
    training_root = Path(training_root).resolve() if training_root else baseline_root.parent
    directories = [baseline_root / name for name in RUNS[:2]] + [starvla_dir]
    provenance = [read_provenance(directory) for directory in directories]
    hashes = [item[0]['checkpoint_sha256'] for item in provenance]
    require(tuple(hashes[:2]) == tuple(expected_baseline_hashes), 'Baseline exports differ from the fixed comparison')
    runs = [read_run(directory, checksum) for directory, checksum in zip(directories, hashes)]
    manifests = [run['manifest'] for run in runs]
    common, options = validate_three_backbones([manifest['policy_config'] for manifest in manifests])
    for key, expected in {'horizon': 16, 'n_obs_steps': 2, 'n_action_steps': 8, 'num_inference_steps': 10}.items():
        require(common.get(key) == expected, f'Unexpected inference setting: {key}')
    for manifest in manifests:
        for key in PROTOCOL:
            require(key in manifest and key in manifests[0] and manifest[key] == manifests[0][key], f'Protocol mismatch: {key}')
        require(manifest['max_episode_steps'] == 550 and manifest['settle_steps'] == 10
                and manifest['seed'] == 20260911, 'Unexpected planned evaluation limits or seed')
        require(manifest['source_sha256'] and manifest['software_versions'], 'Missing code/software provenance')
    def identities(manifest):
        return {row['episode_id']: tuple(row[key] for key in (*IDENTITY_KEYS, 'official_state_sha256'))
                for row in manifest['episodes']}
    require(all(identities(manifest) == identities(manifests[0]) for manifest in manifests),
            'Episode task/state/environment/policy-noise assignments differ')
    training, training_evidence = [], []
    for name in RUNS:
        path = training_root / name / '.hydra/config.yaml'
        raw = path.read_bytes()
        training.append(yaml.safe_load(raw))
        training_evidence.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
    for key in TRAINING:
        require(all(key in cfg and cfg[key] == training[0].get(key) for cfg in training), f'Training recipe differs: {key}')
    require(training[0]['seed'] == 42, 'Expected matched training seed 42')
    validate_three_backbones([cfg['policy'] for cfg in training])
    # Resolve the saved recipe to bind evaluation policy metadata to each actual training configuration.
    from omegaconf import OmegaConf
    from oat.common.hydra_util import register_new_resolvers
    register_new_resolvers()
    for cfg, manifest, evidence in zip(training, manifests, training_evidence):
        resolved = OmegaConf.to_container(OmegaConf.create(cfg).policy, resolve=True)
        require(resolved == manifest['policy_config'], 'Saved training policy differs from evaluated policy')
        require(file_sha256(evidence['path']) == evidence['sha256'], 'Training configuration changed while reporting')
    export_evidence = [verify_export_chain(run, record, sidecar) for run, (record, sidecar) in zip(runs, provenance)]
    deltas, paired = {}, {}
    starvla = runs[2]
    for name, baseline in zip(RUNS[:2], runs[:2]):
        a, b = baseline['full_500'], starvla['full_500']
        deltas[name] = {'successes': b['successes'] - a['successes'],
                        'percentage_points': (b['successes'] - a['successes']) / 5.,
                        'per_task': {task: {'successes': b['per_task'][task]['successes'] - a['per_task'][task]['successes'],
                                           'percentage_points': 2 * (b['per_task'][task]['successes'] - a['per_task'][task]['successes'])}
                                     for task in baseline['task_names']}}
        paired[name] = paired_outcomes(baseline['rows'], starvla['rows'])
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'models': {name: {key: value for key, value in run.items() if key not in ('manifest', 'rows')}
                   for name, run in zip(RUNS, runs)},
        'starvla_minus_baseline': deltas, 'paired_outcomes': paired,
        'common_policy_config': common, 'backbone_options': options,
        'training_position': POSITION, 'completed_epochs': 16, 'optimizer_updates': 31024,
        'common_training_recipe': {key: training[0][key] for key in TRAINING},
        'training_configs': dict(zip(RUNS, training_evidence)), 'export_provenance': dict(zip(RUNS, export_evidence)),
        'evaluation_protocol': {key: manifests[0][key] for key in PROTOCOL},
        'integrity': {'only_backbone_configuration_differs': True, 'training_recipes_match': True,
                      'episode_assignments_match': True, 'source_export_snapshot_chain_verified': True,
                      'error_episodes': 0, 'missing_episodes': 0, 'episodes_per_policy': 500, 'states_per_task': 50},
        'limitations': 'One training seed and different parameter counts; this is not a capacity-matched comparison. '
                       'These initial states were evaluated in earlier comparisons and are not a fresh holdout. '
                       'All three policies retain the exact-heading source; this does not test the Gaussian-prior change. '
                       'StarVLA-DiT names the velocity backbone here, not a pretrained vision-language-action model.',
    }


def markdown(report):
    models = report['models']
    def score(summary):
        return f"{summary['mean_success_rate']:.1%} ({summary['successes']}/{summary['completed_episodes']})"
    lines = ['**' + ' · '.join(f"{label}: {score(models[name]['full_500'])}" for name, label in zip(RUNS, LABELS)) + '**', '']
    for name, label in zip(RUNS[:2], LABELS[:2]):
        delta = report['starvla_minus_baseline'][name]
        lines.append(f"StarVLA-DiT minus {label}: **{delta['percentage_points']:+.1f} percentage points** ({delta['successes']:+d} successes).")
    lines += ['', 'All three use epoch-15 EMA weights: 16 completed epochs and 31,024 optimizer updates, training seed 42. '
              'Each policy uses one checkpoint across all ten tasks.', '',
              '| Task | Transformer | U-Net | StarVLA-DiT | StarVLA − Transformer | StarVLA − U-Net |',
              '|---|---:|---:|---:|---:|---:|']
    for task in models[RUNS[0]]['task_names']:
        scores = [models[name]['full_500']['per_task'][task] for name in RUNS]
        cells = [f"{item['successes']}/50 ({item['success_rate']:.0%})" for item in scores]
        deltas = [f"{report['starvla_minus_baseline'][name]['per_task'][task]['percentage_points']:+d} pp" for name in RUNS[:2]]
        lines.append('| ' + ' | '.join([task.replace('_', ' '), *cells, *deltas]) + ' |')
    lines += ['', 'The saved recipes differ only in velocity-backbone type and its options. StarVLA-DiT uses 32 register tokens. '
              'The shared HeadingZeroFlowPolicy predicts heading from image/state features and uses it both as conditioning and as the exact-heading XY source. '
              'The models use the same ResNet18 observation-encoder and heading-head architectures, trained separately, and the same 116-pixel crops, dataset split, optimizer, schedule and EMA.', '',
              'Evaluation uses the same official initial-state indices 0–49 per task, state hashes and paired environment/policy-noise seeds derived from 20260911. '
              'Each episode has **550 policy-action steps** after **10 settling steps**. The policy consumes two observations, predicts 16 actions, '
              'executes eight before replanning, and integrates ten Euler steps. Evaluator code hashes and software versions match.', '',
              report['limitations'], '',
              'Counts are recomputed from 500 unique error-free episodes per policy. Export/source/snapshot hashes and checkpoint metadata were verified; '
              'export provenance records exact selected-state equality. JSON includes paired outcomes and the reused indices 10–49 subset, without treating it as a fresh holdout.', '']
    for name, label in zip(RUNS, LABELS):
        model = models[name]
        lines += [f"[{label} checkpoint]({model['checkpoint']}): `{model['checkpoint_sha256']}`", '']
    return '\n'.join(lines)


def plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks = report['models'][RUNS[0]]['task_names']
    fig, ax = plt.subplots(figsize=(16, 13))
    for name, label, offset, color in zip(RUNS, LABELS, [-.25, 0, .25], ['#64748b', '#2563eb', '#d97706']):
        values = [report['models'][name]['full_500']['per_task'][task] for task in tasks]
        bars = ax.barh([i + offset for i in range(10)], [100 * value['success_rate'] for value in values],
                       height=.22, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value['successes']}/50" for value in values], padding=3, fontsize=7)
    ax.set_yticks(range(10), [textwrap.fill(task.replace('_', ' '), 55) for task in tasks], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('Success rate (%)')
    ax.set_title('LIBERO-10 — matched epoch-15 backbones, exact-heading prior, 500 episodes each')
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, default=ROOT / 'output/heading_goal/backbone_500')
    parser.add_argument('--starvla-dir', type=Path, default=ROOT / 'output/heading_goal/starvla_dit_500')
    parser.add_argument('--training-root', type=Path, default=ROOT / 'output/heading_goal')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    try:
        require(args.starvla_dir.resolve() not in {args.baseline_root.resolve(), *(args.baseline_root.resolve() / name for name in RUNS[:2])},
                'Output must not overwrite an earlier comparison directory')
        report = build_report(args.baseline_root, args.starvla_dir, args.training_root)
        for name, content in {'comparison.json': json.dumps(report, indent=2) + '\n', 'RESULTS.md': markdown(report)}.items():
            path = args.starvla_dir / name
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(content)
            temporary.replace(path)
        if args.plot:
            plot(report, args.starvla_dir / 'success_by_task.png')
        print(f"Validated comparison: {args.starvla_dir / 'RESULTS.md'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Comparison refused: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
