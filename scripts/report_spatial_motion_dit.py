#!/usr/bin/env python3
"""Report verified 500-epoch spatial-motion DiT results, including partial runs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill
from omegaconf import OmegaConf
import torch
from oat.env_runner.libero_official_eval import file_sha256
from oat.env_runner.libero_official_runner import LiberoOfficialRunner

EPOCHS = tuple(range(50, 501, 50))


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def verify_epoch(run_dir, epoch):
    directory = run_dir / 'rollouts' / f'epoch_{epoch:03d}'
    try:
        evidence = read_json(directory / 'snapshot.json')
        require(evidence['completed_epochs'] == epoch, 'snapshot epoch mismatch')
        require(evidence['training_position']['epoch'] == epoch - 1, 'training position mismatch')
        require(evidence['weights'] == 'ema_model', 'evaluation must use EMA weights')
        for key, value in dict(suite='libero_10', n_per_task=50, init_start=0, seed=20260911,
                               max_episode_steps=550, settle_steps=10, video_per_task=1).items():
            require(evidence['protocol'].get(key) == value, f'protocol mismatch: {key}')
        runner = LiberoOfficialRunner(str(run_dir), **evidence['protocol'],
                                      eval_gpu=evidence['eval_gpu'], torch_threads=evidence['torch_threads'])
        checkpoint = directory / 'policy.ckpt'
        failures = []
        for attempt in sorted(directory.glob('attempt_[0-9][0-9][0-9]')):
            record = attempt / 'runner_verified.json'
            if not record.is_file():
                continue
            try:
                # Recompute success from episode rows and recheck protocol,
                # copied checkpoint hashes, selected videos and manifest.
                metrics = runner._verify_result(attempt, checkpoint, evidence)
                require(read_json(record) == {'checkpoint_sha256': evidence['checkpoint_sha256'],
                                               'returncode': 0, 'metrics': metrics},
                        'runner verification record mismatch')
                summary = read_json(attempt / 'summary.json')
                require(metrics['eval/episodes'] == 500 and len(summary['per_task']) == 10,
                        'evaluation must contain 500 episodes across ten tasks')
                require(all(value['completed'] == 50 for value in summary['per_task'].values()),
                        'each task must contain fifty completed episodes')
                archive = run_dir / 'checkpoints' / f'epoch-{epoch:04d}.ckpt'
                return dict(epoch=epoch, status='verified', success_rate=metrics['mean_success_rate'],
                            successes=metrics['eval/successes'], episodes=500, per_task=summary['per_task'],
                            attempt=str(attempt), snapshot_checkpoint=str(checkpoint),
                            checkpoint_sha256=evidence['checkpoint_sha256'], protocol=evidence['protocol'],
                            training_checkpoint=str(archive) if archive.is_file() else None,
                            videos=runner.media_artifacts())
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                failures.append(f'{attempt.name}: {error}')
        return dict(epoch=epoch, status='incomplete', reason='; '.join(failures) or 'no complete verified attempt')
    except FileNotFoundError:
        return dict(epoch=epoch, status='missing', reason='snapshot/evaluation evidence not available')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return dict(epoch=epoch, status='invalid', reason=str(error))


def verify_training_checkpoint(path, evaluation, dataset, *, selected_topk=False):
    """Match a resumable training boundary to its independently verified EMA."""
    epoch = evaluation['epoch']
    require(path.is_file(), f'epoch-{epoch} training checkpoint missing: {path}')
    digest = file_sha256(path)
    payload = torch.load(path, map_location='cpu', pickle_module=dill)
    saved = {key: dill.loads(value) for key, value in payload['pickles'].items()}
    require(saved['epoch'] == epoch - 1 and saved.get('next_epoch') == epoch
            and saved.get('pending_rollout') is None,
            f'training checkpoint must complete epoch {epoch} with no pending evaluation')
    cfg = payload['cfg']
    require(cfg.training.num_epochs == 500 and cfg.training.max_train_steps is None,
            'checkpoint has an incomplete training schedule')
    require(cfg.training.gradient_accumulate_every == 1 and cfg.dataloader.batch_size == 64
            and cfg.dataloader.drop_last, 'unexpected optimizer-step accounting configuration')
    require(dataset is not None, 'training split metadata missing')
    expected_updates = (dataset['train_window_count'] // 64) * epoch
    require(saved.get('optimizer_step') == expected_updates,
            f'optimizer update count differs from {expected_updates}')
    require('optimizer' in payload['state_dicts'], 'training checkpoint lacks optimizer state')
    require(saved.get('ema_state') is not None and saved.get('lr_scheduler_state') is not None,
            'training checkpoint lacks EMA/scheduler continuation state')
    if selected_topk:
        require(saved.get('topk_state', {}).get(str(path)) == evaluation['success_rate'],
                'checkpoint does not record its selected top-K score/path')
    snapshot = torch.load(evaluation['snapshot_checkpoint'], map_location='cpu', pickle_module=dill)
    actual, expected = payload['state_dicts']['ema_model'], snapshot['state_dicts']['ema_model']
    require(actual.keys() == expected.keys(), 'training EMA keys differ from evaluated snapshot')
    for key, value in expected.items():
        require(actual[key].dtype == value.dtype and actual[key].shape == value.shape
                and torch.equal(actual[key], value), f'training EMA tensor differs from evaluation: {key}')
    require(file_sha256(path) == digest, 'training checkpoint changed during verification')
    require(file_sha256(evaluation['snapshot_checkpoint']) == evaluation['checkpoint_sha256'],
            'evaluated snapshot changed during training-checkpoint verification')
    return dict(path=str(path), sha256=digest, completed_epochs=epoch,
                optimizer_updates=expected_updates, matches_evaluation=True,
                selected_topk=selected_topk)


def verify_final(run_dir, evaluation, dataset):
    require(evaluation['epoch'] == 500, 'final evaluation must be epoch 500')
    result = verify_training_checkpoint(run_dir / 'checkpoints' / 'epoch-0500.ckpt', evaluation, dataset)
    result['matches_final_evaluation'] = True
    return result


def verify_best(run_dir, evaluation, dataset, cfg, epoch_rows):
    require(cfg is not None, 'resolved configuration missing for top-K selection')
    options = cfg.checkpoint.topk
    require(options.monitor_key == 'mean_success_rate' and options.mode == 'max' and options.k == 1,
            'best checkpoint requires the configured success-rate maximum with k=1')
    epoch = evaluation['epoch']
    metrics = {key.replace('/', '_'): value for key, value in epoch_rows.get(epoch, {}).items()}
    metrics.update(epoch=epoch - 1, completed_epochs=epoch,
                   mean_success_rate=evaluation['success_rate'])
    path = run_dir / 'checkpoints' / options.format_str.format(**metrics)
    result = verify_training_checkpoint(path, evaluation, dataset, selected_topk=True)
    result['selection_rule'] = 'highest verified success rate; earliest epoch wins ties'
    return result


def run_timing(run_dir, launcher, observed_at):
    if launcher is None:
        return None
    started = datetime.fromisoformat(launcher['started_at_utc'])
    require(started.tzinfo is not None, 'launcher start timestamp must include a timezone')
    successful = []
    for path in run_dir.glob('launcher_exit_*.json'):
        record = read_json(path)
        if record.get('returncode') == 0:
            finished = datetime.fromisoformat(record['finished_at_utc'])
            require(finished.tzinfo is not None, 'launcher exit timestamp must include a timezone')
            require(finished >= started, 'successful exit precedes original launch')
            successful.append((finished, path))
    if successful:
        finished, path = max(successful, key=lambda entry: entry[0])
        return dict(status='successful_exit', started_at_utc=started.isoformat(),
                    finished_at_utc=finished.isoformat(), observed_at_utc=observed_at.isoformat(),
                    total_wall_clock_seconds=(finished - started).total_seconds(),
                    successful_exit_record=str(path))
    require(observed_at >= started, 'report observation precedes original launch')
    return dict(status='elapsed_to_observation', started_at_utc=started.isoformat(),
                observed_at_utc=observed_at.isoformat(),
                elapsed_seconds=(observed_at - started).total_seconds())


def build_report(run_dir):
    require(run_dir.is_dir(), f'run directory does not exist: {run_dir}')
    issues = []
    cfg = None
    observed_at = datetime.now(timezone.utc)
    config_path = run_dir / 'resolved_config.yaml'
    if not config_path.is_file():
        config_path = run_dir / '.hydra' / 'config.yaml'
    if config_path.is_file():
        cfg = OmegaConf.load(config_path)
        checks = [(cfg.training.num_epochs == 500, 'training budget is not 500 epochs'),
                  (cfg.logging.mode == 'online', 'W&B mode is not online'),
                  (not cfg.training.resume, 'original config is not a scratch start'),
                  (cfg.policy._target_.endswith('.SpatialMotionFlowPolicy'), 'unexpected policy'),
                  (cfg.policy.backbone_type == 'starvla_dit', 'unexpected flow backbone'),
                  (not cfg.task.policy.lazy_eval, 'synchronous evaluation is disabled')]
        issues += [message for okay, message in checks if not okay]
    else:
        issues.append('resolved training configuration missing')
    data_path = run_dir / 'training_dataset.json'
    dataset = read_json(data_path) if data_path.is_file() else None
    if dataset is None or dataset.get('train_episode_count') != 450 or dataset.get('validation_episode_count') != 50:
        issues.append('450/50 training/validation split not verified')
    epoch_rows = {}
    evaluation_durations = {}
    log_path = run_dir / 'logs.json'
    if log_path.is_file():
        lines = log_path.read_text().splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break  # The trainer may be appending the last record.
                raise ValueError(f'malformed training log at line {index+1}')
            if 'completed_epochs' in row:
                epoch_rows[int(row['completed_epochs'])] = row
            if 'eval/duration_seconds' in row:
                epoch = int(row.get('eval/completed_epochs', row.get('completed_epochs')))
                seconds = float(row['eval/duration_seconds'])
                require(math.isfinite(seconds) and seconds >= 0, 'invalid logged evaluation duration')
                # A recovered boundary can log a verified cached evaluation again.
                # Include all recorded execution time for that completed epoch.
                evaluation_durations[epoch] = evaluation_durations.get(epoch, 0.0) + seconds
    completed = max(epoch_rows, default=0)
    evaluations = [verify_epoch(run_dir, epoch) for epoch in EPOCHS]
    verified = [item for item in evaluations if item['status'] == 'verified']
    for item in verified:
        item['duration_seconds'] = evaluation_durations.get(item['epoch'])
        item['duration_source'] = 'logs.json eval/duration_seconds, summed by completed epoch'
        if item['duration_seconds'] is None:
            issues.append(f"epoch-{item['epoch']:04d} logged evaluation duration missing")
    if len(verified) != 10:
        issues.append(f'{len(verified)}/10 scheduled 500-episode evaluations verified')
    for item in verified:
        if item['training_checkpoint'] is None:
            issues.append(f"epoch-{item['epoch']:04d} training checkpoint missing")
    final = None
    if evaluations[-1]['status'] == 'verified':
        try:
            final = verify_final(run_dir, evaluations[-1], dataset)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            issues.append(f'final checkpoint: {error}')
    else:
        issues.append('final epoch-500 evaluation/checkpoint pair not verified')
    # TopKCheckpointManager replaces its k=1 entry only on strict improvement;
    # preserve the earlier checkpoint when success rates tie.
    best_evaluation = max(verified, key=lambda item: (item['success_rate'], -item['epoch']), default=None)
    best_checkpoint = None
    if best_evaluation is not None:
        try:
            best_checkpoint = verify_best(run_dir, best_evaluation, dataset, cfg, epoch_rows)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
            issues.append(f'best checkpoint: {error}')
    if completed < 500:
        issues.append(f'training log records {completed}/500 completed epochs')
    wandb_url = None
    for name in ('wandb_run.json', 'wandb.json'):
        if (run_dir / name).is_file():
            record = read_json(run_dir / name)
            wandb_url = record.get('url', record.get('run_url')) or wandb_url
    if not wandb_url and (run_dir / 'console.log').is_file():
        urls = re.findall(r'https://wandb\.ai/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/runs/[A-Za-z0-9_-]+',
                          (run_dir / 'console.log').read_text())
        wandb_url = urls[-1] if urls else None
    if not wandb_url:
        issues.append('online W&B run URL not recorded')
    launcher = read_json(run_dir / 'launcher.json') if (run_dir / 'launcher.json').is_file() else None
    timing = None
    try:
        timing = run_timing(run_dir, launcher, observed_at)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        issues.append(f'wall-clock timing: {error}')
    if timing is None:
        issues.append('original launcher timing evidence missing')
    elif timing['status'] != 'successful_exit':
        issues.append('successful launcher exit not yet recorded; wall-clock time is elapsed to observation')
    duration = sum(row.get('train/epoch_duration_seconds', 0) for row in epoch_rows.values())
    return dict(generated_at_utc=observed_at.isoformat(), run_dir=str(run_dir),
                complete=not issues, issues=issues, requested_epochs=500, logged_completed_epochs=completed,
                evaluations=evaluations, final_checkpoint=final,
                best_evaluation=best_evaluation, best_checkpoint=best_checkpoint,
                wall_clock=timing, recorded_evaluation_seconds=sum(evaluation_durations.values()),
                wandb_url=wandb_url, launcher=launcher, dataset=dataset,
                recorded_training_seconds=duration, epoch_rows=list(epoch_rows.values()))


def markdown(report):
    status = 'complete' if report['complete'] else 'incomplete'
    lines = ['# Spatial motion-query DiT on LIBERO-10', '', f'**Status: {status}.**', '',
             f"Budget: 500 epochs from scratch; training log records {report['logged_completed_epochs']} completed epochs.",
             'Architecture: two spatial ResNet18 encoders, four motion queries, segmented XY Gaussian sources, and DiT flow.', '',
             'Protocol: ten tasks, initial states 0–49, seed 20260911, ten settling actions, '
             '550 policy actions maximum, EMA weights and one video per task. '
             'Evaluations after every 50 completed epochs total 5,000 scheduled episodes.', '']
    if report['wandb_url']:
        lines += [f"Online curves: [W&B run]({report['wandb_url']}).", '']
    if report['dataset']:
        data = report['dataset']
        lines += [f"Data: {data['train_episode_count']} training / {data['validation_episode_count']} validation demonstrations; "
                  f"{data['train_window_count']:,} training windows; raw XY RMS {data.get('heading_xy_rms', 'unavailable')}.", '']
    lines += ['| Epoch | Verified success | Episodes | Status |', '|---:|---:|---:|---|']
    verified = [item for item in report['evaluations'] if item['status'] == 'verified']
    for item in report['evaluations']:
        score = f"{item['success_rate']:.1%} ({item['successes']}/500)" if item['status'] == 'verified' else '—'
        lines.append(f"| {item['epoch']} | {score} | {item.get('episodes', '—')} | {item['status']} |")
    if report['best_evaluation']:
        best = report['best_evaluation']
        lines += ['', f"Best observed: epoch {best['epoch']}, {best['success_rate']:.1%}. "
                  f"[Evaluated EMA snapshot]({best['snapshot_checkpoint']}); SHA-256 `{best['checkpoint_sha256']}`."]
    if report['best_checkpoint']:
        best = report['best_checkpoint']
        lines += ['', f"[Best resumable training checkpoint]({best['path']}): epoch {best['completed_epochs']}, "
                  f"{best['optimizer_updates']:,} optimizer updates; EMA tensors match the best evaluated snapshot. "
                  f"SHA-256 `{best['sha256']}`. Equal success rates retain the earlier epoch."]
    if report['final_checkpoint']:
        final = report['final_checkpoint']
        lines += ['', f"[Final training checkpoint]({final['path']}): {final['optimizer_updates']:,} optimizer updates; "
                  f"EMA tensors match the final evaluation. SHA-256 `{final['sha256']}`."]
    if verified:
        names = sorted(verified[0]['per_task'])
        lines += ['', '## Per-task success', '', '| Task | ' + ' | '.join(str(item['epoch']) for item in verified) + ' |',
                  '|---|' + '---:|' * len(verified)]
        for name in names:
            lines.append('| ' + name.replace('_', ' ') + ' | ' + ' | '.join(
                f"{item['per_task'][name]['success_rate']:.0%}" for item in verified) + ' |')
    lines += ['', '## Artifacts and timing', '',
              f"Recorded training epoch time: {report['recorded_training_seconds']/3600:.2f} hours "
              '(excludes evaluation and time outside recorded epochs).',
              f"Recorded evaluation time: {report['recorded_evaluation_seconds']/3600:.2f} hours "
              '(from logged evaluation durations, including logged boundary retries).',
              'Exact split indices: `training_dataset.json`. Original config: `resolved_config.yaml`. '
              'Hardware, package versions and run ID: `launcher.json`. Code snapshots and diffs accompany the run.',
              'Benchmark, per-task and epoch curves are in the CSV files. `report.json` records all verified '
              'evaluation directories, episode artifacts, checkpoint hashes and video paths.']
    timing = report['wall_clock']
    if timing is not None:
        if timing['status'] == 'successful_exit':
            lines += [f"Total wall-clock duration: {timing['total_wall_clock_seconds']/3600:.2f} hours "
                      '(original launch to successful exit; includes evaluation, setup and interruptions).',
                      f"Launch: {timing['started_at_utc']}. Successful exit: {timing['finished_at_utc']}. "
                      f"[Exit evidence]({timing['successful_exit_record']})."]
        else:
            lines += [f"Elapsed wall-clock time to report observation: {timing['elapsed_seconds']/3600:.2f} hours "
                      '(no successful exit recorded; this is not a final run duration).',
                      f"Launch: {timing['started_at_utc']}. Observation: {timing['observed_at_utc']}."]
    if report['issues']:
        lines += ['', '## Outstanding evidence', ''] + [f'- {issue}' for issue in report['issues']]
    for item in report['evaluations']:
        if item.get('reason') and item['status'] != 'missing':
            lines.append(f"- Epoch {item['epoch']}: {item['reason']}")
    lines += ['', 'This run measures the combined design. It does not isolate gains from spatial features, '
              'query decoding or segmented targets.', '']
    return '\n'.join(lines)


def write_csv(path, rows, fields):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verified = [item for item in report['evaluations'] if item['status'] == 'verified']
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot([item['epoch'] for item in verified], [100*item['success_rate'] for item in verified], marker='o')
    axis.set(xlim=(0, 510), ylim=(0, 100), xlabel='Completed epochs', ylabel='Success (%)',
             title='LIBERO-10: verified 500-episode evaluations')
    axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(directory / 'success_curve.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, names in zip(axes, [('train_loss', 'train/flow_loss', 'val_loss'),
                                 ('train/heading_loss', 'train/validity_loss')]):
        for name in names:
            rows = [row for row in report['epoch_rows'] if name in row]
            if rows:
                axis.plot([row['completed_epochs'] for row in rows], [row[name] for row in rows], label=name)
        axis.set(xlabel='Completed epochs', ylabel='Loss')
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(directory / 'learning_curves.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='write standalone PNG success and learning curves')
    parser.add_argument('--require-complete', action='store_true', help='exit 2 when the run is incomplete')
    args = parser.parse_args()
    try:
        directory = args.run_dir.resolve()
        report = build_report(directory)
        atomic_text(directory / 'report.json', json.dumps(report, indent=2) + '\n')
        atomic_text(directory / 'RESULTS.md', markdown(report))
        write_csv(directory / 'success_curve.csv', report['evaluations'],
                  ['epoch', 'status', 'success_rate', 'successes', 'episodes', 'duration_seconds', 'checkpoint_sha256', 'reason'])
        task_rows = [dict(epoch=item['epoch'], task=name, **value) for item in report['evaluations']
                     if item['status'] == 'verified' for name, value in item['per_task'].items()]
        write_csv(directory / 'per_task_success.csv', task_rows, ['epoch', 'task', 'successes', 'completed', 'success_rate'])
        fields = ['completed_epochs'] + sorted({key for row in report['epoch_rows'] for key, value in row.items()
                                                if isinstance(value, (int, float)) and key != 'completed_epochs'})
        write_csv(directory / 'learning_curves.csv', report['epoch_rows'], fields)
        if args.plot:
            plot(report, directory)
        print(f"{'Complete' if report['complete'] else 'Incomplete'} report: {directory / 'RESULTS.md'}")
        return 2 if args.require_complete and not report['complete'] else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f'Report failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
