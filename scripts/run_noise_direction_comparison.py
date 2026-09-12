#!/usr/bin/env python3
"""Paired, frozen-checkpoint LIBERO noise/direction experiment; no retraining."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oat.env_runner.libero_official_eval import array_sha256, file_sha256, stable_seed

DEFAULT_OUTPUT = ROOT / 'output/noise_direction_best_sr_20260912'
EVAL_DIRS = {
    'baseline_transformer': 'output/eval_comparison_ep50_ep55_20260910T234741Z/baseline_transformer',
    'heading_zero_dit': 'output/heading_goal/starvla_dit_150_online_20260910/rollouts/epoch_030/attempt_001',
    'heading_gaussian_dit': 'output/20260911/001512_train_flowpolicy_headinggaussian_starvlaDiT_libero10_N500/rollouts/epoch_060/attempt_001',
}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_arrays(path, **arrays):
    path = Path(path)
    with open(str(path) + '.tmp', 'wb') as f:
        np.savez_compressed(f, **arrays)
    Path(str(path) + '.tmp').replace(path)


def checkpoint_config(path):
    import dill
    import torch
    from oat.common.hydra_util import register_new_resolvers
    from omegaconf import OmegaConf
    register_new_resolvers()
    with open(path, 'rb') as f:
        payload = torch.load(f, map_location='cpu', pickle_module=dill)
    config = OmegaConf.to_container(payload['cfg'], resolve=True)
    if 'dataset' not in config.get('task', {}).get('policy', {}):
        # Official rollout exports contain inference settings only.
        candidates = [parent / '.hydra/config.yaml' for parent in Path(path).parents]
        training_path = next((p for p in candidates if p.is_file()), None)
        if training_path is None:
            raise ValueError(f'Cannot recover training dataset config for {path}')
        training = OmegaConf.load(training_path)
        config['task']['policy']['dataset'] = OmegaConf.to_container(training.task.policy.dataset, resolve=True)
        config['dataset_config_provenance'] = str(training_path)
    return config


def select_windows(ends, tasks, val_mask, count=128, seed=20260912):
    """Equal task budgets, near-equal episode budgets; complete unpadded targets."""
    starts = np.r_[0, ends[:-1]]
    windows, representatives = [], []
    for task in sorted(np.unique(tasks)):
        episodes = np.flatnonzero((tasks == task) & val_mask)
        if not len(episodes):
            raise ValueError(f'Task {task} has no held-out episodes')
        rng = np.random.default_rng(stable_seed(seed, 'selection', int(task)))
        order = rng.permutation(episodes)
        representative_episode = int(order[0])
        selected, mandatory = {}, {}
        for rank, episode in enumerate(order):
            length = int(ends[episode] - starts[episode])
            available = np.arange(1, length - 15)  # t-1,t observations; t..t+15 actions.
            quota = count // len(episodes) + int(rank < count % len(episodes))
            required = []
            if episode == representative_episode:
                for phase, fraction in [('early', .15), ('middle', .50), ('late', .85)]:
                    frame = int(available[round((len(available) - 1) * fraction)])
                    required.append(frame)
                    mandatory[(int(episode), frame)] = phase
            candidates = np.setdiff1d(available, required)
            if quota < len(required) or len(available) < quota:
                raise ValueError('Insufficient unpadded frames for the planned episode allocation')
            chosen = np.r_[required, rng.permutation(candidates)[:quota-len(required)]]
            selected[int(episode)] = sorted(map(int, chosen))
        for episode in sorted(selected):
            for frame in selected[episode]:
                row = {'window_index': len(windows), 'task_uid': int(task),
                       'episode_index': int(episode), 'frame_index': frame,
                       'global_index': int(starts[episode]) + frame,
                       'episode_start': int(starts[episode]), 'episode_end': int(ends[episode])}
                windows.append(row)
                if (episode, frame) in mandatory:
                    representatives.append({**row, 'phase': mandatory[(episode, frame)]})
    for i, row in enumerate(representatives):
        row['representative_index'] = i
    return windows, representatives


def prepare(args):
    import torch
    import zarr
    from oat.common.seq_sampler import get_val_mask
    from oat.env.libero.env import task_name_to_suite_and_ids
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    if (out / 'selection.json').exists():
        raise FileExistsError('A selection already exists; do not silently replace the experiment')
    models, dataset_configs = {}, []
    for name, relative in EVAL_DIRS.items():
        directory = ROOT / relative
        manifest = json.loads((directory / 'manifest.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        checkpoint = directory / 'policy.ckpt'
        digest = file_sha256(checkpoint)
        assert digest == manifest['checkpoint_sha256'] == summary['checkpoint_sha256']
        rows = [json.loads(x) for x in (directory / 'episodes.jsonl').read_text().splitlines() if x.strip()]
        assert summary['complete'] and len(rows) == len({x['episode_id'] for x in rows}) == 500
        assert sum(bool(x['success']) for x in rows) == summary['successes']
        config = checkpoint_config(checkpoint)
        dataset_configs.append(config['task']['policy']['dataset'])
        models[name] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                        'evaluation_summary': str(directory / 'summary.json'),
                        'success_rate': summary['successes'] / 500,
                        'successes': summary['successes'], 'evaluation_episodes': 500,
                        'training_position': manifest['training_position'], 'weights': manifest['weights'],
                        'policy_config': config['policy'], 'dataset_config': dataset_configs[-1]}
    first = dataset_configs[0]
    for config in dataset_configs[1:]:
        for key in ['zarr_path', 'seed', 'val_ratio', 'n_obs_steps', 'n_action_steps', 'obs_keys']:
            if config[key] != first[key]:
                raise ValueError(f'Checkpoint data definitions disagree: {key}')
    path = (ROOT / first['zarr_path']).resolve()
    z = zarr.open(str(path), mode='r')
    ends = np.asarray(z['meta/episode_ends'][:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    task_frames = np.asarray(z['data/task_uid'][:]).reshape(-1)
    tasks = task_frames[starts]
    assert np.array_equal(task_frames, np.repeat(tasks, ends - starts))
    val_mask = get_val_mask(len(ends), first['val_ratio'], first['seed'])
    windows, representatives = select_windows(ends, tasks, val_mask, args.windows_per_task, args.seed)
    names = {uid: name for name, (_, _, uid) in task_name_to_suite_and_ids.items()}
    for row in windows + representatives:
        row['task_name'] = names[row['task_uid']]
    assert len(np.unique(tasks)) == 10 and len(representatives) == 30
    arrays = {'actions_gt': np.stack([z['data/action'][w['global_index']:w['global_index']+16] for w in windows]).astype(np.float32)}
    for key in first['obs_keys']:
        arrays['obs__' + key] = np.stack([z['data/' + key][w['global_index']-1:w['global_index']+1] for w in windows])
        print(f'Prepared {key}: {arrays["obs__" + key].shape}', flush=True)
    save_arrays(out / 'data.npz', **arrays)
    noises, representative_noises = [], {}
    rep_indices = {r['window_index'] for r in representatives}
    for row in windows:
        seed = stable_seed(args.seed, 'gaussian', row['task_uid'], row['episode_index'], row['frame_index'])
        row['noise_seed'] = seed
        n = 1024 if row['window_index'] in rep_indices else 32
        noise = torch.randn(n, 16, 7, generator=torch.Generator().manual_seed(seed)).numpy()
        noises.append(noise[:32])
        if n == 1024:
            representative_noises[row['window_index']] = noise
    noise = np.stack(noises)
    rep_noise = np.stack([representative_noises[r['window_index']] for r in representatives])
    for r, rn in zip(representatives, rep_noise):
        np.testing.assert_array_equal(rn[:32], noise[r['window_index']])
    np.save(out / 'noise.npy', noise)
    np.save(out / 'representative_noise.npy', rep_noise)
    metadata = {
        'models': models, 'dataset_path': str(path), 'selection_seed': args.seed,
        'windows_per_task': args.windows_per_task, 'draws_per_window': 32,
        'representative_source_draws': 1024, 'representative_action_draws': 128,
        'windows': windows, 'representatives': representatives,
        'train_episode_indices': np.flatnonzero(~val_mask).tolist(),
        'validation_episode_indices': np.flatnonzero(val_mask).tolist(),
        'episode_ends': ends.tolist(), 'task_ids': sorted(map(int, np.unique(tasks))),
        'selection': 'Seeded near-equal allocation across held-out episodes within each task; unpadded two-frame observations and 16-action targets. One seeded episode/task supplies 15%,50%,85% visual frames, included in the 128 windows. Selection precedes all inference.',
        'data_sha256': file_sha256(out / 'data.npz'), 'noise_sha256': file_sha256(out / 'noise.npy'),
        'representative_noise_sha256': file_sha256(out / 'representative_noise.npy'),
        'dataset_episode_ends_sha256': array_sha256(ends),
        'dataset_actions_sha256': array_sha256(z['data/action'][:]),
        'comparison_scope': 'Exactly three native models, best recorded SR checkpoints; backbone and training duration differ; no source ablations or retraining.',
    }
    write_json(out / 'selection.json', metadata)
    print(f'Selection complete: {len(windows)} windows and {len(representatives)} visual frames', flush=True)


def encode(policy, obs):
    if hasattr(policy, 'heading_targets'):
        return policy.obs_encoder.encode_with_heading(obs)
    return policy.obs_encoder(obs), None


def source_from_noise(policy, noise, prediction):
    import torch
    z = policy.prior_noise_scale * noise
    if prediction is None:
        return z, torch.zeros(len(z), device=z.device, dtype=torch.bool)
    return policy._transform_source(z, prediction, angular_jitter=None)


def integrate(policy, source, cond):
    import torch
    x = source.clone()
    middle = None
    for i in range(policy.num_inference_steps):
        t = torch.full((len(x),), i / policy.num_inference_steps, device=x.device, dtype=cond.dtype)
        x = x + policy.model(x, policy._scale_t(t), cond) / policy.num_inference_steps
        if i + 1 == policy.num_inference_steps // 2:
            middle = policy.normalizer['action'].unnormalize(x).detach()
    return policy.normalizer['action'].unnormalize(x), middle


def native_parity_check(policy, obs, noise):
    """Verify diagnostic instrumentation against the actual native inference method."""
    import torch
    from unittest.mock import patch
    cond, prediction = encode(policy, obs)
    source, _ = source_from_noise(policy, noise, prediction)
    actual, _ = integrate(policy, source, cond)
    if prediction is not None:
        expected = policy.predict_action(obs, noise=noise, return_source=True)
        torch.testing.assert_close(source, expected['source'], rtol=0, atol=0)
    else:
        with patch('torch.randn', return_value=noise):
            expected = policy.predict_action(obs)
    torch.testing.assert_close(actual, expected['action_pred'], rtol=2e-5, atol=2e-6)
    return float((actual - expected['action_pred']).abs().max())


def infer(args):
    import torch
    from omegaconf import OmegaConf
    from oat.env_runner.libero_official_eval import load_policy, verify_repository_imports
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = args.output
    selection = json.loads((out / 'selection.json').read_text())
    model_meta = selection['models'][args.model]
    path = out / 'models' / f'{args.model}.npz'
    if path.exists():
        raise FileExistsError(f'Inference result already exists: {path}')
    checkpoint = model_meta['checkpoint']
    assert file_sha256(checkpoint) == model_meta['checkpoint_sha256']
    data = np.load(out / 'data.npz')
    observations = {key[5:]: data[key] for key in data.files if key.startswith('obs__')}
    noise = np.load(out / 'noise.npy', mmap_mode='r')
    rep_noise = np.load(out / 'representative_noise.npy', mmap_mode='r')
    policy, cfg, weights = load_policy(checkpoint, args.device)
    assert (policy.horizon, policy.n_obs_steps, policy.n_action_steps, policy.num_inference_steps) == (16, 2, 8, 10)
    verify_repository_imports(ROOT)
    begin = time.monotonic()
    with torch.inference_mode():
        obs = {key: torch.tensor(value[:2], device=args.device).to(policy.dtype) for key, value in observations.items()}
        parity_error = native_parity_check(policy, obs, torch.tensor(noise[:2, 0], device=args.device))
    print(f'{args.model}: native inference parity max error {parity_error:.3g}', flush=True)
    def sample(window_indices, draws, n_actions, label):
        n, d = len(window_indices), draws.shape[1]
        result = {
            'action_pred': np.empty((n, n_actions, 16, 7), np.float32),
            'source_raw': np.empty((n, d, 16, 7), np.float32),
            'source_norm': np.empty((n, d, 16, 7), np.float32),
            'heading_direction': np.full((n, 2), np.nan, np.float32),
            'heading_confidence': np.full(n, np.nan, np.float32),
            'source_active': np.zeros((n, d), bool),
        }
        if label == 'representatives':
            result['flow_mid_raw'] = np.empty((n, n_actions, 16, 7), np.float32)
        with torch.inference_mode():
            for start in range(0, n, args.windows_batch):
                ix = np.array(window_indices[start:start+args.windows_batch])
                obs = {key: torch.tensor(value[ix], device=args.device).to(policy.dtype) for key, value in observations.items()}
                cond, prediction = encode(policy, obs)
                if prediction is not None:
                    result['heading_direction'][start:start+len(ix)] = prediction['direction'].cpu().numpy()
                    result['heading_confidence'][start:start+len(ix)] = prediction['confidence'].cpu().numpy()
                for offset, wi in enumerate(ix):
                    r = start + offset
                    pred = None if prediction is None else {key: value[offset:offset+1].expand(d, *value.shape[1:]) for key, value in prediction.items()}
                    eps = torch.tensor(np.array(draws[r]), device=args.device)
                    source, active = source_from_noise(policy, eps, pred)
                    result['source_norm'][r] = source.cpu().numpy()
                    result['source_raw'][r] = policy.normalizer['action'].unnormalize(source).cpu().numpy()
                    result['source_active'][r] = active.cpu().numpy()
                    torch.testing.assert_close(source[..., 2:], (eps * policy.prior_noise_scale)[..., 2:], rtol=0, atol=0)
                    if args.model == 'heading_zero_dit' and bool(active.any()):
                        xy = policy.normalizer['action'].unnormalize(source)[active, :, :2].sum(1)
                        xy = xy / xy.norm(dim=-1, keepdim=True)
                        torch.testing.assert_close(xy, pred['direction'][active], atol=4e-5, rtol=4e-5)
                    for draw_start in range(0, n_actions, args.draws_batch):
                        count = min(args.draws_batch, n_actions - draw_start)
                        pred_action, mid = integrate(policy, source[draw_start:draw_start+count], cond[offset:offset+1].expand(count, -1, -1))
                        if not torch.isfinite(pred_action).all():
                            raise ValueError('Nonfinite generated actions')
                        result['action_pred'][r, draw_start:draw_start+count] = pred_action.cpu().numpy()
                        if 'flow_mid_raw' in result:
                            result['flow_mid_raw'][r, draw_start:draw_start+count] = mid.cpu().numpy()
                if start % 64 == 0 or start + len(ix) == n:
                    print(f'{args.model} {label}: {start+len(ix)}/{n}; elapsed {time.monotonic()-begin:.1f}s', flush=True)
        return result
    if args.smoke:
        sample([0, 1], noise[:2, :2], 2, 'smoke')
        print('SMOKE PASSED', flush=True)
        return
    result = sample(list(range(len(selection['windows']))), noise, 32, 'metrics')
    save_arrays(path, **result)
    rep_indices = [r['window_index'] for r in selection['representatives']]
    representatives = sample(rep_indices, rep_noise, 128, 'representatives')
    # Inference grouping may produce floating rounding differences; source pairing is exact.
    np.testing.assert_allclose(representatives['source_norm'][:, :32], result['source_norm'][rep_indices], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(representatives['action_pred'][:, :32], result['action_pred'][rep_indices], rtol=3e-5, atol=3e-6)
    save_arrays(out / 'models' / f'{args.model}_representatives.npz', **representatives)
    normalizer = policy.normalizer['action'].params_dict
    metadata = {**model_meta, 'model_id': args.model, 'weights': weights,
                'heading_xy_rms': float(policy.heading_xy_rms) if hasattr(policy, 'heading_xy_rms') else None,
                'min_target_confidence': float(policy.min_target_confidence) if hasattr(policy, 'min_target_confidence') else None,
                'min_source_confidence': float(policy.min_source_confidence) if hasattr(policy, 'min_source_confidence') else None,
                'normalizer_scale': normalizer['scale'].detach().cpu().tolist(),
                'normalizer_offset': normalizer['offset'].detach().cpu().tolist(),
                'native_inference_max_absolute_difference': parity_error,
                'paired_source_nonxy_bit_identical': True, 'representative_first32_draws_match': True,
                'inference_inputs': 'Saved observation windows and explicit independent Gaussian noise; no future actions or geometry.',
                'device': args.device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'elapsed_seconds': time.monotonic()-begin, 'script_sha256': file_sha256(__file__),
                'data_sha256': file_sha256(out / 'data.npz'), 'noise_sha256': file_sha256(out / 'noise.npy'),
                'result_sha256': file_sha256(path), 'representative_result_sha256': file_sha256(out / 'models' / f'{args.model}_representatives.npz'),
                'complete': True}
    assert file_sha256(checkpoint) == model_meta['checkpoint_sha256']
    write_json(out / 'models' / f'{args.model}.json', metadata)
    print(f'COMPLETE {args.model}: {metadata["elapsed_seconds"]:.1f}s', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'infer'])
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--model', choices=list(EVAL_DIRS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--windows-per-task', type=int, default=128)
    p.add_argument('--windows-batch', type=int, default=8)
    p.add_argument('--draws-batch', type=int, default=32)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.stage == 'prepare':
        prepare(args)
    else:
        if args.model is None:
            p.error('--model is required for inference')
        infer(args)


if __name__ == '__main__':
    main()
