"""Verify released MimicGen demos against the training rollout adapter."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import multiprocessing as mp
import traceback

TASKS = ['stack_three_d1','square_d2','threading_d1','hammer_cleanup_d1','mug_cleanup_d1','coffee_d2']

def run_one(item):
    task_name, gpu, dataset_root, demo_index = item
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    from oat.env.mimicgen.env import MimicGenEnv, initial_state_sha256
    import h5py
    import numpy as np
    import cv2
    env = None
    try:
        source = Path(dataset_root) / (task_name+'.hdf5')
        with h5py.File(source,'r') as f:
            group = f[f'data/demo_{demo_index}']
            states = group['states'][:]
            actions = group['actions'][:]
            model = group.attrs['model_file']
            env_meta = json.loads(f['data'].attrs['env_args'])
            images = {cam: group['obs'][cam+'_image'][0] for cam in ['agentview','robot0_eye_in_hand']}
            proprio = {port: group['obs'][port][0] for port in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}
        digest = initial_state_sha256(states[0])
        with tempfile.TemporaryDirectory(prefix='mimicgen-smoke-') as td:
            target = Path(td)/'eval.hdf5'
            with h5py.File(target,'w') as f:
                group = f.create_group(f'data/{task_name}/demo_{demo_index}')
                group.create_dataset('states',data=states[0])
                group.attrs['model_file'] = model
            task = {'task_name':task_name,'task_uid':TASKS.index(task_name),'horizon':len(actions)+1,'env_meta':env_meta,'train_initial_state_sha256':[], 'eval_demo_names':[f'demo_{demo_index}']}
            env = MimicGenEnv(task,target,render_gpu_device_id=gpu,native_image_size=84,image_size=128)
            episode = {'task_name':task_name,'demo_name':f'demo_{demo_index}','env_seed':20260912,'initial_state_sha256':digest}
            env.set_episode(episode)
            obs,_ = env.reset()
            metrics = {}
            for cam, source_image in images.items():
                expected = cv2.resize(source_image,(128,128),interpolation=cv2.INTER_LINEAR)
                rendered = obs[cam+'_rgb']
                metrics[cam+'_correct_mae'] = float(np.abs(rendered.astype(float)-expected).mean())
                metrics[cam+'_wrong_flip_mae'] = float(np.abs(rendered[::-1].astype(float)-expected).mean())
            metrics['initial_proprio_max_error'] = max(float(np.abs(obs[k]-v).max()) for k,v in proprio.items())
            first_step_error = None
            for index, action in enumerate(actions):
                obs,reward,done,_,info = env.step(action)
                if index == 0:
                    actual = env.env.sim.get_state().flatten()
                    first_step_error = float(np.max(np.abs(actual-states[1])))
                if done:
                    break
            return {'task':task_name,'gpu':gpu,'source_env_version':env_meta.get('env_version'),'source_steps':len(actions),'replay_steps':index+1,'success':bool(reward),'first_step_state_max_error':first_step_error,**metrics}
    except Exception:
        return {'task':task_name,'gpu':gpu,'error':traceback.format_exc()}
    finally:
        if env is not None:
            env.close()

if __name__ == '__main__':
    import argparse
    from importlib.metadata import version
    parser = argparse.ArgumentParser(description='Replay held-out MimicGen source actions and compare observations.')
    parser.add_argument('--dataset-root', default='data/mimicgen/hdf5/core')
    parser.add_argument('--demo-index', type=int, default=100)
    parser.add_argument('--gpus', type=int, nargs='+', default=[4, 5])
    parser.add_argument('--output', default='output/mimicgen6/verification/replay.json')
    args = parser.parse_args()
    ctx = mp.get_context('spawn')
    results = []
    for start in range(0, len(TASKS), len(args.gpus)):
        assigned = TASKS[start:start + len(args.gpus)]
        with ctx.Pool(len(assigned), maxtasksperchild=1) as pool:
            result = pool.map(run_one, [(name, args.gpus[index], str(Path(args.dataset_root).resolve()), args.demo_index)
                                       for index, name in enumerate(assigned)])
        results.extend(result)
        print(json.dumps(result), flush=True)
    report = {'versions': {name: version(name) for name in ['robosuite', 'mujoco', 'numpy']},
              'demo_index': args.demo_index, 'results': results,
              'complete': all(row.get('success', False) and 'error' not in row for row in results)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['complete'] else 1)
