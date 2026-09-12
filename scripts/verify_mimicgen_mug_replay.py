"""Characterize open-loop replay sensitivity on six fixed held-out Mug demos."""
import os,json,sys,tempfile,traceback,multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def probe(pair):
    index,gpu=pair
    os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl';os.environ['MUJOCO_EGL_DEVICE_ID']=str(gpu)
    import h5py,numpy as np
    from oat.env.mimicgen.env import MimicGenEnv,initial_state_sha256
    env=None
    try:
        task_name='mug_cleanup_d1';demo=f'demo_{index}'
        with h5py.File(Path(__file__).resolve().parents[1]/'data/mimicgen/hdf5/core/mug_cleanup_d1.hdf5','r') as f:
            g=f['data/'+demo];states=g['states'][:];actions=g['actions'][:];model=g.attrs['model_file'];meta=json.loads(f['data'].attrs['env_args'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'eval.hdf5'
            with h5py.File(path,'w') as f:
                g=f.create_group(f'data/{task_name}/{demo}');g.create_dataset('states',data=states[0]);g.attrs['model_file']=model
            task={'task_name':task_name,'task_uid':4,'horizon':len(actions)+1,'env_meta':meta,'train_initial_state_sha256':[],'eval_demo_names':[demo]}
            env=MimicGenEnv(task,path,render_gpu_device_id=gpu)
            ep={'task_name':task_name,'demo_name':demo,'env_seed':20260912,'initial_state_sha256':initial_state_sha256(states[0])}
            env.set_episode(ep);env.reset()
            success=False
            for action in actions:
                _,reward,_,_,_=env.step(action);success=success or bool(reward)
            # Source final state and last action distinguish replay drift from bad task success logic.
            env.env.sim.set_state_from_flattened(states[-1]);env.env.sim.forward()
            for robot in env.env.robots:
                robot.controller.update(force=True);robot.controller.reset_goal()
            final_state_success=bool(env.env._check_success())
            env.env.step(actions[-1]);final_step_success=bool(env.env._check_success())
            # Exact stored float64 controls isolate precision loss from wrapper/dataset float32 actions.
            env.reset();native_success=False
            for action in actions:
                env.env.step(action);native_success=native_success or bool(env.env._check_success())
            return {'demo':demo,'source_action_dtype':str(actions.dtype),'float32_openloop_success':success,'source_final_state_success':final_state_success,'source_final_step_success':final_step_success,'native_dtype_openloop_success':native_success}
    except Exception:
        return {'demo':index,'error':traceback.format_exc()}
    finally:
        if env is not None:env.close()

if __name__=='__main__':
    results=[]
    for start in range(100,106,2):
        with mp.get_context('spawn').Pool(2,maxtasksperchild=1) as pool:rows=pool.map(probe,[(start,4),(start+1,5)])
        results+=rows;print(json.dumps(rows),flush=True)
    (Path(__file__).resolve().parents[1]/'output/mimicgen6/verification/mug_replay_probe.json').write_text(json.dumps(results,indent=2)+'\n')
