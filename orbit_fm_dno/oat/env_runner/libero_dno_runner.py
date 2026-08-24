"""
A LIBERO runner that lets the policy use gradients.

``LiberoRunner.run`` is ``@torch.inference_mode()`` and additionally wraps the policy call
in ``with torch.inference_mode():``.  That is correct and fast for a normal policy, and it
makes stage 2 of orbit DNO impossible -- tensors created under inference mode cannot enter
an autograd graph, so backpropagating through the sampler raises at the first op.

This subclass overrides ``run`` with the same rollout loop minus the inference-mode guards.
Everything else -- env construction, chunking, seeding, video capture, logging -- is
inherited untouched, so a DNO rollout is comparable to a stock rollout episode for episode.

Only needed when ``OrbitDNO.n_grad_steps > 0``.  Stage-1-only DNO (the orbit grid search) is
pure forward evaluation and runs under the stock runner; prefer that first, since it is also
the configuration that isolates the coupling's contribution.

Costs more memory than a normal rollout: the graph through N Euler steps of the transformer
is held for every parallel env at once.  If it OOMs, lower ``n_parallel_envs`` before
lowering ``num_inference_steps`` -- the latter changes what you are evaluating.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import tqdm
import wandb

from oat.common.pytorch_util import dict_apply
from oat.env_runner.libero_runner import LiberoRunner, maybe_to_torch
from oat.policy.base_policy import BasePolicy


class OrbitDnoLiberoRunner(LiberoRunner):
    """``LiberoRunner`` with the rollout loop run under ``torch.enable_grad()``."""

    def run(self, policy: BasePolicy, **kwargs):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_success = [False] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            self.env.call_each(
                "run_dill_function", args_list=[(x,) for x in this_init_fns]
            )

            obs, _ = self.env.reset()
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} in Libero::{self.task_name} {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            while not done and pbar.n < pbar.total:
                obs_dict = dict_apply(
                    obs, lambda x: maybe_to_torch(x, device=device, dtype=dtype)
                )

                # >>> the only substantive difference from LiberoRunner.run <<<
                with torch.enable_grad():
                    action = (
                        policy.predict_action(
                            {p: obs_dict[p] for p in policy.get_observation_ports()},
                            **kwargs,
                        )["action"]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN of Inf action")

                obs, reward, done, _, _ = self.env.step(action)
                done = np.logical_or(
                    done[this_local_slice],
                    all_success[this_global_slice][this_local_slice],
                )
                done = np.all(done[this_local_slice])

                all_success[this_global_slice] = np.logical_or(
                    all_success[this_global_slice],
                    [r >= 1 for r in reward[this_local_slice]],
                )

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = self.env.render()[this_local_slice]

        _ = self.env.reset()

        log_data = dict()
        for task_name in set(self.env_task_names):
            task_success = [
                all_success[i]
                for i in range(n_inits)
                if self.env_task_names[i] == task_name
            ]
            log_data[f"{task_name}/mean_success_rate"] = np.mean(task_success)

        for i in range(n_inits):
            seed = self.env_seeds[i]
            task_name = self.env_task_names[i]
            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[f"{task_name}/video_{seed}"] = wandb.Video(video_path, format="mp4")

        log_data["mean_success_rate"] = np.mean(all_success)
        return log_data
