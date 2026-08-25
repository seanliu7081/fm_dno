#!/usr/bin/env python3
"""
LIBERO-10 evaluation of a frozen policy under orbit-structured noise optimization.

This is the arm of the experiment that has to run on the benchmark rather than on
synthetic chunks: the 2x2 of {iid, orbit-coupled} checkpoint x {DNO on, off}, reported as
success rate on LIBERO-10, with wall-clock per control cycle next to it.

Run the SAME checkpoint twice -- once with ``--no-dno`` and once without -- so the DNO
comparison is exactly matched (same weights, same env seeds, same everything).

Stage-1-only (``--n-grad-steps 0``) is the configuration to run first:
  * it needs no gradients, so the stock ``LiberoRunner`` drives it;
  * it costs one batched forward pass of K rotations per cycle;
  * it is the configuration that isolates whether the coupling made the noise space
    steerable, which is the design's central claim.
Adding ``--n-grad-steps > 0`` switches automatically to ``OrbitDnoLiberoRunner``, which is
the same rollout loop without ``torch.inference_mode``.

Usage:
    # baseline arm (no DNO), 100 episodes
    MUJOCO_GL=egl python scripts/eval_orbit_dno_libero.py \
        -c output/<run>/checkpoints/latest.ckpt -o output/dno/P5_nodno \
        --no-dno --n-test 100

    # stage-1 orbit search, K=32
    MUJOCO_GL=egl python scripts/eval_orbit_dno_libero.py \
        -c output/<run>/checkpoints/latest.ckpt -o output/dno/P5_stage1 \
        --n-orbit 32 --n-grad-steps 0 --n-test 100

    # stage 1 + stage 2
    MUJOCO_GL=egl python scripts/eval_orbit_dno_libero.py \
        -c output/<run>/checkpoints/latest.ckpt -o output/dno/P5_stage12 \
        --n-orbit 32 --n-grad-steps 6 --lr 0.05 --n-test 100 --n-parallel-envs 10
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf, open_dict

from oat.dno.orbit_dno import OrbitDNO
from oat.dno.task_losses import CompositeTaskLoss
from oat.policy.base_policy import BasePolicy
from oat.policy.dno_policy import DNOPolicy


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-n", "--num_exp", default=1, help="repeat the whole eval N times")
# ---- DNO ----------------------------------------------------------------------------
@click.option("--dno/--no-dno", "use_dno", default=True, help="the {DNO on, off} half of the 2x2")
@click.option("--inference-prior", default=None, type=click.Choice(["standard", "matched"]),
              help="P5: re-impose the coupling's own trained source heading law. Same weights "
                   "as P4, different prior -- a perfectly controlled comparison.")
@click.option("--num-inference-steps", default=None, type=int,
              help="override the policy's sampler steps for the low-step-budget sweep. Works "
                   "for plain FlowPolicy checkpoints too (with --no-dno).")
@click.option("--n-orbit", default=32, help="stage-1 angular grid size; 0 disables stage 1")
@click.option("--n-grad-steps", default=0, help="stage-2 iterations; >0 requires the DNO runner")
@click.option("--lr", default=0.05)
@click.option("--typicality-weight", default=0.02)
@click.option("--trust-weight", default=0.01)
@click.option("--optimizer", default="irrep_adam",
              type=click.Choice(["irrep_adam", "adam", "sgd"]))
@click.option("--no-warm-start", is_flag=True)
@click.option("--dno-steps", default=None, type=int, help="sampler steps inside DNO (default: policy's)")
# ---- task loss (Tier 0 by default: no privileged state) -------------------------------
@click.option("--w-smoothness", default=1.0)
@click.option("--w-seam", default=2.0)
@click.option("--w-step-limit", default=5.0)
@click.option("--w-table", default=0.0,
              help="absolute table plane. DEFAULT 0 (OFF) on LIBERO-10: the demonstrated eef "
                   "height is bimodal (gap at 0.815-0.889), so no scalar --table-z is both "
                   "safe and active. Use --w-descent instead.")
@click.option("--w-descent", default=10.0,
              help="relative clearance: penalise descending more than --max-descent below the "
                   "CURRENT eef height. Scene-height agnostic, so it works on both LIBERO-10 "
                   "scene families at one setting.")
@click.option("--max-descent", default=0.15,
              help="metres of descent allowed below the current eef height before penalty")
@click.option("--w-workspace", default=0.0)
@click.option("--w-gripper", default=0.0)
@click.option("--table-z", default=0.44,
              help="absolute table plane, only used when --w-table > 0. 0.44 sits below the "
                   "measured p01 (0.448) of demonstrated eef height; the plan's 0.82 "
                   "placeholder penalises every step of five of the ten tasks.")
@click.option("--trans-gain", default=0.05, help="OSC_POSE output_max for position -- VERIFY THIS")
@click.option("--workspace-lo", default="-0.35,-0.45,0.80")
@click.option("--workspace-hi", default="0.35,0.45,1.35")
# ---- runner -------------------------------------------------------------------------
@click.option("--n-test", default=100)
@click.option("--n-parallel-envs", default=10)
@click.option("--n-test-vis", default=4)
@click.option("--max-batch", default=256, help="chunking for the stage-1 grid")
def main(checkpoint, output_dir, device, num_exp, use_dno, inference_prior,
         num_inference_steps, n_orbit, n_grad_steps, lr,
         typicality_weight, trust_weight, optimizer, no_warm_start, dno_steps,
         w_smoothness, w_seam, w_step_limit, w_table, w_descent, max_descent,
         w_workspace, w_gripper,
         table_z, trans_gain, workspace_lo, workspace_hi,
         n_test, n_parallel_envs, n_test_vis, max_batch):

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    device = torch.device(device)
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    if use_dno and not hasattr(policy, "sample_chunk"):
        raise SystemExit(
            "This checkpoint's policy has no sample_chunk(); DNO needs an OrbitFlowPolicy. "
            "Use --no-dno for a plain FlowPolicy baseline."
        )

    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)
        print(f"sampler steps overridden to {policy.num_inference_steps}")

    if inference_prior is not None:
        if not hasattr(policy, "inference_prior"):
            raise SystemExit("--inference-prior needs an OrbitFlowPolicy checkpoint.")
        if inference_prior == "matched":
            total = float(policy.source_heading_hist.sum())
            if total <= 0:
                raise SystemExit(
                    "--inference-prior=matched but source_heading_hist is empty. Only a "
                    "rotation-applying arm (mode=rot / perm_rot) fills it."
                )
            nz = int((policy.source_heading_hist > 0).sum())
            print(f"matched prior: {total:.0f} samples over {nz}/"
                  f"{policy.source_heading_hist.numel()} bins")
        policy.inference_prior = inference_prior

    if w_table > 0:
        print(f"!! --w-table={w_table} with --table-z={table_z}: LIBERO-10's demonstrated "
              f"eef height is bimodal, so a scalar plane is either unsafe or inert. "
              f"--w-descent is the scene-height-agnostic term.")
    task_loss = CompositeTaskLoss(
        weights={k: v for k, v in dict(
            smoothness=w_smoothness, seam=w_seam, step_limit=w_step_limit,
            table=w_table, descent=w_descent, workspace=w_workspace, gripper=w_gripper,
        ).items() if v > 0},
        trans_gain=trans_gain,
        table_z=table_z,
        max_descent=max_descent,
        workspace_lo=[float(v) for v in workspace_lo.split(",")],
        workspace_hi=[float(v) for v in workspace_hi.split(",")],
    )

    eval_policy = policy
    dno_policy = None
    if use_dno:
        dno = OrbitDNO(
            policy, task_loss,
            n_orbit=n_orbit, n_grad_steps=n_grad_steps, lr=lr,
            typicality_weight=typicality_weight, trust_weight=trust_weight,
            n_steps=dno_steps, optimizer=optimizer,
            warm_start=not no_warm_start, max_batch=max_batch,
        )
        dno_policy = DNOPolicy(policy, dno, enabled=True)
        eval_policy = dno_policy.to(device)

    # ---- runner: stock unless stage 2 needs gradients --------------------------------
    runner_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.policy.env_runner, resolve=True))
    with open_dict(runner_cfg):
        runner_cfg.n_test = n_test
        runner_cfg.n_parallel_envs = n_parallel_envs
        runner_cfg.n_test_vis = n_test_vis
        if use_dno and n_grad_steps > 0:
            runner_cfg._target_ = "oat.env_runner.libero_dno_runner.OrbitDnoLiberoRunner"
    print(f"runner: {runner_cfg._target_}  n_test={n_test}  n_parallel_envs={n_parallel_envs}")
    env_runner = hydra.utils.instantiate(runner_cfg, output_dir=str(out))

    runs = []
    for i in range(num_exp):
        if dno_policy is not None:
            dno_policy.clear_log()
        log = env_runner.run(eval_policy)
        sr = float(log["mean_success_rate"])
        entry = {"mean_success_rate": sr}
        entry.update({k: float(v) for k, v in log.items()
                      if k.endswith("mean_success_rate") and k != "mean_success_rate"})
        if dno_policy is not None:
            entry.update(dno_policy.summarize())
        runs.append(entry)
        print(f"exp {i+1}/{num_exp}: success rate = {sr:.4f}"
              + (f"  |  dno p50 {entry.get('dno/wall_time_p50', float('nan')):.3f}s"
                 f"  p95 {entry.get('dno/wall_time_p95', float('nan')):.3f}s" if dno_policy else ""))
    env_runner.close()

    keys = sorted({k for r in runs for k in r})
    summary = {
        "checkpoint": checkpoint,
        "use_dno": use_dno,
        "inference_prior": inference_prior or "standard",
        "num_inference_steps": int(getattr(policy, "num_inference_steps", -1)),
        "dno": {"n_orbit": n_orbit, "n_grad_steps": n_grad_steps, "lr": lr,
                "optimizer": optimizer, "warm_start": not no_warm_start,
                "n_steps": dno_steps} if use_dno else None,
        "task_loss_weights": task_loss.weights,
        "trans_gain": trans_gain, "table_z": table_z, "max_descent": max_descent,
        "n_test": n_test, "num_exp": num_exp,
    }
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        summary[f"{k}_mean"] = float(np.mean(vals))
        if len(vals) > 1:
            summary[f"{k}_std"] = float(np.std(vals, ddof=1))
            summary[f"{k}_stderr"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    path = out / "dno_eval_log.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"\nwrote {path}")
    print(f"mean_success_rate = {summary['mean_success_rate_mean']:.4f}")


if __name__ == "__main__":
    main()
