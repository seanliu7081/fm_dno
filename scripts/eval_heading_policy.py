#!/usr/bin/env python3
"""Evaluate one immutable flow-policy checkpoint on official LIBERO states.

Example (100-episode pilot):
  CUDA_VISIBLE_DEVICES=3 /venv/fm_dno/bin/python scripts/eval_heading_policy.py \
    -c output/run/checkpoints/ep-0200.ckpt -o output/eval/pilot \
    --n-per-task 10 --seed 20260910 --workers 4

The final 500-episode evaluation uses --n-per-task 50. The checkpoint and
inference configuration are fixed for every task. Partial/failed runs never
report a complete benchmark success rate. --dry-run validates the complete
protocol and saves a checkpoint snapshot without constructing GPU simulators.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from oat.env_runner.libero_official_eval import (
    PROTOCOL_VERSION, array_sha256, assign_video_paths, evaluate_episode, file_sha256,
    initialize_worker, load_official_states, make_episode_plan, summarize_results,
    verify_repository_imports,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10", choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"])
    parser.add_argument("--n-per-task", type=int, default=50)
    parser.add_argument("--init-start", type=int, default=0, help="first official initial-state index; never wraps/repeats")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--video-per-task", type=int, default=0, help="record first N planned episodes/task using existing agentview observations")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device inside CUDA_VISIBLE_DEVICES")
    parser.add_argument("--egl-device", type=int, default=None, help="physical EGL GPU index (defaults to the selected visible GPU)")
    parser.add_argument("--max-episode-steps", type=int, default=550)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or args.torch_threads < 1 or args.max_episode_steps < 1 or args.settle_steps < 0:
        raise ValueError("workers, threads and episode steps must be positive; settle steps nonnegative")
    if not 0 <= args.video_per_task <= args.n_per_task:
        raise ValueError("video_per_task must be between zero and n_per_task")
    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("--checkpoint must be one checkpoint file, not a directory")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new run directory: {output}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    elif "MUJOCO_EGL_DEVICE_ID" not in os.environ and visible:
        ordinal = int(args.device.split(":")[-1]) if args.device.startswith("cuda:") else 0
        devices = visible.split(",")
        if ordinal < len(devices) and devices[ordinal].isdigit():
            os.environ["MUJOCO_EGL_DEVICE_ID"] = devices[ordinal]

    # All imports that may select a renderer occur after EGL configuration.
    import dill
    import torch
    from libero.libero import benchmark, get_libero_path
    from oat.common.hydra_util import register_new_resolvers
    from omegaconf import OmegaConf
    register_new_resolvers()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    names = suite.get_task_names()
    plan = make_episode_plan(names, args.n_per_task, args.seed, args.init_start)
    plan = assign_video_paths(plan, args.video_per_task, output / "videos")
    states_by_task = {}
    initial_files = {}
    for task_index, task_name in enumerate(names):
        states = load_official_states(suite, task_index)
        if args.init_start + args.n_per_task > len(states):
            raise ValueError(f"{task_name} has only {len(states)} official states; requested through {args.init_start + args.n_per_task - 1}")
        states_by_task[task_index] = states
        task = suite.get_task(task_index)
        state_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        initial_files[task_name] = {"path": str(state_path), "sha256": file_sha256(state_path), "available_states": len(states)}
    for episode in plan:
        episode["official_state_sha256"] = array_sha256(states_by_task[episode["task_index"]][episode["init_index"]])

    output.mkdir(parents=True)
    snapshot = output / "policy.ckpt"
    original_hash = file_sha256(checkpoint)
    shutil.copyfile(checkpoint, snapshot)
    checkpoint_hash = file_sha256(snapshot)
    if checkpoint_hash != original_hash or file_sha256(checkpoint) != original_hash:
        raise RuntimeError("source checkpoint changed while snapshotting; retry with a finished checkpoint")
    snapshot.chmod(0o444)
    with snapshot.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    weight_name = "ema_model" if bool(cfg.training.get("use_ema", False)) else "model"
    if weight_name not in payload["state_dicts"]:
        raise ValueError(f"checkpoint has no configured {weight_name} state")
    policy_config = OmegaConf.to_container(cfg.policy, resolve=True)
    training_position = {key: dill.loads(payload["pickles"][key]) for key in ("epoch", "global_step") if key in payload.get("pickles", {})}
    del payload
    source_files = sorted((ROOT_DIR / "oat").rglob("*.py")) + [Path(__file__).resolve()]
    code_hashes = {str(path.relative_to(ROOT_DIR)): file_sha256(path) for path in source_files}
    versions = {}
    for package in ("torch", "torchvision", "mujoco", "robosuite", "libero", "numpy", "hydra-core"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    settings = {
        "checkpoint": str(snapshot), "checkpoint_sha256": checkpoint_hash,
        "repository_root": str(ROOT_DIR),
        "suite": args.suite, "device": args.device, "torch_threads": args.torch_threads,
        "max_episode_steps": args.max_episode_steps, "settle_steps": args.settle_steps,
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint), "checkpoint": str(snapshot),
        "checkpoint_sha256": checkpoint_hash, "weights": weight_name,
        "training_position": training_position, "policy_config": policy_config,
        "suite": args.suite, "n_per_task": args.n_per_task, "init_start": args.init_start,
        "seed": args.seed, "workers": args.workers, "device": args.device,
        "cuda_visible_devices": visible, "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "max_episode_steps": args.max_episode_steps, "settle_steps": args.settle_steps,
        "video_per_task": args.video_per_task,
        "video": {"camera_observation": "agentview_rgb", "fps": 20,
                  "selection": "first N planned episodes per task, chosen before outcomes",
                  "frames": "existing observation after settling and each executed physical step; no extra renders"},
        "official_initial_state_files": initial_files, "episodes": plan,
        "software_versions": versions, "source_sha256": code_hashes,
        "actual_oat_module_paths": verify_repository_imports(ROOT_DIR),
        "checkpoint_selection": "One fixed snapshot for every task; evaluator performs no checkpoint or action-candidate selection.",
        "uncertainty_note": "Wilson interval covers episode sampling only, not training-seed or model-selection uncertainty.",
    }
    write_json(output / "manifest.json", manifest)
    print(f"Checkpoint SHA256: {checkpoint_hash}", flush=True)
    print(f"Protocol: {len(names)} tasks x {args.n_per_task} official initial states; indices {args.init_start}..{args.init_start + args.n_per_task - 1}", flush=True)
    if args.dry_run:
        print(f"Validated plan and checkpoint snapshot: {output / 'manifest.json'}", flush=True)
        return 0

    results = []
    try:
        with (output / "episodes.jsonl").open("w", buffering=1) as log:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker, initargs=(settings,)) as pool:
                pending = {pool.submit(evaluate_episode, episode): episode for episode in plan}
                for future in as_completed(pending):
                    episode = pending[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {**episode, "success": False, "error": f"worker failed: {type(error).__name__}: {error}"}
                    results.append(result)
                    log.write(json.dumps(result, sort_keys=True) + "\n")
                    summary = summarize_results(plan, results)
                    summary.update({"checkpoint_sha256": checkpoint_hash, "weights": weight_name, "manifest": str(output / "manifest.json")})
                    write_json(output / "summary.json", summary)
                    status = result["error"] or ("success" if result["success"] else "failure")
                    print(f"[{len(results)}/{len(plan)}] {episode['episode_id']}: {status}; successes {summary['successes']}/{summary['completed_episodes']}", flush=True)
    finally:
        summary = summarize_results(plan, results)
        summary.update({"checkpoint_sha256": checkpoint_hash, "weights": weight_name, "manifest": str(output / "manifest.json")})
        if file_sha256(snapshot) != checkpoint_hash:
            summary["complete"] = False
            summary["mean_success_rate"] = None
            summary["checkpoint_integrity_error"] = True
        write_json(output / "summary.json", summary)
    if summary["complete"]:
        print(f"Task-balanced success rate: {summary['mean_success_rate']:.3%} ({summary['successes']}/{summary['expected_episodes']}); {output / 'summary.json'}", flush=True)
        return 0
    print(f"Evaluation incomplete; benchmark success rate withheld. See {output / 'summary.json'}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
