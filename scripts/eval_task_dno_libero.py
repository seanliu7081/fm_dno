#!/usr/bin/env python3
"""Matched, sequential LIBERO evaluation of frozen E/F task-noise policies.

This evaluator intentionally obtains goal geometry from the simulator. Its scores
are a privileged-state, local kinematic proxy, not predicted task success. It
never launches the training workspace, optimizer, vector workers or W&B runs.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODES = ("baseline", "best_of_k", "dno", "amortized", "amortized_dno")
DEFAULT_TASK = "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"


def atomic_write(path: Path, writer) -> None:
    """Publish a complete artifact atomically; never replace an existing path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # link fails atomically if destination exists
    finally:
        os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, lambda stream: stream.write(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    ))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_episode(env, seed: int):
    # LiberoEnv.reset(seed=...) currently ignores its seed argument.
    seed_all(seed)
    env.env.seed(seed)
    obs, info = env.reset()
    # LiberoEnv.reset returns its observation before _let_objects_fall(). Use
    # the actual state after settling for both the policy and oracle context.
    obs = env._extract_obs()
    return obs, info


def state_fingerprint(env) -> str:
    simulator = env.env.sim
    digest = hashlib.sha256()
    arrays = {"state": simulator.get_state().flatten()}
    # Fixtures can be placed in model coordinates rather than free-joint qpos.
    for name in ("body_pos", "body_quat"):
        if hasattr(simulator.model, name):
            arrays[name] = getattr(simulator.model, name)
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def batch_observation(history, n_obs_steps: int, device, dtype):
    result = {}
    recent = list(history)[-n_obs_steps:]
    recent = [recent[0]] * (n_obs_steps - len(recent)) + recent
    for key, value in recent[-1].items():
        if isinstance(value, np.ndarray):
            stacked = np.stack([item[key] for item in recent])[None]
            result[key] = torch.as_tensor(stacked, device=device).to(dtype=dtype)
        else:
            # Existing runner exposes one prompt string per environment.
            result[key] = [value]
    return result


def batch_context(context, device, dtype):
    result = {}
    for key, value in context.items():
        if isinstance(value, (np.ndarray, np.number, float, int, bool)):
            tensor = torch.as_tensor(np.asarray(value), device=device).unsqueeze(0)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            result[key] = tensor
        else:
            result[key] = value
    return result


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def quantiles(values):
    return {"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))} if values else {"p50": 0.0, "p95": 0.0}


def _as_cpu_row(value):
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor[0].clone()


def run_episode(env, policy, context_builder, *, mode: str, episode_id: int,
                seed: int, max_episode_steps: int, expected_state_hash=None,
                collect_teacher=False):
    """Run only the returned action prefix, keeping true t -> t+m feedback."""
    obs, _ = reset_episode(env, seed)
    initial_hash = state_fingerprint(env)
    if expected_state_hash is not None and initial_hash != expected_state_hash:
        raise RuntimeError(f"Unmatched reset for episode {episode_id}: {initial_hash} != "
                           f"{expected_state_hash}. Results must not be compared.")
    policy.reset(seed=seed)
    policy.clear_log()
    history = deque([obs], maxlen=policy.n_obs_steps)
    context_state = {}
    records, latencies = [], []
    env_steps, episode_return, success = 0, 0.0, False
    terminal, truncated = False, False
    env_reported_terminated, time_limit_reached = False, False
    while env_steps < max_episode_steps and not (terminal or truncated):
        context = context_builder(env, state=context_state)
        batched_obs = batch_observation(history, policy.n_obs_steps, policy.device, policy.dtype)
        batched_context = batch_context(context, policy.device, policy.dtype)
        synchronize(policy.device)
        started = time.perf_counter()
        # No inference_mode/no_grad here: DNO differentiates the sampler input.
        output = policy.predict_action(batched_obs, task_context=batched_context)
        synchronize(policy.device)
        latencies.append(time.perf_counter() - started)
        action = output["action"].detach().cpu().numpy()[0]
        if action.ndim != 2 or action.shape[-1] != 7 or not np.isfinite(action).all():
            raise ValueError("Policy must return a finite [B, execution_prefix, 7] action.")
        if len(action) == 0 or len(action) > policy.n_action_steps:
            raise ValueError("Policy returned an empty prefix or exceeded n_action_steps.")
        action = action[:max_episode_steps - env_steps]
        start_step, chunk_return = env_steps, 0.0
        executed = []
        for act in action:
            obs, reward, env_reported_terminated, env_truncated, info = env.step(act)
            env_steps += 1
            history.append(obs)
            executed.append(act.copy())
            chunk_return += float(reward)
            episode_return += float(reward)
            success = success or bool(float(reward) >= 1.0)
            time_limit_reached = env_steps >= max_episode_steps
            # LiberoEnv folds its time limit into done. Preserve that raw flag,
            # but distinguish failed timeouts for consumers that bootstrap value
            # estimates. Success on the final allowed step remains terminal.
            timeout = time_limit_reached and not success
            terminal = bool(env_reported_terminated and not timeout)
            truncated = bool(env_truncated or timeout)
            if terminal or truncated or env_steps >= max_episode_steps:
                break
        if collect_teacher:
            required = ("cond", "base_noise", "optimized_noise", "context_features")
            missing = [key for key in required if key not in output]
            if missing:
                raise ValueError(f"Teacher recording requested but policy omitted {missing}.")
            row = {key: _as_cpu_row(output[key]) for key in required}
            row.update({
                "episode_id": torch.tensor(episode_id),
                "env_t": torch.tensor(start_step),
                "next_env_t": torch.tensor(env_steps),
                "executed_steps": torch.tensor(env_steps - start_step),
                "chunk_return": torch.tensor(chunk_return),
                "terminated": torch.tensor(bool(terminal)),
                "truncated": torch.tensor(truncated),
                "env_reported_terminated": torch.tensor(bool(env_reported_terminated)),
                "time_limit_reached": torch.tensor(time_limit_reached),
                "next_eef_pos": torch.as_tensor(np.array(obs["robot0_eef_pos"], copy=True)),
                "accepted": _as_cpu_row(output.get("teacher_improved", torch.tensor([False]))).bool(),
            })
            for key in ("initial_objective", "final_objective"):
                if key in output:
                    row[key] = _as_cpu_row(output[key])
            # Padding is only storage. executed_steps is the authoritative mask.
            padded_action = np.zeros((policy.n_action_steps, 7), dtype=np.float32)
            padded_action[:len(executed)] = np.asarray(executed)
            row["executed_action"] = torch.as_tensor(padded_action)
            records.append(row)
    for row in records:
        row["episode_return"] = torch.tensor(episode_return)
        row["episode_success"] = torch.tensor(success)
    result = {
        "mode": mode, "episode_id": episode_id, "seed": seed,
        "initial_state_sha256": initial_hash, "success": success,
        "episode_return": episode_return, "env_steps": env_steps,
        "control_cycles": len(latencies), "latency_seconds": quantiles(latencies),
        "latency_samples_seconds": latencies,
        "policy_stats": policy.summarize(),
        "terminated": bool(terminal),
        "truncated": truncated,
        "env_reported_terminated": bool(env_reported_terminated),
        "time_limit_reached": time_limit_reached,
    }
    return result, records


def load_frozen_checkpoint(path: Path, device):
    """Load only the policy state; fingerprint the same file handle we deserialize."""
    import dill
    import hydra
    from oat.common.hydra_util import register_new_resolvers
    register_new_resolvers()
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", pickle_module=dill, weights_only=False)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Checkpoint changed while loading. Use a completed epoch checkpoint.")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if bool(cfg.training.get("use_ema", False)) else "model"
    policy.load_state_dict(payload["state_dicts"][state_key], strict=True)
    source = {"condition": "E", "both": "F"}.get(getattr(policy, "reference_use", None))
    if source is None:
        raise ValueError("This evaluator requires an E (condition) or F (both) checkpoint.")
    if not bool(policy.obs_encoder.reference_loaded):
        raise ValueError("Checkpoint does not contain the initialized frozen heading reference.")
    policy.to(device).eval().requires_grad_(False)
    identity = {"source_policy": source, "base_checkpoint_sha256": digest.hexdigest()}
    return policy, cfg, identity, state_key


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML defaults; explicit CLI flags take precedence")
    parser.add_argument("-c", "--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--modes", default="baseline,best_of_k,dno")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-source", choices=("E", "F"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=550)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--num-grad-steps", type=int, default=3)
    parser.add_argument("--best-of-k-candidates", type=int,
                        help="Defaults to K*(grad_steps+2), matching DNO forward NFE; backward work differs")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--trust-weight", type=float, default=0.01)
    parser.add_argument("--trust-radius", type=float, default=0.5)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--gripper-weight", type=float, default=0.001)
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--teacher-mode", choices=("dno", "amortized_dno"),
                        help="Export nontrivial residual teachers; best_of_k only selects an unchanged source noise")
    parser.add_argument("--cpu-threads", type=int, default=4)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        import yaml
        with preliminary.config.open() as stream:
            defaults = yaml.safe_load(stream)
        if not isinstance(defaults, dict):
            parser.error("--config must contain a YAML mapping")
        known = {action.dest for action in parser._actions} - {"help", "config", "checkpoint", "output_dir"}
        unknown = set(defaults) - known
        if unknown:
            parser.error(f"Unknown config settings: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not args.modes or len(set(args.modes)) != len(args.modes) or any(m not in MODES for m in args.modes):
        parser.error(f"--modes must be a comma-separated unique subset of {MODES}")
    if any(m.startswith("amortized") for m in args.modes) and args.initializer is None:
        parser.error("amortized modes require --initializer")
    if args.teacher_mode not in (None, "dno", "amortized_dno"):
        parser.error("--teacher-mode must be dno or amortized_dno; best_of_k selects an unchanged source and cannot teach residual updates")
    if args.teacher_mode and args.teacher_mode not in args.modes:
        parser.error("--teacher-mode must be included in --modes")
    if min(args.episodes, args.max_episode_steps, args.num_candidates, args.cpu_threads) <= 0:
        parser.error("episodes, max-episode-steps, num-candidates and cpu-threads must be positive")
    if args.best_of_k_candidates is None:
        args.best_of_k_candidates = args.num_candidates * (args.num_grad_steps + 2)
    if args.best_of_k_candidates < 1:
        parser.error("--best-of-k-candidates must be positive")
    if any(mode in ("dno", "amortized_dno") for mode in args.modes) and args.num_grad_steps < 1:
        parser.error("DNO modes require at least one gradient step")
    if args.num_grad_steps < 0 or args.lr <= 0 or args.trust_weight < 0 or args.trust_radius < 0:
        parser.error("Invalid optimization hyperparameters")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(args.cpu_threads)
    from oat.env.libero.env import LiberoEnv
    from oat.dno.task_objectives import TaskGeometryObjective
    from oat.env.libero.dno_context import build_task_context
    from oat.policy.task_noise_policy import TaskNoisePolicy
    from oat.dno.noise_initializer import load_noise_initializer

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    base_policy, cfg, identity, state_key = load_frozen_checkpoint(args.checkpoint, args.device)
    if args.expected_source and identity["source_policy"] != args.expected_source:
        raise ValueError(f"Expected {args.expected_source}, checkpoint is {identity['source_policy']}")
    initializer = None
    if args.initializer:
        initializer = load_noise_initializer(args.initializer, expected_identity=identity,
                                             map_location=args.device).to(args.device)
    objective = TaskGeometryObjective(n_action_steps=base_policy.n_action_steps,
                                     position_weight=args.position_weight,
                                     gripper_weight=args.gripper_weight)
    runner_cfg = cfg.task.policy.env_runner
    env = LiberoEnv(task_name=args.task_name,
                    image_size=int(runner_cfg.get("image_size", 128)), seed=args.seed,
                    camera_names=list(runner_cfg.camera_names),
                    state_ports=list(runner_cfg.state_ports),
                    max_episode_steps=args.max_episode_steps)
    results, teacher_records, expected_hashes = [], [], {}
    metadata = {
        **identity, "checkpoint": str(args.checkpoint.resolve()), "weights": state_key,
        "task_name": args.task_name, "privileged_state": True,
        "evaluator": "simulator_goal_geometry_with_local_kinematic_proxy",
        "reward_source": "actual_LIBERO_check_success_after_each_executed_action",
        "comparison": "matched_env_reset_seeds_and_candidate_zero",
        "oracle_proxy_is_success_predictor": False,
        "termination_semantics": "unsuccessful evaluation time limits are truncated; env_reported_terminated preserves the raw LIBERO done flag",
        "n_obs_steps": base_policy.n_obs_steps, "n_action_steps": base_policy.n_action_steps,
        "horizon": base_policy.horizon,
        "sampler_steps": int(base_policy.num_inference_steps),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "manifest.json", metadata)
        for mode in args.modes:
            policy = TaskNoisePolicy(base_policy, objective, mode=mode,
                                     num_candidates=(args.best_of_k_candidates if mode == "best_of_k" else args.num_candidates),
                                     num_grad_steps=args.num_grad_steps, lr=args.lr,
                                     trust_weight=args.trust_weight, trust_radius=args.trust_radius,
                                     seed=args.seed, initializer=initializer,
                                     record_training_data=(mode == args.teacher_mode))
            for episode_id in range(args.episodes):
                result, records = run_episode(
                    env, policy, build_task_context, mode=mode, episode_id=episode_id,
                    seed=args.seed + episode_id, max_episode_steps=args.max_episode_steps,
                    expected_state_hash=expected_hashes.get(episode_id),
                    collect_teacher=(mode == args.teacher_mode),
                )
                expected_hashes.setdefault(episode_id, result["initial_state_sha256"])
                results.append(result)
                teacher_records.extend(records)
                write_json(args.output_dir / f"episode_{mode}_{episode_id:04d}.json", result)
                print(f"{mode} {episode_id + 1}/{args.episodes}: success={result['success']} "
                      f"steps={result['env_steps']} latency_p50={result['latency_seconds']['p50']:.3f}s",
                      flush=True)
        summary = {}
        for mode in args.modes:
            subset = [item for item in results if item["mode"] == mode]
            summary[mode] = {
                "episodes": len(subset),
                "success_rate": float(np.mean([item["success"] for item in subset])),
                "mean_env_steps": float(np.mean([item["env_steps"] for item in subset])),
                "latency_seconds": quantiles([x for item in subset for x in item["latency_samples_seconds"]]),
                "total_control_cycles": sum(item["control_cycles"] for item in subset),
                "total_forward_nfe": sum(item["policy_stats"].get("dno/nfe", 0) * item["control_cycles"] for item in subset),
                "total_backward_steps": sum(item["policy_stats"].get("dno/backward_steps", 0) * item["control_cycles"] for item in subset),
                "nfe_definition": "candidate-weighted Euler forward evaluations; excludes backward FLOPs",
            }
        if teacher_records:
            archive = {key: torch.stack([row[key] for row in teacher_records])
                       for key in teacher_records[0]}
            archive["meta"] = {**metadata, "teacher_mode": args.teacher_mode,
                               "accepted_semantics": "proxy_objective_improved; not actual task success",
                               "actual_feedback": "episode_return/episode_success and chunk_return are executed feedback"}
            atomic_write(args.output_dir / "teacher_data.pt", lambda stream: torch.save(archive, stream))
        write_json(args.output_dir / "results.json", {**metadata, "summary": summary, "episodes": results})
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
