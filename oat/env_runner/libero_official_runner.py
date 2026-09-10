"""Synchronous official LIBERO evaluation of a caller-selected policy snapshot.

The training process never instantiates another policy or simulator. Each completed
rollout is reusable only after its immutable tensors, protocol and raw rows pass
validation again. Incomplete attempts are retained and a new attempt is created.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import dill
from omegaconf import OmegaConf
import torch

from oat.env_runner.base_runner import BaseRunner
from oat.env_runner.libero_official_eval import (
    PROTOCOL_VERSION, assign_video_paths, file_sha256, make_episode_plan,
    summarize_results,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE_TASKS = {"libero_10": 10, "libero_spatial": 10, "libero_object": 10,
               "libero_goal": 10, "libero_90": 90}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _ordinary_path(path):
    _require(not path.is_symlink(), f"refusing symlink artifact: {path}")


@contextmanager
def _exclusive_epoch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".runner.lock"
    _ordinary_path(lock)
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another evaluation owns {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class LiberoOfficialRunner(BaseRunner):
    def __init__(self, output_dir, suite="libero_10", n_per_task=50,
                 init_start=0, seed=20260911, workers=8, eval_gpu=1,
                 max_episode_steps=550, settle_steps=10, video_per_task=1,
                 torch_threads=1):
        super().__init__(output_dir)
        if suite not in SUITE_TASKS:
            raise ValueError(f"unsupported official suite: {suite}")
        values = (n_per_task, init_start, seed, workers, eval_gpu,
                  max_episode_steps, settle_steps, video_per_task, torch_threads)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("runner counts, seeds and GPU index must be integers")
        if min(n_per_task, workers, max_episode_steps, torch_threads) < 1:
            raise ValueError("episode counts, workers, steps and threads must be positive")
        if min(init_start, eval_gpu, settle_steps, video_per_task) < 0 or video_per_task > n_per_task:
            raise ValueError("invalid initial state, GPU, settle or video setting")
        self.protocol = dict(suite=suite, n_per_task=n_per_task, init_start=init_start,
                             seed=seed, workers=workers, max_episode_steps=max_episode_steps,
                             settle_steps=settle_steps, video_per_task=video_per_task)
        self.eval_gpu = eval_gpu
        self.torch_threads = torch_threads

    def run(self, policy, **kwargs):
        raise RuntimeError("use run_checkpoint(policy, cfg, epoch, global_step) to fix the evaluation context")

    def close(self):
        pass

    @staticmethod
    def _inference_config(cfg):
        # Resolve only fields consumed by the official evaluator. This also makes
        # reuse independent of logging paths or the training resume flag.
        runner = cfg.task.policy.env_runner
        return {
            "policy": OmegaConf.to_container(cfg.policy, resolve=True),
            "training": {"use_ema": bool(cfg.training.get("use_ema", False))},
            "task": {"policy": {"env_runner": {
                "image_size": int(runner.get("image_size", 128)),
                "camera_names": list(runner.get("camera_names", ["agentview", "robot0_eye_in_hand"])),
                "state_ports": list(runner.get("state_ports", ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"])),
            }}},
        }

    def _snapshot(self, directory, policy, cfg, epoch, global_step):
        inference = self._inference_config(cfg)
        state_name = "ema_model" if inference["training"]["use_ema"] else "model"
        position = {"epoch": int(epoch), "global_step": int(global_step)}
        # Copy buffers as well as parameters; detach/clone never initializes a
        # model or advances any Python, NumPy, CPU or CUDA random generator.
        selected = policy.state_dict()
        _require(selected and all(isinstance(value, torch.Tensor) for value in selected.values()),
                 "selected policy must have a nonempty tensor state_dict")
        selected = {name: value.detach().cpu().clone() for name, value in selected.items()}
        checkpoint = directory / "policy.ckpt"
        _ordinary_path(checkpoint)
        if not checkpoint.exists():
            payload = {"cfg": OmegaConf.create(inference), "state_dicts": {state_name: selected},
                       "pickles": {name: dill.dumps(value) for name, value in position.items()}}
            temporary = directory / "policy.ckpt.pending"
            with temporary.open("xb") as stream:
                torch.save(payload, stream, pickle_module=dill)
            temporary.rename(checkpoint)
            checkpoint.chmod(0o444)
        initial_hash = file_sha256(checkpoint)
        with checkpoint.open("rb") as stream:
            loaded = torch.load(stream, map_location="cpu", pickle_module=dill)
        _require(OmegaConf.to_container(loaded["cfg"], resolve=True) == inference,
                 "conflicting snapshot inference configuration")
        _require(set(loaded["state_dicts"]) == {state_name}, "conflicting snapshot selected weights")
        _require({name: dill.loads(loaded["pickles"][name]) for name in position} == position,
                 "conflicting snapshot training position")
        actual = loaded["state_dicts"][state_name]
        _require(set(actual) == set(selected), "snapshot tensor keys differ")
        for name, expected in selected.items():
            value = actual[name]
            _require(value.device.type == "cpu" and value.dtype == expected.dtype
                     and value.shape == expected.shape and torch.equal(value, expected),
                     f"snapshot tensor differs: {name}")
        _require(file_sha256(checkpoint) == initial_hash, "snapshot changed during verification")
        evidence = {"checkpoint_sha256": initial_hash, "weights": state_name,
                    "training_position": position, "completed_epochs": int(epoch) + 1,
                    "tensor_count": len(selected), "exact_tensor_roundtrip": True,
                    "inference_config": inference, "protocol": self.protocol,
                    "eval_gpu": self.eval_gpu, "torch_threads": self.torch_threads}
        record = directory / "snapshot.json"
        _ordinary_path(record)
        if record.exists():
            _require(_read_json(record) == evidence, "conflicting snapshot/protocol evidence")
        else:
            _write_new_json(record, evidence)
        return checkpoint, evidence

    def _verify_result(self, output, checkpoint, evidence):
        for filename in ("manifest.json", "summary.json", "episodes.jsonl", "policy.ckpt"):
            path = output / filename
            _ordinary_path(path)
            _require(path.is_file(), f"missing evaluation artifact: {path}")
        manifest = _read_json(output / "manifest.json")
        summary = _read_json(output / "summary.json")
        digest = evidence["checkpoint_sha256"]
        _require(file_sha256(checkpoint) == digest and file_sha256(output / "policy.ckpt") == digest,
                 "evaluation checkpoint hash mismatch")
        expected_fields = {**self.protocol, "protocol_version": PROTOCOL_VERSION,
                           "checkpoint_sha256": digest, "weights": evidence["weights"],
                           "training_position": evidence["training_position"],
                           "policy_config": evidence["inference_config"]["policy"],
                           "source_checkpoint": str(checkpoint), "checkpoint": str(output / "policy.ckpt"),
                           "device": "cuda:0", "cuda_visible_devices": str(self.eval_gpu),
                           "egl_device": str(self.eval_gpu)}
        for name, expected in expected_fields.items():
            _require(manifest.get(name) == expected, f"evaluation manifest mismatch: {name}")
        plan = manifest["episodes"]
        task_names = {}
        for entry in plan:
            index, name = entry["task_index"], entry["task_name"]
            _require(index not in task_names or task_names[index] == name, "conflicting task names")
            task_names[index] = name
        count = SUITE_TASKS[self.protocol["suite"]]
        _require(set(task_names) == set(range(count)), "evaluation does not cover the complete suite")
        expected_plan = make_episode_plan([task_names[index] for index in range(count)],
                                         self.protocol["n_per_task"], self.protocol["seed"],
                                         self.protocol["init_start"])
        expected_plan = assign_video_paths(expected_plan, self.protocol["video_per_task"], output / "videos")
        _require(len(plan) == len(expected_plan), "evaluation episode count mismatch")
        for actual, expected in zip(plan, expected_plan):
            _require(all(actual.get(key) == value for key, value in expected.items()),
                     "evaluation episode plan mismatch")
            _require(("video_path" in actual) == ("video_path" in expected), "video selection mismatch")
            _require(isinstance(actual.get("official_state_sha256"), str)
                     and len(actual["official_state_sha256"]) == 64, "missing official initial-state hash")
        results = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines() if line.strip()]
        planned = {entry["episode_id"]: entry for entry in plan}
        for result in results:
            expected = planned.get(result["episode_id"])
            _require(expected is not None, "unknown episode result")
            for field in ("task_index", "task_name", "init_index", "env_seed", "policy_seed", "official_state_sha256"):
                _require(result.get(field) == expected[field], f"episode assignment mismatch: {field}")
            _require(isinstance(result.get("success"), bool), "episode success must be boolean")
            _require("error" in result and result["error"] is None, "evaluation contains episode errors")
            _require(result.get("video_error") is None, "evaluation contains video errors")
        computed = summarize_results(plan, results)
        _require(computed["complete"], "evaluation incomplete; success rate withheld")
        for key, expected in computed.items():
            _require(summary.get(key) == expected, f"evaluation summary mismatch: {key}")
        _require(summary.get("checkpoint_sha256") == digest and summary.get("weights") == evidence["weights"]
                 and not summary.get("checkpoint_integrity_error"), "summary checkpoint integrity mismatch")
        return {"mean_success_rate": float(computed["mean_success_rate"]),
                "eval/successes": int(computed["successes"]),
                "eval/episodes": int(computed["completed_episodes"]),
                "eval/completed_epochs": int(evidence["completed_epochs"]),
                **{f"eval/task/{name}/success_rate": float(value["success_rate"])
                   for name, value in computed["per_task"].items()}}

    def run_checkpoint(self, policy, cfg, epoch, global_step):
        if int(epoch) != epoch or int(global_step) != global_step or min(epoch, global_step) < 0:
            raise ValueError("epoch and global_step must be nonnegative integers")
        rollout_root = Path(self.output_dir).absolute() / "rollouts"
        _ordinary_path(rollout_root)
        directory = rollout_root / f"epoch_{int(epoch) + 1:03d}"
        _ordinary_path(directory)
        with _exclusive_epoch(directory):
            checkpoint, evidence = self._snapshot(directory, policy, cfg, epoch, global_step)
            attempts = sorted(directory.glob("attempt_[0-9][0-9][0-9]"))
            for attempt in attempts:
                _ordinary_path(attempt)
                verified = attempt / "runner_verified.json"
                _ordinary_path(verified)
                if verified.exists():
                    metrics = self._verify_result(attempt, checkpoint, evidence)
                    _require(_read_json(verified) == {"checkpoint_sha256": evidence["checkpoint_sha256"],
                                                     "returncode": 0, "metrics": metrics},
                             "conflicting completed evaluation evidence")
                    print(f"Official LIBERO evaluation reused: completed epoch {epoch + 1}, "
                          f"success {metrics['mean_success_rate']:.1%} "
                          f"({metrics['eval/successes']}/{metrics['eval/episodes']}), {attempt}", flush=True)
                    return metrics
            # Never modify an incomplete attempt, including its console log.
            index = max([int(path.name[-3:]) for path in attempts] + [0]) + 1
            while (directory / f"attempt_{index:03d}.log").exists():
                index += 1
            output = directory / f"attempt_{index:03d}"
            command = [sys.executable, str(ROOT / "scripts/eval_heading_policy.py"),
                       "--checkpoint", str(checkpoint), "--output", str(output),
                       "--device", "cuda:0", "--egl-device", str(self.eval_gpu),
                       "--torch-threads", str(self.torch_threads)]
            for name, value in self.protocol.items():
                command.extend(["--" + name.replace("_", "-"), str(value)])
            env = dict(os.environ)
            env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=str(self.eval_gpu),
                       MUJOCO_EGL_DEVICE_ID=str(self.eval_gpu), MUJOCO_GL="egl",
                       PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS=str(self.torch_threads),
                       MKL_NUM_THREADS=str(self.torch_threads), CUBLAS_WORKSPACE_CONFIG=":4096:8")
            expected_episodes = SUITE_TASKS[self.protocol["suite"]] * self.protocol["n_per_task"]
            print(f"Official LIBERO evaluation starting: completed epoch {epoch + 1}, "
                  f"{expected_episodes} episodes, GPU {self.eval_gpu}, {output}", flush=True)
            with (directory / f"attempt_{index:03d}.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            _require(result.returncode == 0, f"official evaluation exited {result.returncode}; artifacts retained at {output}")
            metrics = self._verify_result(output, checkpoint, evidence)
            _write_new_json(output / "runner_verified.json",
                            {"checkpoint_sha256": evidence["checkpoint_sha256"], "returncode": 0, "metrics": metrics})
            print(f"Official LIBERO evaluation complete: completed epoch {epoch + 1}, "
                  f"success {metrics['mean_success_rate']:.1%} "
                  f"({metrics['eval/successes']}/{metrics['eval/episodes']}), {output}", flush=True)
            return metrics
