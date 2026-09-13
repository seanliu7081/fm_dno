#!/usr/bin/env python3
"""Evaluate exact 10k/20k/30k Heading Gaussian checkpoints alongside training.

This coordinator owns only periodic inference/simulator children and hardlinked
checkpoint snapshots. The existing training and final-best evaluation workflow
continues independently on its original devices.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_starvla_experiment import (
    InterruptedExperiment, OwnedProcess, aggregate_results, evaluation_command,
    file_hash, load_instruction_catalog, read_json, seed_configuration,
    server_command, validate_experiment, validate_metadata, verify_evaluation, write_json,
)

STEPS = (10000, 20000, 30000)
EPISODES = {"libero": 2000, "libero_plus": 10030}
CHECKPOINT_FILES = ("weights.pt", "config.json", "dataset_manifest.json", "metadata.json")


def complete_events(path):
    """Ignore only an unfinished trailing append; malformed committed rows fail."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    complete = data[:data.rfind(b"\n") + 1]
    events = [json.loads(line) for line in complete.splitlines() if line.strip()]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError(f"Events must be JSON objects: {path}")
    return events


def read_committed_steps(seed_output):
    steps = set()
    for event in complete_events(Path(seed_output) / "metrics.jsonl"):
        if event.get("event") != "checkpoint" or event.get("optimizer_state_saved") is not True:
            continue
        step = event.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("Committed checkpoint event has an invalid step")
        steps.add(step)
    return steps


def _regular_file(path):
    mode = Path(path).lstat().st_mode
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"Expected a regular checkpoint file, refusing symlinks: {path}")


def validate_snapshot(checkpoint, seed_output, seed, step, expected_config, *, require_weights=True):
    checkpoint, seed_output = Path(checkpoint), Path(seed_output)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RuntimeError("Checkpoint snapshot must be an owned directory, not a symlink")
    names = CHECKPOINT_FILES if require_weights else CHECKPOINT_FILES[1:]
    for name in names:
        _regular_file(checkpoint / name)
    saved = read_json(seed_output / "run_config.json")
    if {key: value for key, value in saved.items() if key != "statistics"} != expected_config:
        raise RuntimeError("Seed run configuration differs from the immutable experiment")
    if read_json(checkpoint / "config.json") != saved:
        raise RuntimeError("Checkpoint configuration differs from the seed run")
    metadata = read_json(checkpoint / "metadata.json")
    validate_metadata(metadata, benchmark="libero_plus")
    if (any(not isinstance(metadata.get(name), int) or isinstance(metadata.get(name), bool)
            for name in ("seed", "step", "world_size"))
            or (metadata.get("seed"), metadata.get("step"), metadata.get("world_size")) != (seed, step, 6)
            or metadata.get("full_finetuning") is not True):
        raise RuntimeError("Checkpoint belongs to a different seed, step, or training method")
    score = metadata.get("validation_normalized_action_mse")
    if (not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(score) or score < 0):
        raise RuntimeError("Checkpoint lacks its finite original-LIBERO validation result")
    if file_hash(checkpoint / "config.json") != metadata.get("config_sha256"):
        raise RuntimeError("Checkpoint configuration hash differs from metadata")
    if require_weights and file_hash(checkpoint / "weights.pt") != metadata["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint weights hash differs from metadata")
    if not metadata.get("dataset_manifest_sha256"):
        raise RuntimeError("Checkpoint lacks its frozen training-manifest identity")
    load_instruction_catalog(checkpoint / "dataset_manifest.json", metadata["dataset_manifest_sha256"])
    return {"checkpoint": str(checkpoint.resolve()), "metadata": metadata}


def capture_checkpoint(seed_output, destination, seed, step, expected_config):
    """Pin an exact optimizer-committed export before the trainer prunes it.

    Trainer checkpoint events are appended after optimizer commit and export
    publication. Four hardlinks preserve the immutable files independently of
    later trainer pruning; hashes are checked after acquiring those links.
    """
    seed_output, destination = Path(seed_output).resolve(), Path(destination)
    if destination.name != "checkpoint":
        raise ValueError("Periodic snapshots must use a checkpoint directory")
    if destination.absolute() != destination.resolve():
        raise RuntimeError("Refusing a symlinked periodic snapshot path")
    if destination.resolve().is_relative_to((seed_output / "exports").resolve()):
        raise RuntimeError("Periodic snapshots cannot live inside trainer-pruned exports")
    committed = read_committed_steps(seed_output)
    if step not in committed:
        if any(value > step for value in committed):
            raise RuntimeError(f"Required step {step} has no committed checkpoint event in {seed_output}")
        status = read_json(seed_output / "status.json") if (seed_output / "status.json").is_file() else {}
        if status.get("status") == "completed" and status.get("step", -1) >= step:
            raise RuntimeError(f"Completed training is missing required checkpoint step {step}")
        return None
    source = seed_output / "exports" / f"step_{step:08d}"
    receipt_path = destination.parent / "capture_receipt.json"
    if destination.exists():
        verification = destination.parent / "verification.json"
        released = (verification.is_file() and read_json(verification).get("status") == "completed"
                    and read_json(verification).get("server_stopped") is True)
        info = validate_snapshot(destination, seed_output, seed, step, expected_config,
                                 require_weights=not released or (destination / "weights.pt").exists())
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if (receipt.get("seed"), receipt.get("step"), receipt.get("metadata"), receipt.get("source")) != (
                    seed, step, info["metadata"], str(source)):
                raise RuntimeError("Existing snapshot capture receipt differs from its immutable identity")
        else:
            if not (destination / "weights.pt").is_file():
                raise RuntimeError("Released snapshot has no capture receipt")
            write_json(receipt_path, {"seed": seed, "step": step, "source": str(source),
                       "metadata": info["metadata"], "captured_at_unix": time.time(),
                       "method": "hardlink_then_atomic_directory_publish", "recovered_published_snapshot": True})
        return info
    if not source.is_dir() or source.is_symlink() or source.resolve().parent != (seed_output / "exports").resolve():
        raise RuntimeError(f"Required committed export was pruned or is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ".checkpoint.capture.tmp"
    if staging.is_symlink():
        raise RuntimeError("Refusing a symlinked snapshot staging directory")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        for name in CHECKPOINT_FILES:
            _regular_file(source / name)
            os.link(source / name, staging / name, follow_symlinks=False)
        info = validate_snapshot(staging, seed_output, seed, step, expected_config)
        os.replace(staging, destination)
        info["checkpoint"] = str(destination.resolve())
        write_json(receipt_path, {"seed": seed, "step": step, "source": str(source),
                   "metadata": info["metadata"], "captured_at_unix": time.time(),
                   "method": "hardlink_then_atomic_directory_publish"})
        return info
    except FileNotFoundError as error:
        raise RuntimeError(f"Required export disappeared during capture: {source}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _verified_identity(event):
    if event.get("event") != "evaluation_verified":
        raise ValueError("Expected an evaluation_verified event")
    for field in ("seed", "step", "episodes"):
        value = event.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid evaluation event {field}")
    if event["step"] not in STEPS or event.get("benchmark") not in EPISODES:
        raise ValueError("Evaluation event is outside the periodic schedule")
    if event["episodes"] != EPISODES[event["benchmark"]]:
        raise ValueError("Evaluation event does not cover the full benchmark")
    for field in ("success_rate", "time_unix"):
        value = event.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Invalid evaluation event {field}")
    if not 0 <= event["success_rate"] <= 1 or event["time_unix"] < 0:
        raise ValueError("Evaluation event has an invalid rate or timestamp")
    if not re.fullmatch(r"[0-9a-f]{64}", event.get("checkpoint_sha256", "")):
        raise ValueError("Evaluation event lacks a checkpoint SHA-256")
    return event["seed"], event["step"], event["benchmark"]


def append_event(events_path, event):
    """Repair only our own torn final append before writing the next durable row.

    The coordinator flock guarantees a single writer. Valid complete rows are
    retained byte-for-byte so an online logger can verify its consumed prefix.
    """
    path = Path(events_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        data = stream.read()
        end = data.rfind(b"\n") + 1
        if end != len(data):
            stream.truncate(end)
        stream.write((json.dumps(event, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def record_verified_event(events_path, event):
    """Append each verified checkpoint/benchmark score once, rejecting conflicts."""
    identity = _verified_identity(event)
    found = False
    for previous in complete_events(events_path):
        if previous.get("event") != "evaluation_verified":
            continue
        if _verified_identity(previous) != identity:
            continue
        fields = ("checkpoint_sha256", "success_rate", "episodes")
        if any(previous[field] != event[field] for field in fields):
            raise ValueError("Conflicting duplicate periodic evaluation identity")
        found = True
    if found:
        return False
    append_event(events_path, event)
    return True


def release_snapshot_weights(job_directory, info, reports):
    """Release only an owned snapshot's weights after both verified evaluations."""
    job = Path(job_directory)
    checkpoint = job / "checkpoint"
    if job.absolute() != job.resolve() or checkpoint.is_symlink() or checkpoint.resolve() != Path(info["checkpoint"]):
        raise RuntimeError("Refusing to release weights outside the owned periodic snapshot")
    metadata = info["metadata"]
    if job.name != f"step_{metadata['step']:08d}" or job.parent.name != f"seed{metadata['seed']}":
        raise RuntimeError("Refusing to release weights outside an exact seed/step job directory")
    receipt = read_json(job / "verification.json")
    if (receipt.get("status") != "completed" or receipt.get("server_stopped") is not True
            or receipt.get("metadata") != metadata
            or receipt.get("checkpoint_sha256") != metadata["checkpoint_sha256"]):
        raise RuntimeError("Cannot release snapshot before complete verification and server shutdown")
    for benchmark in EPISODES:
        verified = verify_evaluation(job / benchmark, benchmark, info)
        if verified is None or reports.get(benchmark) != verified:
            raise RuntimeError("Cannot release snapshot before both complete benchmark evaluations")
    weights = checkpoint / "weights.pt"
    if weights.exists() or weights.is_symlink():
        _regular_file(weights)
        if file_hash(weights) != metadata["checkpoint_sha256"]:
            raise RuntimeError("Snapshot weights changed before release")
        weights.unlink()
    if not receipt.get("weights_released_at_unix"):
        receipt["weights_released_at_unix"] = time.time()
        write_json(job / "verification.json", receipt)


class PeriodicExperiment:
    def __init__(self, args, experiment):
        self.args, self.experiment = args, experiment
        self.root = args.output_root.resolve()
        self.output = self.root / "periodic_evaluation"
        self.config = experiment["config"]
        self.seeds = list(self.config.get("seeds", [42, 43, 44]))
        self.children = []
        self.stopping = False
        self.infos, self.reports = {}, {}
        self.last_capture = -float("inf")
        self.server_url = f"http://127.0.0.1:{args.server_port}"
        self.current_job = None
        self.failed_job = None
        self.current_benchmark = None
        self.phase = "waiting_for_checkpoint"

    def event(self, event, **fields):
        value = {"event": event, "time_unix": time.time(), **fields}
        if event == "evaluation_verified":
            added = record_verified_event(self.output / "events.jsonl", value)
            if not added:
                return
        else:
            append_event(self.output / "events.jsonl", value)
        print(json.dumps(value, allow_nan=False), flush=True)

    def job_directory(self, seed, step):
        return self.output / f"seed{seed}" / f"step_{step:08d}"

    def on_signal(self, signum, frame):
        self.stopping = True

    def update_status(self, status="running", **fields):
        pending = sorted(set(self.infos) - set(self.reports))
        current = ({"seed": self.current_job[0], "step": self.current_job[1],
                    "benchmark": self.current_benchmark} if self.current_job else None)
        write_json(self.output / "status.json", {"status": status, "phase": self.phase,
            "current_job": current, "seeds": self.seeds, "steps": list(STEPS),
            "requested_jobs": len(self.seeds) * len(STEPS), "captured_jobs": len(self.infos),
            "completed_jobs": len(self.reports), "queued_jobs": len([key for key in pending if key != self.current_job]),
            "awaiting_checkpoint_jobs": len(self.seeds) * len(STEPS) - len(self.infos),
            "free_disk_gib": shutil.disk_usage(self.root).free / 2**30,
            "observed_at_unix": time.time(), **fields})

    def capture_due(self, force=False):
        if not force and time.monotonic() - self.last_capture < self.args.poll_seconds:
            return
        self.last_capture = time.monotonic()
        for seed in self.seeds:
            source = self.root / f"seed{seed}"
            for step in STEPS:
                key = seed, step
                if key in self.infos:
                    continue
                job = self.job_directory(seed, step)
                try:
                    info = capture_checkpoint(source, job / "checkpoint", seed, step,
                                              seed_configuration(self.config, seed, source))
                except Exception:
                    self.failed_job = seed, step
                    raise
                if info is None:
                    continue
                self.infos[key] = info
                self.event("checkpoint_captured", seed=seed, step=step, checkpoint=info["checkpoint"],
                           checkpoint_sha256=info["metadata"]["checkpoint_sha256"])
                verification = job / "verification.json"
                if verification.is_file() and read_json(verification).get("status") == "completed":
                    self.accept_completed_job(seed, step, info)
                else:
                    write_json(job / "status.json", {"status": "queued", "seed": seed, "step": step})
        self.update_status()

    def verified_event(self, seed, step, benchmark, info, report):
        self.event("evaluation_verified", seed=seed, step=step, benchmark=benchmark,
                   success_rate=report["official_success_rate"], episodes=EPISODES[benchmark],
                   checkpoint_sha256=info["metadata"]["checkpoint_sha256"])

    def accept_completed_job(self, seed, step, info):
        job = self.job_directory(seed, step)
        reports = {benchmark: verify_evaluation(job / benchmark, benchmark, info) for benchmark in EPISODES}
        if not all(report is not None for report in reports.values()):
            raise RuntimeError("Completed periodic job has incomplete official benchmark evidence")
        release_snapshot_weights(job, info, reports)
        for benchmark, report in reports.items():
            self.verified_event(seed, step, benchmark, info, report)
        self.reports[seed, step] = reports
        write_json(job / "status.json", {"status": "completed", "seed": seed, "step": step,
                   "checkpoint_sha256": info["metadata"]["checkpoint_sha256"]})
        self.update_summary()

    def update_summary(self):
        summaries = {str(step): aggregate_results(self.seeds, {
            seed: self.reports[seed, step] for seed in self.seeds if (seed, step) in self.reports}) for step in STEPS}
        write_json(self.output / "summary.json", {"variant": "heading_gaussian", "steps": list(STEPS),
                   "seeds": self.seeds, "complete": len(self.reports) == len(self.seeds) * len(STEPS),
                   "selection_policy": "Original LIBERO validation MSE only; periodic success rates never select checkpoints",
                   "by_step": summaries})
        for seed in self.seeds:
            completed = [step for step in STEPS if (seed, step) in self.reports]
            write_json(self.output / f"seed{seed}" / "status.json", {
                "status": "completed" if len(completed) == len(STEPS) else "pending",
                "seed": seed, "requested_steps": list(STEPS), "completed_steps": completed})

    def start(self, command, kind, log_path):
        if self.stopping:
            raise InterruptedExperiment("Periodic evaluation interrupted")
        if kind not in ("server", "evaluation"):
            raise ValueError("Periodic coordinator cannot launch training processes")
        child = OwnedProcess(command, kind, log_path)
        self.children.append(child)
        self.event("process_started", kind=kind, pid=child.process.pid, command=command, log=str(log_path))
        return child

    def run_command(self, command, kind, log_path):
        child = self.start(command, kind, log_path)
        try:
            while child.process.poll() is None:
                self.capture_due()
                if self.stopping:
                    raise InterruptedExperiment("Periodic evaluation interrupted")
                time.sleep(.5)
            if child.process.returncode:
                raise RuntimeError(f"{kind} exited with status {child.process.returncode}; inspect {log_path}")
        finally:
            child.stop()
        if self.stopping:
            raise InterruptedExperiment("Periodic evaluation interrupted")

    def wait_for_server(self, server, info):
        deadline = time.monotonic() + self.args.server_startup_timeout
        while time.monotonic() < deadline:
            self.capture_due()
            if self.stopping:
                raise InterruptedExperiment("Interrupted while loading periodic checkpoint")
            if server.process.poll() is not None:
                raise RuntimeError(f"Periodic inference server exited; inspect {server.log_path}")
            try:
                with urlopen(self.server_url + "/metadata", timeout=2) as response:
                    metadata = json.load(response)
                if metadata != info["metadata"]:
                    raise RuntimeError("Owned periodic server loaded a different checkpoint")
                return
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(.5)
        raise RuntimeError("Periodic inference server did not become ready before its startup deadline")

    def run_job(self, seed, step):
        self.current_job = seed, step
        self.current_benchmark = None
        self.phase = "loading_checkpoint"
        self.update_status()
        job = self.job_directory(seed, step)
        info = self.infos[seed, step]
        if not (job / "checkpoint/weights.pt").is_file():
            raise RuntimeError("Unfinished periodic job has no retained checkpoint weights")
        reports = {benchmark: verify_evaluation(job / benchmark, benchmark, info) for benchmark in EPISODES}
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.args.server_port)) == 0:
                raise RuntimeError("Periodic server port is occupied; refusing to adopt another process")
        write_json(job / "status.json", {"status": "running", "seed": seed, "step": step})
        server = self.start(server_command(info["checkpoint"], self.args.server_port, self.args.server_device),
                            "server", job / "inference.log")
        try:
            self.wait_for_server(server, info)
            write_json(job / "checkpoint_loaded.json", {"checkpoint": info["checkpoint"], "metadata": info["metadata"],
                       "verification": "owned_inference_server_loaded_checkpoint", "verified_at_unix": time.time()})
            for benchmark in EPISODES:
                self.capture_due()
                if reports[benchmark] is None:
                    self.current_benchmark = benchmark
                    self.phase = "evaluating"
                    self.update_status()
                    self.event("evaluation_started", seed=seed, step=step, benchmark=benchmark,
                               checkpoint_sha256=info["metadata"]["checkpoint_sha256"])
                    self.run_command(evaluation_command(benchmark, job / benchmark, self.server_url,
                        job / "checkpoint/dataset_manifest.json", self.args.eval_workers, self.args.render_gpu_device_id),
                        "evaluation", job / f"evaluation_{benchmark}.log")
                    reports[benchmark] = verify_evaluation(job / benchmark, benchmark, info)
                if reports[benchmark] is None:
                    raise RuntimeError(f"{benchmark} exited without complete periodic benchmark results")
                self.verified_event(seed, step, benchmark, info, reports[benchmark])
        finally:
            server.stop()
        self.current_benchmark = None
        self.phase = "verifying_results"
        self.update_status()
        write_json(job / "verification.json", {"status": "completed", "seed": seed, "step": step,
                   "checkpoint_sha256": info["metadata"]["checkpoint_sha256"], "metadata": info["metadata"],
                   "server_stopped": True, "verified_at_unix": time.time(),
                   "episodes": EPISODES, "selection_policy": "Reporting only; never used for checkpoint selection"})
        self.accept_completed_job(seed, step, info)
        self.event("job_completed", seed=seed, step=step, checkpoint_sha256=info["metadata"]["checkpoint_sha256"])
        self.current_job = None
        self.phase = "waiting_for_checkpoint"
        self.update_status()

    def run(self):
        validate_experiment(self.config, self.seeds)
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "coordinator.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signature = {"protocol_version": 1, "steps": list(STEPS), "seeds": self.seeds,
                "benchmarks": EPISODES, "source_experiment_config_sha256": file_hash(self.root / "experiment_config.json"),
                "server_device": self.args.server_device, "render_device": self.args.render_gpu_device_id,
                "server_port": self.args.server_port, "eval_workers": self.args.eval_workers}
            config_path = self.output / "config.json"
            if config_path.exists() and read_json(config_path) != signature:
                raise RuntimeError("Periodic evaluation settings changed; resume with the frozen configuration")
            write_json(config_path, signature)
            handlers = {sig: signal.signal(sig, self.on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
            self.event("periodic_evaluation_started", seeds=self.seeds, steps=list(STEPS))
            self.update_status()
            try:
                self.capture_due(force=True)
                self.update_summary()
                while len(self.reports) < len(self.seeds) * len(STEPS):
                    if self.stopping:
                        raise InterruptedExperiment("Periodic evaluation interrupted")
                    self.capture_due()
                    pending = sorted(set(self.infos) - set(self.reports))
                    if pending:
                        self.run_job(*pending[0])
                    else:
                        time.sleep(.5)
                self.phase = "completed"
                self.update_status("completed", completed_at_unix=time.time())
                self.event("periodic_evaluation_completed", seeds=self.seeds, steps=list(STEPS))
                return 0
            except Exception as error:
                status = "interrupted" if isinstance(error, InterruptedExperiment) else "failed"
                fields = {"status": status, "error": f"{type(error).__name__}: {error}", "time_unix": time.time()}
                self.update_status(**fields)
                for key in {self.current_job, self.failed_job} - {None}:
                    seed, step = key
                    write_json(self.job_directory(seed, step) / "status.json", {**fields, "seed": seed, "step": step})
                    write_json(self.output / f"seed{seed}" / "status.json", {**fields, "seed": seed})
                self.event("periodic_evaluation_stopped", **fields)
                return 130 if isinstance(error, InterruptedExperiment) else 1
            finally:
                for child in reversed(self.children):
                    child.stop()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/starvla_heading_gaussian")
    parser.add_argument("--server-device", default="cuda:6")
    parser.add_argument("--render-gpu-device-id", type=int, default=7)
    parser.add_argument("--server-port", type=int, default=18087)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--server-startup-timeout", type=float, default=600)
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    experiment = read_json(args.output_root / "experiment_config.json")
    match = re.fullmatch(r"cuda:(\d+)", args.server_device)
    if not match or args.render_gpu_device_id < 0 or args.eval_workers < 1:
        parser.error("Expected cuda:N, a nonnegative render GPU, and positive worker count")
    server_gpu = int(match.group(1))
    reserved = set(experiment["training_gpus"])
    reserved.add(int(experiment["server_device"].split(":")[1]))
    reserved.add(experiment["render_device"])
    if server_gpu == args.render_gpu_device_id or {server_gpu, args.render_gpu_device_id} & reserved:
        parser.error("Periodic inference and rendering must use separate GPUs disjoint from the original experiment")
    if (not 1 <= args.server_port <= 65535 or args.server_port == experiment["server_port"]
            or not 0 < args.poll_seconds <= 60 or not math.isfinite(args.server_startup_timeout)
            or args.server_startup_timeout <= 0):
        parser.error("Invalid periodic port or polling/startup timeout")
    return PeriodicExperiment(args, experiment).run()


if __name__ == "__main__":
    raise SystemExit(main())
