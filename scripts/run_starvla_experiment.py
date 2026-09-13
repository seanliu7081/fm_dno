#!/usr/bin/env python3
"""Run the fixed Gaussian experiment: three full-finetuning seeds, then two benchmarks.

Supervisor manages this coordinator. It owns every training, inference and
simulator child process group; nothing is launched as an independent daemon.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oat.starvla_heading.evaluation import (
    audit_prompt_mapping, json_hash, load_instruction_catalog, load_results, summarize_results, validate_metadata,
)

TRAIN_PYTHON = "/workspace/venvs/starvla-heading/bin/python"
SIMULATORS = {
    "libero": ("/venv/fm_dno/bin/python", "/workspace/libero_config/original"),
    "libero_plus": ("/workspace/venvs/libero-plus/bin/python", "/workspace/libero_config/plus"),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_configuration(config, seed, output):
    resolved = copy.deepcopy(config)
    resolved["seed"] = int(seed)
    resolved["output_dir"] = str(Path(output).resolve())
    return resolved


def validate_experiment(config, seeds):
    if config.get("variant") != "heading_gaussian":
        raise ValueError("Only Heading Gaussian belongs to this experiment")
    training = config["training"]
    if (training.get("world_size"), training.get("max_steps"), training.get("full_finetuning")) != (6, 30000, True):
        raise ValueError("The experiment requires six GPUs and the fixed 30,000-step full-finetuning budget")
    microbatch = training.get("micro_batch_size", 0)
    accumulation = training.get("gradient_accumulation_steps", 0)
    if (not isinstance(microbatch, int) or not isinstance(accumulation, int) or
            min(microbatch, accumulation) < 1 or 6 * microbatch * accumulation != 60):
        raise ValueError("The production experiment requires a measured microbatch/accumulation configuration with effective batch60")
    if config["validation"].get("benchmark") != "libero":
        raise ValueError("Only original LIBERO can select checkpoints")
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 - 6 for seed in seeds):
        raise ValueError("Training seeds must be distinct and valid NumPy/PyTorch seeds")


def training_command(config_path, seed, output):
    output = Path(output)
    command = ["bash", str(ROOT / "scripts/run_starvla_heading.sh"), "--config", str(config_path),
               "--seed", str(seed), "--output-dir", str(output)]
    if (output / "resume/INCOMPLETE").exists():
        raise RuntimeError(f"Optimizer checkpoint is INCOMPLETE: {output / 'resume'}")
    if (output / "resume/commit.json").is_file():
        command.append("--resume")
    elif (output / "run_config.json").exists():
        raise RuntimeError(f"Run has no committed optimizer state: {output}; refusing to restart training from scratch")
    return command


def evaluation_command(benchmark, output, server_url, training_manifest, workers=4, render_gpu=1):
    python, config = SIMULATORS[benchmark]
    return [python, str(ROOT / "scripts/eval_starvla_heading.py"), "--benchmark", benchmark,
            "--libero-config-path", config, "--output", str(output), "--server-url", server_url,
            "--training-manifest", str(training_manifest),
            "--workers", str(workers), "--render-gpu-device-id", str(render_gpu),
            "--trials", "50" if benchmark == "libero" else "1"]


def server_command(checkpoint, port=18080, device="cuda:0"):
    return [TRAIN_PYTHON, str(ROOT / "scripts/serve_starvla_heading.py"), "--checkpoint",
            str(Path(checkpoint).resolve(strict=True)), "--host", "127.0.0.1", "--port", str(port),
            "--device", device, "--max-batch-size", "4"]


def verify_completed_training(output, expected_config):
    """Return an immutable, hashed best checkpoint only after full training."""
    output = Path(output).resolve()
    status_path = output / "status.json"
    if not status_path.is_file():
        return None
    status = read_json(status_path)
    if status.get("status") != "completed":
        return None
    if status.get("step") != expected_config["training"]["max_steps"]:
        raise RuntimeError("Completed training status does not match the fixed training budget")
    actual_config = read_json(output / "run_config.json")
    without_statistics = {key: value for key, value in actual_config.items() if key != "statistics"}
    if without_statistics != expected_config:
        raise RuntimeError("Completed run configuration differs from this experiment")
    checkpoint = (output / "best").resolve(strict=True)
    if checkpoint.parent != (output / "exports").resolve() or not re.fullmatch(r"step_\d{8}", checkpoint.name):
        raise RuntimeError("Best checkpoint must resolve to an immutable export inside this seed directory")
    metadata = read_json(checkpoint / "metadata.json")
    validate_metadata(metadata, benchmark="libero_plus")
    metric = metadata.get("validation_normalized_action_mse")
    if not isinstance(metric, (int, float)) or not math.isfinite(metric):
        raise RuntimeError("Best checkpoint lacks a finite original-LIBERO selection result")
    if status.get("best_validation_mse") != metric:
        raise RuntimeError("Best export selection metric differs from completed training status")
    if (metadata.get("seed"), metadata.get("world_size"), metadata.get("full_finetuning")) != (expected_config["seed"], 6, True):
        raise RuntimeError("Best checkpoint belongs to a different seed or training method")
    if not 0 < metadata.get("step", 0) <= status["step"]:
        raise RuntimeError("Best checkpoint has an invalid training step")
    if metadata["checkpoint_sha256"] != file_hash(checkpoint / "weights.pt"):
        raise RuntimeError("Best checkpoint weights changed")
    if metadata.get("config_sha256") != file_hash(checkpoint / "config.json"):
        raise RuntimeError("Best checkpoint configuration changed")
    if read_json(checkpoint / "config.json") != actual_config:
        raise RuntimeError("Export configuration differs from its completed training run")
    if not metadata.get("dataset_manifest_sha256"):
        raise RuntimeError("Selected checkpoint lacks its frozen training-manifest identity")
    load_instruction_catalog(checkpoint / "dataset_manifest.json", metadata["dataset_manifest_sha256"])
    return {"checkpoint": str(checkpoint), "metadata": metadata, "training_status": status}


def verify_evaluation(output, benchmark, checkpoint_info):
    """Recompute coverage and scores from every manifest-bound episode result."""
    output = Path(output)
    if not (output / "manifest.json").is_file():
        if any((output / name).exists() for name in ("results.jsonl", "summary.json")):
            raise RuntimeError("Evaluation artifacts exist without an immutable manifest")
        return None
    manifest = read_json(output / "manifest.json")
    settings = manifest["settings"]
    if settings.get("benchmark") != benchmark or settings.get("smoke"):
        raise RuntimeError("Evaluation is a smoke run or belongs to another benchmark")
    if settings.get("policy") != checkpoint_info["metadata"]:
        raise RuntimeError("Evaluation uses a different frozen checkpoint or policy settings")
    scope = manifest["scope"]
    if not scope.get("all_available_selected") or not scope.get("all_four_suites"):
        raise RuntimeError("Partial evaluation cannot satisfy this experiment")
    if manifest.get("trials") != (50 if benchmark == "libero" else 1):
        raise RuntimeError("Evaluation trial count differs from the official protocol")
    expected_tasks = 40 if benchmark == "libero" else 10030
    expected_episodes = expected_tasks * manifest["trials"]
    planned = manifest["plan"]
    catalog = load_instruction_catalog(Path(checkpoint_info["checkpoint"]) / "dataset_manifest.json",
                                       checkpoint_info["metadata"]["dataset_manifest_sha256"])
    if settings.get("instruction_catalog") != catalog:
        raise RuntimeError("Evaluation uses a different or obsolete instruction protocol")
    prompt_audit = audit_prompt_mapping(planned, catalog)
    if manifest.get("prompt_audit") != prompt_audit or prompt_audit["mapped_tasks"] != expected_tasks:
        raise RuntimeError("Evaluation prompt-source audit differs from its complete task plan")
    counts = scope.get("suite_counts", {})
    if (set(counts) != {"libero_spatial", "libero_object", "libero_goal", "libero_10"} or
            sum(counts.values()) != expected_tasks or len(planned) != expected_episodes or
            scope.get("available_episodes") != expected_episodes or scope.get("selected_episodes") != expected_episodes or
            len({entry["episode_id"] for entry in planned}) != expected_episodes):
        raise RuntimeError("Full-benchmark manifest coverage is inconsistent")
    for suite, count in counts.items():
        positions = {(entry["task_index"], entry["init_index"]) for entry in planned if entry["suite"] == suite}
        if positions != {(task, trial) for task in range(count) for trial in range(manifest["trials"])}:
            raise RuntimeError("Full-benchmark manifest has missing or duplicate task/trial positions")
    results = load_results(output / "results.jsonl", json_hash(manifest))
    summary = summarize_results(manifest["plan"], results, official_scope=True, smoke=False)
    if not summary["official_benchmark_complete"]:
        return None
    if not (output / "summary.json").is_file() or read_json(output / "summary.json") != summary:
        raise RuntimeError("Saved evaluation summary differs from its complete episode results")
    return summary


def aggregate_results(seeds, reports):
    def distribution(rates):
        values = list(rates.values())
        return {"seed_success_rates": {str(seed): value for seed, value in rates.items()},
                "completed_training_seeds": len(values),
                "mean_success_rate": statistics.mean(values) if values else None,
                "std_across_training_seeds": statistics.stdev(values) if len(values) > 1 else None}

    benchmarks = {}
    for benchmark in SIMULATORS:
        available = {seed: reports[seed][benchmark] for seed in seeds
                     if reports.get(seed, {}).get(benchmark) is not None}
        item = distribution({seed: report["official_success_rate"] for seed, report in available.items()})
        item["breakdowns"] = {}
        for kind in ("suite", "category", "difficulty_level"):
            names = sorted({name for report in available.values() for name in report["breakdowns"][kind]})
            item["breakdowns"][kind] = {
                name: distribution({seed: report["breakdowns"][kind][name]["success_rate"]
                                    for seed, report in available.items() if name in report["breakdowns"][kind]})
                for name in names}
        benchmarks[benchmark] = item
    completed = [seed for seed in seeds if all(reports.get(seed, {}).get(benchmark) is not None for benchmark in SIMULATORS)]
    return {"variant": "heading_gaussian", "requested_training_seeds": list(seeds),
            "completed_training_seeds": completed, "complete": len(completed) == len(seeds),
            "uncertainty": "Sample standard deviation across independently trained seeds; null with fewer than two seeds",
            "benchmarks": benchmarks}


def verify_server_receipt(output, info):
    path = Path(output) / "evaluation/checkpoint_verified.json"
    if not path.is_file():
        return False
    receipt = read_json(path)
    return (receipt.get("checkpoint") == info["checkpoint"] and receipt.get("metadata") == info["metadata"]
            and receipt.get("verification") == "owned_inference_server_loaded_checkpoint")


def prune_completed_optimizer(output, info, reports):
    """Explicit retention option; preserve selected policy and all evaluation evidence."""
    output = Path(output).resolve()
    if not verify_server_receipt(output, info):
        raise RuntimeError("Cannot prune optimizer state before checkpoint reload verification")
    status = read_json(output / "status.json")
    if status != info["training_status"] or status.get("status") != "completed" or status.get("step") != 30000:
        raise RuntimeError("Cannot prune an unfinished training run")
    best = Path(info["checkpoint"]).resolve(strict=True)
    if best.parent != (output / "exports").resolve() or file_hash(best / "weights.pt") != info["metadata"]["checkpoint_sha256"]:
        raise RuntimeError("Cannot prune before verifying the retained selected policy")
    for benchmark in SIMULATORS:
        verified = verify_evaluation(output / "evaluation" / benchmark, benchmark, info)
        if verified is None or reports.get(benchmark) != verified:
            raise RuntimeError("Cannot prune optimizer state before both full benchmark evaluations")
    targets = []
    resume = output / "resume"
    if resume.exists():
        if resume.is_symlink() or resume.resolve().parent != output:
            raise RuntimeError("Refusing to prune a resume path outside its generated seed directory")
        targets.append(resume)
    latest = output / "latest"
    unlink_latest = False
    if latest.is_symlink():
        target = latest.resolve(strict=True)
        if target != best:
            if target.parent != (output / "exports").resolve() or not re.fullmatch(r"step_\d{8}", target.name):
                raise RuntimeError("Refusing to prune a non-export latest checkpoint")
            targets.append(target)
            unlink_latest = True
    # Validate all targets before removing any generated artifacts.
    removed = []
    if unlink_latest:
        latest.unlink()
    for target in targets:
        shutil.rmtree(target)
        removed.append(str(target))
    previous = read_json(output / "retention.json") if (output / "retention.json").is_file() else {"removed": []}
    write_json(output / "retention.json", {"removed": sorted(set(previous["removed"] + removed)),
               "preserved_best_checkpoint": str(best), "checkpoint_sha256": info["metadata"]["checkpoint_sha256"],
               "reason": "Explicit --prune-completed-optimizer after completed training, checkpoint reload, and both full evaluations"})


class InterruptedExperiment(RuntimeError):
    pass


class OwnedProcess:
    def __init__(self, command, kind, log_path, env=None):
        self.kind = kind
        self.stopped = False
        self.worker_groups = set()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", buffering=1)
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)

    def _group_signal(self, sig):
        for group in {self.process.pid, *self.worker_groups}:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass

    def track_training_workers(self, proc_root=Path("/proc")):
        """torchrun creates a separate session/process group for each rank."""
        if self.kind != "training":
            return []
        workers = []
        for path in Path(proc_root).iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                ppid, pgrp = int(fields[1]), int(fields[2])
                if ppid != self.process.pid:
                    continue
                cmdline = (path / "cmdline").read_bytes().split(b"\0")
                if any(Path(arg.decode(errors="replace")).name == "train_starvla_heading.py" for arg in cmdline):
                    workers.append(int(path.name))
                    self.worker_groups.add(pgrp)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return workers

    def _signal_training_workers(self):
        # Signal rank PIDs only, preserving their loader workers while each
        # rank saves. Torchrun's default launcher shutdown is much shorter.
        workers = self.track_training_workers()
        for pid in workers:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return bool(workers)

    def stop(self, training_grace=900):
        if self.stopped:
            return
        if self.process.poll() is None:
            graceful = self.kind == "training" and self._signal_training_workers()
            if not graceful:
                self._group_signal(signal.SIGTERM)
            deadline = time.monotonic() + (training_grace if graceful else 30)
            while self.process.poll() is None and time.monotonic() < deadline:
                time.sleep(.25)
            if self.process.poll() is None:
                self._group_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._group_signal(signal.SIGKILL)
                    self.process.wait(timeout=10)
        # Reap lingering descendants even if their launcher has exited.
        self._group_signal(signal.SIGTERM)
        # Child process groups belong to this coordinator even after their
        # launcher exits. Bound cleanup and never revisit a reaped PID later.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            alive = False
            for group in {self.process.pid, *self.worker_groups}:
                try:
                    os.killpg(group, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            if not alive:
                break
            time.sleep(.1)
        else:
            self._group_signal(signal.SIGKILL)
        self.log.close()
        self.stopped = True


class Experiment:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.output = args.output_root
        self.seeds = args.seeds or config.get("seeds", [42, 43, 44])
        self.reports = {}
        self.stopping = False
        self.children = []
        self.server_url = f"http://127.0.0.1:{args.server_port}"

    def event(self, event, **fields):
        value = {"event": event, "time_unix": time.time(), **fields}
        print(json.dumps(value, allow_nan=False), flush=True)
        with (self.output / "experiment_events.jsonl").open("a") as log:
            log.write(json.dumps(value, allow_nan=False) + "\n")

    def on_signal(self, signum, frame):
        self.stopping = True

    def start(self, command, kind, log, env=None):
        if self.stopping:
            raise InterruptedExperiment("Experiment interrupted")
        child = OwnedProcess(command, kind, log, env)
        self.children.append(child)
        self.event("process_started", kind=kind, pid=child.process.pid, command=command, log=str(log))
        return child

    def run_command(self, command, kind, log, env=None):
        child = self.start(command, kind, log, env)
        last_worker_scan = 0.
        while child.process.poll() is None:
            if child.kind == "training" and time.monotonic() - last_worker_scan >= 2:
                child.track_training_workers()
                last_worker_scan = time.monotonic()
            if self.stopping:
                child.stop()
                raise InterruptedExperiment("Experiment interrupted; owned process shutdown completed")
            time.sleep(.5)
        code = child.process.returncode
        child.stop()
        if self.stopping:
            raise InterruptedExperiment("Experiment interrupted")
        if code:
            raise RuntimeError(f"{kind} exited with status {code}; inspect {log}")

    def wait_for_server(self, server, info):
        deadline = time.monotonic() + self.args.server_startup_timeout
        while time.monotonic() < deadline:
            if self.stopping:
                raise InterruptedExperiment("Interrupted while loading inference checkpoint")
            if server.process.poll() is not None:
                raise RuntimeError(f"Inference server exited; inspect {server.log_path}")
            try:
                with urlopen(self.server_url + "/metadata", timeout=2) as response:
                    metadata = json.load(response)
                if metadata != info["metadata"]:
                    raise RuntimeError("Owned inference server loaded a different checkpoint")
                return
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(.5)
        raise RuntimeError("Inference server did not become ready before its startup deadline")

    def update_summary(self):
        result = aggregate_results(self.seeds, self.reports)
        write_json(self.output / "experiment_summary.json", result)
        return result

    def run_seed(self, seed):
        output = self.output / f"seed{seed}"
        expected_config = seed_configuration(self.config, seed, output)
        info = verify_completed_training(output, expected_config)
        if info is None:
            self.event("training_started", seed=seed, budget_steps=30000)
            env = dict(os.environ, STARVLA_TRAIN_GPUS="0,1,2,3,4,5")
            self.run_command(training_command(self.args.config, seed, output), "training", output / "training.log", env)
            info = verify_completed_training(output, expected_config)
            if info is None:
                raise RuntimeError("Training exited without verified completion of the 30,000-step budget")
        self.event("training_verified", seed=seed, checkpoint=info["checkpoint"], checkpoint_sha256=info["metadata"]["checkpoint_sha256"])
        reports = self.reports[seed] = {
            benchmark: verify_evaluation(output / "evaluation" / benchmark, benchmark, info)
            for benchmark in SIMULATORS}
        self.update_summary()
        server = None
        if not all(reports.values()) or not verify_server_receipt(output, info):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", self.args.server_port)) == 0:
                    raise RuntimeError(f"Port {self.args.server_port} is occupied; refusing to adopt another server")
            try:
                server = self.start(server_command(info["checkpoint"], self.args.server_port, self.args.server_device), "server", output / "inference.log")
                self.wait_for_server(server, info)
                write_json(output / "evaluation/checkpoint_verified.json", {**info,
                    "verification": "owned_inference_server_loaded_checkpoint", "verified_at_unix": time.time()})
                for benchmark in SIMULATORS:
                    if reports[benchmark] is not None:
                        continue
                    self.event("evaluation_started", seed=seed, benchmark=benchmark, checkpoint=info["checkpoint"])
                    destination = output / "evaluation" / benchmark
                    self.run_command(evaluation_command(
                        benchmark, destination, self.server_url, Path(info["checkpoint"]) / "dataset_manifest.json",
                        self.args.eval_workers, self.args.render_gpu_device_id),
                        "evaluation", output / f"evaluation_{benchmark}.log")
                    reports[benchmark] = verify_evaluation(destination, benchmark, info)
                    if reports[benchmark] is None:
                        raise RuntimeError(f"{benchmark} exited without complete official benchmark results")
                    self.update_summary()
                    self.event("evaluation_verified", seed=seed, benchmark=benchmark,
                               success_rate=reports[benchmark]["official_success_rate"])
            finally:
                if server is not None:
                    server.stop()
        if self.args.prune_completed_optimizer:
            prune_completed_optimizer(output, info, reports)
        write_json(output / "experiment_seed_status.json", {"status": "completed", "seed": seed,
                   "checkpoint": info["checkpoint"], "checkpoint_sha256": info["metadata"]["checkpoint_sha256"]})
        self.event("seed_completed", seed=seed)
        self.update_summary()

    def run(self):
        validate_experiment(self.config, self.seeds)
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.output / "experiment_config.json"
            signature = {"config": self.config, "config_path": str(self.args.config),
                         "training_gpus": [0, 1, 2, 3, 4, 5], "server_device": self.args.server_device,
                         "render_device": self.args.render_gpu_device_id, "eval_workers": self.args.eval_workers,
                         "server_port": self.args.server_port}
            if path.exists() and read_json(path) != signature:
                raise RuntimeError("Experiment configuration changed; use its original settings or another output root")
            write_json(path, signature)
            handlers = {sig: signal.signal(sig, self.on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
            self.event("experiment_started", seeds=self.seeds, prune_completed_optimizer=self.args.prune_completed_optimizer)
            try:
                for seed in self.seeds:
                    if self.stopping:
                        raise InterruptedExperiment("Experiment interrupted")
                    self.run_seed(seed)
                summary = self.update_summary()
                if not summary["complete"]:
                    raise RuntimeError("Experiment ended with missing required results")
                write_json(self.output / "experiment_status.json", {"status": "completed", "seeds": self.seeds})
                return 0
            except Exception as error:
                write_json(self.output / "experiment_status.json", {
                    "status": "interrupted" if isinstance(error, InterruptedExperiment) else "failed",
                    "error": f"{type(error).__name__}: {error}"})
                self.update_summary()
                self.event("experiment_stopped", error=str(error))
                return 130 if isinstance(error, InterruptedExperiment) else 1
            finally:
                for child in reversed(self.children):
                    child.stop()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "oat/config/starvla_heading_gaussian.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, help="Default: all three configured training seeds")
    parser.add_argument("--server-port", type=int, default=18080)
    parser.add_argument("--server-startup-timeout", type=float, default=600)
    parser.add_argument("--server-device", default="cuda:0", help="Inference GPU after six-GPU training has exited")
    parser.add_argument("--render-gpu-device-id", type=int, default=1, help="EGL device used after training has exited")
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--prune-completed-optimizer", action="store_true",
                        help="After both verified full evaluations, delete generated optimizer state and an unselected latest export")
    args = parser.parse_args(argv)
    import yaml
    args.config = args.config.resolve()
    config = yaml.safe_load(args.config.read_text())
    args.output_root = (args.output_root or Path(config["output_dir"]).parent).resolve()
    if not 1 <= args.server_port <= 65535 or args.server_startup_timeout <= 0:
        parser.error("Invalid server port or startup timeout")
    if not re.fullmatch(r"cuda:\d+", args.server_device) or args.render_gpu_device_id < 0 or args.eval_workers < 1:
        parser.error("Expected --server-device cuda:N, a nonnegative render GPU, and positive eval workers")
    return Experiment(args, config).run()


if __name__ == "__main__":
    raise SystemExit(main())
