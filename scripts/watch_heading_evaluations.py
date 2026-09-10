#!/usr/bin/env python3
"""Evaluate scheduled training epochs using verified immutable checkpoint snapshots.

Run under supervisor with the intended CUDA_VISIBLE_DEVICES and PYTHONPATH.
Epochs are checkpoint epoch labels, not counts of completed training epochs.
The evaluator and this watcher preserve failed attempts and never call partial
metrics a completed stage. Restarting reuses the stage ledger and snapshots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{utc_now()}] {message}", flush=True)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def signature(path):
    stat = path.stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path):
    # Explicit CPU map; do not initialize CUDA or instantiate a policy/renderer.
    import dill
    import torch
    from oat.common.hydra_util import register_new_resolvers
    register_new_resolvers()
    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    weight_name = "ema_model" if cfg.training.get("use_ema", False) else "model"
    if weight_name not in payload["state_dicts"]:
        raise ValueError(f"checkpoint lacks configured {weight_name} weights")
    epoch = dill.loads(payload["pickles"]["epoch"])
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"invalid checkpoint epoch: {epoch!r}")
    return {"epoch": epoch, "weights": weight_name}


class MetadataCache:
    """Load each signature once, only after it stays unchanged across polls."""
    def __init__(self):
        self.entries = {}

    def inspect(self, path):
        try:
            current = signature(path)
        except FileNotFoundError:
            self.entries.pop(str(path), None)
            return None
        key = str(path)
        entry = self.entries.get(key)
        if entry is None or entry["signature"] != current:
            self.entries[key] = {"signature": current, "checked": False}
            return None
        if not entry["checked"]:
            entry["checked"] = True
            try:
                metadata = read_metadata(path)
                if signature(path) != current:
                    self.entries.pop(key, None)
                    return None
                entry["metadata"] = metadata
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                log(f"Checkpoint not readable yet: {path}: {entry['error']}")
        if "metadata" in entry:
            return {"path": str(path), "signature": current, **entry["metadata"]}
        return None


def checkpoint_candidates(checkpoint_dir, target):
    exact = []
    for path in checkpoint_dir.glob("*.ckpt"):
        match = re.search(r"(?:ep-|epoch[=-])(\d+)", path.name)
        if match and int(match.group(1)) == target:
            exact.append(path)
    return sorted(exact) + [checkpoint_dir / "latest.ckpt"]


def choose_checkpoint(candidates, target):
    ready = [item for item in candidates if item and item["epoch"] >= target]
    exact = [item for item in ready if item["epoch"] == target and Path(item["path"]).name != "latest.ckpt"]
    latest = [item for item in ready if Path(item["path"]).name == "latest.ckpt"]
    return (exact or latest or ready or [None])[0]


def make_snapshot(candidate, destination):
    """Verify a complete private copy before launching any evaluation workers."""
    source = Path(candidate["path"])
    before = signature(source)
    if before != tuple(candidate["signature"]):
        raise RuntimeError("checkpoint changed since metadata validation")
    temporary = destination.with_suffix(".partial")
    try:
        digest = hashlib.sha256()
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                digest.update(block)
                outgoing.write(block)
        copied_hash = digest.hexdigest()
        if signature(source) != before or sha256(source) != copied_hash or signature(source) != before:
            raise RuntimeError("checkpoint changed while making the private snapshot")
        metadata = read_metadata(temporary)
        if metadata["epoch"] != candidate["epoch"]:
            raise RuntimeError("snapshot epoch differs from validated source metadata")
        temporary.chmod(0o444)
        temporary.replace(destination)
        return {**metadata, "sha256": copied_hash, "path": str(destination)}
    finally:
        if temporary.exists():
            temporary.unlink()


def completed_attempt(output, stage, args):
    """A process exit code or directory alone is never proof of completion."""
    try:
        manifest = json.loads((output / "manifest.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        success_rate = summary.get("mean_success_rate")
        expected = len(manifest["episodes"])
        return (
            summary.get("complete") is True
            and isinstance(success_rate, (float, int)) and math.isfinite(success_rate)
            and 0 <= success_rate <= 1
            and summary.get("expected_episodes") == expected
            and summary.get("completed_episodes") == expected
            and summary.get("checkpoint_sha256") == stage["checkpoint_sha256"]
            and manifest.get("checkpoint_sha256") == stage["checkpoint_sha256"]
            and manifest.get("training_position", {}).get("epoch") == stage["actual_epoch"]
            and manifest.get("n_per_task") == args.n_per_task
            and manifest.get("init_start") == args.init_start
            and manifest.get("seed") == args.seed
            and manifest.get("suite") == args.suite
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def training_status(program):
    result = subprocess.run(["supervisorctl", "status", program], capture_output=True, text=True, timeout=15)
    fields = result.stdout.strip().split()
    known = {"RUNNING", "STARTING", "BACKOFF", "STOPPED", "STOPPING", "EXITED", "FATAL", "UNKNOWN"}
    if len(fields) < 2 or fields[1] not in known:
        raise RuntimeError(f"Cannot inspect training program {program!r}: {(result.stdout + result.stderr).strip()}")
    return fields[1]


def child_still_running(pid, output):
    # Verify the command as well as PID to avoid waiting on a reused PID.
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return str(output).encode() in command and any(b"eval_heading_policy.py" in arg for arg in command)
    except OSError:
        return False


def terminate_evaluator(process):
    # Stay in supervisor's process group for stopasgroup=true, and explicitly
    # reclaim multiprocessing workers when an individual evaluation times out.
    import psutil
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.terminate()
    _, survivors = psutil.wait_procs(descendants, timeout=10)
    for child in survivors:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(survivors, timeout=5)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", default="5,15,30,60")
    parser.add_argument("--n-per-task", type=int, default=10)
    parser.add_argument("--init-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--training-program")
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-snapshot-attempts", type=int, default=3)
    parser.add_argument("--max-idle-seconds", type=float, default=7200)
    parser.add_argument("--max-eval-seconds", type=float, default=14400)
    parser.add_argument("--training-stop-grace", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        args.epochs = sorted(set(int(value) for value in args.epochs.split(",")))
    except ValueError:
        parser.error("--epochs must be comma-separated integer checkpoint epoch labels")
    if not args.epochs or min(args.epochs) < 0 or args.init_start < 0:
        parser.error("epochs and init-start must be nonnegative")
    if min(args.n_per_task, args.workers, args.max_attempts, args.max_snapshot_attempts) < 1:
        parser.error("episode/worker/attempt counts must be positive")
    if not 0 < args.poll_seconds <= 60 or min(args.max_idle_seconds, args.max_eval_seconds) <= 0 or args.training_stop_grace < 0:
        parser.error("poll interval must be 0..60 seconds and timeouts positive")
    return args


def run(args):
    run_dir, output_root = args.run_dir.resolve(), args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Keep this file descriptor alive for the complete watcher lifetime.
    with (output_root / ".watcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another watcher already owns {output_root}") from error
        settings = {"run_dir": str(run_dir), "epochs": args.epochs, "n_per_task": args.n_per_task,
                    "init_start": args.init_start, "seed": args.seed, "suite": args.suite}
        ledger_path = output_root / "watcher_state.json"
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            if ledger["settings"] != settings:
                raise ValueError("Existing watcher settings differ; use a new output-root for a different protocol")
        else:
            ledger = {"settings": settings, "created_utc": utc_now(), "stages": {}}
        snapshots = output_root / ".snapshots"
        snapshots.mkdir(exist_ok=True)
        cache = MetadataCache()
        last_activity = time.monotonic()
        last_files = None
        stopped_since = None
        try:
            for target in args.epochs:
                stage = ledger["stages"].setdefault(str(target), {"requested_epoch": target, "attempts": [], "snapshot_failures": 0})
                while True:
                    write_json(ledger_path, ledger)
                    complete = next((attempt for attempt in stage["attempts"]
                                     if attempt.get("returncode") in (None, 0)
                                     and completed_attempt(Path(attempt["output"]), stage, args)), None)
                    if complete:
                        stage.update(status="complete", output=complete["output"], completed_utc=utc_now())
                        write_json(ledger_path, ledger)
                        log(f"Stage {target} complete at actual epoch {stage['actual_epoch']}: {stage['output']}")
                        break
                    if stage.get("status") == "complete":
                        raise RuntimeError(f"Stage {target} was marked complete but its final metrics are missing or inconsistent")
                    active = next((attempt for attempt in reversed(stage["attempts"])
                                   if attempt.get("pid") and child_still_running(attempt["pid"], attempt["output"])), None)
                    if active:
                        if time.time() - active["started_timestamp"] > args.max_eval_seconds:
                            raise RuntimeError(f"Existing evaluator PID {active['pid']} exceeded timeout; inspect {active['output']} before restarting")
                        time.sleep(args.poll_seconds)
                        continue
                    if len(stage["attempts"]) >= args.max_attempts:
                        raise RuntimeError(f"Stage {target} failed after {args.max_attempts} attempts; inspect {stage['attempts'][-1]['output']}")
                    if "snapshot" not in stage:
                        paths = checkpoint_candidates(run_dir / "checkpoints", target)
                        files = []
                        for path in paths:
                            try:
                                files.append((str(path), signature(path)))
                            except FileNotFoundError:
                                pass  # A top-k checkpoint can disappear between glob/stat.
                        if files != last_files:
                            last_files, last_activity = files, time.monotonic()
                        candidate = choose_checkpoint([cache.inspect(path) for path in paths], target)
                        if candidate is None:
                            if args.training_program:
                                status = training_status(args.training_program)
                                if status in {"RUNNING", "STARTING", "BACKOFF"}:
                                    stopped_since = None
                                elif stopped_since is None:
                                    stopped_since = time.monotonic()
                                    log(f"Training is {status}; allowing {args.training_stop_grace}s for final checkpoint writes")
                                elif time.monotonic() - stopped_since >= args.training_stop_grace:
                                    raise RuntimeError(f"Training {args.training_program} is {status}, with no complete checkpoint reaching epoch {target}; resume training or revise --epochs in a new output-root")
                            if time.monotonic() - last_activity > args.max_idle_seconds:
                                raise RuntimeError(f"No checkpoint progress for {args.max_idle_seconds}s while waiting for epoch {target}; inspect training and checkpoint writes")
                            log(f"Waiting for a complete checkpoint at epoch >= {target}")
                            time.sleep(args.poll_seconds)
                            continue
                        try:
                            snapshot = make_snapshot(candidate, snapshots / f"target{target:03d}.ckpt")
                        except Exception as error:
                            stage["snapshot_failures"] += 1
                            stage["last_error"] = f"{type(error).__name__}: {error}"
                            log(f"Snapshot not ready, stage {target}: {stage['last_error']}")
                            if stage["snapshot_failures"] >= args.max_snapshot_attempts:
                                raise RuntimeError(f"Stage {target} snapshot failed {args.max_snapshot_attempts} times: {error}") from error
                            time.sleep(args.poll_seconds)
                            continue
                        stage.update(snapshot=snapshot["path"], actual_epoch=snapshot["epoch"],
                                     checkpoint_sha256=snapshot["sha256"], source_checkpoint=candidate["path"])
                    snapshot = Path(stage["snapshot"])
                    if not snapshot.is_file() or sha256(snapshot) != stage["checkpoint_sha256"]:
                        raise RuntimeError(f"Stage {target} snapshot missing or modified: {snapshot}")
                    number = len(stage["attempts"]) + 1
                    name = f"target{target:03d}_actual{stage['actual_epoch']:03d}"
                    output = output_root / (name if number == 1 else f"{name}_attempt{number:02d}")
                    if output.exists():
                        raise RuntimeError(f"Untracked evaluation directory exists: {output}; inspect it before restarting")
                    attempt = {"number": number, "output": str(output), "started_utc": utc_now(),
                               "started_timestamp": time.time(), "status": "starting"}
                    stage["attempts"].append(attempt)
                    stage["status"] = "evaluating"
                    command = [sys.executable, str(ROOT_DIR / "scripts/eval_heading_policy.py"),
                               "--checkpoint", str(snapshot), "--output", str(output), "--suite", args.suite,
                               "--n-per-task", str(args.n_per_task), "--init-start", str(args.init_start),
                               "--seed", str(args.seed), "--workers", str(args.workers)]
                    attempt["command"] = command
                    write_json(ledger_path, ledger)
                    log(f"Evaluating requested epoch {target}, actual {stage['actual_epoch']}, attempt {number}: {output}")
                    with (output_root / f"{output.name}.log").open("a") as evaluation_log:
                        process = subprocess.Popen(command, cwd=ROOT_DIR, env=os.environ.copy(), stdout=evaluation_log, stderr=subprocess.STDOUT)
                        attempt.update(pid=process.pid, status="running")
                        write_json(ledger_path, ledger)
                        try:
                            returncode = process.wait(timeout=args.max_eval_seconds)
                        except subprocess.TimeoutExpired:
                            terminate_evaluator(process)
                            returncode = -1
                            attempt["error"] = "evaluation timeout"
                        except BaseException:
                            terminate_evaluator(process)
                            raise
                    attempt.update(returncode=returncode, finished_utc=utc_now())
                    attempt["status"] = "complete" if returncode == 0 and completed_attempt(output, stage, args) else "failed"
                    if attempt["status"] == "failed":
                        log(f"Evaluation failed/incomplete, stage {target}, return code {returncode}: {output}; full log {output_root / (output.name + '.log')}")
                    last_activity = time.monotonic()
                    write_json(ledger_path, ledger)
            ledger.update(status="complete", completed_utc=utc_now())
            write_json(ledger_path, ledger)
            log(f"All requested stages completed: {ledger_path}")
            return 0
        except BaseException as error:
            ledger.update(status="failed", last_error=f"{type(error).__name__}: {error}", failed_utc=utc_now())
            write_json(ledger_path, ledger)
            raise


def main():
    def stop(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    try:
        return run(parse_args())
    except Exception as error:
        log(f"ERROR: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
