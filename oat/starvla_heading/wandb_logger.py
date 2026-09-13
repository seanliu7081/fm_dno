"""Mirror the append-only experiment metrics to W&B without touching training."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def read_events(path, previous=None):
    data = Path(path).read_bytes()
    if previous:
        count = previous["byte_count"]
        if len(data) < count or hashlib.sha256(data[:count]).hexdigest() != previous["sha256"]:
            raise ValueError(f"Metrics source was truncated or rewritten: {path}")
    end = data.rfind(b"\n") + 1
    complete = data[:end]
    events = [json.loads(line) for line in complete.splitlines()]
    return events, {"byte_count": end, "sha256": hashlib.sha256(complete).hexdigest()}


def pending_events(events, next_step):
    if next_step < 0 or next_step > len(events):
        raise ValueError("W&B history position is inconsistent with the metrics source")
    return enumerate(events[next_step:], start=next_step)


def numeric_fields(value, prefix, output):
    if isinstance(value, dict):
        for key, item in value.items():
            numeric_fields(item, f"{prefix}/{key}", output)
    elif isinstance(value, (int, float, bool)):
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite metric: {prefix}")
        output[prefix] = value


def event_metrics(event, index):
    for name, value in (("optimizer_step", event["step"]), ("source/event_index", index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    kind = event["event"]
    result = {"source/event_index": index, "optimizer_step": event["step"], "event_type": kind}
    if "time_unix" in event:
        numeric_fields(event["time_unix"], "source/time_unix", result)
    prefix = {"training": "train", "validation": "validation/libero",
              "gradient_audit": "gradient_audit", "started": "initialization"}.get(kind, kind)
    for key, value in event.items():
        if key not in {"event", "step", "time_unix", "gpu_memory", "learning_rates"}:
            numeric_fields(value, f"{prefix}/{key}", result)
    numeric_fields(event.get("learning_rates", {}), "lr", result)
    for memory in event.get("gpu_memory", []):
        rank = int(memory["rank"])
        for source, target in (("max_allocated_gib", "peak_allocated_gib"),
                               ("max_reserved_gib", "peak_reserved_gib")):
            numeric_fields(memory[source], f"gpu/rank_{rank}/{target}", result)
    return result


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


TRAINING_STREAM = "metrics.jsonl"
PERIODIC_STREAM = "periodic_evaluation/events.jsonl"


def periodic_identity(event):
    """Validate a verified evaluation; retries may differ only in timestamp."""
    for name in ("seed", "step", "episodes"):
        value = event.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Periodic evaluation {name} must be a nonnegative integer")
    if event.get("event") != "evaluation_verified" or event["step"] not in (10000, 20000, 30000):
        raise ValueError("Periodic evaluations must verify checkpoints at 10000, 20000, or 30000 steps")
    expected_episodes = {"libero": 2000, "libero_plus": 10030}
    benchmark = event.get("benchmark")
    if benchmark not in expected_episodes or event["episodes"] != expected_episodes[benchmark]:
        raise ValueError("Periodic evaluation must contain a complete supported benchmark")
    rate = event.get("success_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Periodic success rate must be finite and in [0, 1]")
    digest = event.get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Periodic evaluation requires a checkpoint SHA256")
    timestamp = event.get("time_unix")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError("Periodic evaluation requires a finite timestamp")
    return (event["seed"], event["step"], benchmark, digest)


def periodic_metrics(event, source_index):
    periodic_identity(event)
    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
        raise ValueError("Periodic source index must be a nonnegative integer")
    prefix = f"evaluation/periodic/{event['benchmark']}"
    return {"source/event_index": source_index, "source/time_unix": event["time_unix"],
            "optimizer_step": event["step"], "event_type": "periodic_evaluation_verified",
            f"{prefix}/success_rate": event["success_rate"], f"{prefix}/episodes": event["episodes"],
            f"{prefix}/checkpoint_sha256": event["checkpoint_sha256"]}


def ledger_metrics(row, ledger_index):
    stream = row["stream"]
    if stream == TRAINING_STREAM:
        metrics = event_metrics(row["event"], row["source_index"])
    elif stream == PERIODIC_STREAM:
        metrics = periodic_metrics(row["event"], row["source_index"])
    else:
        raise ValueError(f"Unknown W&B history source: {stream}")
    return {**metrics, "source/stream": stream, "source/ledger_index": ledger_index}


class EventLedger:
    """Durable ordering across two append-only inputs; W&B step is its row index.

    Initial migration copies every complete training row first, preserving all
    legacy W&B steps. Each refresh is fsynced before any row can be uploaded.
    """

    def __init__(self, directory, root, seed, identity, legacy_fingerprint=None, required=False):
        self.directory, self.root, self.seed = directory, root, seed
        self.path = directory / "wandb_history_ledger.json"
        self.identity = identity
        self.state = read_json(self.path)
        if self.state is None:
            if required:
                raise ValueError("The existing W&B history ledger is missing; refusing to reorder history")
            self.state = {"version": 1, "identity": identity, "rows": [], "training_count": 0,
                          "periodic_count": 0, "training_fingerprint": legacy_fingerprint,
                          "periodic_fingerprint": None}
            self.new = True
        else:
            if self.state.get("version") != 1 or self.state.get("identity") != identity:
                raise ValueError("W&B history ledger identity or version differs from the run")
            self.new = False
        self.refresh()

    def refresh(self):
        old = self.state
        training, training_fp = read_events(self.directory / TRAINING_STREAM, old["training_fingerprint"])
        periodic_path = self.root / PERIODIC_STREAM
        if periodic_path.exists():
            periodic, periodic_fp = read_events(periodic_path, old["periodic_fingerprint"])
        else:
            if old["periodic_fingerprint"] and old["periodic_fingerprint"]["byte_count"]:
                raise ValueError("Periodic evaluation source disappeared after being recorded")
            periodic, periodic_fp = [], {"byte_count": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        for name, source in (("training_count", training), ("periodic_count", periodic)):
            count = old[name]
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(source):
                raise ValueError(f"Invalid W&B ledger source position: {name}")
        rows = list(old["rows"])
        training_indices = []
        periodic_indices = []
        seen = {}
        for row in rows:
            index, stream = row["source_index"], row["stream"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("Invalid W&B ledger source index")
            source = training if stream == TRAINING_STREAM else periodic if stream == PERIODIC_STREAM else None
            count = old["training_count"] if stream == TRAINING_STREAM else old["periodic_count"]
            if source is None or index >= count or row["event"] != source[index]:
                raise ValueError("W&B history ledger disagrees with its source")
            ledger_metrics(row, 0)  # Validate stored events before any upload.
            if stream == TRAINING_STREAM:
                training_indices.append(index)
            else:
                event = row["event"]
                identity = periodic_identity(event)
                if identity[0] != self.seed or identity[:3] in seen:
                    raise ValueError("Duplicate or wrong-seed periodic event in W&B ledger")
                seen[identity[:3]] = event
                periodic_indices.append(index)
        if training_indices != list(range(old["training_count"])) or periodic_indices != sorted(set(periodic_indices)):
            raise ValueError("W&B ledger source rows were omitted or reordered")

        # A completed training prefix always comes first during legacy migration.
        for index in range(old["training_count"], len(training)):
            event_metrics(training[index], index)
            rows.append({"stream": TRAINING_STREAM, "source_index": index, "event": training[index]})

        # Recheck the complete evaluation prefix as well as new rows: a retry's
        # conflicting result must not slip through a previously advanced cursor.
        expected_indices = []
        source_seen = {}
        for index, event in enumerate(periodic):
            if event.get("event") != "evaluation_verified" or event.get("seed") != self.seed:
                continue
            identity = periodic_identity(event)
            slot = identity[:3]
            previous = source_seen.get(slot)
            if previous is not None:
                fields = ("seed", "step", "benchmark", "checkpoint_sha256", "episodes", "success_rate")
                if any(event[k] != previous[k] for k in fields):
                    raise ValueError(f"Conflicting periodic evaluation for {slot}")
                continue
            source_seen[slot] = event
            expected_indices.append(index)
            if index >= old["periodic_count"]:
                rows.append({"stream": PERIODIC_STREAM, "source_index": index, "event": event})
        if periodic_indices != [i for i in expected_indices if i < old["periodic_count"]]:
            raise ValueError("Verified periodic evaluation was omitted from W&B ledger")
        updated = {"version": 1, "identity": self.identity, "rows": rows,
                   "training_count": len(training), "periodic_count": len(periodic),
                   "training_fingerprint": training_fp, "periodic_fingerprint": periodic_fp}
        if self.new or updated != old:
            write_json(self.path, updated)
            self.state = updated
            self.new = False
        return training

    @property
    def rows(self):
        return self.state["rows"]


def periodic_progress(root, seed, rows):
    """Separate periodic metrics and gate completion on all scheduled results."""
    schedule = read_json(root / "periodic_evaluation/config.json")
    events = [r["event"] for r in rows if r["stream"] == PERIODIC_STREAM]
    summary = {}
    for benchmark in ("libero", "libero_plus"):
        latest = max((e for e in events if e["benchmark"] == benchmark),
                     key=lambda e: e["step"], default=None)
        if latest:
            prefix = f"evaluation/periodic/{benchmark}"
            summary.update({f"{prefix}/latest_success_rate": latest["success_rate"],
                            f"{prefix}/latest_checkpoint_step": latest["step"],
                            f"{prefix}/latest_checkpoint_sha256": latest["checkpoint_sha256"]})
    if schedule is None or seed not in schedule["seeds"]:
        return True, summary
    expected = {(step, benchmark, episodes) for step in schedule["steps"]
                for benchmark, episodes in schedule["benchmarks"].items()}
    observed = {(e["step"], e["benchmark"], e["episodes"]) for e in events}
    status = read_json(root / f"periodic_evaluation/seed{seed}/status.json", {})
    coordinator = read_json(root / "periodic_evaluation/status.json", {})
    complete = status.get("status") == "completed" and expected <= observed
    summary.update({"evaluation/periodic/completed": complete,
                    "evaluation/periodic/interval_steps": 10000,
                    "evaluation/periodic/coordinator_status": coordinator.get("status", "not_started"),
                    "evaluation/periodic/coordinator_error": coordinator.get("error"),
                    "evaluation/periodic/seed_status": status.get("status", "pending"),
                    "evaluation/periodic/seed_error": status.get("error"),
                    "evaluation/periodic/verified_jobs": len(expected & observed),
                    "evaluation/periodic/expected_jobs": len(expected)})
    return complete, summary


def supervisor_running(service):
    try:
        result = subprocess.run(["supervisorctl", "status", service], capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode not in (0, 3):
        return None
    fields = result.stdout.split()
    return fields[1] == "RUNNING" if len(fields) > 1 else None


def verified_evaluations(root, seed):
    path = root / "experiment_events.jsonl"
    if not path.exists():
        return {}
    events, _ = read_events(path)
    return {f"evaluation/{e['benchmark']}/success_rate": e["success_rate"]
            for e in events if e.get("event") == "evaluation_verified" and e.get("seed") == seed}


class SeedMirror:
    def __init__(self, root, seed, entity, project, group, wandb):
        self.directory = root / f"seed{seed}"
        self.root, self.seed = root, seed
        self.metadata_path = self.directory / "wandb_online.json"
        self.config = read_json(self.directory / "run_config.json")
        digest = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.identity = {"entity": entity, "project": project, "run_id": f"svhg{seed}-{digest[:12]}",
                         "config_sha256": digest}
        existing = read_json(self.metadata_path, {})
        if existing and any(existing.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Online logger identity differs from the existing seed run")
        self.fingerprint = existing.get("source_fingerprint")
        # Preserve the legacy prefix audit and create durable source ordering
        # before starting or resuming a remote run.
        read_events(self.directory / "metrics.jsonl", self.fingerprint)
        self.ledger = EventLedger(self.directory, root, seed, self.identity, self.fingerprint,
                                  required=existing.get("history_ledger_version") is not None)
        self.run = wandb.init(
            entity=entity, project=project, id=self.identity["run_id"], resume="allow",
            name=f"Qwen2.5-VL-3B-DiT-HeadingGaussian-seed{seed}", group=group,
            job_type="training-metrics", mode="online", force=True, reinit="create_new",
            dir=str(root / "wandb_tracking"), config=self.config, save_code=False,
            tags=["starvla", "heading_gaussian", "full_finetuning", "libero", "libero-plus-zero-shot"],
            notes="Live mirror of local training and verified periodic evaluation events. Durable merged "
                  "ledger index is the W&B history step; optimizer_step is the chart axis. "
                  "GPU memory values are trainer-reported lifetime peaks. "
                  "Training and checkpoint files are managed independently of this logger.",
            settings=wandb.Settings(console="off", disable_code=True, disable_git=True,
                                    x_disable_stats=True, x_save_requirements=False,
                                    init_timeout=60, finish_timeout=45))
        self.run.define_metric("optimizer_step")
        self.run.define_metric("*", step_metric="optimizer_step", step_sync=False)
        # This is W&B's acknowledged resume position, not the local last-queued cursor.
        self.next_step = self.run.step
        list(pending_events(self.ledger.rows, self.next_step))
        self.finished = False
        self.last_summary = None
        self.run.summary["tracking/source"] = "metrics.jsonl + periodic_evaluation/events.jsonl"
        self.run.summary["tracking/system_metrics"] = "trainer-reported per-rank peak GPU memory"
        self.run.summary["tracking/config_sha256"] = digest
        print(json.dumps({"event": "wandb_run_connected", "seed": seed, "url": self.run.url,
                          "resume_history_step": self.next_step}), flush=True)

    def poll(self, service_live):
        events = self.ledger.refresh()
        fingerprint = self.ledger.state["training_fingerprint"]
        queued = 0
        for index, row in pending_events(self.ledger.rows, self.next_step):
            self.run.log(ledger_metrics(row, index), step=index, commit=True)
            self.next_step = index + 1
            queued += 1
        self.fingerprint = fingerprint
        latest = next((e for e in reversed(events) if e["event"] == "training"), None)
        status = read_json(self.directory / "status.json", {})
        experiment_completed = read_json(self.directory / "experiment_seed_status.json", {}).get("status") == "completed"
        periodic_complete, periodic_summary = periodic_progress(self.root, self.seed, self.ledger.rows)
        completed = experiment_completed and periodic_complete
        training_complete = (status.get("status") == "completed" and
                             status.get("step") == self.config["training"]["max_steps"])
        stage = ("completed" if completed else "periodic_evaluation" if experiment_completed
                 else "evaluation" if training_complete else "training")
        queued_rows = self.ledger.rows[:self.next_step]
        training_queued = sum(row["stream"] == TRAINING_STREAM for row in queued_rows)
        summary = {"experiment/stage": stage, "tracking/logger_status": "running",
                   "tracking/coordinator_running": service_live,
                   "tracking/source_events_queued": training_queued,
                   "tracking/history_events_queued": self.next_step,
                   "tracking/periodic_events_queued": self.next_step - training_queued,
                   **verified_evaluations(self.root, self.seed), **periodic_summary}
        if latest:
            summary["training/latest_step"] = latest["step"]
            summary["training/latest_event_time_unix"] = latest["time_unix"]
        commit = read_json(self.directory / "resume/commit.json", {})
        if commit:
            summary["checkpoint/committed_step"] = commit["step"]
            summary["checkpoint/best_export_step"] = commit["best_export_step"]
            summary["validation/libero/best_normalized_action_mse"] = commit["best_validation_mse"]
        summary["checkpoint/save_in_progress"] = (self.directory / "resume/INCOMPLETE").exists()
        for role in ("best", "latest"):
            path = self.directory / role
            if path.exists():
                summary[f"checkpoint/{role}_path"] = str(path.resolve())
        if summary != self.last_summary:
            self.run.summary.update(summary)
            self.last_summary = summary
        metadata = {**self.identity, "url": self.run.url, "source_fingerprint": fingerprint,
                    "last_queued_source_event_index": training_queued - 1,
                    "history_ledger_version": 1,
                    "last_queued_history_event_index": self.next_step - 1,
                    "periodic_source_fingerprint": self.ledger.state["periodic_fingerprint"],
                    "last_observed_training_step": latest["step"] if latest else None,
                    "experiment_stage": stage, "logger_observed_at_unix": time.time()}
        write_json(self.metadata_path, metadata)
        if queued:
            print(json.dumps({"event": "wandb_events_queued", "seed": self.seed,
                              "count": queued, "source_events": training_queued,
                              "history_events": self.next_step,
                              "training_step": metadata["last_observed_training_step"]}), flush=True)
        if completed:
            self.run.summary["tracking/logger_status"] = "complete"
            self.run.finish(exit_code=0)
            self.finished = True
        return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="fm_dno_starvla")
    parser.add_argument("--service", default="fm-starvla-heading-experiment")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    tracking = root / "wandb_tracking"
    tracking.mkdir(parents=True, exist_ok=True)
    lock = (tracking / "logger.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import wandb
    config = read_json(root / "experiment_config.json")["config"]
    seeds = config.get("seeds", [42, 43, 44])
    group = "heading-gaussian-" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    mirrors = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    failed = True
    try:
        while not stopping:
            live = supervisor_running(args.service)
            for seed in seeds:
                directory = root / f"seed{seed}"
                if seed not in mirrors and (directory / "metrics.jsonl").exists() and (directory / "run_config.json").exists():
                    mirrors[seed] = SeedMirror(root, seed, args.entity, args.project, group, wandb)
                mirror = mirrors.get(seed)
                if mirror is not None and not mirror.finished:
                    mirror.poll(live)
            if len(mirrors) == len(seeds) and all(m.finished for m in mirrors.values()):
                failed = False
                break
            deadline = time.monotonic() + args.poll_seconds
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(.5, max(0, deadline - time.monotonic())))
    finally:
        for mirror in mirrors.values():
            if not mirror.finished:
                mirror.run.summary["tracking/logger_status"] = "stopped" if stopping else "error"
                mirror.run.finish(exit_code=255 if stopping else 1 if failed else 0)
        lock.close()


if __name__ == "__main__":
    main()
