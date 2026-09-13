"""Six-GPU full fine-tuning and original-LIBERO-only checkpoint selection.

Run through torchrun. Every training, inference and consolidated-save operation
is collective under ZeRO-3. Optimizer checkpoints are overwritten in place under
an INCOMPLETE marker; exported inference checkpoints commit metadata last.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from .data import LiberoHDF5Dataset, ORIGINAL_SUITES, collate_libero_samples, load_manifest

STARVLA_REVISION = "3422b9f2387b6f682cf02802904a77b23ab13afd"


def rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def rank_zero_call(function):
    error = None
    if rank() == 0:
        try:
            function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if torch.distributed.is_initialized():
        message = [error]
        torch.distributed.broadcast_object_list(message, src=0)
        error = message[0]
    if error:
        raise RuntimeError(error)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def validate_configuration(config, manifest, actual_world_size=None):
    if config.get("variant") != "heading_gaussian":
        raise ValueError("Only Heading Gaussian is supported by this experiment")
    if (set(manifest["suites"]) != set(ORIGINAL_SUITES) or len(manifest["tasks"]) != 40 or
            len(manifest["episodes"]) != 2000 or not manifest.get("require_complete") or
            manifest.get("expected_demos_per_task") != 50 or manifest.get("missing_tasks")):
        raise ValueError("Full training requires all 40 original LIBERO tasks and 2000 demonstrations")
    if manifest["dataset_origin"] != "original_libero":
        raise ValueError("LIBERO-Plus must never influence training or checkpoint selection")
    if manifest["train_episode_count"] < 1 or manifest["val_episode_count"] < 1:
        raise ValueError("Both original LIBERO training and validation splits are required")
    training = config["training"]
    for key in ("world_size", "micro_batch_size", "gradient_accumulation_steps", "max_steps",
                "checkpoint_interval", "log_interval"):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if training["world_size"] != 6:
        raise ValueError("This full-fine-tuning configuration requires exactly six GPUs")
    if actual_world_size is not None and actual_world_size != training["world_size"]:
        raise ValueError(f"Expected six torchrun processes, got {actual_world_size}")
    if not training.get("full_finetuning", False):
        raise ValueError("All Qwen vision/language/projector and action expert parameters must train")
    for key in ("num_samples", "interval", "micro_batch_size"):
        if int(config["validation"][key]) < 1:
            raise ValueError(f"validation.{key} must be positive")
    if config["validation"]["benchmark"] != "libero":
        raise ValueError("Checkpoint selection may use original LIBERO only")
    if config["validation"]["metric"] != "normalized_action_mse":
        raise ValueError("Select using generated normalized action MSE on original LIBERO validation")
    framework = config["framework"]
    expert = framework["action_model"]
    if (framework["n_obs_steps"], expert["state_dim"], expert["action_dim"],
            expert["action_horizon"], framework["execution_horizon"]) != (2, 9, 7, 16, 8):
        raise ValueError("Expected two-frame state9, seven actions, prediction16 and execution8")
    if framework["image_size"] != 224:
        raise ValueError("Change image resolution only after a separate memory profile")
    if expert["num_inference_timesteps"] != 10:
        raise ValueError("The planned sampler uses ten Euler steps")
    return training["world_size"] * training["micro_batch_size"] * training["gradient_accumulation_steps"]


def deepspeed_configuration(config):
    training = config["training"]
    return {
        "train_batch_size": training["world_size"] * training["micro_batch_size"] * training["gradient_accumulation_steps"],
        "train_micro_batch_size_per_gpu": training["micro_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "bf16": {"enabled": True}, "fp16": {"enabled": False},
        "gradient_clipping": training.get("gradient_clipping", 1.0),
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 3, "overlap_comm": True, "contiguous_gradients": True,
            "reduce_bucket_size": 5_000_000, "stage3_prefetch_bucket_size": 5_000_000,
            "stage3_param_persistence_threshold": 10_000,
            "stage3_max_live_parameters": 100_000_000,
            "stage3_max_reuse_distance": 100_000_000,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "steps_per_print": 1_000_000_000,
        "wall_clock_breakdown": False,
    }


def optimizer_groups(model, settings):
    grouped = {}
    counts = {"qwen": 0, "vision": 0, "expert": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"Full fine-tuning unexpectedly froze {name}")
        family = "vision" if name.startswith("qwen.visual.") else (
            "qwen" if name.startswith("qwen.") else "expert")
        shape = getattr(parameter, "ds_shape", parameter.shape)
        decay = len(shape) >= 2 and not name.endswith("bias")
        key = (family, decay)
        if key not in grouped:
            grouped[key] = {"params": [], "lr": float(settings[family + "_lr"]),
                            "weight_decay": float(settings["weight_decay"]) if decay else 0.0,
                            "name": family + ("_decay" if decay else "_no_decay")}
        grouped[key]["params"].append(parameter)
        counts[family] += int(getattr(parameter, "ds_numel", parameter.numel()))
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Expected trainable language, vision and expert parameters: {counts}")
    return list(grouped.values()), counts


def verify_statistics_buffers(model, statistics, expected_device=None):
    actual = model.action_model.statistics
    expected = {f"{family}_{key}": statistics[family][key]
                for family in ("action", "state") for key in ("scale", "offset")}
    expected["heading_xy_rms"] = statistics["heading_xy_rms"]
    for name, value in expected.items():
        tensor = getattr(actual, name)
        if expected_device is not None and tensor.device != torch.device(expected_device):
            raise ValueError(f"Training-only statistics are on the wrong device: {name} is on {tensor.device}, "
                             f"expected {expected_device}")
        reference = torch.tensor(value, dtype=torch.float32)
        if tensor.dtype != torch.float32 or not torch.equal(tensor.detach().cpu(), reference):
            raise ValueError(f"Training-only statistics lost FP32 fidelity: {name}")


def representative_gradient_parameters(model):
    prefixes = {"vision": "qwen.visual.blocks.", "visual_merger": "qwen.visual.merger.",
                "language": "qwen.language_model.layers.",
                "dit": "action_model.model.transformer_blocks.",
                "heading": "action_model.heading_head."}
    selected = {}
    for name, parameter in model.named_parameters():
        shape = getattr(parameter, "ds_shape", parameter.shape)
        if len(shape) != 1:
            continue
        for family, prefix in prefixes.items():
            if family not in selected and name.startswith(prefix):
                selected[family] = (name, parameter)
    if set(selected) != set(prefixes):
        raise ValueError(f"Cannot audit all trainable components: found {sorted(selected)}")
    return selected


def gradient_audit(model):
    from deepspeed.utils import safe_get_full_grad
    result = {}
    for family, (name, parameter) in sorted(representative_gradient_parameters(model).items()):
        gradient = safe_get_full_grad(parameter)
        if gradient is None:
            raise ValueError(f"No full-fine-tuning gradient for {name}")
        norm = float(gradient.float().norm().item())
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"Expected finite nonzero full-fine-tuning gradient for {name}, got {norm}")
        result[family] = {"parameter": name, "gradient_norm": norm}
    return result


def warmup_multiplier(step, warmup_steps):
    return min(1.0, (step + 1) / max(1, warmup_steps))


def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def isolated_rng(seed=None):
    state = capture_rng_state()
    try:
        if seed is not None:
            seed_everything(seed)
        yield
    finally:
        restore_rng_state(state)


class ResumableBatchSampler:
    """Skip already-consumed batches without decoding their images again."""
    def __init__(self, sampler, micro_batch_size, start_batch=0):
        self.batches = BatchSampler(sampler, micro_batch_size, drop_last=True)
        if not 0 <= start_batch <= len(self.batches):
            raise ValueError("Invalid resume batch cursor")
        self.start_batch = start_batch

    def __iter__(self):
        return itertools.islice(iter(self.batches), self.start_batch, None)

    def __len__(self):
        return len(self.batches) - self.start_batch


def validation_schedule(length, count, *, seed, rank_id, world, micro_batch_size):
    """Equal collective calls on all ranks; duplicates carry zero metric weight."""
    if min(length, count, world, micro_batch_size) < 1 or not 0 <= rank_id < world:
        raise ValueError("Invalid validation schedule dimensions")
    selected = np.random.default_rng(seed).choice(length, size=min(count, length), replace=False).tolist()
    block = world * micro_batch_size
    for start in range(0, len(selected), block):
        batch = []
        for local in range(micro_batch_size):
            offset = start + rank_id * micro_batch_size + local
            batch.append((selected[offset] if offset < len(selected) else selected[0], offset < len(selected)))
        yield batch


def normalized_error(prediction, examples, statistics, included):
    labels = torch.as_tensor(np.stack([item["action"] for item in examples]),
                             dtype=torch.float32, device=prediction.device)
    scale = torch.as_tensor(statistics["action"]["scale"], device=prediction.device, dtype=torch.float32)
    offset = torch.as_tensor(statistics["action"]["offset"], device=prediction.device, dtype=torch.float32)
    mask = torch.as_tensor(np.stack([item["action_mask"] for item in examples]),
                           device=prediction.device, dtype=torch.bool)
    mask = mask & torch.as_tensor(included, device=prediction.device, dtype=torch.bool)[:, None]
    error = (prediction.float() - (labels * scale + offset)).square()
    return (error * mask[..., None]).sum(dtype=torch.float64), mask.sum(dtype=torch.float64) * labels.shape[-1]


@torch.no_grad()
def evaluate_original_libero(engine, dataset, settings, device):
    total = torch.zeros(2, dtype=torch.float64, device=device)
    was_training = engine.training
    engine.eval()
    try:
        with isolated_rng(settings["seed"] + rank()):
            for scheduled in validation_schedule(len(dataset), settings["num_samples"],
                    seed=settings["seed"], rank_id=rank(), world=world_size(),
                    micro_batch_size=settings["micro_batch_size"]):
                examples = [dataset[index] for index, _ in scheduled]
                # Root and child forward hooks must execute to gather ZeRO-3 partitions.
                output = engine(examples, inference=True)
                numerator, denominator = normalized_error(output["normalized_actions"], examples,
                                                           dataset.statistics, [valid for _, valid in scheduled])
                total += torch.stack((numerator, denominator))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total)
        if not torch.isfinite(total).all() or total[1] <= 0:
            raise ValueError("Validation returned non-finite generated actions or no valid labels")
        return float((total[0] / total[1]).item())
    finally:
        engine.train(was_training)


def checkpoint_metadata(config, manifest, step, validation_mse, weights_path, config_path):
    return {"protocol_version": 1, "variant": "heading_gaussian",
            "training_benchmark": "libero", "selection_benchmark": "libero" if validation_mse is not None else None,
            "selection_metric": "normalized_action_mse", "validation_normalized_action_mse": validation_mse,
            "checkpoint_sha256": file_sha256(weights_path), "config_sha256": file_sha256(config_path),
            "dataset_manifest_sha256": manifest["sha256"], "step": step, "seed": config["seed"],
            "image_orientation": "vertical_flip", "image_size": config["framework"]["image_size"],
            "camera_render_size": 128, "image_resize": "bicubic",
            "camera_names": ["agentview", "robot0_eye_in_hand"], "n_obs_steps": 2,
            "state_dim": 9, "action_dim": 7, "prediction_horizon": 16, "execution_horizon": 8,
            "num_inference_timesteps": 10, "full_finetuning": True,
            "world_size": config["training"]["world_size"], "starvla_revision": STARVLA_REVISION}


def _replace_link(path, target):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target)
    os.replace(temporary, path)


def export_checkpoint(engine, output_dir, config, manifest, step, validation_mse, is_best, *, publish=True):
    output_dir = Path(output_dir)
    exports = output_dir / "exports"
    name = f"step_{step:08d}"
    staging, final = exports / ("." + name + ".tmp"), exports / name
    def prepare():
        exports.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(f"Refusing to overwrite committed inference checkpoint {final}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
    rank_zero_call(prepare)
    with isolated_rng():
        saved = engine.save_16bit_model(str(staging), save_filename="weights.pt")
    if not saved:
        raise RuntimeError("ZeRO-3 refused consolidated export; enable gather_16bit_weights_on_model_save")
    barrier()
    def commit():
        write_json(staging / "config.json", config)
        write_json(staging / "dataset_manifest.json", manifest)
        metadata = checkpoint_metadata(config, manifest, step, validation_mse,
                                       staging / "weights.pt", staging / "config.json")
        # Publishing metadata is the commit marker; incomplete weights are never served.
        write_json(staging / "metadata.json", metadata)
        os.replace(staging, final)
    rank_zero_call(commit)
    if publish:
        def publish_export():
            best_step = step if is_best else None
            if not is_best and (output_dir / "best").is_symlink():
                best_step = json.loads((output_dir / "best" / "metadata.json").read_text())["step"]
            publish_checkpoint_links(output_dir, step, best_step)
        rank_zero_call(publish_export)
    return final


def publish_checkpoint_links(output_dir, latest_step, best_step):
    """Publish only exports referenced by a committed optimizer state.

    Calling this during resume also removes completed but uncommitted future
    exports left by interruption between weight export and optimizer commit.
    """
    output_dir = Path(output_dir)
    exports = output_dir / "exports"
    keep = set()
    for label, step in (("latest", latest_step), ("best", best_step)):
        link = output_dir / label
        if step is None:
            if link.is_symlink():
                link.unlink()
            continue
        name = f"step_{step:08d}"
        path = exports / name
        if not (path / "metadata.json").is_file():
            raise FileNotFoundError(f"Committed optimizer state references missing export {path}")
        if json.loads((path / "metadata.json").read_text())["step"] != step:
            raise ValueError(f"Export metadata step differs from the optimizer commit: {path}")
        _replace_link(link, Path("exports") / name)
        keep.add(path.resolve())
    for path in exports.glob("step_*"):
        if path.is_dir() and path.resolve() not in keep:
            shutil.rmtree(path)
    for path in exports.glob(".step_*.tmp"):
        if path.is_dir():
            shutil.rmtree(path)


APPLICATION_STATE_FIELDS = frozenset({
    "step", "epoch", "batch_cursor", "best_validation_mse", "config_sha256",
    "dataset_manifest_sha256", "world_size", "latest_export_step", "best_export_step",
})
APPLICATION_STATE_REQUIRED_FIELDS = frozenset({
    "step", "epoch", "batch_cursor", "config_sha256", "dataset_manifest_sha256", "world_size",
})


def application_training_state(state):
    """Exclude DeepSpeed's reserved metadata from the application resume state.

    load_checkpoint returns its entire checkpoint mapping, including DeepSpeed
    metadata. Passing that mapping back as client_state collides with reserved
    fields on the next save. The commit marker contains only these owned fields.
    """
    result = {key: value for key, value in state.items() if key in APPLICATION_STATE_FIELDS}
    missing = APPLICATION_STATE_REQUIRED_FIELDS - result.keys()
    if missing:
        raise ValueError(f"Missing application checkpoint fields: {sorted(missing)}")
    for key in ("step", "epoch", "batch_cursor", "world_size"):
        value = result[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < (1 if key == "world_size" else 0):
            raise ValueError(f"Invalid application checkpoint integer: {key}")
    for key in ("config_sha256", "dataset_manifest_sha256"):
        value = result[key]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"Invalid application checkpoint SHA256: {key}")
    score = result.get("best_validation_mse")
    if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool)
                              or not math.isfinite(score) or score < 0):
        raise ValueError("Invalid best validation MSE in application checkpoint")
    for key in ("latest_export_step", "best_export_step"):
        value = result.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                  or not 0 <= value <= result["step"]):
            raise ValueError(f"Invalid application checkpoint export step: {key}")
    if (result.get("best_export_step") is not None and result.get("latest_export_step") is not None
            and result["best_export_step"] > result["latest_export_step"]):
        raise ValueError("Best checkpoint step exceeds latest checkpoint step")
    return result


def save_training_state(engine, output_dir, state):
    state = application_training_state(state)
    resume = Path(output_dir) / "resume"
    marker = resume / "INCOMPLETE"
    def mark_incomplete():
        resume.mkdir(parents=True, exist_ok=True)
        marker.write_text("A save is in progress. This optimizer state cannot be resumed until the marker disappears.\n")
    rank_zero_call(mark_incomplete)
    rng = capture_rng_state()
    with isolated_rng():
        engine.save_checkpoint(str(resume), tag="state", client_state=state, save_latest=False)
    torch.save(rng, resume / "state" / f"rng_rank_{rank():05d}.pt")
    barrier()
    rank_zero_call(lambda: (write_json(resume / "commit.json", state), marker.unlink()))


def load_training_state(engine, output_dir, expected_config_hash, expected_manifest_hash):
    resume = Path(output_dir) / "resume"
    def check():
        if (resume / "INCOMPLETE").exists():
            raise RuntimeError("Optimizer checkpoint is INCOMPLETE; recover from a completed checkpoint")
        if not (resume / "commit.json").is_file():
            raise FileNotFoundError("No committed optimizer checkpoint is available")
        state = application_training_state(json.loads((resume / "commit.json").read_text()))
        check_resume_contract(state, expected_config_hash, expected_manifest_hash, world_size())
        if state.get("latest_export_step") is not None:
            publish_checkpoint_links(output_dir, state["latest_export_step"], state.get("best_export_step"))
    rank_zero_call(check)
    committed_state = application_training_state(json.loads((resume / "commit.json").read_text()))
    load_path, loaded_state = engine.load_checkpoint(str(resume), tag="state", load_module_strict=True)
    if load_path is None:
        raise RuntimeError("DeepSpeed failed to load the committed optimizer checkpoint")
    state = application_training_state(loaded_state)
    check_resume_contract(state, expected_config_hash, expected_manifest_hash, world_size())
    if state != committed_state:
        raise ValueError("DeepSpeed application state does not match the authoritative optimizer commit")
    rng = torch.load(resume / "state" / f"rng_rank_{rank():05d}.pt", map_location="cpu", weights_only=False)
    restore_rng_state(rng)
    return state


def check_resume_contract(state, config_hash, manifest_hash, world):
    expected = {"config_sha256": config_hash, "dataset_manifest_sha256": manifest_hash, "world_size": world}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Resume {key} differs from the original training run")
    if state.get("step", -1) < 0 or state.get("epoch", -1) < 0 or state.get("batch_cursor", -1) < 0:
        raise ValueError("Invalid saved sampler cursor")


def repair_completed_status(output_dir, config, state):
    """Finish the last commit after interruption before status publication.

    Only a committed optimizer step exactly at the configured budget can finish
    training. Both its final export and selected best export must be intact and
    validated on original LIBERO. This performs no model or optimizer work.
    """
    maximum = int(config["training"]["max_steps"])
    if state["step"] > maximum:
        raise ValueError("Committed optimizer step exceeds the configured training budget")
    if state["step"] < maximum:
        return False
    output_dir = Path(output_dir)
    resume = output_dir / "resume"
    if (resume / "INCOMPLETE").exists():
        raise RuntimeError("Cannot complete training from an INCOMPLETE optimizer checkpoint")
    committed = application_training_state(json.loads((resume / "commit.json").read_text()))
    if committed != application_training_state(state):
        raise ValueError("Completion state differs from the authoritative optimizer commit")
    check_resume_contract(committed, config_sha256(config), state["dataset_manifest_sha256"],
                          config["training"]["world_size"])
    if committed.get("latest_export_step") != maximum or committed.get("best_export_step") is None:
        raise ValueError("Completion requires the committed final export and a selected best export")
    from .evaluation import validate_metadata
    inspected = {}
    for role, step in (("final", maximum), ("best", committed["best_export_step"])):
        if step in inspected:
            continue
        checkpoint = output_dir / "exports" / f"step_{step:08d}"
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        validate_metadata(metadata, benchmark="libero_plus")
        if (metadata.get("step"), metadata.get("seed"), metadata.get("world_size"),
                metadata.get("full_finetuning"), metadata.get("dataset_manifest_sha256")) != (
                step, config["seed"], config["training"]["world_size"], True,
                committed["dataset_manifest_sha256"]):
            raise ValueError(f"The {role} export does not match the committed training run")
        score = metadata.get("validation_normalized_action_mse")
        if (not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) or score < 0):
            raise ValueError(f"The {role} export lacks a finite original-LIBERO validation score")
        if metadata.get("selection_metric") != "normalized_action_mse":
            raise ValueError(f"The {role} export uses a different checkpoint selection metric")
        if metadata.get("checkpoint_sha256") != file_sha256(checkpoint / "weights.pt"):
            raise ValueError(f"The {role} export weights do not match their SHA256")
        if metadata.get("config_sha256") != file_sha256(checkpoint / "config.json"):
            raise ValueError(f"The {role} export configuration does not match its SHA256")
        if json.loads((checkpoint / "config.json").read_text()) != config:
            raise ValueError(f"The {role} export configuration differs from the completed run")
        inspected[step] = metadata
    best = inspected[committed["best_export_step"]]
    if best["validation_normalized_action_mse"] != committed.get("best_validation_mse"):
        raise ValueError("Selected best validation score differs from the optimizer commit")
    publish_checkpoint_links(output_dir, maximum, committed["best_export_step"])
    write_json(output_dir / "status.json", {"status": "completed", "step": maximum,
                                           "best_validation_mse": committed["best_validation_mse"]})
    return True


def append_metrics(output_dir, event, device):
    memory = {"rank": rank(), "max_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
              "max_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
    memories = [None] * world_size()
    if torch.distributed.is_initialized():
        torch.distributed.all_gather_object(memories, memory)
    else:
        memories = [memory]
    if rank() == 0:
        event = {"time_unix": time.time(), **event, "gpu_memory": memories}
        with (Path(output_dir) / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event, allow_nan=False), flush=True)


def run_training(config, *, resume=False, skip_resume_save=False, stop_after_steps=None):
    import fcntl
    import deepspeed
    from transformers.integrations import HfDeepSpeedConfig
    manifest = load_manifest(config["dataset"]["manifest"])
    validate_configuration(config, manifest, int(os.environ.get("WORLD_SIZE", "1")))
    if not torch.cuda.is_available():
        raise RuntimeError("Full Qwen fine-tuning requires six CUDA GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    torch.cuda.reset_peak_memory_stats(device)
    config = copy.deepcopy(config)
    config["statistics"] = manifest["statistics"]
    signature = config_sha256(config)
    output_dir = Path(config["output_dir"]).resolve()
    lock_handle = None
    def prepare_output():
        nonlocal lock_handle
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = (output_dir / "run.lock").open("a")
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = output_dir / "run_config.json"
        if existing.is_file():
            if not resume:
                raise FileExistsError("Output already contains a run; use --resume or a fresh output directory")
            if config_sha256(json.loads(existing.read_text())) != signature:
                raise ValueError("Resolved configuration differs from the original run")
        elif resume:
            raise FileNotFoundError("Cannot resume without a saved run configuration")
        if not resume and not skip_resume_save:
            minimum = float(config["training"].get("minimum_free_disk_gib", 75)) * 2**30
            available = shutil.disk_usage(output_dir).free
            if available < minimum:
                raise RuntimeError(f"Need {minimum / 2**30:.1f} GiB free for optimizer state and exports; "
                                   f"only {available / 2**30:.1f} GiB free")
        write_json(existing, config)
        write_json(output_dir / "dataset_manifest.json", manifest)
    rank_zero_call(prepare_output)
    training, validation = config["training"], config["validation"]
    ds_config = deepspeed_configuration(config)
    rank_zero_call(lambda: write_json(output_dir / "deepspeed_config.json", ds_config))
    seed_everything(config["seed"])
    # Keep this strong reference alive for the entire run, before from_pretrained.
    hf_deepspeed_config = HfDeepSpeedConfig(ds_config)
    from .model import QwenHeadingGaussian
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = QwenHeadingGaussian(config, statistics=manifest["statistics"], pretrained=True)
    groups, counts = optimizer_groups(model, training["optimizer"])
    optimizer = torch.optim.AdamW(groups, betas=tuple(training["optimizer"]["betas"]),
                                  eps=training["optimizer"]["eps"], foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step: warmup_multiplier(step, training["optimizer"]["warmup_steps"]))
    engine, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer,
                                                          lr_scheduler=scheduler, config=ds_config)
    # ZeRO Init can place all parameters on CUDA while leaving FP32 buffers on
    # CPU; DeepSpeed then skips the usual whole-module device transfer.
    engine.module.action_model.statistics.to(device=device)
    verify_statistics_buffers(engine.module, manifest["statistics"], expected_device=device)
    seed_everything(config["seed"] + rank())
    train_data = LiberoHDF5Dataset(config["dataset"]["manifest"], "train", image_size=224)
    val_data = LiberoHDF5Dataset(config["dataset"]["manifest"], "val", image_size=224)
    sampler = DistributedSampler(train_data, num_replicas=world_size(), rank=rank(),
                                  shuffle=True, seed=config["seed"], drop_last=True)
    state = {"step": 0, "epoch": 0, "batch_cursor": 0, "best_validation_mse": None,
             "config_sha256": signature, "dataset_manifest_sha256": manifest["sha256"],
             "world_size": world_size(), "latest_export_step": None, "best_export_step": None}
    if resume:
        state = load_training_state(engine, output_dir, signature, manifest["sha256"])
        verify_statistics_buffers(engine.module, manifest["statistics"], expected_device=device)
        if engine.global_steps != state["step"]:
            raise ValueError("DeepSpeed optimizer step differs from the saved sampler state")
    stopping = {"requested": False}
    def on_signal(signum, frame):
        stopping["requested"] = True
    old_handlers = {name: signal.signal(name, on_signal) for name in (signal.SIGTERM, signal.SIGINT)}
    engine.train()
    append_metrics(output_dir, {"event": "started", "step": state["step"], "parameter_counts": counts,
                   "trainable_parameters": sum(counts.values()), "effective_batch": ds_config["train_batch_size"],
                   "optimizer_resume_enabled": not skip_resume_save}, device)
    accumulated, micro_count = {}, 0
    started = time.monotonic()
    try:
        if state["step"] >= training["max_steps"]:
            rank_zero_call(lambda: repair_completed_status(output_dir, config, state))
            return state
        while state["step"] < training["max_steps"]:
            epoch, cursor = state["epoch"], state["batch_cursor"]
            sampler.set_epoch(epoch)
            batches = ResumableBatchSampler(sampler, training["micro_batch_size"], cursor)
            total_batches = len(batches.batches)
            # A dedicated loader generator keeps worker seeds from consuming the model RNG.
            loader_generator = torch.Generator().manual_seed(config["seed"] + epoch * 1000 + rank())
            loader = DataLoader(train_data, batch_sampler=batches, collate_fn=collate_libero_samples,
                                num_workers=training["num_workers"], generator=loader_generator,
                                persistent_workers=False)
            for batch_index, examples in enumerate(loader, start=cursor):
                boundary = engine.is_gradient_accumulation_boundary()
                metrics = engine(examples)
                loss = metrics["action_loss"]
                finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Non-finite training loss on at least one GPU")
                engine.backward(loss)
                if boundary and engine.global_steps + 1 in training.get("gradient_audit_steps", [1]):
                    audit = gradient_audit(engine.module)
                    append_metrics(output_dir, {"event": "gradient_audit", "step": engine.global_steps + 1,
                                               "components": audit}, device)
                engine.step()
                for key, value in metrics.items():
                    accumulated[key] = accumulated.get(key, 0.0) + float(value.detach().float().item())
                micro_count += 1
                state["epoch"] = epoch + (batch_index + 1 == total_batches)
                state["batch_cursor"] = 0 if batch_index + 1 == total_batches else batch_index + 1
                if not boundary:
                    continue
                state["step"] = int(engine.global_steps)
                step = state["step"]
                stop_tensor = torch.tensor(int(stopping["requested"]), device=device)
                torch.distributed.all_reduce(stop_tensor, op=torch.distributed.ReduceOp.MAX)
                scheduled_pause = stop_after_steps is not None and step >= stop_after_steps
                should_stop = bool(stop_tensor.item()) or scheduled_pause
                final_step = step >= training["max_steps"]
                if step % training["log_interval"] == 0 or final_step or should_stop:
                    keys = sorted(accumulated)
                    values = torch.tensor([accumulated[key] for key in keys] + [micro_count],
                                           device=device, dtype=torch.float64)
                    torch.distributed.all_reduce(values)
                    reduced = {key: float(values[i] / values[-1]) for i, key in enumerate(keys)}
                    append_metrics(output_dir, {"event": "training", "step": step, "epoch": epoch,
                        "elapsed_seconds": time.monotonic() - started, **reduced,
                        "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}}, device)
                    accumulated, micro_count = {}, 0
                validation_mse, is_best = None, False
                if step % validation["interval"] == 0 or final_step or scheduled_pause:
                    validation_mse = evaluate_original_libero(engine, val_data, validation, device)
                    is_best = state["best_validation_mse"] is None or validation_mse < state["best_validation_mse"]
                    if is_best:
                        state["best_validation_mse"] = validation_mse
                    append_metrics(output_dir, {"event": "validation", "step": step,
                        "benchmark": "libero", "normalized_action_mse": validation_mse,
                        "best": is_best, "sample_count": min(validation["num_samples"], len(val_data))}, device)
                save_now = (step % training["checkpoint_interval"] == 0 or validation_mse is not None
                            or final_step or should_stop)
                if save_now:
                    export_checkpoint(engine, output_dir, config, manifest, step, validation_mse, is_best,
                                      publish=skip_resume_save)
                    state["latest_export_step"] = step
                    if is_best:
                        state["best_export_step"] = step
                    if not skip_resume_save:
                        save_training_state(engine, output_dir, state)
                        rank_zero_call(lambda: publish_checkpoint_links(output_dir, state["latest_export_step"],
                                                                        state["best_export_step"]))
                    append_metrics(output_dir, {"event": "checkpoint", "step": step,
                                               "optimizer_state_saved": not skip_resume_save}, device)
                if final_step or should_stop:
                    rank_zero_call(lambda: write_json(output_dir / "status.json",
                        {"status": "interrupted" if should_stop and not final_step else "completed",
                         "step": step, "best_validation_mse": state["best_validation_mse"]}))
                    return state
            if not total_batches:
                raise ValueError("Dataset is too small for one distributed microbatch")
        return state
    finally:
        train_data.close()
        val_data.close()
        for name, handler in old_handlers.items():
            signal.signal(name, handler)
        if lock_handle is not None:
            lock_handle.close()
        # Preserve HF's ZeRO configuration until all model/checkpoint work is finished.
        _ = hf_deepspeed_config
