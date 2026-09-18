#!/usr/bin/env python3
"""Prepare and sample the exact shared held-out Heading Zero/Gaussian/SMQ subset.

All actions are produced by each checkpoint's native predict_action. Diagnostic
hooks observe conditions/sources/midpoint states; they do not modify inference.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "output/shared_holdout_heading_smq_20260914"
PARENT = ROOT / "output/noise_direction_best_sr_20260912"
INTAKE = ROOT / "output/smq_intake_20260914"
SMQ_RUN = ROOT / "output/spatial_motion_dit/20260913_500ep_seed42"
MODEL_IDS = ("heading_zero_dit", "heading_gaussian_dit", "smq_dit")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_npz(path, values):
    path = Path(path)
    with path.with_suffix(path.suffix + ".tmp").open("wb") as stream:
        np.savez_compressed(stream, **values)
    path.with_suffix(path.suffix + ".tmp").replace(path)


def select_representatives(windows):
    """Three frames per task, without using any model predictions or GT directions."""
    result = []
    for task in sorted({w["task_uid"] for w in windows}):
        task_windows = [w for w in windows if w["task_uid"] == task]
        episode_ids = sorted({w["episode_index"] for w in task_windows})
        # Select the episode with the most available windows; earliest index breaks ties.
        episode = min(episode_ids, key=lambda e: (-sum(w["episode_index"] == e for w in task_windows), e))
        candidates = [w for w in task_windows if w["episode_index"] == episode]
        used = set()
        for phase, fraction in (("early", .15), ("middle", .5), ("late", .85)):
            length = candidates[0]["episode_end"] - candidates[0]["episode_start"]
            target_frame = 1 + fraction * (length - 17)
            chosen = min((w for w in candidates if w["window_index"] not in used),
                         key=lambda w: (abs(w["frame_index"] - target_frame), w["frame_index"]))
            used.add(chosen["window_index"])
            result.append({**chosen, "representative_index": len(result), "phase": phase,
                           "target_fraction_of_unpadded_episode": fraction})
    return result


def prepare(args):
    import torch
    out = args.output
    if (out / "selection.json").exists():
        raise FileExistsError("Selection already exists; reuse it for infer rather than overwriting")
    audit_path = INTAKE / "episode_split_audit.json"
    audit = json.loads(audit_path.read_text())
    for rec in audit["inputs"].values():
        if digest(rec["path"]) != rec["sha256"]:
            raise ValueError(f"Changed split-audit input: {rec['path']}")
    old = json.loads((PARENT / "selection.json").read_text())
    for filename, hash_key in (("data.npz", "data_sha256"), ("noise.npy", "noise_sha256")):
        if digest(PARENT / filename) != old[hash_key]:
            raise ValueError(f"Changed parent input: {filename}")
    indices = audit["reusable_saved_window_indices"]
    if len(indices) != 239 or len(set(indices)) != 239:
        raise ValueError("Expected exactly the approved 239 unique shared held-out windows")
    mapping = {r["server_episode_index"]: r for r in audit["episode_mapping"]}
    windows = []
    for original_index in indices:
        row = old["windows"][original_index]
        match = mapping[row["episode_index"]]
        if match["heading_split"] != "validation" or match["smq_split"] != "validation":
            raise ValueError("A selected episode was used for training")
        if match["task_uid"] != row["task_uid"]:
            raise ValueError("Task identity mismatch")
        windows.append({**row, "window_index": len(windows), "original_window_index": original_index,
                        "smq_episode_index": match["smq_episode_index"]})
    representatives = select_representatives(windows)
    models = {}
    for name in MODEL_IDS[:2]:
        models[name] = dict(old["models"][name])
        models[name]["num_motion_segments"] = 1
    smq_dir = SMQ_RUN / "rollouts/epoch_200/attempt_001"
    manifest = json.loads((smq_dir / "manifest.json").read_text())
    summary = json.loads((smq_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (smq_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if not summary["complete"] or len(rows) != 500 or len({r["episode_id"] for r in rows}) != 500:
        raise ValueError("SMQ evaluation is incomplete or has duplicate episodes")
    if sum(bool(r["success"]) for r in rows) != summary["successes"] or any(r.get("error") for r in rows):
        raise ValueError("SMQ evaluation summary disagrees with its episode records")
    if summary["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("SMQ evaluation hashes disagree")
    models["smq_dit"] = {
        "checkpoint": str(smq_dir / "policy.ckpt"), "checkpoint_sha256": manifest["checkpoint_sha256"],
        "evaluation_summary": str(smq_dir / "summary.json"), "success_rate": summary["successes"] / 500,
        "successes": summary["successes"], "evaluation_episodes": 500,
        "training_position": manifest["training_position"], "weights": manifest["weights"],
        "policy_config": manifest["policy_config"],
        "dataset_config": json.loads((SMQ_RUN / "dataset_config_20260913T200831Z_1522023.json").read_text()),
        "num_motion_segments": 4,
    }
    for name, model in models.items():
        if digest(model["checkpoint"]) != model["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash differs for {name}")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("models", "logs", "provenance", "reproduce"):
        (out / name).mkdir(exist_ok=True)
    with np.load(PARENT / "data.npz", allow_pickle=False) as archive:
        data = {key: archive[key][indices] for key in archive.files}
    save_npz(out / "data.npz", data)
    noise = np.array(np.load(PARENT / "noise.npy", mmap_mode="r")[indices])
    np.save(out / "noise.npy", noise)
    rep_noise = []
    for rep in representatives:
        draws = torch.randn(1024, 16, 7, generator=torch.Generator().manual_seed(rep["noise_seed"])).numpy()
        np.testing.assert_array_equal(draws[:32], noise[rep["window_index"]])
        rep_noise.append(draws)
    np.save(out / "representative_noise.npy", np.stack(rep_noise))
    selection = {
        "schema_version": 1, "models": models, "model_order": list(MODEL_IDS),
        "windows": windows, "representatives": representatives,
        "window_count": 239, "task_ids": sorted({w["task_uid"] for w in windows}),
        "excluded_task_ids": [30, 32, 33, 34, 37],
        "common_validation_episode_indices": sorted({w["episode_index"] for w in windows}),
        "train_episode_indices": old["train_episode_indices"],
        "validation_episode_indices": old["validation_episode_indices"],
        "episode_ends": old["episode_ends"], "dataset_path": old["dataset_path"],
        "source_comparison": str(PARENT), "source_selection_sha256": digest(PARENT / "selection.json"),
        "split_audit": str(audit_path), "split_audit_sha256": digest(audit_path),
        "selection_seed": old["selection_seed"], "draws_per_window": 32,
        "representative_source_draws": 1024, "representative_action_draws": 128,
        "common_direction_xy_rms": 0.32195183634757996, "common_direction_min_confidence": .05,
        "representative_selection": "For each task choose the common held-out episode with most original selected windows (ties lowest episode index); choose distinct original windows nearest 15%,50%,85% of its unpadded frame range. No inference results used.",
        "comparison_scope": "Exactly 239 original windows from 8 episodes held out by Heading Zero, Heading Gaussian, and SMQ. Five tasks only, equal task weight. New native inference with paired original Gaussian draws; fixed best-recorded-SR EMA checkpoints. No retraining or new rollout SR evaluation.",
        "rgb_identity_caveat": "Cross-dataset identities verified by complete action and proprioception hashes. RGB bytes were not transferred/hashed; all models receive identical saved server observations.",
        "data_sha256": digest(out / "data.npz"), "noise_sha256": digest(out / "noise.npy"),
        "representative_noise_sha256": digest(out / "representative_noise.npy"),
    }
    shutil.copy2(audit_path, out / "provenance/episode_split_audit.json")
    for name, model in models.items():
        directory = Path(model["evaluation_summary"]).parent
        dest = out / "provenance" / name
        dest.mkdir(exist_ok=True)
        for filename in ("summary.json", "manifest.json", "episodes.jsonl"):
            shutil.copy2(directory / filename, dest / filename)
    shutil.copy2(SMQ_RUN / "training_dataset.json", out / "provenance/smq_training_dataset.json")
    write_json(out / "selection.json", selection)
    print(f"Prepared {len(windows)} windows, {len(representatives)} representative frames, {len(selection['task_ids'])} tasks", flush=True)


def instrumented_prediction(policy, obs, noise, smq=False):
    """Capture native intermediate tensors without changing sampling arithmetic."""
    import torch
    captured = {}
    encoder = policy if smq else policy.obs_encoder
    encode_name = "encode_condition" if smq else "encode_with_heading"
    source_name = "make_source" if smq else "_transform_source"
    original_encode, original_source = getattr(encoder, encode_name), getattr(policy, source_name)
    calls = [0]
    def encode(*a, **kw):
        result = original_encode(*a, **kw)
        captured["prediction"] = result[1]
        return result
    def source(*a, **kw):
        result = original_source(*a, **kw)
        captured["source"], captured["active"] = result
        return result
    def before_velocity(module, inputs):
        calls[0] += 1
        if calls[0] == policy.num_inference_steps // 2 + 1:
            captured["middle"] = policy.normalizer["action"].unnormalize(inputs[0]).detach().clone()
    def paired_randn(*a, **kw):
        shape = tuple(a[0]) if len(a) == 1 and isinstance(a[0], (tuple, list)) else tuple(a)
        if shape != tuple(noise.shape):
            raise ValueError(f"Unexpected random draw request: {shape}; expected {tuple(noise.shape)}")
        if kw.get("dtype", noise.dtype) != noise.dtype:
            raise ValueError("Native noise dtype differs from the saved paired Gaussian dtype")
        return noise
    with ExitStack() as stack:
        stack.enter_context(patch.object(encoder, encode_name, side_effect=encode))
        stack.enter_context(patch.object(policy, source_name, side_effect=source))
        stack.enter_context(patch("torch.randn", side_effect=paired_randn))
        hook = policy.model.register_forward_pre_hook(before_velocity)
        stack.callback(hook.remove)
        result = policy.predict_action(obs)
    if calls[0] != policy.num_inference_steps or "middle" not in captured:
        raise ValueError("Native flow step count differs from configured inference steps")
    return result, captured


def check_native_parity(policy, obs, noise, smq):
    import torch
    actual, capture = instrumented_prediction(policy, obs, noise, smq)
    with patch("torch.randn", return_value=noise):
        expected = policy.predict_action(obs)
    torch.testing.assert_close(actual["action_pred"], expected["action_pred"], rtol=0, atol=0)
    if not smq:
        explicit = policy.predict_action(obs, noise=noise, return_source=True)
        torch.testing.assert_close(explicit["source"], capture["source"], rtol=0, atol=0)
        torch.testing.assert_close(explicit["action_pred"], expected["action_pred"], rtol=0, atol=0)
    return float((actual["action_pred"] - expected["action_pred"]).abs().max())


def sample(policy, observations, indices, draws, action_draws, smq, label):
    import torch
    n, source_draws = len(indices), draws.shape[1]
    segments = 4 if smq else 1
    result = {
        "action_pred": np.empty((n, action_draws, 16, 7), np.float32),
        "source_raw": np.empty((n, source_draws, 16, 7), np.float32),
        "source_norm": np.empty((n, source_draws, 16, 7), np.float32),
        "heading_direction": np.empty((n, segments, 2), np.float32),
        "heading_confidence": np.empty((n, segments), np.float32),
        "source_active": np.empty((n, source_draws, segments), bool),
    }
    if label == "representatives":
        result["flow_mid_raw"] = np.empty((n, action_draws, 16, 7), np.float32)
    start_time = time.monotonic()
    for r, wi in enumerate(indices):
        base_obs = {key: torch.from_numpy(value[wi:wi+1].copy()).to(device=policy.device, dtype=policy.dtype)
                    for key, value in observations.items()}
        first_prediction = None
        for offset in range(0, action_draws, 32):
            count = min(32, action_draws - offset)
            obs = {key: value.expand(count, *value.shape[1:]).contiguous() for key, value in base_obs.items()}
            noise = torch.tensor(np.array(draws[r, offset:offset+count]), device=policy.device)
            output, captured = instrumented_prediction(policy, obs, noise, smq)
            prediction = captured["prediction"]
            direction = prediction["direction"] if smq else prediction["direction"][:, None, :]
            confidence = prediction["confidence"] if smq else prediction["confidence"][:, None]
            active = captured["active"] if smq else captured["active"][:, None]
            torch.testing.assert_close(direction, direction[:1].expand_as(direction), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(confidence, confidence[:1].expand_as(confidence), atol=1e-6, rtol=1e-6)
            if first_prediction is None:
                first_prediction = {key: value[:1].clone() for key, value in prediction.items()}
                result["heading_direction"][r] = direction[0].float().cpu().numpy()
                result["heading_confidence"][r] = confidence[0].float().cpu().numpy()
            else:
                torch.testing.assert_close(direction[0], torch.tensor(result["heading_direction"][r],device=direction.device).to(direction.dtype), rtol=0, atol=0)
            source = captured["source"]
            raw_source = policy.normalizer["action"].unnormalize(source)
            for tensor in (output["action_pred"], source, raw_source, direction, confidence):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite native model output at window {wi}")
            torch.testing.assert_close(source[..., 2:], (noise * policy.prior_noise_scale)[..., 2:], rtol=0, atol=0)
            result["action_pred"][r, offset:offset+count] = output["action_pred"].float().cpu().numpy()
            result["source_norm"][r, offset:offset+count] = source.float().cpu().numpy()
            result["source_raw"][r, offset:offset+count] = raw_source.float().cpu().numpy()
            result["source_active"][r, offset:offset+count] = active.cpu().numpy()
            if "flow_mid_raw" in result:
                result["flow_mid_raw"][r, offset:offset+count] = captured["middle"].float().cpu().numpy()
        if source_draws > action_draws:
            count = source_draws - action_draws
            noise = torch.tensor(np.array(draws[r, action_draws:]), device=policy.device)
            pred = {key: value.expand(count, *value.shape[1:]) for key, value in first_prediction.items()}
            if smq:
                source, active = policy.make_source(noise, pred)
            else:
                source, active = policy._transform_source(noise * policy.prior_noise_scale, pred, angular_jitter=None)
                active = active[:, None]
            raw_source = policy.normalizer["action"].unnormalize(source)
            if not torch.isfinite(source).all() or not torch.isfinite(raw_source).all():
                raise ValueError("Nonfinite supplemental source sample")
            torch.testing.assert_close(source[..., 2:], (noise * policy.prior_noise_scale)[..., 2:], rtol=0, atol=0)
            result["source_norm"][r, action_draws:] = source.float().cpu().numpy()
            result["source_raw"][r, action_draws:] = raw_source.float().cpu().numpy()
            result["source_active"][r, action_draws:] = active.cpu().numpy()
        if r % 20 == 0 or r + 1 == n:
            print(f"{label}: {r+1}/{n} windows, {time.monotonic()-start_time:.1f}s", flush=True)
    return result


def infer(args):
    import torch
    from oat.env_runner.libero_official_eval import load_policy
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.20, device=args.device)
    out = args.output
    selection = json.loads((out / "selection.json").read_text())
    meta = selection["models"][args.model]
    path = out / "models" / f"{args.model}.npz"
    if path.exists() and not args.smoke:
        raise FileExistsError(f"Model result already exists: {path}")
    for filename,key in (("data.npz","data_sha256"),("noise.npy","noise_sha256"),("representative_noise.npy","representative_noise_sha256")):
        if digest(out / filename) != selection[key]:
            raise ValueError(f"Input hash differs: {filename}")
    if digest(meta["checkpoint"]) != meta["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs")
    policy,cfg,weights = load_policy(meta["checkpoint"], args.device)
    if (policy.horizon, policy.n_obs_steps, policy.n_action_steps, policy.num_inference_steps)!=(16,2,8,10):
        raise ValueError("Checkpoint action/observation/inference settings are incompatible")
    smq = args.model == "smq_dit"
    with np.load(out / "data.npz",allow_pickle=False) as data:
        observations = {key[5:]:data[key] for key in data.files if key.startswith("obs__")}
    noise=np.load(out / "noise.npy",mmap_mode="r")
    rep_noise=np.load(out / "representative_noise.npy",mmap_mode="r")
    begin=time.monotonic()
    with torch.inference_mode():
        obs = {key:torch.tensor(value[:1],device=args.device).to(policy.dtype).expand(32,*value.shape[1:]).contiguous()
               for key,value in observations.items()}
        parity = check_native_parity(policy,obs,torch.tensor(np.array(noise[0]),device=args.device),smq)
        print(f"{args.model}: native instrumentation parity max error {parity}",flush=True)
        if args.smoke:
            sample(policy, observations, [0], noise[:1], 32, smq, "smoke")
            print("SMOKE PASSED",flush=True)
            return
        result=sample(policy,observations,list(range(len(selection["windows"]))),noise,32,smq,"main")
        rep_indices=[r["window_index"] for r in selection["representatives"]]
        reps=sample(policy,observations,rep_indices,rep_noise,128,smq,"representatives")
    for key in ("action_pred","source_norm","source_raw","source_active"):
        np.testing.assert_array_equal(reps[key][:,:32],result[key][rep_indices])
    for key in ("heading_direction","heading_confidence"):
        np.testing.assert_array_equal(reps[key],result[key][rep_indices])
    save_npz(path,result)
    rep_path=out / "models" / f"{args.model}_representatives.npz"
    save_npz(rep_path,reps)
    params=policy.normalizer["action"].params_dict
    metadata={**meta,"model_id":args.model,"weights":weights,"complete":True,
              "native_inference_max_absolute_difference":parity,"representative_first32_draws_match":True,
              "paired_source_nonxy_bit_identical":True,"heading_xy_rms":float(policy.heading_xy_rms),
              "min_target_confidence":float(policy.min_target_confidence),
              "min_source_confidence":float(policy.min_source_confidence),
              "normalizer_scale":params["scale"].detach().cpu().tolist(),
              "normalizer_offset":params["offset"].detach().cpu().tolist(),
              "device":str(policy.device),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "inference_bf16_enabled":bool(getattr(policy,"inference_bf16",False) and policy.device.type=="cuda" and torch.cuda.is_bf16_supported()),
              "sampling":"Unmodified native predict_action; fixed32-draw batches of repeated identical observations; torch.randn supplies saved paired Gaussian arrays. Observer hooks capture native source, head and midpoint tensors.",
              "source_supplement":"Representative source draws128:1024 use the same captured observation-only prediction and native source function; first128 sources are captured from native action inference.",
              "elapsed_seconds":time.monotonic()-begin,"script_sha256":digest(__file__),
              "data_sha256":selection["data_sha256"],"noise_sha256":selection["noise_sha256"],
              "result_sha256":digest(path),"representative_result_sha256":digest(rep_path),
              "torch_version":torch.__version__,"numpy_version":np.__version__}
    if not smq:
        original=[w["original_window_index"] for w in selection["windows"]]
        with np.load(PARENT / "models" / f"{args.model}.npz") as previous:
            metadata["previous_saved_heading_action_max_absolute_difference"] = float(np.max(np.abs(result["action_pred"]-previous["action_pred"][original])))
    if digest(meta["checkpoint"]) != meta["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during inference")
    write_json(out / "models" / f"{args.model}.json",metadata)
    print(f"COMPLETE {args.model}: {metadata['elapsed_seconds']:.1f}s",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","infer"))
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--model",choices=MODEL_IDS)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if args.stage=="infer" and args.model is None:
        parser.error("infer requires --model")
    args.output=args.output.resolve()
    (prepare if args.stage=="prepare" else infer)(args)


if __name__=="__main__":
    main()
