#!/usr/bin/env python3
"""Verify shared-holdout artifacts without model loading, GPU work, or rendering.

Writes verification.json. Missing downstream stages are explicitly incomplete;
--require-plots requires every stage, including all 108 final figure assets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/shared_holdout_heading_smq_20260914"
MODELS = ["heading_zero_dit", "heading_gaussian_dit", "smq_dit"]
TASK_COUNTS = {31: 77, 35: 64, 36: 30, 38: 26, 39: 42}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonstandard nonfinite JSON constant {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def episode_key(episode):
    return (episode["task_uid"], episode["length"],
            tuple((key, tuple(value["shape"]), value["sha256"])
                  for key, value in sorted(episode["arrays"].items())))


def verify_inputs(output, selection):
    parent = Path(selection["source_comparison"])
    require(digest(parent / "selection.json") == selection["source_selection_sha256"], "Parent selection hash changed")
    original = read_json(parent / "selection.json")
    audit_path = Path(selection["split_audit"])
    require(digest(audit_path) == selection["split_audit_sha256"], "Split audit hash changed")
    require(digest(output / "provenance/episode_split_audit.json") == selection["split_audit_sha256"], "Copied split audit differs")
    audit = read_json(audit_path)
    for record in audit["inputs"].values():
        require(digest(record["path"]) == record["sha256"], f"Audit input changed: {record['path']}")
    server = read_json(audit["inputs"]["server_manifest"]["path"])
    smq = read_json(audit["inputs"]["smq_manifest"]["path"])
    training = read_json(audit["inputs"]["smq_training_split"]["path"])
    require(digest(output / "provenance/smq_training_dataset.json") == audit["inputs"]["smq_training_split"]["sha256"], "Copied SMQ split differs")
    require(len(server["episodes"]) == len(smq["episodes"]) == 500, "Expected 500 episodes per dataset")
    server_index = {episode_key(e): e["episode_index"] for e in server["episodes"]}
    smq_index = {episode_key(e): e["episode_index"] for e in smq["episodes"]}
    require(len(server_index) == len(smq_index) == 500, "Duplicate episode content")
    require(server_index.keys() == smq_index.keys(), "Episode content mismatch")
    mapping = {value: smq_index[key] for key, value in server_index.items()}
    require([e["end"] for e in server["episodes"]] == original["episode_ends"], "Server boundaries differ")
    require([e["end"] for e in smq["episodes"]] == training["episode_ends"], "SMQ boundaries differ")
    heading_val, smq_val = set(original["validation_episode_indices"]), set(training["validation_episodes"])
    shared = sorted(i for i in heading_val if mapping[i] in smq_val)
    require(len(shared) == 8 and shared == selection["common_validation_episode_indices"], "Shared validation episodes differ")
    indices = [i for i, row in enumerate(original["windows"]) if row["episode_index"] in shared]
    require(len(indices) == len(set(indices)) == 239, "Expected exact 239 unique windows")
    require(indices == audit["reusable_saved_window_indices"], "Audit window indices differ from independent matching")
    require(indices == [row["original_window_index"] for row in selection["windows"]], "Selection window indices differ")
    require(Counter(row["task_uid"] for row in selection["windows"]) == TASK_COUNTS, "Task window counts differ")
    require(selection["model_order"] == MODELS and sorted(selection["models"]) == sorted(MODELS), "Unexpected model set")
    require(selection["window_count"] == 239 and selection["task_ids"] == sorted(TASK_COUNTS), "Selection scope differs")
    for index, row in enumerate(selection["windows"]):
        require(row["window_index"] == index, "Window indices not contiguous")
        require(row["smq_episode_index"] == mapping[row["episode_index"]], "SMQ episode mapping differs")
        for key, value in original["windows"][indices[index]].items():
            if key != "window_index":
                require(row[key] == value, f"Original window field changed: {key}")
        require(1 <= row["frame_index"] <= row["episode_end"] - row["episode_start"] - 16, "Padded window selected")
    for filename, key in (("data.npz", "data_sha256"), ("noise.npy", "noise_sha256")):
        require(digest(parent / filename) == original[key], f"Parent {filename} changed")
        require(digest(output / filename) == selection[key], f"Selected {filename} changed")
    data = arrays(output / "data.npz")
    with np.load(parent / "data.npz", allow_pickle=False) as old:
        require(set(data) == set(old.files), "Observation/target fields differ")
        for key, value in data.items():
            require(np.isfinite(value).all(), f"Nonfinite observation or GT: {key}")
            np.testing.assert_array_equal(value, old[key][indices])
    noise = np.load(output / "noise.npy", allow_pickle=False)
    np.testing.assert_array_equal(noise, np.load(parent / "noise.npy", mmap_mode="r", allow_pickle=False)[indices])
    require(digest(output / "representative_noise.npy") == selection["representative_noise_sha256"], "Representative noise changed")
    rep_noise = np.load(output / "representative_noise.npy", allow_pickle=False)
    reps = selection["representatives"]
    require(len(reps) == 15, "Expected 15 representatives")
    expected_reps = []
    for task in sorted(TASK_COUNTS):
        task_windows = [row for row in selection["windows"] if row["task_uid"] == task]
        counts = Counter(row["episode_index"] for row in task_windows)
        episode = min(counts, key=lambda key: (-counts[key], key))
        candidates = [row for row in task_windows if row["episode_index"] == episode]
        used = set()
        for phase, fraction in (("early", .15), ("middle", .5), ("late", .85)):
            length = candidates[0]["episode_end"] - candidates[0]["episode_start"]
            chosen = min((row for row in candidates if row["window_index"] not in used),
                         key=lambda row: (abs(row["frame_index"] - (1 + fraction * (length - 17))), row["frame_index"]))
            used.add(chosen["window_index"])
            expected_reps.append({**chosen, "representative_index": len(expected_reps), "phase": phase,
                                  "target_fraction_of_unpadded_episode": fraction})
    require(reps == expected_reps, "Representative selection not independently reproducible")
    rep_indices = [row["window_index"] for row in reps]
    require(noise.shape == (239, 32, 16, 7) and rep_noise.shape == (15, 1024, 16, 7), "Gaussian shapes differ")
    require(noise.dtype == rep_noise.dtype == np.float32, "Paired noise must be float32")
    require(np.isfinite(noise).all() and np.isfinite(rep_noise).all(), "Nonfinite noise")
    np.testing.assert_array_equal(rep_noise[:, :32], noise[rep_indices])
    return {"matched_episodes": 500, "shared_validation_episodes": 8, "selected_windows": 239,
            "task_window_counts": TASK_COUNTS, "representatives": 15,
            "all_observations_and_targets_exact_parent_subset": True, "all_noise_exact_parent_subset": True,
            "rgb_identity_across_training_datasets_verified": False}, data, noise, rep_noise


def zero_alignment(archive):
    resultant = archive["source_raw"][..., :2].astype(np.float64).sum(axis=2)
    heading = archive["heading_direction"][:, None, 0].astype(np.float64)
    cross = resultant[..., 0] * heading[..., 1] - resultant[..., 1] * heading[..., 0]
    dot = (resultant * heading).sum(axis=-1)
    angle = np.rad2deg(np.arctan2(np.abs(cross), dot))
    active = archive["source_active"][..., 0]
    require(active.any(), "No active Heading Zero samples to validate alignment")
    maximum = float(angle[active].max())
    require(maximum < .1, f"Heading Zero active H16 source misaligned: {maximum} degrees")
    return maximum


def verify_models(output, selection, noise, rep_noise):
    ri = [row["window_index"] for row in selection["representatives"]]
    results = {}
    for name in MODELS:
        meta = read_json(output / "models" / f"{name}.json")
        selected = selection["models"][name]
        for key, value in selected.items():
            require(meta[key] == value, f"Model metadata differs from selection: {name}/{key}")
        require(meta["complete"] is True and meta["model_id"] == name, f"Incomplete model {name}")
        require(meta["native_inference_max_absolute_difference"] == 0, f"Native parity failed: {name}")
        require(meta["weights"] == "ema_model", f"Unexpected weights: {name}")
        require(meta["data_sha256"] == selection["data_sha256"] and meta["noise_sha256"] == selection["noise_sha256"], "Models do not share observations/noise")
        require(digest(meta["checkpoint"]) == meta["checkpoint_sha256"], f"Checkpoint changed: {name}")
        require(digest(ROOT / "scripts/run_shared_holdout_comparison.py") == meta["script_sha256"], "Inference script changed since sampling")
        main_path = output / "models" / f"{name}.npz"
        rep_path = output / "models" / f"{name}_representatives.npz"
        require(digest(main_path) == meta["result_sha256"] and digest(rep_path) == meta["representative_result_sha256"], f"Model output hash changed: {name}")
        main, reps = arrays(main_path), arrays(rep_path)
        segments = 4 if name == "smq_dit" else 1
        require(meta["num_motion_segments"] == segments, "Head segment metadata differs")
        errors = []
        for archive, rows, draws, action_draws, gaussian in ((main, 239, 32, 32, noise), (reps, 15, 1024, 128, rep_noise)):
            expected = {"action_pred": (rows, action_draws, 16, 7), "source_norm": (rows, draws, 16, 7),
                        "source_raw": (rows, draws, 16, 7), "heading_direction": (rows, segments, 2),
                        "heading_confidence": (rows, segments), "source_active": (rows, draws, segments)}
            if rows == 15:
                expected["flow_mid_raw"] = (rows, 128, 16, 7)
            require(set(archive) == set(expected), "Unexpected model output fields")
            for key, shape in expected.items():
                require(archive[key].shape == shape and np.isfinite(archive[key]).all(), f"Shape or finiteness failure: {name}/{key}")
            require(archive["source_active"].dtype == np.bool_, "Source gate must be boolean")
            require(((archive["heading_confidence"] >= 0) & (archive["heading_confidence"] <= 1)).all(), "Confidence outside [0,1]")
            np.testing.assert_allclose(np.linalg.norm(archive["heading_direction"], axis=-1), 1, rtol=2e-5, atol=2e-5)
            paired = gaussian * selected["policy_config"]["prior_noise_scale"]
            np.testing.assert_array_equal(archive["source_norm"][..., 2:], paired[..., 2:])
            gate = np.repeat(archive["source_active"], 16 // segments, axis=-1)
            np.testing.assert_array_equal(archive["source_norm"][~gate], paired[~gate])
            scale, offset = np.array(meta["normalizer_scale"], np.float32), np.array(meta["normalizer_offset"], np.float32)
            require(scale.shape == offset.shape == (7,) and (scale > 0).all(), "Invalid action normalizer")
            forward = archive["source_raw"] * scale + offset
            inverse = (archive["source_norm"] - offset) / scale
            np.testing.assert_allclose(forward, archive["source_norm"], rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(inverse, archive["source_raw"], rtol=2e-6, atol=2e-6)
            errors.append(float(np.max(np.abs(forward - archive["source_norm"]))))
        for key in ("action_pred", "source_norm", "source_raw", "source_active"):
            np.testing.assert_array_equal(main[key][ri], reps[key][:, :32])
        for key in ("heading_direction", "heading_confidence"):
            np.testing.assert_array_equal(main[key][ri], reps[key])
        results[name] = {"native_parity_recorded_max_error": 0, "all_arrays_finite": True,
                         "head_segments": segments, "representative_first32_bit_exact": True,
                         "all_1024_representative_nonxy_draws_bit_exact": True,
                         "affine_roundtrip_max_absolute_error": max(errors),
                         "inference_bf16_enabled": meta["inference_bf16_enabled"],
                         "result_sha256": meta["result_sha256"], "representative_result_sha256": meta["representative_result_sha256"]}
        if name == "heading_zero_dit":
            results[name]["active_h16_source_alignment_max_degrees"] = max(zero_alignment(main), zero_alignment(reps))
        if name == "smq_dit":
            require(meta["inference_bf16_enabled"] is True, "Expected native SMQ BF16 inference")
    return results


def verify_projection(output, selection, data):
    projection, meta = arrays(output / "projection.npz"), read_json(output / "projection.json")
    reps = selection["representatives"]
    ri = [row["window_index"] for row in reps]
    require(len(meta["representatives"]) == 15 and meta["matched_episode_count"] == 8, "Projection scope differs")
    for key in ("source_hdf5_index_matches_all_selected_windows", "comparison_data_current_observations_exactly_verified", "all_eef_kinematic_checks_passed", "all_image_orientation_checks_passed"):
        require(meta[key] is True, f"Projection audit failed: {key}")
    for key, value in projection.items():
        require(np.isfinite(value).all(), f"Nonfinite projection array {key}")
    np.testing.assert_array_equal(projection["representative_window_indices"], ri)
    np.testing.assert_array_equal(projection["image"], data["obs__agentview_rgb"][ri, -1])
    np.testing.assert_array_equal(projection["ee_pos"].astype(np.float32), data["obs__robot0_eef_pos"][ri, -1])
    homogeneous = np.concatenate([projection["ee_pos"], np.ones((15, 1))], axis=-1)
    projected = np.einsum("bij,bj->bi", projection["world_to_pixel"], homogeneous)
    pixels = projected[:, :2] / projected[:, 2:3]
    np.testing.assert_allclose(pixels, projection["projected_eef_xy"], rtol=1e-12, atol=1e-10)
    require((projected[:, 2] > 0).all() and ((pixels >= 0) & (pixels < 128)).all(), "EEF outside calibrated image")
    image, reconstructed = projection["image"].astype(float), projection["reconstructed_image"].astype(float)
    mae = np.abs(image - reconstructed).mean(axis=(1, 2, 3))
    flipped_mae = np.abs(image - reconstructed[:, ::-1]).mean(axis=(1, 2, 3))
    require((mae < flipped_mae).all(), "Projection orientation differs")
    for i, (record, rep) in enumerate(zip(meta["representatives"], reps)):
        for key, value in rep.items():
            require(record[key] == value, f"Projection representative differs: {key}")
        require(record["eef_kinematic_error_pixels"] < .5 and record["eef_kinematic_error_m"] < .005, "Projection alignment tolerance exceeded")
        np.testing.assert_allclose([record["render_image_mae"], record["vertically_flipped_render_image_mae"]], [mae[i], flipped_mae[i]], rtol=0, atol=1e-10)
        np.testing.assert_allclose(record["projected_eef_xy_pixels"], pixels[i], atol=1e-10, rtol=1e-12)
    require(float(projection["controller_translation_scale"]) == .05, "Controller translation scale differs")
    require(float(projection["controller_input_min"]) == -1 and float(projection["controller_input_max"]) == 1, "Controller clipping differs")
    return {"representative_observations_exact": True, "independent_camera_projection_matches": True,
            "max_recorded_eef_kinematic_error_pixels": max(r["eef_kinematic_error_pixels"] for r in meta["representatives"]),
            "image_orientation_recomputed": True, "projection_sha256": digest(output / "projection.npz")}


def verify_metrics(output, selection, data):
    meta = read_json(output / "metrics.json")
    require(meta["model_order"] == MODELS, "Metrics model order differs")
    require(meta["selection"]["window_count"] == 239 and meta["selection"]["episode_count"] == 8 and meta["selection"]["task_count"] == 5, "Metrics scope differs")
    require(meta["split_audit"]["sha256"] == selection["split_audit_sha256"], "Metrics split audit differs")
    for filename, expected in meta["provenance_sha256"].items():
        require(digest(output / filename) == expected, f"Metric input changed: {filename}")
    require(digest(meta["analysis_script"]["path"]) == meta["analysis_script"]["sha256"], "Analysis script changed")
    per_window = arrays(output / "per_window_metrics.npz")
    np.testing.assert_array_equal(per_window["model_ids"], MODELS)
    for key in ("window_index", "original_window_index", "episode_index", "smq_episode_index", "frame_index", "task_uid"):
        np.testing.assert_array_equal(per_window[key], [row[key] for row in selection["windows"]])
    for key, value in per_window.items():
        if np.issubdtype(value.dtype, np.number):
            require(not np.isinf(value).any(), f"Infinite metric values: {key}")
    for i, name in enumerate(MODELS):
        require(meta["models"][name]["own_heading_segment_count"] == (4 if name == "smq_dit" else 1), "Metrics head segments differ")
        predictions = arrays(output / "models" / f"{name}.npz")["action_pred"].astype(float)
        difference = predictions[:, :, :8] - data["actions_gt"][:, None, :8].astype(float)
        mse = (difference ** 2).mean(axis=(1, 2, 3))
        np.testing.assert_allclose(mse, per_window["action_h8_mse"][i], rtol=1e-12, atol=1e-14)
        balanced = np.mean([mse[per_window["task_uid"] == task].mean() for task in sorted(TASK_COUNTS)])
        np.testing.assert_allclose(balanced, meta["models"][name]["task_balanced"]["action_h8_mse"]["mean"], rtol=1e-12, atol=1e-14)
    require(meta["bootstrap"]["episode_count"] == 8 and meta["bootstrap"]["single_episode_task_uids"] == [35, 38, 39], "Bootstrap strata differ")
    require(meta["bootstrap"]["task_weight"] == .2, "Expected equal task weighting")
    return {"provenance_hashes_match": True, "per_window_identities_match": True,
            "h8_mse_and_equal_task_mean_independently_recomputed": True,
            "undefined_direction_metric_nan_allowed": True, "metrics_sha256": digest(output / "metrics.json")}


def verify_plots(output, selection):
    from PIL import Image
    manifest = read_json(output / "plot_manifest.json")
    quality = read_json(output / "plot_quality_checks.json")
    require(manifest["model_order"] == MODELS, "Plot model order differs")
    require(manifest["figure_count"] == quality["figure_count"] == quality["expected_figure_count"] == 36, "Expected 36 figures")
    require(set(manifest["formats"]) == {"png", "pdf", "svg"}, "Expected PNG/PDF/SVG")
    require(len(manifest["figures"]) == 36, "Figure inventory incomplete")
    for filename, expected in manifest["input_sha256"].items():
        require(digest(output / filename) == expected, f"Plot input changed: {filename}")
    require(digest(manifest["plotting_script"]["path"]) == manifest["plotting_script"]["sha256"], "Plot script changed")
    for key, value in quality.items():
        if key.endswith("_checks"):
            require(value == "passed", f"Recorded plot check failed: {key}")
    assets = {}
    for formats in manifest["figures"].values():
        require(set(formats) == {"png", "pdf", "svg"}, "Missing figure format")
        for extension, filename in formats.items():
            path = output / filename
            require(path.resolve().is_relative_to(output.resolve()) and path.stat().st_size > 100, "Invalid or empty figure")
            require(filename not in assets, "Duplicate figure asset")
            if extension == "png":
                with Image.open(path) as img:
                    require(img.width > 100 and img.height > 100, "Invalid PNG dimensions")
                    img.verify()
            elif extension == "pdf":
                with path.open("rb") as stream:
                    require(stream.read(5) == b"%PDF-", "Invalid PDF header")
            else:
                require(ET.parse(path).getroot().tag.endswith("svg"), "Invalid SVG")
            assets[filename] = digest(path)
    require(len(assets) == 108, "Expected all 108 figure assets")
    read_json(output / "direction_plot_summary.json")
    gallery = (output / "gallery.html").read_text()
    for filename in assets:
        require(filename in gallery, f"Gallery does not link asset {filename}")
    for suffix, steps in (("first_action", 1), ("first8", 8)):
        heatmap = arrays(output / f"heatmap_{suffix}_arrays.npz")
        np.testing.assert_array_equal(heatmap["model_ids"], MODELS)
        density = heatmap["density"]
        require(density.shape[:2] == (15, 3), "Heatmap representative/model dimensions differ")
        require(np.isfinite(density).all() and (density >= 0).all(), "Invalid heatmap density")
        require(int(heatmap["steps"]) == steps and float(heatmap["translation_scale"]) == .05, "Heatmap displacement settings differ")
        mass = density.sum(axis=(-1, -2))
        visible = heatmap["visible"].mean(axis=-1)
        require((mass <= visible + 1e-10).all() and (mass <= 1 + 1e-10).all(), "Heatmap exceeds probability mass")
        np.testing.assert_allclose(mass, manifest["heatmap_summary"][suffix]["density_mass"], rtol=1e-12, atol=1e-12)
    return {"figures": 36, "assets": 108, "png_integrity_pdf_header_svg_parse_passed": True,
            "input_hashes_match": True, "gallery_links_all_assets": True, "heatmap_probability_checks_passed": True,
            "visual_design_review_not_certified_by_this_script": True, "asset_sha256": assets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-plots", action="store_true", help="Require every stage, including metrics, projection, and all 108 plotted assets")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "output_dir": str(output), "verifier_sha256": digest(__file__), "stages": {},
              "final_complete": False, "native_parity_scope": "Checks saved exact-parity record; does not rerun GPU inference"}
    stage = "inputs"
    try:
        selection = read_json(output / "selection.json")
        check, data, noise, rep_noise = verify_inputs(output, selection)
        result["selection_sha256"] = digest(output / "selection.json")
        result["stages"][stage] = {"status": "passed", **check}
        stage = "models"
        model_files = [f"models/{name}{suffix}" for name in MODELS for suffix in (".json", ".npz", "_representatives.npz")]
        stages = [
            ("models", model_files, lambda: verify_models(output, selection, noise, rep_noise)),
            ("projection", ["projection.json", "projection.npz", "hdf5_episode_matches.json"], lambda: verify_projection(output, selection, data)),
            ("metrics", ["metrics.json", "per_window_metrics.npz", "per_task_metrics.csv", "REPORT.md"], lambda: verify_metrics(output, selection, data)),
            ("plots", ["plot_manifest.json", "plot_quality_checks.json", "direction_plot_summary.json", "gallery.html",
                       "heatmap_first_action_arrays.npz", "heatmap_first8_arrays.npz"], lambda: verify_plots(output, selection)),
        ]
        for stage, files, check in stages:
            missing = [filename for filename in files if not (output / filename).is_file()]
            if missing:
                result["stages"][stage] = {"status": "missing", "missing_files": missing}
            else:
                result["stages"][stage] = {"status": "passed", **check()}
        result["final_complete"] = all(s["status"] == "passed" for s in result["stages"].values())
    except Exception as error:
        result["stages"][stage] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    result["available_stage_checks_passed"] = all(s["status"] != "failed" for s in result["stages"].values())
    path = output / "verification.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"final_complete": result["final_complete"],
                      "stages": {name: value["status"] for name, value in result["stages"].items()},
                      "verification": str(path)}, indent=2))
    if not result["available_stage_checks_passed"]:
        print(result["stages"][stage].get("error", "Verification failed"), file=sys.stderr)
        return 1
    return 1 if args.require_plots and not result["final_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
