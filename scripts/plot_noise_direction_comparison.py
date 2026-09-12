#!/usr/bin/env python3
"""Export the approved three-model LIBERO-10 noise/direction comparison.

Reads recorded samples only: never loads a checkpoint or runs a policy.  Raw
noise, normalized noise, and denormalized generated actions have distinct axis
labels. Image overlays are first-eight-step intended command endpoint densities,
not rollouts, attention maps, or source-only causal effects.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ("baseline_transformer", "heading_zero_dit", "heading_gaussian_dit")
MODEL_LABELS = ("Transformer Flow", "Heading Zero DiT", "Heading Gaussian DiT")
MODEL_SUBTITLES = ("Best SR 76.8% · ep-0055", "Best SR 80.0% · epoch 030", "Best SR 79.0% · epoch 060")
COLORS = ("#3479B6", "#D27135", "#775AA6")
DARK = "#263444"
GT_COLOR = "#18754B"
HEAD_COLOR = "#AD2761"
PHASES = ("early", "middle", "late")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        if keys is None:
            keys = archive.files
        return {key: archive[key] for key in keys}


def task_title(name):
    return str(name).replace("_", " ").replace("SCENE", "scene")


def wrap_title(name, width=105):
    return "\n".join(textwrap.wrap(task_title(name), width=width))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def style_axes(ax):
    ax.set_facecolor("#FAFCFD")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BAC6CE")
    ax.tick_params(colors=DARK, labelsize=8)
    ax.grid(color="#DCE4E9", linewidth=.5, alpha=.65, zorder=0)


def arrow(ax, vector, limit, color, dashed=False):
    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or magnitude < 1e-10:
        return
    endpoint = vector / magnitude * limit * .76
    # Keep the heading-aligned ray visible as sample dots above its own arrow.
    ax.annotate("", xy=endpoint, xytext=(0, 0), zorder=1.5,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                linestyle="--" if dashed else "-", mutation_scale=13))


def figure_footer(fig, text, y=.015, fontsize=9):
    fig.text(.055, y, text, color=DARK, fontsize=fontsize, va="bottom", linespacing=1.45)


def save_figure(fig, output, relative, inventory, formats=("png", "pdf", "svg"), dpi=180):
    stem = output / "figures" / relative
    stem.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, facecolor="white")
        result[extension] = str(path.relative_to(output))
    inventory[str(relative)] = result
    plt.close(fig)
    print(f"Rendered {relative}", flush=True)


def noise_limit(reps_data, representative_indices, key, resultant):
    # Never choose per-model limits or hide long tails using quantiles.
    extent = 0.
    for model in reps_data:
        xy = model[key][representative_indices, ..., :2]
        if resultant:
            xy = xy.sum(axis=-2)
        extent = max(extent, float(np.max(np.abs(xy))))
    return max(extent * 1.08, .1)


def noise_panel(ax, xy, color, limit, resultant, gt=None, heading=None, label=None,
                normalized=False, compact=False):
    style_axes(ax)
    if resultant:
        dots = xy[..., :2].sum(axis=-2)
        size, alpha = 5.4, .32
    else:
        indices = np.linspace(0, xy.shape[0] - 1, min(256, xy.shape[0]), dtype=int)
        dots = xy[indices, ..., :2].reshape(-1, 2)
        size, alpha = 2.5, .18
    ax.scatter(dots[:, 0], dots[:, 1], s=size, color=color, alpha=alpha,
               linewidths=0, rasterized=True, zorder=2)
    ax.axhline(0, color="#9DAFB9", lw=.65, zorder=1)
    ax.axvline(0, color="#9DAFB9", lw=.65, zorder=1)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    if gt is not None:
        arrow(ax, gt, limit, GT_COLOR)
    if heading is not None:
        arrow(ax, heading, limit, HEAD_COLOR, dashed=True)
    if label:
        ax.text(.04, .96, label, va="top", transform=ax.transAxes, fontsize=7.8,
                color=DARK, bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
    if not compact:
        units = "normalized source" if normalized else "raw action-coordinate source"
        kind = "Σ₁₆ " if resultant else ""
        ax.set_xlabel(f"{kind}X · {units}", fontsize=8, color=DARK)
        ax.set_ylabel(f"{kind}Y · {units}", fontsize=8, color=DARK)


def noise_legend(fig, normalized=False, y=.07):
    if normalized:
        return
    handles = [Line2D([0], [0], color=GT_COLOR, lw=2, label="GT raw XY direction (Σ first 16 actions)"),
               Line2D([0], [0], color=HEAD_COLOR, lw=2, ls="--", label="Predicted heading head")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=2,
               frameon=False, fontsize=9)


def render_noise_tasks(output, representatives, groups, reps_data, rep_gt, rep_gt_valid,
                       inventory, normalized=False, dpi=180):
    key = "source_norm" if normalized else "source_raw"
    suffix = "normalized" if normalized else "raw"
    for task_number, (task_uid, indices) in enumerate(groups, 1):
        name = representatives[indices[0]]["task_name"]
        fig, axes = plt.subplots(6, 3, figsize=(14.8, 22), squeeze=False)
        fig.subplots_adjust(left=.09, right=.96, bottom=.11, top=.92, hspace=.5, wspace=.27)
        fig.suptitle(f"Task {task_number:02d} · {'Normalized' if normalized else 'Raw'} source-noise dot distributions\n"
                     + wrap_title(name, 98), x=.055, y=.98, ha="left", color=DARK,
                     fontsize=16, fontweight="bold")
        step_limit = noise_limit(reps_data, indices, key, False)
        sum_limit = noise_limit(reps_data, indices, key, True)
        for phase_idx, r in enumerate(indices):
            rep = representatives[r]
            for m, model in enumerate(reps_data):
                for row_in_phase, resultant in enumerate((False, True)):
                    row = phase_idx * 2 + row_in_phase
                    detail = f"{rep['phase'].title()} · frame {rep['frame_index']}\n"
                    detail += "Chunk resultants · 1,024 dots" if resultant else "Per step · 4,096 dots"
                    if resultant and m:
                        detail += f"\nSource active {model['source_active'][r].mean() * 100:.0f}%"
                    noise_panel(axes[row, m], model[key][r], COLORS[m], sum_limit if resultant else step_limit,
                                resultant, None if normalized or not rep_gt_valid[r] else rep_gt[r, :, :2].sum(0),
                                None if normalized else model["heading_direction"][r], detail, normalized)
                    if row == 0:
                        axes[row, m].set_title(MODEL_LABELS[m] + "\n" + MODEL_SUBTITLES[m],
                                              fontsize=11, color=DARK, fontweight="bold", pad=14)
        noise_legend(fig, normalized, y=.065)
        if normalized:
            note = ("Normalized coordinates are the actual flow-network source coordinates; affine action normalization may be anisotropic.\n"
                    "Physical GT/head direction arrows are omitted here because they live in raw action coordinates; see the paired raw-source figure.\n"
                    "Identical axis ranges across models and all three phases within each row type. Every resultant draw is shown; per-step dots use 256 paired chunks.")
        else:
            note = ("Per-step dots and chunk-resultant dots answer different questions: Heading Zero aligns the summed 16-step raw XY direction.\n"
                    "Arrows show direction only, with equal display length. GT arrows are omitted if the target direction is below the shared motion threshold.\n"
                    "Identical axis ranges across models and phases within each row type. Every resultant draw is shown; per-step dots use 256 paired chunks.")
        figure_footer(fig, note, y=.017, fontsize=8.5)
        save_figure(fig, output, Path("tasks") / f"task_{task_uid}_noise_{suffix}", inventory, dpi=dpi)


def render_noise_overview(output, representatives, groups, reps_data, rep_gt, rep_gt_valid,
                          projection, inventory, dpi=180):
    mids = [indices[len(indices) // 2] for _, indices in groups]
    limit = noise_limit(reps_data, mids, "source_raw", True)
    fig, axes = plt.subplots(len(groups), 4, figsize=(16, 31), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
    fig.subplots_adjust(left=.04, right=.965, top=.945, bottom=.052, hspace=.47, wspace=.24)
    fig.suptitle("All 10 LIBERO-10 tasks · source-noise chunk resultants",
                 x=.04, y=.985, ha="left", fontsize=18, fontweight="bold", color=DARK)
    fig.text(.04, .966, "Fixed middle-demonstration observations · 1,024 paired draws per panel · raw XY coordinates",
             fontsize=11, color=DARK)
    for row, ((task_uid, indices), r) in enumerate(zip(groups, mids)):
        ax = axes[row, 0]
        ax.imshow(projection["image"][r])
        ax.axis("off")
        title = f"{row + 1:02d} · {task_title(representatives[r]['task_name'])}"
        ax.set_title("\n".join(textwrap.wrap(title, 40)), loc="left", fontsize=8.4, color=DARK, pad=5)
        for m, model in enumerate(reps_data):
            noise_panel(axes[row, m + 1], model["source_raw"][r], COLORS[m], limit, True,
                        rep_gt[r, :, :2].sum(0) if rep_gt_valid[r] else None,
                        model["heading_direction"][r], compact=True)
            if row == 0:
                axes[row, m + 1].set_title(MODEL_LABELS[m] + "\n" + MODEL_SUBTITLES[m],
                                          fontsize=10.5, color=DARK, fontweight="bold", pad=11)
            if row == len(groups) - 1:
                axes[row, m + 1].set_xlabel("Σ₁₆ raw source X", fontsize=9)
            if m == 0:
                axes[row, m + 1].set_ylabel("Σ₁₆ raw source Y", fontsize=9)
    noise_legend(fig, y=.023)
    figure_footer(fig, "All noise axes share one scale. The individual per-step distributions and three phases per task are in the detailed figures.", y=.01, fontsize=9)
    save_figure(fig, output, "noise_overview_all10", inventory, dpi=dpi)


def project(world, matrix):
    world = np.asarray(world, dtype=float)
    homogeneous = np.concatenate((world, np.ones((*world.shape[:-1], 1))), axis=-1)
    transformed = homogeneous @ np.asarray(matrix).T
    depth = transformed[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = transformed[..., :2] / depth[..., None]
    valid = np.isfinite(pixels).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    return pixels, valid


def command_path(actions, ee_pos, scale, steps=8):
    delta = np.clip(np.asarray(actions)[..., :steps, :3], -1., 1.) * scale
    start_shape = (*delta.shape[:-2], 1, 3)
    start = np.broadcast_to(np.asarray(ee_pos), start_shape)
    return np.concatenate((start, start + np.cumsum(delta, axis=-2)), axis=-2)


def endpoint_density(pixels, camera_valid, width, height, bandwidth):
    # No border clipping: out-of-view endpoints do not become edge hotspots.
    inside = camera_valid & (pixels[..., 0] >= -.5) & (pixels[..., 0] < width - .5)
    inside &= (pixels[..., 1] >= -.5) & (pixels[..., 1] < height - .5)
    points = pixels[inside]
    hist = np.histogram2d(points[:, 1], points[:, 0],
                          bins=(np.arange(height + 1) - .5, np.arange(width + 1) - .5))[0]
    density = gaussian_filter(hist / len(pixels), sigma=bandwidth, mode="constant", cval=0.)
    return density, inside


def make_heatmaps(reps_data, rep_gt, projection, bandwidth=2., steps=8):
    r_count = len(rep_gt)
    height, width = projection["image"].shape[1:3]
    scale = float(np.asarray(projection["controller_translation_scale"]).reshape(-1)[0])
    density = np.zeros((r_count, 3, height, width), dtype=np.float64)
    endpoint_pixels, visible, gt_paths, starts, crops, camera_valid = [], [], [], [], [], []
    for r in range(r_count):
        matrix, ee_pos = projection["world_to_pixel"][r], projection["ee_pos"][r]
        per_model_points, per_model_visible, per_model_camera_valid = [], [], []
        for m, model in enumerate(reps_data):
            paths = command_path(model["action_pred"][r], ee_pos, scale, steps)
            pixels, valid = project(paths[:, -1], matrix)
            density[r, m], inside = endpoint_density(pixels, valid, width, height, bandwidth)
            per_model_points.append(pixels)
            per_model_visible.append(inside)
            per_model_camera_valid.append(valid)
        gt_pixels, gt_valid = project(command_path(rep_gt[r], ee_pos, scale, steps), matrix)
        gt_pixels[~gt_valid] = np.nan
        gt_paths.append(gt_pixels)
        starts.append(gt_pixels[0])
        endpoint_pixels.append(per_model_points)
        visible.append(per_model_visible)
        camera_valid.append(per_model_camera_valid)
        # Common local crop, centered at the observed EEF and containing the
        # majority of finite visible endpoint mass for all three models.
        finite_sets = [points[inside] for points, inside in zip(per_model_points, per_model_visible)]
        finite_sets.append(gt_pixels[np.isfinite(gt_pixels).all(axis=1)])
        all_points = np.concatenate(finite_sets, axis=0)
        if len(all_points):
            lo, hi = np.quantile(all_points, [.01, .99], axis=0)
            center = (lo + hi) / 2
            side = min(max(float(np.max(hi - lo)) + 12., 28.), float(min(width, height)))
            center = np.clip(center, side / 2 - .5, np.array([width, height]) - .5 - side / 2)
            crops.append([center[0] - side / 2, center[0] + side / 2,
                          center[1] + side / 2, center[1] - side / 2])
        else:
            crops.append([-.5, width - .5, height - .5, -.5])
    density_max = max(float(density.max()), 1e-10)
    differences = density[:, 1:] - density[:, :1]
    difference_max = max(float(np.abs(differences).max()), 1e-10)
    return {"density": density, "difference": differences, "density_max": density_max,
            "difference_max": difference_max, "pixels": endpoint_pixels, "visible": visible,
            "gt_path": gt_paths, "start": starts, "crop": crops,
            "bandwidth_px": bandwidth, "translation_scale": scale, "steps": steps, "camera_valid": camera_valid}


def heatmap_panel(ax, image, values, vmax, gt_path, signed=False, visible=None, crop=None):
    ax.imshow(image, interpolation="nearest", zorder=0)
    if signed:
        cmap = "RdBu_r"
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)
    else:
        cmap = "inferno"
        norm = Normalize(vmin=0., vmax=vmax)
    # A shared threshold/alpha maps equal density to equal visual strength.
    alpha = np.minimum(np.abs(values) / vmax * 2.5, .78)
    ax.imshow(values, cmap=cmap, norm=norm, alpha=alpha, interpolation="bilinear", zorder=1)
    line, = ax.plot(gt_path[:, 0], gt_path[:, 1], color="#5DF6AE", lw=1.6, zorder=3)
    line.set_path_effects([pe.Stroke(linewidth=2.8, foreground="#102D23"), pe.Normal()])
    if np.isfinite(gt_path[0]).all():
        ax.scatter(*gt_path[0], s=21, c="white", edgecolors="#102D23", linewidths=.8, zorder=4)
    if np.isfinite(gt_path[-1]).all():
        ax.scatter(*gt_path[-1], s=38, c="#5DF6AE", marker="*", edgecolors="#102D23", linewidths=.6, zorder=4)
    if visible is not None:
        ax.text(.025, .035, f"Endpoints in view: {int(np.sum(visible))}/{len(visible)}",
                transform=ax.transAxes, va="bottom", color="white", fontsize=7,
                bbox=dict(facecolor="black", alpha=.62, edgecolor="none", pad=2))
    # Plotting an offscreen GT endpoint must not expand the displayed image.
    if crop is None:
        height, width = image.shape[:2]
        crop = [-.5, width - .5, height - .5, -.5]
    if crop is not None:
        ax.set_xlim(crop[:2])
        ax.set_ylim(crop[2:])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def heatmap_colorbars(fig, heatmaps, bottom=.078, width=.27):
    left = fig.add_axes([.13, bottom, width, .012])
    right = fig.add_axes([.62, bottom, width, .012])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, heatmaps["density_max"]), cmap="inferno"), cax=left, orientation="horizontal")
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -heatmaps["difference_max"], heatmaps["difference_max"]), cmap="RdBu_r"), cax=right, orientation="horizontal")
    cb1.set_label("Endpoint probability / image pixel", fontsize=8)
    cb2.set_label("Δ probability / pixel · red: increase; blue: decrease", fontsize=8)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=7)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()


def render_heatmap_tasks(output, representatives, groups, projection, heatmaps, inventory, dpi=180):
    titles = (*MODEL_LABELS, "Heading Zero − Transformer", "Heading Gaussian − Transformer")
    first_action = heatmaps["steps"] == 1
    suffix = "_first_action" if first_action else ""
    for task_number, (task_uid, indices) in enumerate(groups, 1):
        fig, axes = plt.subplots(6, 5, figsize=(17.5, 20), squeeze=False)
        fig.subplots_adjust(left=.045, right=.985, top=.915, bottom=.155, hspace=.3, wspace=.05)
        fig.suptitle(f"Task {task_number:02d} · {'first-action goal' if first_action else '8-action endpoint'} prediction heatmaps\n" + wrap_title(representatives[indices[0]]["task_name"], 115),
                     x=.045, y=.98, ha="left", fontsize=16, fontweight="bold", color=DARK)
        fig.text(.045, .938, ("Projected desired controller goal from first action" if first_action else "Projected intended command endpoints after 8 actions") + " · 128 paired draws · full task image and shared local zoom",
                 fontsize=10, color=DARK)
        for phase_idx, r in enumerate(indices):
            rep = representatives[r]
            for local in range(2):
                row = phase_idx * 2 + local
                for col in range(5):
                    signed = col >= 3
                    values = heatmaps["difference"][r, col - 3] if signed else heatmaps["density"][r, col]
                    vmax = heatmaps["difference_max"] if signed else heatmaps["density_max"]
                    heatmap_panel(axes[row, col], projection["image"][r], values, vmax, heatmaps["gt_path"][r], signed,
                                  None if signed else heatmaps["visible"][r][col], heatmaps["crop"][r] if local else None)
                    if row == 0:
                        axes[row, col].set_title(titles[col], fontsize=9.5, color=DARK, fontweight="bold", pad=9)
                    if col == 0:
                        axes[row, col].text(-.06, .5, f"{rep['phase'].title()} · f{rep['frame_index']}\n" + ("Local zoom" if local else "Full image"),
                                            rotation=90, transform=axes[row, col].transAxes, ha="center", va="center", fontsize=9, color=DARK)
        heatmap_colorbars(fig, heatmaps, bottom=.105)
        figure_footer(fig,
                      ("Green line/star: GT desired first-command displacement/goal. White dot: observed EEF. Raw XYZ commands are clipped to [−1,1] and scaled by 0.05 m.\n" if first_action else "Green path/star: GT intended displacement/endpoint. White dot: observed EEF. Commands are clipped to [−1,1], scaled by 0.05 m, then cumulatively summed.\n") +
                      "One global density scale and one symmetric difference scale cover every model, frame, and task; Gaussian smoothing σ = 2 image pixels.\n"
                      "Out-of-view endpoints are excluded without moving them onto borders; probability denominator retains all 128 samples.\n"
                      "These are native-model differences: backbone and best-SR checkpoint differ. The maps are not physical rollouts, attention, or source-only causal effects.",
                      y=.017, fontsize=8.5)
        save_figure(fig, output, Path("tasks") / f"task_{task_uid}_prediction_heatmaps{suffix}", inventory, dpi=dpi)


def render_heatmap_overview(output, representatives, groups, projection, heatmaps, inventory, dpi=180):
    first_action = heatmaps["steps"] == 1
    columns = 6 if first_action else 5
    fig, axes = plt.subplots(len(groups), columns, figsize=(20 if first_action else 18, 30), squeeze=False)
    fig.subplots_adjust(left=.04, right=.985, top=.94, bottom=.07, hspace=.45, wspace=.05)
    fig.suptitle("All 10 LIBERO-10 tasks · " + ("predicted first-action controller goals" if first_action else "changes in predicted 8-action endpoints"), x=.04, y=.984,
                 ha="left", fontsize=18, fontweight="bold", color=DARK)
    fig.text(.04, .963, "Fixed middle-demonstration images · " + ("task context and shared local zooms" if first_action else "intended displacement over 8 actions") + " · 128 paired samples per model",
             fontsize=10.5, color=DARK)
    titles = (*MODEL_LABELS, "Heading Zero − Transformer", "Heading Gaussian − Transformer")
    for row, (task_uid, indices) in enumerate(groups):
        r = indices[len(indices) // 2]
        offset = int(first_action)
        if first_action:
            context = axes[row, 0]
            context.imshow(projection["image"][r], interpolation="nearest")
            x0, x1, y1, y0 = heatmaps["crop"][r]
            box = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor="white", linewidth=1.5, linestyle="--")
            box.set_path_effects([pe.Stroke(linewidth=2.5, foreground="#263444"), pe.Normal()])
            context.add_patch(box)
            context.set_xticks([])
            context.set_yticks([])
            for spine in context.spines.values():
                spine.set_visible(False)
            if row == 0:
                context.set_title("Task image · zoom region", fontsize=9.5, color=DARK, fontweight="bold", pad=12)
        for col in range(5):
            signed = col >= 3
            values = heatmaps["difference"][r, col - 3] if signed else heatmaps["density"][r, col]
            heatmap_panel(axes[row, col + offset], projection["image"][r], values,
                          heatmaps["difference_max"] if signed else heatmaps["density_max"],
                          heatmaps["gt_path"][r], signed,
                          None if signed else heatmaps["visible"][r][col],
                          heatmaps["crop"][r] if first_action else None)
            if row == 0:
                axes[row, col + offset].set_title(titles[col], fontsize=9.5, color=DARK, fontweight="bold", pad=12)
        name = f"{row+1:02d} · {task_title(representatives[r]['task_name'])}"
        axes[row, 0].text(0., -.035, "\n".join(textwrap.wrap(name, 48 if first_action else 53)), transform=axes[row, 0].transAxes,
                          ha="left", va="top", fontsize=7.8, color=DARK)
    heatmap_colorbars(fig, heatmaps, bottom=.04)
    note = ("White box: shared zoom region. Green: GT desired first-command displacement. " if first_action else "Green: GT intended command path. ")
    figure_footer(fig, note + "Global scales; σ = 2 original-image px. Differences include model/checkpoint changes and do not isolate the noise source.", y=.01, fontsize=8.5)
    save_figure(fig, output, "prediction_heatmaps_first_action_overview_all10" if first_action else "prediction_heatmaps_overview_all10", inventory, dpi=dpi)


def angle_error(vectors, target):
    a = np.arctan2(vectors[..., 1], vectors[..., 0])
    b = np.arctan2(target[..., 1], target[..., 0])
    delta = a - b
    return np.degrees(np.arctan2(np.sin(delta), np.cos(delta)))


def render_direction_tasks(output, representatives, groups, reps_data, rep_gt, rms, inventory, dpi=180):
    for task_number, (task_uid, indices) in enumerate(groups, 1):
        fig, axes = plt.subplots(3, 3, figsize=(14.5, 12.5), squeeze=False)
        fig.subplots_adjust(left=.085, right=.96, top=.88, bottom=.15, hspace=.55, wspace=.27)
        fig.suptitle(f"Task {task_number:02d} · generated action direction versus demonstration\n"
                     + wrap_title(representatives[indices[0]]["task_name"], 100),
                     x=.055, y=.975, ha="left", fontsize=16, fontweight="bold", color=DARK)
        for row, r in enumerate(indices):
            for m, model in enumerate(reps_data):
                ax = axes[row, m]
                style_axes(ax)
                pred = model["action_pred"][r]
                row_specs = ((8, "Generated first 8", .62), (16, "Generated full 16", 1.62))
                for horizon, label, ypos in row_specs:
                    target = rep_gt[r, :horizon, :2].sum(0)
                    gt_valid = np.linalg.norm(target) >= .05 * rms * np.sqrt(horizon)
                    resultants = pred[:, :horizon, :2].sum(-2)
                    valid = np.linalg.norm(resultants, axis=-1) >= .05 * rms * np.sqrt(horizon)
                    if gt_valid:
                        errors = angle_error(resultants, target)
                        # Deterministic display jitter, unrelated to model/noise RNG.
                        jitter = np.sin(np.arange(len(errors)) * 2.399963229728653) * .17
                        ax.scatter(errors[valid], ypos + jitter[valid], s=8, color=COLORS[m], alpha=.45, linewidths=0, rasterized=True)
                        ax.text(.99, ypos / 3.2 + .075, f"{valid.sum()}/{len(valid)} valid", transform=ax.transAxes,
                                ha="right", fontsize=7, color=DARK)
                    else:
                        ax.text(0, ypos, "GT direction below motion threshold", ha="center", fontsize=8, color="#6A7984")
                target16 = rep_gt[r, :16, :2].sum(0)
                head = model["heading_direction"][r]
                valid_head = np.isfinite(head).all() and np.linalg.norm(head) > 1e-10
                if valid_head and np.linalg.norm(target16) >= .05 * rms * 4:
                    error = angle_error(head, target16)
                    ax.scatter(error, 2.62, color=HEAD_COLOR, marker="D", s=37, zorder=5)
                    ax.text(error, 2.87, f"{error:+.1f}°", ha="center", fontsize=8, color=HEAD_COLOR)
                elif m == 0:
                    ax.text(0, 2.62, "No heading head", ha="center", fontsize=8, color="#6A7984")
                ax.axvspan(-30, 30, alpha=.07, color=GT_COLOR)
                ax.axvline(0, color=GT_COLOR, lw=1.4)
                ax.set(xlim=(-180, 180), ylim=(0, 3.2), xticks=[-180, -90, 0, 90, 180],
                       yticks=[.62, 1.62, 2.62], yticklabels=["First 8", "Full 16", "Heading head"])
                ax.set_xlabel("Signed direction error relative to GT (degrees)", fontsize=8)
                if row == 0:
                    ax.set_title(MODEL_LABELS[m], fontsize=11, fontweight="bold", color=DARK, pad=12)
                if m == 0:
                    ax.set_ylabel(f"{representatives[r]['phase'].title()} · frame {representatives[r]['frame_index']}", fontsize=10, color=DARK)
        figure_footer(fig,
                      "Each colored dot is one generated action chunk (128 paired draws). Green line: GT direction; pale band: within ±30°. Pink diamond: heading head.\n"
                      "Generated directions use the sum of raw XY actions over the stated horizon; each row uses its matching demonstration target.\n"
                      "A direction is valid only when ||Σ XY|| / (training XY RMS × √horizon) ≥ 0.05. Vertical dot jitter is solely for legibility.",
                      y=.035, fontsize=9)
        save_figure(fig, output, Path("tasks") / f"task_{task_uid}_action_directions", inventory, dpi=dpi)


def render_direction_overview(output, selection, groups, all_data, gt, rms, inventory, dpi=180):
    fig, axes = plt.subplots(1, 3, figsize=(19, 8.8), sharey=True)
    fig.subplots_adjust(left=.24, right=.975, top=.8, bottom=.19, wspace=.18)
    fig.suptitle("All 10 LIBERO-10 tasks · predicted directions versus ground truth", x=.04, y=.96,
                 ha="left", fontsize=18, fontweight="bold", color=DARK)
    fig.text(.04, .908, "Per-task mean absolute angular error · same 128 held-out observation windows and 32 paired samples per model\n"
             "Points average samples within observations, then observations within each task; errors are conditional on valid predicted/GT motion.",
             fontsize=10.5, color=DARK, linespacing=1.5, va="top")
    names = [task_title(selection["representatives"][indices[0]]["task_name"]) for _, indices in groups]
    window_tasks = np.array([row["task_uid"] for row in selection["windows"]])
    overview_stats = []
    for col, horizon in enumerate((8, 16, 16)):
        ax = axes[col]
        style_axes(ax)
        for m, model in enumerate(all_data):
            if col == 2 and m == 0:
                continue
            target = gt[:, :horizon, :2].sum(1)
            threshold = .05 * rms * np.sqrt(horizon)
            gt_valid = np.linalg.norm(target, axis=-1) >= threshold
            if col < 2:
                vectors = model["action_pred"][:, :, :horizon, :2].sum(2)
                valid = (np.linalg.norm(vectors, axis=-1) >= threshold) & gt_valid[:, None]
                values = np.abs(angle_error(vectors, target[:, None]))
                values = np.where(valid, values, np.nan)
                with np.errstate(invalid="ignore", divide="ignore"):
                    counts = np.sum(valid, axis=1)
                    window_values = np.divide(np.nansum(values, axis=1), counts,
                                              out=np.full(len(gt), np.nan), where=counts > 0)
            else:
                vectors = model["heading_direction"]
                valid = np.isfinite(vectors).all(-1) & (np.linalg.norm(vectors, axis=-1) > 1e-10) & gt_valid
                window_values = np.where(valid, np.abs(angle_error(vectors, target)), np.nan)
            for task_idx, (task_uid, _) in enumerate(groups):
                mask = window_tasks == task_uid
                valid_values = window_values[mask]
                mean = float(np.nanmean(valid_values)) if np.isfinite(valid_values).any() else np.nan
                ypos = task_idx + (m - 1) * .18
                ax.scatter(mean, ypos, color=COLORS[m], s=40, zorder=4, label=MODEL_LABELS[m] if task_idx == 0 else None)
                if np.isfinite(mean):
                    ax.text(mean + 1.5, ypos, f"{mean:.1f}°", fontsize=7.5, color=COLORS[m], va="center")
                overview_stats.append(dict(task_uid=int(task_uid), model_id=MODEL_IDS[m],
                                           metric=("heading_head_16" if col == 2 else f"generated_{horizon}"),
                                           angular_error_deg=mean, valid_windows=int(np.isfinite(valid_values).sum())))
        ax.set_xlim(0, 190)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.set_ylim(len(groups) - .5, -.5)
        ax.axvline(30, color=GT_COLOR, lw=.8, ls="--", alpha=.6)
        ax.set_xlabel("Mean angular error (degrees) ↓", fontsize=10, color=DARK)
        ax.set_title(("Generated first 8 actions", "Generated full 16-action chunk", "Predicted heading head (16-step GT)")[col],
                     fontsize=11, color=DARK, fontweight="bold", pad=17)
    # These are task means, not individual angles: use one shared range that
    # includes every displayed mean with room for value labels.
    finite_means = [row["angular_error_deg"] for row in overview_stats if np.isfinite(row["angular_error_deg"])]
    mean_limit = min(195., max(30., float(np.ceil((max(finite_means) + 10.) / 15.) * 15.)))
    for ax in axes:
        ax.set_xlim(0, mean_limit)
        ax.set_xticks(np.arange(0, mean_limit + .1, 15.))
    axes[0].set_yticks(np.arange(len(groups)))
    axes[0].set_yticklabels([f"{i+1:02d} · " + "\n".join(textwrap.wrap(name, 39)) for i, name in enumerate(names)], fontsize=8)
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", color=COLORS[m], label=MODEL_LABELS[m]) for m in range(3)],
               loc="lower center", bbox_to_anchor=(.61, .1), ncol=3, frameon=False, fontsize=10)
    figure_footer(fig,
                  "Checkpoint selection uses each model's best recorded rollout SR. Backbone and training duration differ; this is a native-model comparison.\n"
                  "Motion threshold: ||Σ XY|| / (training XY RMS × √horizon) ≥ 0.05. See metrics.json / REPORT.md for coverage, cosine, ±15°/±30° rates and paired episode confidence intervals.",
                  y=.027, fontsize=9)
    save_figure(fig, output, "action_direction_overview_all10", inventory, dpi=dpi)
    return overview_stats


def gallery(output, representatives, groups, inventory):
    def links(stem):
        files = inventory[stem]
        return " · ".join(f'<a href="{html.escape(path)}">{ext.upper()}</a>' for ext, path in files.items())

    def card(stem, title, description):
        png = inventory[stem]["png"]
        return (f'<article><h3>{html.escape(title)}</h3><p>{html.escape(description)}</p>'
                f'<a href="{html.escape(png)}"><img loading="lazy" src="{html.escape(png)}" alt="{html.escape(title)}"></a>'
                f'<p class="downloads">{links(stem)}</p></article>')

    content = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
               "<title>LIBERO-10 noise and action comparison</title><style>",
               "body{margin:0;background:#f6f8fa;color:#263444;font:16px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:40px 28px}h1,h2,h3{line-height:1.25}h1{font-size:36px}h2{margin-top:52px}a{color:#225e92}p{max-width:1000px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px 18px;border-bottom:1px solid #dbe3e8;text-align:left}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:22px}article{background:white;border:1px solid #dbe3e8;border-radius:10px;padding:20px}article img{max-width:100%;height:auto;max-height:650px;object-fit:contain;object-position:top;display:block;margin:auto}article p{font-size:14px}.downloads{font-size:14px}.note{padding:18px;background:#eaf1f5;border-left:4px solid #3479B6}nav{display:flex;flex-wrap:wrap;gap:8px}nav a{padding:5px 9px;background:white;border:1px solid #dbe3e8;border-radius:4px}footer{font-size:13px;color:#647582;margin:40px 0} @media(max-width:700px){main{padding:24px 14px}h1{font-size:28px}}</style><main>",
               "<h1>LIBERO-10: noise shape and predicted action direction</h1>",
               "<p>Three native models at their best recorded success-rate checkpoint, evaluated on paired held-out demonstration observations across all ten tasks.</p>",
               "<table><thead><tr><th>Model</th><th>Recorded rollout SR</th><th>EMA checkpoint</th></tr></thead><tbody>",
               "<tr><td>Transformer Flow baseline</td><td>76.8% (384/500)</td><td>ep-0055</td></tr>",
               "<tr><td>Heading Zero DiT</td><td>80.0% (400/500)</td><td>30 completed epochs</td></tr>",
               "<tr><td>Heading Gaussian DiT</td><td>79.0% (395/500)</td><td>60 completed epochs</td></tr></tbody></table>",
               "<p><a href='REPORT.md'>Findings and methods report</a> · <a href='metrics.json'>Numerical metrics</a> · <a href='selection.json'>Selected observations</a> · <a href='plot_manifest.json'>Figure provenance</a></p>",
               "<p class='note'>Noise plots show actual recorded source samples. Direction comparisons use raw XY action sums and matching demonstration horizons. Ground-truth indices follow the training dataset convention: observation images are post-action while the target starts at that same dataset index. Heatmaps show projected intended command displacement over eight actions, not physical robot rollouts or attention. Model and checkpoint differences are included; the comparison does not isolate a causal noise-source effect.</p>",
               "<h2>All-ten-task overviews</h2><div class='grid'>"]
    content.append(card("noise_overview_all10", "Source noise: chunk-resultant dot plots", "One fixed middle-demonstration image per task. All model/task axes use the same raw XY scale. Individual per-step dots are in the task details."))
    content.append(card("action_direction_overview_all10", "Generated and predicted-heading directions", "First eight actions, full sixteen-action chunks, and heading-head predictions against ground truth, for all held-out observations."))
    content.append(card("prediction_heatmaps_first_action_overview_all10", "First-action desired-goal heatmaps", "Task context with a marked crop and shared local zooms show desired first-action controller goals on the same preselected images. This immediate-command view remains useful when an eight-command endpoint leaves the camera view."))
    content.append(card("prediction_heatmaps_overview_all10", "Eight-action prediction-change heatmaps", "Endpoint densities on task images and signed model differences relative to Transformer Flow; identical density scales and smoothing across all tasks."))
    content.append("</div><h2>Three preselected phases for every task</h2><nav>")
    for i, (uid, _) in enumerate(groups, 1):
        content.append(f'<a href="#task-{uid}">Task {i:02d}</a>')
    content.append("</nav>")
    for i, (uid, indices) in enumerate(groups, 1):
        name = task_title(representatives[indices[0]]["task_name"])
        content.append(f'<section id="task-{uid}"><h2>{i:02d} · {html.escape(name)}</h2><div class="grid">')
        for suffix, title, description in (
            ("noise_raw", "Raw source noise", "Per-step and full-chunk-resultant dot distributions, with GT and predicted-heading direction arrows. Equal scales across models and phases."),
            ("noise_normalized", "Normalized source noise", "The corresponding flow-network source coordinates. Raw physical direction arrows are omitted because normalization may change angles."),
            ("action_directions", "Predicted direction versus ground truth", "Every dot is one generated chunk. First-eight and full-sixteen action directions use their own GT target; diamonds mark the heading head."),
            ("prediction_heatmaps_first_action", "First-action prediction changes", "Full images and identical local zooms for the same early, middle, and late observations. Densities show the desired controller goal from the first clipped/scaled XYZ command."),
            ("prediction_heatmaps", "Eight-action prediction changes on task images", "Full images plus local zoom for early, middle, and late observations. Green paths show ground-truth intended command displacement."),
        ):
            content.append(card(f"tasks/task_{uid}_{suffix}", title, description))
        content.append("</div></section>")
    content.append("<footer>Scientific figure gallery. PNG, PDF, and SVG exports are available for every figure. See the report and saved arrays for definitions, exclusions, reproducibility, and the limits of each visualization.</footer></main></html>")
    (output / "gallery.html").write_text("\n".join(content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "output/noise_direction_best_sr_20260912")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--only", choices=("all", "noise", "heatmaps", "directions", "first-action-overview"), default="all")
    args = parser.parse_args()
    output = args.run_dir.resolve()
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text())
    representatives = sorted(selection["representatives"], key=lambda row: row["representative_index"])
    require([r["representative_index"] for r in representatives] == list(range(30)), "Expected exactly 30 ordered representatives")
    groups = []
    for uid in sorted({r["task_uid"] for r in representatives}):
        indices = [i for i, r in enumerate(representatives) if r["task_uid"] == uid]
        indices.sort(key=lambda r: PHASES.index(representatives[r]["phase"]))
        require(len(indices) == 3, f"Task {uid} needs three phases")
        groups.append((uid, indices))
    require(len(groups) == 10, "Expected all ten LIBERO-10 tasks")
    gt = load_npz(output / "data.npz", ["actions_gt"])["actions_gt"]
    rep_gt = gt[[r["window_index"] for r in representatives]]
    reps_data = [load_npz(output / "models" / f"{model_id}_representatives.npz") for model_id in MODEL_IDS]
    projection = load_npz(output / "projection.npz")
    for model_id, model in zip(MODEL_IDS, reps_data):
        require(model["action_pred"].shape == (30, 128, 16, 7), f"{model_id}: wrong representative action shape")
        for key in ("source_raw", "source_norm"):
            require(model[key].shape == (30, 1024, 16, 7), f"{model_id}: wrong {key} shape")
            require(np.isfinite(model[key]).all(), f"{model_id}: nonfinite {key}")
        require(np.isfinite(model["action_pred"]).all(), f"{model_id}: nonfinite action predictions")
    rms_values = []
    for model_id in MODEL_IDS[1:]:
        metadata = json.loads((output / "models" / f"{model_id}.json").read_text())
        # Producer stores policy-specific scalar metadata alongside its NPZ.
        rms = metadata.get("heading_xy_rms", metadata.get("policy", {}).get("heading_xy_rms"))
        require(rms is not None and float(rms) > 0, f"{model_id}: missing positive training XY RMS")
        rms_values.append(float(rms))
    require(np.allclose(rms_values[0], rms_values[1], atol=1e-7, rtol=1e-6), "Heading models use different motion thresholds")
    rms = rms_values[0]
    rep_gt_valid = np.linalg.norm(rep_gt[:, :, :2].sum(1), axis=1) >= .05 * rms * 4
    inventory = {}
    old_manifest = output / "plot_manifest.json"
    if args.only != "all" and old_manifest.exists():
        inventory.update(json.loads(old_manifest.read_text()).get("figures", {}))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.labelcolor": DARK, "text.color": DARK, "savefig.facecolor": "white"}):
        if args.only == "first-action-overview":
            heatmaps = make_heatmaps(reps_data, rep_gt, projection, steps=1)
            render_heatmap_overview(output, representatives, groups, projection, heatmaps, inventory, args.dpi)
        if args.only in ("all", "noise"):
            render_noise_overview(output, representatives, groups, reps_data, rep_gt, rep_gt_valid, projection, inventory, args.dpi)
            render_noise_tasks(output, representatives, groups, reps_data, rep_gt, rep_gt_valid, inventory, False, args.dpi)
            render_noise_tasks(output, representatives, groups, reps_data, rep_gt, rep_gt_valid, inventory, True, args.dpi)
        if args.only in ("all", "heatmaps"):
            for steps in (8, 1):
                heatmaps = make_heatmaps(reps_data, rep_gt, projection, steps=steps)
                render_heatmap_overview(output, representatives, groups, projection, heatmaps, inventory, args.dpi)
                render_heatmap_tasks(output, representatives, groups, projection, heatmaps, inventory, args.dpi)
                np.savez_compressed(output / ("heatmap_first_action_arrays.npz" if steps == 1 else "heatmap_arrays.npz"), density=heatmaps["density"], difference=heatmaps["difference"],
                                    visible=np.array(heatmaps["visible"]), endpoint_pixels=np.array(heatmaps["pixels"]),
                                    gt_path=np.array(heatmaps["gt_path"]), crop=np.array(heatmaps["crop"]),
                                    density_max=heatmaps["density_max"], difference_max=heatmaps["difference_max"], bandwidth_px=2.)
                (output / ("heatmap_first_action_summary.json" if steps == 1 else "heatmap_summary.json")).write_text(json.dumps({
                    "definition": ("Projected desired controller goal from first clipped raw XYZ action; not physical rollout" if steps == 1 else "Projected intended controller-command endpoint after first 8 clipped raw XYZ actions; not physical rollout"),
                    "action_steps": steps,
                    "controller_translation_scale_m": heatmaps["translation_scale"], "bandwidth_original_image_px": 2.,
                    "normalization": "Histogram divided by all 128 model samples, including out-of-view endpoints; constant-zero Gaussian boundary",
                    "density_scale_all_tasks": [0., heatmaps["density_max"]],
                    "signed_difference_scale_all_tasks": [-heatmaps["difference_max"], heatmaps["difference_max"]],
                    "comparison": "native heading model minus native Transformer Flow baseline; model/backbone/checkpoint differences included",
                    "rows": [{"representative_index": r, "task_uid": rep["task_uid"], "phase": rep["phase"],
                              "visible_counts": {MODEL_IDS[m]: int(np.sum(heatmaps["visible"][r][m])) for m in range(3)},
                              "behind_camera_or_invalid_counts": {MODEL_IDS[m]: int(np.sum(~heatmaps["camera_valid"][r][m])) for m in range(3)},
                              "offscreen_front_camera_counts": {MODEL_IDS[m]: int(np.sum(heatmaps["camera_valid"][r][m] & ~heatmaps["visible"][r][m])) for m in range(3)},
                              "density_probability_mass": {MODEL_IDS[m]: float(heatmaps["density"][r, m].sum()) for m in range(3)}}
                             for r, rep in enumerate(representatives)],
                }, indent=2) + "\n")
        if args.only in ("all", "directions"):
            all_data = [load_npz(output / "models" / f"{model_id}.npz", ["action_pred", "heading_direction"]) for model_id in MODEL_IDS]
            stats = render_direction_overview(output, selection, groups, all_data, gt, rms, inventory, args.dpi)
            render_direction_tasks(output, representatives, groups, reps_data, rep_gt, rms, inventory, args.dpi)
            (output / "direction_plot_summary.json").write_text(json.dumps({"rows": stats}, indent=2) + "\n")
    expected = {"noise_overview_all10", "prediction_heatmaps_overview_all10", "action_direction_overview_all10", "prediction_heatmaps_first_action_overview_all10"}
    expected |= {f"tasks/task_{uid}_{kind}" for uid, _ in groups
                 for kind in ("noise_raw", "noise_normalized", "prediction_heatmaps", "prediction_heatmaps_first_action", "action_directions")}
    if expected.issubset(inventory):
        gallery(output, representatives, groups, inventory)
    inputs = [selection_path, output / "data.npz", output / "projection.npz"]
    inputs += [output / "models" / f"{model_id}{suffix}" for model_id in MODEL_IDS
               for suffix in (".npz", ".json", "_representatives.npz")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(__file__),
                "input_sha256": {str(p.relative_to(output)): sha256(p) for p in inputs},
                "model_ids": list(MODEL_IDS), "task_count": len(groups), "representative_count": len(representatives),
                "heading_xy_rms": rms, "direction_threshold_ratio": .05,
                "coordinate_notes": {"source_raw": "denormalized source in raw action coordinates",
                                     "source_norm": "actual normalized flow-network source",
                                     "generated": "raw action sums; 8 and 16 steps compared against same-horizon demonstration",
                                     "heatmaps": "cumulative clipped XYZ command*controller scale, not physical rollout"},
                "figures": inventory, "complete": expected == set(inventory),
                "output_sha256": {path: sha256(output / path) for files in inventory.values() for path in files.values()}}
    old_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(inventory)} figures × 3 formats. Complete: {manifest['complete']}", flush=True)


if __name__ == "__main__":
    main()
