#!/usr/bin/env python3
"""Plot the five-task, shared-held-out Heading Zero/Gaussian/SMQ analysis.

Reads recorded arrays and analysis summaries.  Source clouds are latent flow
inputs expressed in raw action coordinates or normalized model coordinates;
neither is a measured robot displacement.  SMQ has four independent H4 heads,
which are never averaged into an invented H16 head.  Camera overlays visualize
clipped intended commands, not physical rollouts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np

try:
    from scripts.analyze_noise_direction_comparison import mean_finite
    from scripts.plot_noise_direction_comparison import (
        arrow, figure_footer, heatmap_panel, load_npz, make_heatmaps, save_figure,
        sha256, style_axes, task_title,
    )
except ModuleNotFoundError:  # Support ``python scripts/plot_...py``.
    from analyze_noise_direction_comparison import mean_finite
    from plot_noise_direction_comparison import (
        arrow, figure_footer, heatmap_panel, load_npz, make_heatmaps, save_figure,
        sha256, style_axes, task_title,
    )

MODEL_IDS = ("heading_zero_dit", "heading_gaussian_dit", "smq_dit")
MODEL_LABELS = ("Heading Zero", "Heading Gaussian", "SMQ")
MODEL_SHORT = ("Zero", "Gaussian", "SMQ")
MODEL_DESCRIPTIONS = ("one 16-action heading", "one 16-action heading", "four 4-action headings")
COLORS = ("#CC7035", "#7957AB", "#167E9A")
DARK = "#263444"
GT_COLOR = "#18754B"
HEAD_COLOR = "#AE2861"
PHASES = ("early", "middle", "late")
SCOPE = "Shared held-out subset · 239 windows · 8 episodes · 5 tasks"
LIMITATION = "Only 8 shared held-out episodes; 3 of the 5 tasks have one episode. Results describe this fixed five-task subset."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wrapped(name, width=85):
    return "\n".join(textwrap.wrap(task_title(name), width=width))


def angle_error(vectors, target):
    """Signed XY angle, with validity handled explicitly by the caller."""
    delta = np.arctan2(vectors[..., 1], vectors[..., 0]) - np.arctan2(target[..., 1], target[..., 0])
    return np.degrees(np.arctan2(np.sin(delta), np.cos(delta)))


def vector_valid(vector, threshold):
    vector = np.asarray(vector)
    return np.isfinite(vector).all(axis=-1) & (np.linalg.norm(vector, axis=-1) >= threshold)


def figure_header(fig, title, subtitle, y=.974, subtitle_y=.934, width=110):
    fig.suptitle(title, x=.055, y=y, ha="left", color=DARK, fontsize=17, fontweight="bold")
    fig.text(.055, subtitle_y, subtitle, color=DARK, fontsize=9.5, va="top", linespacing=1.5)


def legend(fig, y=.058, head_label="Heading Zero/Gaussian: predicted H16 head", fontsize=8.5):
    fig.legend(handles=[
        Line2D([], [], color=GT_COLOR, lw=2, label="GT raw XY direction (matching horizon)"),
        Line2D([], [], color=HEAD_COLOR, lw=2, ls="--", label=head_label),
    ], loc="lower center", bbox_to_anchor=(.5, y), ncol=2, frameon=False, fontsize=fontsize)


def source_limit(models, indices, key, resultant, segment=None):
    extent = 0.
    for model in models:
        values = np.asarray(model[key][indices, ..., :2], dtype=np.float64)
        if segment is not None:
            values = values[..., segment * 4:(segment + 1) * 4, :]
        if resultant:
            values = values.sum(axis=-2)
        extent = max(extent, float(np.abs(values).max()))
    return max(extent * 1.09, .1)


def source_panel(ax, values, color, limit, resultant, label="", gt=None, head=None,
                 normalized=False, compact=False, horizon=16):
    style_axes(ax)
    if resultant:
        dots = values[..., :2].sum(axis=-2)
        dot_size, alpha = 5.5, .35
    else:
        ids = np.linspace(0, len(values) - 1, min(256, len(values)), dtype=int)
        dots = values[ids, ..., :2].reshape(-1, 2)
        dot_size, alpha = 2.8, .21
    require(np.isfinite(dots).all(), "Nonfinite source points")
    require(np.abs(dots).max() < limit, "Source points would be cropped by the chosen axis range")
    ax.scatter(dots[:, 0], dots[:, 1], s=dot_size, color=color, alpha=alpha, linewidths=0, rasterized=True, zorder=2)
    ax.axhline(0, color="#AEBBC4", lw=.6, zorder=1)
    ax.axvline(0, color="#AEBBC4", lw=.6, zorder=1)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    if gt is not None:
        arrow(ax, gt, limit, GT_COLOR)
    if head is not None:
        require(np.asarray(head).shape == (2,), "Display one actual head, never an averaged SMQ head")
        arrow(ax, head, limit, HEAD_COLOR, dashed=True)
    if label:
        ax.text(.04, .96, label, transform=ax.transAxes, va="top", fontsize=7.5, color=DARK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
    if not compact:
        kind = f"sum of {horizon} " if resultant else "per-step "
        coordinates = "normalized source" if normalized else "raw source"
        ax.set_xlabel(f"{kind}X · {coordinates}", fontsize=8, color=DARK)
        ax.set_ylabel(f"{kind}Y · {coordinates}", fontsize=8, color=DARK)
    ax.locator_params(axis="both", nbins=5)


def head16(model, representative_index):
    heads = model["heading_direction"][representative_index]
    return heads[0] if heads.shape == (1, 2) else None


def middle(indices, representatives):
    return next((r for r in indices if representatives[r]["phase"] == "middle"), indices[len(indices) // 2])


def render_noise_overview(output, representatives, groups, models, gt, projection, threshold16, inventory, dpi):
    mids = [middle(indices, representatives) for _, indices in groups]
    limit = source_limit(models, mids, "source_raw", True)
    fig, axes = plt.subplots(5, 4, figsize=(15.5, 18.8), squeeze=False,
                             gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    fig.subplots_adjust(left=.055, right=.975, top=.866, bottom=.105, hspace=.55, wspace=.23)
    figure_header(fig, "Source distributions on the shared held-out subset",
                  SCOPE + "\nMiddle-phase observations · 1,024 draws per model · sum of 16 source XY values")
    for row, ((task_uid, indices), r) in enumerate(zip(groups, mids)):
        ax = axes[row, 0]
        ax.imshow(projection["image"][r])
        ax.axis("off")
        ax.set_title(wrapped(f"Task {task_uid} · {representatives[r]['task_name']}", 40), fontsize=8, loc="left", color=DARK)
        target = gt[r, :, :2].sum(axis=0)
        for m, model in enumerate(models):
            head = head16(model, r)
            source_panel(axes[row, m + 1], model["source_raw"][r], COLORS[m], limit, True,
                         "4 segment heads; see detail" if m == 2 else "",
                         target if vector_valid(target, threshold16) else None, head, compact=True)
            if row == 0:
                axes[row, m + 1].set_title(MODEL_LABELS[m] + "\n" + MODEL_DESCRIPTIONS[m], fontsize=10, fontweight="bold", color=DARK, pad=12)
            if m == 0:
                axes[row, m + 1].set_ylabel("Sum of 16 raw source Y", fontsize=8.5)
            if row == 4:
                axes[row, m + 1].set_xlabel("Sum of 16 raw source X", fontsize=8.5)
    legend(fig, y=.060)
    figure_footer(fig,
                  "All 15 source panels share one scale; every resultant draw is shown. Arrows show direction with equal display length.\n"
                  "SMQ's H4 heads are shown separately in its segment figures; no H16 head is constructed. Sources are inputs to flow, not motion.\n" + LIMITATION,
                  y=.019, fontsize=8.2)
    save_figure(fig, output, "noise_overview_shared5", inventory, dpi=dpi)


def render_noise_tasks(output, representatives, groups, models, gt, threshold16, inventory, dpi):
    for normalized in (False, True):
        key = "source_norm" if normalized else "source_raw"
        suffix = "normalized" if normalized else "raw"
        for task_uid, indices in groups:
            fig, axes = plt.subplots(6, 3, figsize=(14.5, 22), squeeze=False)
            fig.subplots_adjust(left=.09, right=.97, top=.884, bottom=.115, hspace=.6, wspace=.25)
            figure_header(fig, f"Task {task_uid} · {'Normalized' if normalized else 'Raw'} source distributions",
                          wrapped(representatives[indices[0]]["task_name"], 110) + "\n" + SCOPE,
                          subtitle_y=.944)
            step_limit = source_limit(models, indices, key, False)
            sum_limit = source_limit(models, indices, key, True)
            for phase_index, r in enumerate(indices):
                target = gt[r, :, :2].sum(axis=0)
                rep = representatives[r]
                for m, model in enumerate(models):
                    for sum_index, resultant in enumerate((False, True)):
                        row = phase_index * 2 + sum_index
                        detail = f"{rep['phase'].title()} · frame {rep['frame_index']}\n"
                        detail += "1,024 chunk resultants" if resultant else "4,096 per-step dots"
                        if resultant:
                            detail += f"\nGate active {model['source_active'][r].mean() * 100:.1f}%"
                            if m == 2:
                                detail += " (mean of 4 segments)"
                        source_panel(axes[row, m], model[key][r], COLORS[m], sum_limit if resultant else step_limit,
                                     resultant, detail,
                                     None if normalized or not vector_valid(target, threshold16) else target,
                                     None if normalized else head16(model, r), normalized)
                        if row == 0:
                            axes[row, m].set_title(MODEL_LABELS[m] + "\n" + MODEL_DESCRIPTIONS[m], fontsize=11, fontweight="bold", color=DARK, pad=14)
            if not normalized:
                legend(fig, y=.068)
            note = ("Normalized source coordinates are the actual flow-network input; affine normalization can change XY angles.\n"
                    "Raw GT/head arrows are therefore omitted. Compare raw sources in the companion figure.\n") if normalized else (
                    "Green: H16 GT direction; pink dashed: the actual H16 heading head (Heading Zero/Gaussian only). Arrows show direction only.\n"
                    "SMQ shapes four H4 segments; its sum-of-16 cloud need not align with any one segment head. See the SMQ segment figure.\n")
            figure_footer(fig, note + "All models and phases share limits within each row type; no source dots are cropped. Per-step panels display 256 source chunks.\n"
                          "Early/middle/late are selected observation phases within a shared held-out episode; every frame is one of the same 239 windows.",
                          y=.021, fontsize=8.1)
            save_figure(fig, output, Path("tasks") / f"task_{task_uid}_noise_{suffix}", inventory, dpi=dpi)


def render_smq_segments(output, representatives, groups, models, gt, threshold4, inventory, dpi):
    model = models[2]
    for task_uid, indices in groups:
        fig, axes = plt.subplots(3, 4, figsize=(16, 13.5), squeeze=False)
        fig.subplots_adjust(left=.075, right=.97, top=.85, bottom=.16, hspace=.5, wspace=.25)
        figure_header(fig, f"Task {task_uid} · SMQ's four source segments",
                      wrapped(representatives[indices[0]]["task_name"], 105) + "\n"
                      "Each panel compares one H4 source resultant, its own head, and the matching H4 demonstration direction.", subtitle_y=.942)
        limit = max(source_limit([model], indices, "source_raw", True, segment=k) for k in range(4))
        for row, r in enumerate(indices):
            for k in range(4):
                values = model["source_raw"][r, :, k * 4:(k + 1) * 4]
                target = gt[r, k * 4:(k + 1) * 4, :2].sum(axis=0)
                confidence = float(model["heading_confidence"][r, k])
                active = float(model["source_active"][r, :, k].mean())
                detail = f"{representatives[r]['phase'].title()} · f{representatives[r]['frame_index']}\n"
                detail += f"Head confidence {confidence:.2f}\nGate active {active * 100:.1f}%"
                source_panel(axes[row, k], values, COLORS[2], limit, True, detail,
                             target if vector_valid(target, threshold4) else None,
                             model["heading_direction"][r, k], horizon=4)
                if row == 0:
                    axes[row, k].set_title(f"Segment {k + 1} · actions {k * 4 + 1}–{(k + 1) * 4}", fontsize=10.5, color=DARK, fontweight="bold", pad=12)
        legend(fig, y=.095, head_label="SMQ: predicted head of this H4 segment")
        figure_footer(fig,
                      "1,024 resultants per panel; one shared XY scale across all four segments and all three observations. Arrows have equal display length.\n"
                      "Green is omitted when the matching H4 GT motion is below the common threshold. Head confidence and source activation describe this native model.\n"
                      "These are source-coordinate distributions, not generated actions or physical displacements. The four heads are never averaged into one H16 direction.\n" + LIMITATION,
                      y=.025, fontsize=8.1)
        save_figure(fig, output, Path("tasks") / f"task_{task_uid}_smq_segments", inventory, dpi=dpi)


def render_direction_overview(output, selection, groups, per_window, inventory, dpi):
    tasks = np.asarray([row["task_uid"] for row in selection["windows"]])
    episodes = np.asarray([row["episode_index"] for row in selection["windows"]])
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 7.8), sharey=True)
    fig.subplots_adjust(left=.285, right=.98, top=.77, bottom=.2, wspace=.18)
    figure_header(fig, "Generated action direction versus ground truth",
                  SCOPE + "\nPer-task angular error · fixed 32 draws averaged within windows, then equal selected windows within each task",
                  y=.967, subtitle_y=.910)
    specifications = [("direction_h8_angle_deg", "First 8 actions"),
                      ("direction_h16_angle_deg", "Full 16 actions"),
                      ("direction_segment4_angle_deg", "Four H4 segments")]
    names = []
    rows = np.arange(len(groups))
    for uid, indices in groups:
        mask = tasks == uid
        name = wrapped(selection["representatives"][indices[0]]["task_name"], 45)
        names.append(name + f"\n{mask.sum()} windows · {len(np.unique(episodes[mask]))} episode(s)")
    for col, (key, title) in enumerate(specifications):
        ax = axes[col]
        style_axes(ax)
        for m in range(3):
            values = [float(mean_finite(per_window[key][m, tasks == uid])) for uid, _ in groups]
            ax.scatter(values, rows + (m - 1) * .18, color=COLORS[m], s=44, label=MODEL_LABELS[m], zorder=3)
        ax.set_title(title, fontsize=12, color=DARK, fontweight="bold", pad=15)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("Mean absolute angular error (°) ↓", fontsize=8.8)
        ax.set_ylim(4.6, -.6)
        ax.set_yticks(rows, names)
        if col == 0:
            ax.tick_params(axis="y", labelsize=8)
    fig.legend(handles=[Line2D([], [], color=COLORS[m], marker="o", lw=0, label=MODEL_LABELS[m]) for m in range(3)],
               loc="lower center", bbox_to_anchor=(.62, .11), ncol=3, frameon=False, fontsize=9)
    figure_footer(fig,
                  "Angles condition on valid predicted and GT XY motion. The H4 panel scores all models on the same four action segments.\n"
                  "This figure compares generated actions; it does not compare the architectures' different native heading heads.\n" + LIMITATION,
                  y=.023, fontsize=8.2)
    save_figure(fig, output, "action_direction_overview_shared5", inventory, dpi=dpi)


def render_angle_distributions(output, selection, models, gt, rms, confidence, inventory, dpi):
    tasks = np.asarray([row["task_uid"] for row in selection["windows"]])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), sharey=True)
    fig.subplots_adjust(left=.08, right=.965, top=.73, bottom=.25, wspace=.15)
    figure_header(fig, "Distribution of generated direction errors",
                  SCOPE + "\nEach task has equal weight; each eligible window within a task has equal weight.",
                  y=.966, subtitle_y=.898)
    bin_edges = np.linspace(-180, 180, 37)
    summary = {}
    for col, horizon in enumerate((8, 16)):
        ax = axes[col]
        style_axes(ax)
        threshold = confidence * rms * np.sqrt(horizon)
        targets = gt[:, :horizon, :2].sum(axis=1)
        for m, model in enumerate(models):
            resultants = model["action_pred"][:, :, :horizon, :2].sum(axis=2)
            valid = vector_valid(resultants, threshold) & vector_valid(targets, threshold)[:, None]
            errors = angle_error(resultants, targets[:, None])
            weights = np.zeros(valid.shape, dtype=float)
            for task in np.unique(tasks):
                task_mask = tasks == task
                eligible = task_mask & valid.any(axis=1)
                require(eligible.any(), f"No valid directions for {MODEL_IDS[m]} task {task}")
                count = valid.sum(axis=1)
                weights[eligible] = valid[eligible] / count[eligible, None] / eligible.sum() / len(np.unique(tasks))
            require(np.isclose(weights.sum(), 1.), "Direction histogram weights must sum to one")
            frequency, _ = np.histogram(errors[valid], bins=bin_edges, weights=weights[valid])
            ax.stairs(frequency, bin_edges, color=COLORS[m], linewidth=2.1, label=MODEL_LABELS[m])
            summary[f"{MODEL_IDS[m]}_h{horizon}"] = {"bin_edges_deg": bin_edges.tolist(), "probability": frequency.tolist(),
                                                          "eligible_windows": int(valid.any(axis=1).sum()), "valid_draws": int(valid.sum())}
        ax.axvline(0, color=GT_COLOR, lw=1.2, alpha=.8)
        ax.axvspan(-30, 30, color=GT_COLOR, alpha=.055)
        ax.set(xlim=(-180, 180), xticks=[-180, -90, 0, 90, 180], xlabel="Signed XY direction error relative to GT (°)")
        ax.set_title(f"First {horizon} actions", fontsize=12, color=DARK, fontweight="bold", pad=13)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Probability per 10° bin", fontsize=10)
    fig.legend(handles=[Line2D([], [], color=COLORS[m], lw=2, label=MODEL_LABELS[m]) for m in range(3)],
               loc="lower center", bbox_to_anchor=(.5, .13), ncol=3, frameon=False, fontsize=10)
    figure_footer(fig,
                  "Direction distributions condition on valid GT and generated motion; invalid predicted motion is counted as a failure in accuracy metrics.\n"
                  "Sampled action chunks and nearby windows are dependent. Histogram shape does not represent the number of independent episodes.\n" + LIMITATION,
                  y=.025, fontsize=8.3)
    save_figure(fig, output, "action_angle_distributions", inventory, dpi=dpi)
    return summary


def render_paired_comparison(output, metrics, inventory, dpi):
    comparisons = [
        ("heading_gaussian_dit-minus-heading_zero_dit", "Gaussian − Zero"),
        ("smq_dit-minus-heading_zero_dit", "SMQ − Zero"),
        ("smq_dit-minus-heading_gaussian_dit", "SMQ − Gaussian"),
    ]
    specs = [("direction_h8_angle_deg", "First 8 actions", "Δ angular error (°) · lower is better", 1.),
             ("direction_h16_angle_deg", "Full 16 actions", "Δ angular error (°) · lower is better", 1.),
             ("direction_h8_within30", "First 8 within 30°", "Δ accuracy (percentage points) · higher is better", 100.)]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.9), sharey=True)
    fig.subplots_adjust(left=.165, right=.97, top=.7, bottom=.27, wspace=.3)
    figure_header(fig, "Paired differences on the same observations and draws",
                  SCOPE + "\nEqual task weight · fixed checkpoints · episode bootstrap within each task",
                  y=.965, subtitle_y=.902)
    for col, (metric_key, title, xlabel, scale) in enumerate(specs):
        ax = axes[col]
        style_axes(ax)
        extent = 0.
        for row, (pair_key, label) in enumerate(comparisons):
            estimate = metrics["paired_differences"][pair_key]["metrics"][metric_key]
            mean = float(estimate["mean"]) * scale
            ci = estimate.get("ci95_episode_bootstrap")
            if ci is not None:
                low, high = np.asarray(ci, float) * scale
                ax.plot([low, high], [row, row], color=COLORS[row], lw=2.1, solid_capstyle="round", zorder=3)
                extent = max(extent, abs(low), abs(high))
            ax.scatter(mean, row, s=52, color=COLORS[row], edgecolors="white", linewidths=.7, zorder=4)
            extent = max(extent, abs(mean))
            ax.annotate(f"{mean:+.2f}", xy=(mean, row), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8.5, color=DARK)
        ax.axvline(0, color="#778995", lw=1.1)
        extent = max(extent * 1.35, 1.)
        ax.set(xlim=(-extent, extent), ylim=(2.55, -.65), yticks=[0, 1, 2], yticklabels=[label for _, label in comparisons])
        ax.set_title(title, fontsize=12, fontweight="bold", color=DARK, pad=17)
        ax.set_xlabel("\n".join(textwrap.wrap(xlabel, 33)), fontsize=8.5)
        ax.tick_params(axis="y", labelsize=10)
        ax.locator_params(axis="x", nbins=5)
    figure_footer(fig,
                  "Points: paired mean difference. Lines: 95% episode-bootstrap intervals, conditional on these tasks, windows, draws, and checkpoints.\n"
                  "With only 8 episodes, uncertainty is weakly supported; 3 task strata contain one episode and cannot estimate within-task episode variation.\n"
                  "Angles use complete valid model/GT pairs; accuracy retains invalid predicted directions as failures on valid GT.\n"
                  "Differences include checkpoint, training, and native inference changes; they do not isolate a causal effect of the noise source.",
                  y=.03, fontsize=8.7)
    save_figure(fig, output, "paired_model_comparison", inventory, dpi=dpi)


def render_direction_tasks(output, representatives, groups, models, gt, rms, confidence, inventory, dpi):
    for task_uid, indices in groups:
        fig, axes = plt.subplots(3, 3, figsize=(14.5, 13), squeeze=False)
        fig.subplots_adjust(left=.085, right=.975, top=.855, bottom=.145, hspace=.58, wspace=.3)
        figure_header(fig, f"Task {task_uid} · generated action directions",
                      wrapped(representatives[indices[0]]["task_name"], 100) + "\n"
                      "128 generated draws per model and observation · four H4 rows score the same segments for all three models.", subtitle_y=.942)
        for row, r in enumerate(indices):
            for m, model in enumerate(models):
                ax = axes[row, m]
                style_axes(ax)
                specs = [(0, 8, "First 8"), (0, 16, "Full 16")] + [(k * 4, 4, f"H4 seg. {k + 1}") for k in range(4)]
                for ypos, (start, horizon, _) in enumerate(specs):
                    target = gt[r, start:start + horizon, :2].sum(axis=0)
                    vectors = model["action_pred"][r, :, start:start + horizon, :2].sum(axis=1)
                    threshold = confidence * rms * np.sqrt(horizon)
                    valid = vector_valid(vectors, threshold)
                    if vector_valid(target, threshold):
                        errors = angle_error(vectors, target)
                        jitter = np.sin(np.arange(len(errors)) * 2.399963229728653) * .19
                        ax.scatter(errors[valid], ypos + jitter[valid], s=6.2, color=COLORS[m], alpha=.43, linewidths=0, rasterized=True)
                    else:
                        ax.text(0, ypos, "GT below motion threshold", fontsize=7, ha="center", color="#71818D")
                ax.axvline(0, color=GT_COLOR, lw=1.1)
                ax.axvspan(-30, 30, color=GT_COLOR, alpha=.06)
                ax.set(xlim=(-180, 180), ylim=(5.5, -.6), xticks=[-180, -90, 0, 90, 180],
                       yticks=np.arange(6), yticklabels=[label for _, _, label in specs])
                ax.set_xlabel("Signed direction error relative to GT (°)", fontsize=8)
                if row == 0:
                    ax.set_title(MODEL_LABELS[m], fontsize=12, fontweight="bold", color=DARK, pad=15)
                if m == 0:
                    rep = representatives[r]
                    ax.set_ylabel(f"{rep['phase'].title()} · frame {rep['frame_index']}", fontsize=9.5, labelpad=10)
        figure_footer(fig,
                      "Each dot is a generated action-chunk direction relative to the demonstration at the matching horizon/segment; vertical jitter is only for legibility.\n"
                      "The four H4 action rows provide a common comparison; architecture-specific heading heads are not overlaid or averaged.\n"
                      "GT and prediction must exceed the shared motion threshold: 0.05 × training XY RMS × √horizon. Green band: within ±30°.\n" + LIMITATION,
                      y=.03, fontsize=8.6)
        save_figure(fig, output, Path("tasks") / f"task_{task_uid}_action_directions", inventory, dpi=dpi)


def add_density_colorbar(fig, heatmaps, rect=(.35, .087, .32, .012)):
    axis = fig.add_axes(rect)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, heatmaps["density_max"]), cmap="inferno"),
                           cax=axis, orientation="horizontal")
    colorbar.set_label("Endpoint probability per original image pixel", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.formatter.set_powerlimits((-2, 2))
    colorbar.update_ticks()


def density_note(heatmaps):
    first = heatmaps["steps"] == 1
    scale = heatmaps["translation_scale"]
    command = "First-action goal" if first else "First-eight intended command endpoint"
    return (f"{command}: clip raw XYZ commands to [−1, 1], multiply by {scale:g} m, then {'add to the observed EEF position' if first else 'sum from the observed EEF position'}.\n"
            "White dot: observed EEF. Green path/star: GT command/endpoint. Shared density scale across all tasks, models, and phases; smoothing σ = 2 image pixels.\n"
            "Out-of-view endpoints stay outside the image; the denominator retains all 128 draws. These are intended commands, not physical rollouts.")


def render_heatmap_overview(output, representatives, groups, projection, heatmaps, inventory, dpi):
    first = heatmaps["steps"] == 1
    title = "first-action controller goals" if first else "first-eight intended command endpoints"
    suffix = "first_action" if first else "first8"
    fig, axes = plt.subplots(5, 4, figsize=(15.8, 18.2), squeeze=False)
    fig.subplots_adjust(left=.055, right=.97, top=.87, bottom=.14, hspace=.5, wspace=.08)
    figure_header(fig, "Predicted " + title,
                  SCOPE + "\nMiddle-phase observations · 128 action draws per model · shared local zoom within each row")
    for row, (task_uid, indices) in enumerate(groups):
        r = middle(indices, representatives)
        ax = axes[row, 0]
        image = projection["image"][r]
        ax.imshow(image)
        x0, x1, y1, y0 = heatmaps["crop"][r]
        box = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.4, linestyle="--")
        box.set_path_effects([pe.Stroke(linewidth=2.4, foreground=DARK), pe.Normal()])
        ax.add_patch(box)
        ax.axis("off")
        ax.text(0, -.03, wrapped(f"Task {task_uid} · {representatives[r]['task_name']}", 44),
                transform=ax.transAxes, fontsize=7.4, va="top", color=DARK)
        if row == 0:
            ax.set_title("Task image · zoom region", fontsize=10.5, color=DARK, fontweight="bold", pad=12)
        for m in range(3):
            heatmap_panel(axes[row, m + 1], image, heatmaps["density"][r, m], heatmaps["density_max"],
                          heatmaps["gt_path"][r], visible=heatmaps["visible"][r][m], crop=heatmaps["crop"][r])
            if row == 0:
                axes[row, m + 1].set_title(MODEL_LABELS[m], fontsize=11, color=DARK, fontweight="bold", pad=12)
    add_density_colorbar(fig, heatmaps, rect=(.36, .085, .3, .009))
    figure_footer(fig, density_note(heatmaps) + "\nWhite box: shared 1st–99th percentile local zoom; full-image panels are in the per-task figures.",
                  y=.018, fontsize=8.2)
    save_figure(fig, output, f"prediction_heatmaps_{suffix}_overview_shared5", inventory, dpi=dpi)


def render_heatmap_tasks(output, representatives, groups, projection, heatmaps, inventory, dpi):
    first = heatmaps["steps"] == 1
    suffix = "first_action" if first else "first8"
    kind = "first-action goals" if first else "first-eight command endpoints"
    for task_uid, indices in groups:
        fig, axes = plt.subplots(6, 3, figsize=(12.8, 22), squeeze=False)
        fig.subplots_adjust(left=.105, right=.97, top=.89, bottom=.12, hspace=.17, wspace=.045)
        figure_header(fig, f"Task {task_uid} · predicted {kind}",
                      wrapped(representatives[indices[0]]["task_name"], 95) + "\n"
                      "Shared held-out observations · 128 action draws · full image and the same local zoom for all models", subtitle_y=.945)
        for phase_index, r in enumerate(indices):
            for local in range(2):
                row = phase_index * 2 + local
                for m in range(3):
                    heatmap_panel(axes[row, m], projection["image"][r], heatmaps["density"][r, m], heatmaps["density_max"],
                                  heatmaps["gt_path"][r], visible=heatmaps["visible"][r][m],
                                  crop=heatmaps["crop"][r] if local else None)
                    if row == 0:
                        axes[row, m].set_title(MODEL_LABELS[m], fontsize=11, fontweight="bold", color=DARK, pad=12)
                    if m == 0:
                        rep = representatives[r]
                        label = f"{rep['phase'].title()} · f{rep['frame_index']}\n" + ("Shared local zoom" if local else "Full image")
                        axes[row, m].text(-.07, .5, label, rotation=90, transform=axes[row, m].transAxes,
                                          ha="center", va="center", fontsize=9, color=DARK)
        add_density_colorbar(fig, heatmaps, rect=(.38, .078, .29, .009))
        figure_footer(fig, density_note(heatmaps) + "\nLocal crops use pooled 1st–99th percentile endpoints plus a margin; full-image rows retain the complete camera view.",
                      y=.019, fontsize=7.8)
        save_figure(fig, output, Path("tasks") / f"task_{task_uid}_prediction_heatmaps_{suffix}", inventory, dpi=dpi)


def save_heatmap_arrays(output, heatmaps, suffix):
    density = np.asarray(heatmaps["density"])
    require(np.isfinite(density).all() and (density >= 0).all(), "Density must be finite and nonnegative")
    mass = density.sum(axis=(-1, -2))
    visible = np.asarray(heatmaps["visible"])
    require(np.all(mass <= visible.mean(axis=-1) + 1e-10), "Smoothing cannot create endpoint probability mass")
    require(np.all(mass <= 1 + 1e-10), "Endpoint probabilities exceed one")
    np.savez_compressed(output / f"heatmap_{suffix}_arrays.npz", density=density,
                        endpoint_pixels=np.asarray(heatmaps["pixels"]), visible=visible,
                        camera_valid=np.asarray(heatmaps["camera_valid"]),
                        gt_path=np.asarray(heatmaps["gt_path"]), crop=np.asarray(heatmaps["crop"]),
                        density_max=heatmaps["density_max"], bandwidth_px=heatmaps["bandwidth_px"],
                        translation_scale=heatmaps["translation_scale"], steps=heatmaps["steps"],
                        model_ids=np.asarray(MODEL_IDS))
    return {"steps": heatmaps["steps"], "bandwidth_px": heatmaps["bandwidth_px"], "density_max": heatmaps["density_max"],
            "density_mass": mass.tolist(), "visible_fraction": visible.mean(axis=-1).tolist(),
            "finite_nonnegative_density": True, "out_of_view_samples_not_clipped": True,
            "probability_denominator": "all 128 sampled actions, including out-of-view endpoints",
            "camera_coordinates": "original 128 x 128 agentview pixels; shared crop does not rescale density"}


def write_gallery(output, inventory, selection):
    def links(stem):
        files = inventory[stem]
        return " · ".join(f'<a href="{html.escape(path)}">{extension.upper()}</a>' for extension, path in files.items())
    overview = ["noise_overview_shared5", "action_direction_overview_shared5", "action_angle_distributions", "paired_model_comparison",
                "prediction_heatmaps_first_action_overview_shared5", "prediction_heatmaps_first8_overview_shared5"]
    titles = ["Source distributions", "Generated directions by task", "Angle distributions", "Paired model differences",
              "First-action projected goals", "First-eight intended command endpoints"]
    cards = []
    for stem, title in zip(overview, titles):
        files = inventory[stem]
        cards.append(f'<article><h2>{title}</h2><p>{links(stem)}</p><a href="{files["png"]}"><img loading="lazy" src="{files["png"]}" alt="{title}"></a></article>')
    task_sections = []
    task_names = {row["task_uid"]: row["task_name"] for row in selection["representatives"]}
    for task_uid, name in sorted(task_names.items()):
        entries = []
        for suffix, title in [("noise_raw", "Raw sources"), ("noise_normalized", "Normalized sources"), ("smq_segments", "SMQ four H4 segments"),
                              ("action_directions", "Generated action directions"), ("prediction_heatmaps_first_action", "First-action maps"),
                              ("prediction_heatmaps_first8", "First-eight command maps")]:
            stem = f"tasks/task_{task_uid}_{suffix}"
            entries.append(f'<li>{title}: {links(stem)}</li>')
        task_sections.append(f'<details><summary>Task {task_uid} · {html.escape(task_title(name))}</summary><ul>{"".join(entries)}</ul></details>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Shared held-out Heading Zero / Heading Gaussian / SMQ comparison</title>'
    page += '<style>body{max-width:1300px;margin:40px auto;padding:0 24px;background:#f4f7f9;color:#263444;font:16px/1.55 system-ui,sans-serif}h1{line-height:1.15}a{color:#16688a}article{background:white;padding:20px;border-radius:8px;border:1px solid #dce4e9;min-width:0}article img{width:100%;height:420px;object-fit:contain;object-position:top}h2{font-size:20px;margin-top:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:22px}.note{padding:16px 20px;background:#e5edf2;border-radius:6px}details{margin:10px 0;background:white;padding:15px;border:1px solid #dce4e9}summary{cursor:pointer;font-weight:600}footer{margin-top:28px;color:#647782;font-size:14px}</style>'
    page += '<h1>Heading Zero · Heading Gaussian · SMQ</h1>'
    page += f'<p>{SCOPE}. All three checkpoints use the same observations and paired base noise.</p>'
    page += '<p class="note">This is the five-task shared-held-out comparison. The other five LIBERO-10 tasks are absent. '
    page += 'Only eight demonstrations are independent episode units; three task strata have a single episode. '
    page += 'SMQ has four H4 heads; the heading models have one H16 head. Source clouds are flow inputs, while image maps show intended commands and are not rollouts.</p>'
    page += '<p><a href="REPORT.md">Analysis report</a> · <a href="metrics.json">Metrics JSON</a> · <a href="per_task_metrics.csv">Per-task CSV</a> · <a href="selection.json">Exact selected windows</a> · <a href="plot_manifest.json">Figure manifest</a></p>'
    page += '<main class="grid">' + ''.join(cards) + '</main><h2 style="margin-top:32px">All three phases for each task</h2>' + ''.join(task_sections)
    page += '<footer>Every figure is available as PNG, PDF, and SVG. Static scientific figures generated from recorded inference arrays.</footer></html>'
    (output / "gallery.html").write_text(page)


def validate_inputs(output, selection, data, main, representatives, projection, per_window, metrics):
    windows = selection["windows"]
    reps = selection["representatives"]
    require(len(windows) == 239 and len(reps) == 15, "This analysis must use exactly 239 windows and 15 selected representatives")
    tasks = {row["task_uid"] for row in windows}
    episodes = {row["episode_index"] for row in windows}
    require(len(tasks) == 5 and len(episodes) == 8, "Expected five tasks and eight shared held-out episodes")
    require(sum(len({row['episode_index'] for row in windows if row['task_uid'] == uid}) == 1 for uid in tasks) == 3,
            "Scope label assumes exactly three task strata have one episode")
    require([row["window_index"] for row in windows] == list(range(239)), "Windows must have contiguous local indices")
    require(data["actions_gt"].shape == (239, 16, 7), "Unexpected ground-truth shape")
    require(np.isfinite(data["actions_gt"]).all(), "Ground truth contains nonfinite values")
    require([row["representative_index"] for row in reps] == list(range(15)), "Representatives must have contiguous indices")
    require(len({row['window_index'] for row in reps}) == 15, "Representatives must be distinct")
    for rep in reps:
        original = windows[rep["window_index"]]
        for key in ("task_uid", "episode_index", "frame_index", "original_window_index"):
            require(rep[key] == original[key], f"Representative {key} does not match a selected window")
    groups = []
    for uid in sorted(tasks):
        indices = [r for r, row in enumerate(reps) if row["task_uid"] == uid]
        require([reps[r]["phase"] for r in indices] == list(PHASES), "Representatives must be early, middle, late per task")
        groups.append((uid, indices))
    for model_id, model, representative in zip(MODEL_IDS, main, representatives):
        k = 4 if model_id == "smq_dit" else 1
        for arrays, count, draws in ((model, 239, 32), (representative, 15, 128)):
            require(arrays["action_pred"].shape == (count, draws, 16, 7), f"Unexpected {model_id} action shape")
            source_draws = 1024 if count == 15 else 32
            require(arrays["source_raw"].shape == (count, source_draws, 16, 7), f"Unexpected {model_id} source shape")
            require(arrays["source_norm"].shape == arrays["source_raw"].shape, "Source raw/normalized arrays differ in shape")
            require(arrays["heading_direction"].shape == (count, k, 2), f"Unexpected {model_id} heading shape")
            require(arrays["heading_confidence"].shape == (count, k), "Unexpected heading confidence shape")
            require(arrays["source_active"].shape == (count, source_draws, k), "Unexpected source activation shape")
            for key in ("action_pred", "source_raw", "source_norm", "heading_direction", "heading_confidence", "source_active"):
                require(np.isfinite(arrays[key]).all(), f"Nonfinite {model_id} {key}")
    require(projection["image"].shape == (15, 128, 128, 3), "Projection requires representative 128x128 RGB images")
    require(projection["world_to_pixel"].shape in ((15, 4, 4), (15, 3, 4)), "Unexpected projection matrix shape")
    require(projection["ee_pos"].shape == (15, 3), "Unexpected EEF position shape")
    for key in ("world_to_pixel", "ee_pos", "image"):
        require(np.isfinite(projection[key]).all(), f"Nonfinite projection {key}")
    require(np.isclose(float(np.asarray(projection["controller_translation_scale"]).reshape(-1)[0]), .05), "Expected LIBERO controller translation scale 0.05 m")
    require(list(metrics["model_order"]) == list(MODEL_IDS), "Unexpected model order in metrics")
    for key in ("direction_h8_angle_deg", "direction_h16_angle_deg", "direction_segment4_angle_deg"):
        require(per_window[key].shape == (3, 239), f"Unexpected per-window metric shape for {key}")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/shared_holdout_heading_smq_20260914"))
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    output = args.output.resolve()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelcolor": DARK, "text.color": DARK,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    selection = json.loads((output / "selection.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    data = load_npz(output / "data.npz", ["actions_gt"])
    models = [load_npz(output / "models" / f"{model_id}.npz") for model_id in MODEL_IDS]
    reps_models = [load_npz(output / "models" / f"{model_id}_representatives.npz") for model_id in MODEL_IDS]
    projection = load_npz(output / "projection.npz")
    per_window = load_npz(output / "per_window_metrics.npz")
    groups = validate_inputs(output, selection, data, models, reps_models, projection, per_window, metrics)
    reps = selection["representatives"]
    gt = np.asarray(data["actions_gt"])
    rep_gt = gt[[row["window_index"] for row in reps]]
    rms = float(metrics["direction_definition"]["heading_xy_rms"])
    confidence = float(metrics["direction_definition"]["min_target_confidence"])
    inventory = {}
    render_noise_overview(output, reps, groups, reps_models, rep_gt, projection, confidence * rms * 4, inventory, args.dpi)
    render_noise_tasks(output, reps, groups, reps_models, rep_gt, confidence * rms * 4, inventory, args.dpi)
    render_smq_segments(output, reps, groups, reps_models, rep_gt, confidence * rms * 2, inventory, args.dpi)
    render_direction_overview(output, selection, groups, per_window, inventory, args.dpi)
    angle_summary = render_angle_distributions(output, selection, models, gt, rms, confidence, inventory, args.dpi)
    render_paired_comparison(output, metrics, inventory, args.dpi)
    render_direction_tasks(output, reps, groups, reps_models, rep_gt, rms, confidence, inventory, args.dpi)
    heatmap_summary = {}
    for steps, suffix in ((1, "first_action"), (8, "first8")):
        heatmaps = make_heatmaps(reps_models, rep_gt, projection, bandwidth=2., steps=steps)
        heatmap_summary[suffix] = save_heatmap_arrays(output, heatmaps, suffix)
        render_heatmap_overview(output, reps, groups, projection, heatmaps, inventory, args.dpi)
        render_heatmap_tasks(output, reps, groups, projection, heatmaps, inventory, args.dpi)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE, "model_order": MODEL_IDS,
                "limitations": LIMITATION, "figure_count": len(inventory), "formats": ["png", "pdf", "svg"], "figures": inventory,
                "input_sha256": {str(path.relative_to(output)): sha256(path) for path in
                                 [output / "selection.json", output / "data.npz", output / "metrics.json", output / "projection.npz",
                                  output / "per_window_metrics.npz"] +
                                 [output / "models" / f"{model_id}{suffix}.npz" for model_id in MODEL_IDS for suffix in ("", "_representatives")]},
                "plotting_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
                "source_axis_policy": "One model-independent range within each figure and cloud type, based on maximum absolute source coordinate; no source points cropped",
                "smq_head_policy": "Show each H4 head only against its own H4 source and GT; no averaged H16 head",
                "heatmap_summary": heatmap_summary}
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "direction_plot_summary.json").write_text(json.dumps(angle_summary, indent=2) + "\n")
    (output / "plot_quality_checks.json").write_text(json.dumps({
        "input_shape_and_finiteness_checks": "passed", "scope_and_representative_membership_checks": "passed",
        "source_no_crop_checks": "passed", "histogram_unit_mass_checks": "passed", "density_probability_checks": "passed",
        "figure_count": len(inventory), "expected_figure_count": 36,
        "visual_review": "Requires inspection of rendered artifacts; automated checks do not certify typography or camera alignment.",
    }, indent=2) + "\n")
    require(len(inventory) == 36, "Expected six overview and thirty task figures")
    write_gallery(output, inventory, selection)
    print(f"Completed {len(inventory)} figures in PNG/PDF/SVG and {output / 'gallery.html'}", flush=True)


if __name__ == "__main__":
    main()
