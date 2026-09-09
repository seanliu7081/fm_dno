"""Visualize actual paired IID and XY-heading source samples from the mini run.

No new model sampling or training. The top paths are cumulative sums of noise
vectors, not generated actions or robot trajectories. Population angles are
relative to the prior model's own predicted heading, never demonstration labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2878B5"
ORANGE = "#DC7433"
DARK = "#263444"
GREEN = "#41735A"


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def heading(xy):
    resultant = xy.sum(axis=-2)
    if np.any(np.linalg.norm(resultant, axis=-1) < 1e-8):
        raise ValueError("Undefined source direction")
    return np.arctan2(resultant[..., 1], resultant[..., 0])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_paired(run_dir, seed):
    paths = {mode: run_dir / f"seed{seed}_{mode}.npz"
             for mode in ("condition", "condition_prior_xy")}
    data = {}
    for mode, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            data[mode] = {key: archive[key] for key in archive.files}
    iid, prior = data["condition"], data["condition_prior_xy"]
    for key in ("episode", "frame", "task", "epsilon", "angular_jitter"):
        if not np.array_equal(iid[key], prior[key]):
            raise ValueError(f"Pairing mismatch: {key}")
    if not np.all(prior["source_active"]):
        raise ValueError("This plot requires recorded active priors to recover headings")
    np.testing.assert_array_equal(iid["source"], iid["epsilon"])
    np.testing.assert_array_equal(prior["source"][..., 2:], iid["source"][..., 2:])
    z = iid["source"][..., :2].astype(np.float64)
    x0 = prior["source"][..., :2].astype(np.float64)
    inferred = wrap(heading(x0) - prior["angular_jitter"])
    predicted = np.angle(np.exp(1j * inferred).mean(axis=0))
    reconstruction_error = np.max(np.abs(wrap(inferred - predicted[None])))
    if reconstruction_error > 1e-5:
        raise ValueError("Recovered prediction differs across draws of the same observation")
    iid_relative = wrap(heading(z) - predicted[None])
    prior_relative = wrap(heading(x0) - predicted[None])
    np.testing.assert_allclose(wrap(prior_relative - prior["angular_jitter"]), 0., atol=1e-5)
    length_error = np.max(np.abs(np.linalg.norm(z, axis=-1) - np.linalg.norm(x0, axis=-1)))
    if length_error > 1e-5:
        raise ValueError("XY source length was not preserved")
    return iid, prior, z, x0, predicted, iid_relative, prior_relative, paths, {
        "max_prediction_reconstruction_difference_deg": float(np.degrees(reconstruction_error)),
        "max_per_step_xy_length_difference": float(length_error),
        "unselected_dimensions_bit_identical": True,
        "base_epsilon_and_jitter_paired": True,
    }


def style_cartesian(ax):
    ax.set_facecolor("#FCFDFE")
    ax.grid(color="#D9E1E8", alpha=.6, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A6B3BF")
    ax.tick_params(colors=DARK, labelsize=9)


def path_panel(ax, xy, color, title, theta, extent, label):
    path = np.vstack((np.zeros(2), xy.cumsum(axis=0)))
    style_cartesian(ax)
    ax.axhline(0, color="#B5C1CC", linewidth=.75)
    ax.axvline(0, color="#B5C1CC", linewidth=.75)
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, alpha=.95,
            marker="o", markersize=3)
    ax.scatter([0], [0], s=45, color=DARK, zorder=5)
    end = path[-1]
    ax.annotate("", xy=end, xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "lw": 2.6, "color": color,
                            "mutation_scale": 15}, zorder=4)
    predicted_end = extent * .68 * np.array([np.cos(theta), np.sin(theta)])
    ax.annotate("", xy=predicted_end, xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "lw": 1.8, "linestyle": "--",
                            "color": GREEN, "mutation_scale": 14}, zorder=3)
    ax.annotate("start", (0, 0), xytext=(6, -13), textcoords="offset points", fontsize=9, color=DARK)
    ax.annotate("step 16", end, xytext=(7, 7 if end[1] >= 0 else -15), textcoords="offset points", fontsize=9, color=color)
    ax.text(.035, .965, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color=DARK,
            bbox={"facecolor":"white", "edgecolor":"#E2E8ED", "boxstyle":"round,pad=.45", "alpha":.94})
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", fontsize=12, color=DARK, fontweight="bold", pad=12)
    ax.set_xlabel("Cumulative normalized noise X", fontsize=9, color=DARK)
    ax.set_ylabel("Cumulative normalized noise Y", fontsize=9, color=DARK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "output/mini_heading_prior/sampling_robustness_20260909")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/mini_heading_prior/noise_visualization_20260909")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--draw", type=int, default=0)
    args = parser.parse_args()
    iid, prior, z, x0, theta, rel_iid, rel_prior, paths, checks = load_paired(args.run_dir, args.seed)
    draws, windows, horizon, _ = z.shape
    if not (0 <= args.window < windows and 0 <= args.draw < draws):
        parser.error("Window or draw is outside the saved arrays")
    obs, draw = args.window, args.draw
    source_xy, prior_xy = z[draw, obs], x0[draw, obs]
    theta_example = theta[obs]
    h_iid, h_prior = heading(source_xy), heading(prior_xy)
    phi = wrap(h_prior - h_iid)
    cosine, sine = np.cos(phi), np.sin(phi)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    np.testing.assert_allclose(source_xy @ rotation.T, prior_xy, atol=2e-6, rtol=2e-6)
    endpoint = np.cumsum(np.stack((source_xy, prior_xy)), axis=1)
    extent = max(float(np.abs(endpoint).max()) * 1.19, 1.)
    stats = {}
    for name, values in (("iid", rel_iid), ("prior", rel_prior)):
        stats[name] = {
            "mean_absolute_angle_from_prediction_deg": float(np.degrees(np.abs(values)).mean()),
            "fraction_within_30_deg_of_prediction": float((np.abs(values) <= np.pi / 6).mean()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10,
                         "pdf.fonttype":42, "svg.fonttype":"none"}):
        fig = plt.figure(figsize=(13.2, 10.0), facecolor="white")
        grid = fig.add_gridspec(2, 2, left=.09, right=.94, bottom=.15, top=.86,
                               hspace=.47, wspace=.36, height_ratios=[1.14, .9])
        ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        path_panel(ax_a, source_xy, BLUE, "A   IID source", theta_example, extent,
                   f"Chunk direction: {np.degrees(h_iid):.1f}°\nDeviation from prediction: {abs(np.degrees(rel_iid[draw, obs])):.1f}°")
        path_panel(ax_b, prior_xy, ORANGE, "B   Heading-conditioned XY source", theta_example, extent,
                   f"Chunk direction: {np.degrees(h_prior):.1f}°\nDeviation from prediction: {abs(np.degrees(rel_prior[draw, obs])):.1f}°")
        ax_c, ax_d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
        style_cartesian(ax_c)
        bins = np.linspace(-180, 180, 25)
        centers = (bins[:-1] + bins[1:]) / 2
        hist_iid = np.histogram(np.degrees(rel_iid).ravel(), bins=bins)[0] / rel_iid.size * 100
        hist_prior = np.histogram(np.degrees(rel_prior).ravel(), bins=bins)[0] / rel_prior.size * 100
        ax_c.bar(centers, hist_prior, width=14.5, color=ORANGE, alpha=.48, label="Heading prior")
        ax_c.stairs(hist_iid, bins, color=BLUE, linewidth=2, label="IID")
        ax_c.axvline(0, color=GREEN, linewidth=1.3, linestyle="--")
        ax_c.axvspan(-30, 30, color=GREEN, alpha=.055)
        ax_c.set_xlim(-180, 180)
        ax_c.set_xticks([-180, -90, 0, 90, 180])
        ax_c.set_ylim(0, max(hist_prior.max(), hist_iid.max()) * 1.5)
        ax_c.set_xlabel("Chunk direction − predicted heading (degrees)", fontsize=9, color=DARK)
        ax_c.set_ylabel("Samples per 15° bin (%)", fontsize=9, color=DARK)
        ax_c.set_title("C   Distribution across all saved source draws", loc="left", fontsize=11,
                       color=DARK, fontweight="bold", pad=12)
        ax_c.text(.025, .96,
                  f"Within ±30°\nIID: {stats['iid']['fraction_within_30_deg_of_prediction']*100:.1f}%\nPrior: {stats['prior']['fraction_within_30_deg_of_prediction']*100:.1f}%",
                  transform=ax_c.transAxes, va="top", fontsize=9, color=DARK,
                  bbox={"facecolor":"white", "edgecolor":"none", "alpha":.9})
        ax_c.text(.97, .96, f"{windows} observations × {draws} draws\n0° = each observation's prediction",
                  transform=ax_c.transAxes, ha="right", va="top", fontsize=8.5, color=DARK)
        style_cartesian(ax_d)
        steps = np.arange(1, horizon + 1)
        magnitude_before = np.linalg.norm(source_xy, axis=-1)
        magnitude_after = np.linalg.norm(prior_xy, axis=-1)
        ax_d.plot(steps, magnitude_before, color=BLUE, marker="o", markersize=5,
                  linewidth=1.4, label="IID")
        ax_d.plot(steps, magnitude_after, color=ORANGE, marker="x", markersize=6,
                  linestyle="none", markeredgewidth=1.6, label="Heading prior")
        ax_d.set_xlim(.5, horizon + .5)
        ax_d.set_ylim(0, max(magnitude_before.max(), 1.) * 1.3)
        ax_d.set_xticks([1, 4, 8, 12, 16])
        ax_d.set_xlabel("Noise step in the same example chunk", fontsize=9, color=DARK)
        ax_d.set_ylabel("XY noise-vector length", fontsize=9, color=DARK)
        ax_d.set_title("D   Each step keeps its XY magnitude", loc="left", fontsize=11,
                       color=DARK, fontweight="bold", pad=12)
        ax_d.text(.025, .97, f"All 16 steps rotate by the same {np.degrees(phi):+.1f}°\nOther five action dimensions stay unchanged",
                  transform=ax_d.transAxes, va="top", fontsize=9, color=DARK)
        ax_d.legend(loc="upper right", bbox_to_anchor=(1., .8), frameon=False, fontsize=9)
        fig.suptitle("IID noise vs heading-conditioned noise", fontsize=21, fontweight="bold", color=DARK, y=.974)
        fig.text(.5, .927, "Actual paired samples from the completed experiment · XY-only prior · κ = 4",
                 ha="center", fontsize=11, color=DARK)
        fig.text(.5, .897,
                 f"Fixed example: seed {args.seed}, observation {obs}, draw {draw} · predicted heading {np.degrees(theta_example):.1f}° · angular jitter {np.degrees(prior['angular_jitter'][draw, obs]):+.1f}°",
                 ha="center", fontsize=10, color=DARK)
        handles = [Line2D([0], [0], color=BLUE, lw=2, label="IID"),
                   Line2D([0], [0], color=ORANGE, lw=2, label="Heading prior"),
                   Line2D([0], [0], color=GREEN, lw=1.8, linestyle="--", label="Predicted heading (not ground truth)")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .070), ncol=3,
                   frameon=False, fontsize=10)
        fig.text(.5, .023,
                 "Top paths are cumulative sums of 16 NOISE vectors, not robot trajectories. Bold arrows show their resultants.\nThese are initial flow sources x₀, not generated actions; direction deviation is relative to the prediction, not a task-success metric.",
                 ha="center", fontsize=9, color="#566574", linespacing=1.5)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"iid_vs_heading_prior.{ext}", dpi=190, facecolor="white")
        plt.close(fig)
    metadata = {
        "source_run": str(args.run_dir.resolve()), "model_seed": args.seed,
        "inputs_sha256": {str(path):sha256(path) for path in paths.values()},
        "generator_sha256":sha256(__file__),
        "example_selection":"fixed default window=0, draw=0; no selection by error or angle",
        "example":{"window":obs,"draw":draw,"episode":int(prior['episode'][obs]),
                   "frame":int(prior['frame'][obs]),"task":int(prior['task'][obs]),
                   "predicted_heading_deg":float(np.degrees(theta_example)),
                   "iid_heading_deg":float(np.degrees(h_iid)),"prior_heading_deg":float(np.degrees(h_prior)),
                   "rotation_deg":float(np.degrees(phi)),"jitter_deg":float(np.degrees(prior['angular_jitter'][draw,obs]))},
        "population":{"windows":windows,"draws_per_window":draws,"source_samples":int(rel_iid.size),
                      "angular_reference":"each observation's prediction from the same prior model; not ground truth",
                      "statistics":stats},
        "prediction_reconstruction":"circular mean across draws of source resultant heading minus recorded angular jitter; all source gates active",
        "coordinate_system":"normalized action XY; actual fitted scale equal on XY and zero offset; prior_noise_scale=1",
        "checks":checks,
        "limits":["only initial source noise is shown", "cumulative noise sums do not represent executed robot motion", "population observations repeat over four source draws", "illustrative example is not the typical angular deviation"],
    }
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    source_copy=args.output_dir / "source" / Path(__file__).name
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    source_copy.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"output_dir":str(args.output_dir),"example":metadata['example'],
                      "statistics":stats,"checks":checks},indent=2))


if __name__ == "__main__":
    main()
