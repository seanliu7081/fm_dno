#!/usr/bin/env python3
"""
Render stage of the steering probe: overlay N arms' (phi, dtheta) clouds as a VECTOR figure.

The figure carries one claim: an uncoupled policy's output heading is unrelated to the source
rotation, so its cloud fills the box; an orbit-coupled policy's tracks it one-for-one, so its
cloud traces the identity line. Everything quoted in the legend and caption is read from the
measure-stage JSONs, never from memory.

Refuses to render arms whose readout specs differ — under different specs the two arms'
dtheta are different quantities and the overlay would be meaningless.

Usage:
    python scripts/steering_probe_render.py \
        -i output/exp/probe/probe_P1.json -i output/exp/probe/probe_P4.json \
        -o output/exp/probe/steering_probe.svg
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# quiet control first, bright coupled after: the reader's eye should land on the coupled cloud
PALETTE = ["#9aa4b2", "#d1495b", "#2a9d8f", "#e9c46a", "#6a4c93"]
PI = math.pi


def pi_ticks(lo, hi, step=0.5):
    """Ticks at multiples of pi/2 with proper labels."""
    vals, labs = [], []
    k = lo / PI
    while k <= hi / PI + 1e-9:
        vals.append(k * PI)
        if abs(k) < 1e-9:
            labs.append("0")
        elif abs(k - 1) < 1e-9:
            labs.append(r"$\pi$")
        elif abs(k + 1) < 1e-9:
            labs.append(r"$-\pi$")
        elif abs(k - round(k)) < 1e-9:
            labs.append(rf"${int(round(k))}\pi$")
        else:
            num = round(k * 2)
            labs.append(rf"$-\frac{{{abs(num)}\pi}}{{2}}$" if num < 0 else rf"$\frac{{{num}\pi}}{{2}}$")
        k += step
    return vals, labs


@click.command()
@click.option("-i", "--inputs", multiple=True, required=True, help="measure-stage JSONs, repeatable")
@click.option("-o", "--output", default="output/exp/probe/steering_probe.svg")
@click.option("--title", default="Orbit steering probe: does the source phase steer the chunk heading?")
@click.option("--point-size", default=7.0)
@click.option("--alpha", default=0.55)
@click.option("--low-conf-alpha", default=0.12)
@click.option("--also-png", is_flag=True, help="additionally write a PNG preview")
def main(inputs, output, title, point_size, alpha, low_conf_alpha, also_png):
    arms = [json.loads(pathlib.Path(p).read_text()) for p in inputs]

    specs = {a["readout_spec"] for a in arms}
    if len(specs) > 1:
        print("REFUSING TO RENDER: arms were measured under different readout specs.", file=sys.stderr)
        for a in arms:
            print(f"  {a['label']}: {a['readout_spec']}", file=sys.stderr)
        print("Under different specs the dtheta are different quantities and the overlay is\n"
              "meaningless. Re-measure every arm with the same --readout-spec.", file=sys.stderr)
        raise SystemExit(2)

    epochs = [a.get("epoch") for a in arms]
    epochs_match = len({e for e in epochs if e is not None}) <= 1

    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    handles = []
    for i, a in enumerate(arms):
        c = PALETTE[i % len(PALETTE)]
        phi = np.asarray(a["phi"])
        dth = np.asarray(a["dtheta"])
        conf = np.asarray(a["confidence"])
        low = conf < a["conf_threshold"]
        # the control should recede; later (coupled) arms carry the claim
        aa = alpha * (0.55 if i == 0 and len(arms) > 1 else 1.0)
        # faded, not dropped -- how many there are is part of the result
        ax.scatter(phi[low], dth[low], s=point_size, c=c, alpha=low_conf_alpha,
                   linewidths=0, zorder=3 + i)
        ax.scatter(phi[~low], dth[~low], s=point_size, c=c, alpha=aa,
                   linewidths=0, zorder=3 + i)
        handles.append(Line2D([], [], marker="o", ls="", color=c, markersize=6,
                              label=(f"{a['label']}   gain {a['gain']:+.2f},  "
                                     f"consistency {a['phase_consistency']:.2f}")))

    # ideal reference: dtheta = wrap(phi). TWO segments -- it jumps at pi because dtheta is
    # wrapped to (-pi, pi], not because the policy does anything discontinuous.
    # Drawn ON TOP of the clouds so it stays legible where the coupled arm is densest.
    ax.plot([0, PI], [0, PI], ls="--", lw=1.4, color="#111111", zorder=20)
    ax.plot([PI, 2 * PI], [-PI, 0], ls="--", lw=1.4, color="#111111", zorder=20)
    handles.append(Line2D([], [], ls="--", color="#111111", lw=1.4,
                          label=r"ideal  $\Delta\theta=\mathrm{wrap}(\varphi)$"))

    xv, xl = pi_ticks(0, 2 * PI)
    yv, yl = pi_ticks(-PI, PI)
    ax.set_xticks(xv); ax.set_xticklabels(xl)
    ax.set_yticks(yv); ax.set_yticklabels(yl)
    ax.set_xlim(0, 2 * PI); ax.set_ylim(-PI, PI)
    ax.set_xlabel(r"source rotation  $\varphi$")
    ax.set_ylabel(r"change in generated chunk heading  $\Delta\theta$")
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # legend OUTSIDE the axes: inside it covers the identity line, which is the whole point
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=8.5, frameon=False, ncol=1 if len(handles) > 3 else 2,
              handletextpad=0.5, labelspacing=0.4)

    fig.tight_layout()
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")     # vector: cloud stays sharp
    if also_png:
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- caption, written from the JSONs ------------------------------------------------
    a0 = arms[0]
    ep = ("all at epoch %d" % epochs[0] if epochs_match and epochs[0] is not None
          else "epochs " + ", ".join(f"{a['label']}={a.get('epoch')}" for a in arms)
               + "  (NOT epoch-matched)")
    lines = [
        f"Orbit steering probe on LIBERO-10 checkpoints ({ep}).",
        "",
        f"For a frozen policy F, a fixed batch of {a0['n_obs']} validation observations and a "
        f"fixed source noise z, the source is rotated by rho(phi) over {a0['n_angles']} angles "
        f"spanning (0, 2pi) and the change in the generated chunk's heading is recorded: "
        f"dtheta(phi) = wrap(heading(F(rho(phi)z | o)) - heading(F(z | o))). Each arm "
        f"contributes {a0['n_points']} points.",
        "",
        "Arms:",
    ]
    for a in arms:
        mode = a["coupling"].get("mode", "?") if isinstance(a["coupling"], dict) else str(a["coupling"])
        align = a["coupling"].get("align") if isinstance(a["coupling"], dict) else None
        lines.append(
            f"  - {a['label']}: coupling mode={mode}"
            + (f", align={align}" if align and mode in ("rot", "perm_rot") else "")
            + f", prior={a['inference_prior']}, {a['num_inference_steps']}-step sampler; "
              f"gain {a['gain']:+.3f}, phase consistency {a['phase_consistency']:.3f}; "
              f"low-confidence points {a['low_confidence_frac']:.1%}"
            + (f" (gain among the rest {a['gain_high_conf']:+.3f})"
               if a["low_confidence_frac"] > 0.01 and a["gain_high_conf"] == a["gain_high_conf"] else "")
        )
    lines += [
        "",
        f"Readout spec forced identically across arms: {a0['readout_spec_name']} "
        f"({a0['readout_spec']}). Each checkpoint's own trained action_spec is deliberately "
        f"NOT used: chunk_heading weights the vector blocks, so arms trained before and after "
        f"the G4' decision would otherwise report different quantities. rotate_chunk acts on "
        f"vector_blocks and rotates every block regardless of the weights, so forcing the "
        f"readout changes what is measured, never what is done to the noise.",
        "",
        f"Identical across arms: observations (obs_seed={a0['obs_seed']}), source noise "
        f"(z_seed={a0['z_seed']}), readout spec. The checkpoint is the only difference"
        + ("." if len({a["inference_prior"] for a in arms}) == 1
           else ", except where inference_prior is varied as a labelled arm."),
        "",
        "The dashed reference breaks at phi = pi because dtheta is wrapped to (-pi, pi]; it is "
        "not a discontinuity in the policy. Faded points are those whose chunk-heading "
        "confidence falls below "
        f"{a0['conf_threshold']} -- a chunk whose planar deltas cancel over the horizon has no "
        "meaningful heading. They are shown rather than dropped because how many there are is "
        "part of the result.",
        "",
        "Measured on validation observations only: no rollouts, no environment. The probe is "
        "therefore independent of task success and of the operating point.",
        "",
        "This figure shows that the steering channel exists. It does NOT show that using it "
        "improves task success (that is Phase 6's on-benchmark 2x2), it says nothing about the "
        "transport claim (straightness / few_step_gap are separate metrics), and it is "
        "noise-space equivariance only -- the observation is held fixed while the noise "
        "rotates, so it is not equivariance in the textbook sense.",
    ]
    cap = out.with_suffix(".caption.md")
    cap.write_text("\n".join(lines) + "\n")

    print(f"wrote {out}")
    if also_png:
        print(f"wrote {out.with_suffix('.png')}")
    print(f"wrote {cap}")
    print()
    for a in arms:
        print(f"  {a['label']:<22} gain {a['gain']:+.5f}  consistency {a['phase_consistency']:.5f}"
              f"  low-conf {a['low_confidence_frac']:.2%}  epoch {a.get('epoch')}")


if __name__ == "__main__":
    main()
