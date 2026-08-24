#!/usr/bin/env python3
"""
Collect every eval_log.json / dno_eval_log.json / diagnostics json under a directory into
one markdown table, aggregated over seeds.

Expects the layout EXPERIMENT_PLAN.md prescribes:

    output/exp/<ARM>/seed<N>/eval/eval_log.json          from eval_policy_sim.py
    output/exp/<ARM>/seed<N>/diag.json                   from report_coupling_diagnostics.py
    output/exp/<ARM>/seed<N>/dno_<VARIANT>/dno_eval_log.json

Usage:
    python scripts/aggregate_experiment_results.py -r output/exp -o output/exp/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
from collections import defaultdict

MAIN = [
    ("mean_success_rate_mean", "SR", 3, True),
    ("orbit_steering_gain", "steerGain", 3, False),
    ("src_orbit_eq_rel", "srcEqErr", 3, False),
    ("heading_MAE", "headMAE", 3, False),
    ("action_mse_N10", "MSE@10", 5, False),
    ("few_step_gap_N1", "gapN1", 3, False),
    ("few_step_gap_N4", "gapN4", 3, False),
    ("straightness", "straight", 2, False),
]
DNO = [
    ("mean_success_rate_mean", "SR", 3),
    ("dno/wall_time_p50_mean", "t50(s)", 3),
    ("dno/wall_time_p95_mean", "t95(s)", 3),
    ("dno/orbit_loss_spread_mean", "orbitSpread", 3),
    ("dno/final_loss_mean", "taskL", 3),
    ("dno/z_shift_mean", "|dz|", 2),
]


def cell(vals, prec):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "--"
    m = st.mean(vals)
    if len(vals) < 2:
        return f"{m:.{prec}f}"
    return f"{m:.{prec}f} ± {st.stdev(vals):.{prec}f}"


def table(rows, cols, title, key_header="arm"):
    out = [f"### {title}", ""]
    out.append("| " + key_header + " | n | " + " | ".join(c[1] for c in cols) + " |")
    out.append("|" + "---|" * (len(cols) + 2))
    for arm in sorted(rows):
        seeds = rows[arm]
        cs = [cell([s.get(c[0]) for s in seeds], c[2]) for c in cols]
        out.append(f"| `{arm}` | {len(seeds)} | " + " | ".join(cs) + " |")
    out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", default="output/exp")
    ap.add_argument("-o", "--output", default="output/exp/RESULTS.md")
    args = ap.parse_args()
    root = pathlib.Path(args.root)

    main_rows, dno_rows = defaultdict(list), defaultdict(list)
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rel = path.relative_to(root).parts
        arm = rel[0] if rel else "?"
        if path.name == "dno_eval_log.json":
            variant = path.parent.name
            dno_rows[f"{arm}/{variant}"].append(data)
        elif path.name == "eval_log.json":
            main_rows[arm].append(data)
        elif path.name.startswith("diag"):
            # merge diagnostics into whichever eval row shares this seed directory
            main_rows[arm].append(data)

    # merge per-seed dicts that came from different files under the same arm+seed
    merged = defaultdict(list)
    for arm, entries in main_rows.items():
        by_seed = defaultdict(dict)
        for e in entries:
            by_seed[str(e.get("checkpoint", len(by_seed)))].update(e)
        merged[arm] = list(by_seed.values())

    lines = ["# Orbit coupling + DNO -- LIBERO-10 results", ""]
    lines += table(merged, MAIN, "Training arms (LIBERO-10 success rate + offline diagnostics)")
    lines += table(dno_rows, DNO, "Test-time orbit DNO", key_header="arm / variant")
    lines += [
        "### Reading guide",
        "",
        "- `steerGain` near 0 on a coupled arm means the coupling did not take; stage-1 DNO",
        "  will behave as best-of-K resampling and should be reported as such.",
        "- `headMAE` / `MSE@10` are expected to get **worse** for rotation-applying couplings",
        "  (heading delegation). Judge those arms on the DNO table, not this one.",
        "- `gapN1` / `straight` are the few-step payoff; they should improve for any coupling",
        "  that shortens transport, symmetry-aware or not.",
        "- The symmetry claim lives in `P2/P4/P6 vs P3`, not in `P2/P4/P6 vs P1`.",
        "",
    ]
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
