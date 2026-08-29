#!/usr/bin/env python3
"""
The one table PLAN_fewstep_coupling.md S5 asks for: arms x sampler steps, plus mechanism.

    rows     {B10, B2, R1_reflow, P3, (P2)}
    columns  SR@N1 / N2 / N4 / N10, each with per-task rates behind it
    plus     straightness / few_step_gap_N1 as the mechanism check, and the paired ΔSR
             against a chosen reference row AT EACH N

Reads whatever ``eval_orbit_dno_libero.py`` and ``report_coupling_diagnostics.py`` already
wrote -- no re-running, no re-evaluating.  A run is located by its own JSON, not by a naming
convention: ``dno_eval_log.json`` records ``num_inference_steps``, so the N a row lands in is
read out of the file rather than parsed out of a directory name.

WHY THE PAIRED COLUMN IS THE ONE THAT MATTERS
---------------------------------------------
LIBERO-10's aggregate is a mean over 10 tasks and every arm is evaluated on the same 10, so
arm-vs-arm differences are paired samples and their uncertainty is ``std(delta)/sqrt(10)``
-- roughly 5x the repeat stderr the eval log reports.  ``paired_task_compare.py`` is the
authority on that computation; this script imports it rather than reimplementing it, so the
two can never disagree.

Usage:
    python scripts/report_fewstep_table.py -r output/exp -o output/exp/FEWSTEP.md
    python scripts/report_fewstep_table.py -r output/exp --ref P1_curve/seed42 \
        --label P1_curve/seed42=B10 --label R1_reflow/seed42=R1_reflow
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
sys.path.append(str(pathlib.Path(__file__).parent))

from paired_task_compare import (  # noqa: E402
    SUFFIX, _binom_sign_p, _t_crit, _t_sf,
)

STEPS = (1, 2, 4, 10)
DIAG_KEYS = (("straightness", "straight", 3), ("few_step_gap_N1", "gapN1", 3),
             ("few_step_gap_N4", "gapN4", 3), ("action_mse_N10", "MSE@10", 5))


def per_task(d: Dict) -> Dict[str, float]:
    return {k[: -len(SUFFIX)]: float(v) for k, v in d.items() if k.endswith(SUFFIX)}


def paired(a: Dict[str, float], b: Dict[str, float]) -> Optional[Dict[str, float]]:
    """b - a, paired over the tasks both evaluated.  None if they share none."""
    shared = sorted(set(a) & set(b))
    n = len(shared)
    if n < 2:
        return None
    deltas = [b[t] - a[t] for t in shared]
    mean_d = sum(deltas) / n
    var = sum((x - mean_d) ** 2 for x in deltas) / (n - 1)
    sd = var ** 0.5
    se = sd / (n ** 0.5)
    t = mean_d / se if se > 0 else float("nan")
    nz = [x for x in deltas if x != 0.0]
    return {
        "n_tasks": n, "mean_delta": mean_d, "stderr": se,
        "t": t, "p_ttest": _t_sf(abs(t), n - 1) if t == t else float("nan"),
        "p_sign": _binom_sign_p(sum(1 for x in nz if x > 0), len(nz)),
        "mde_p05": _t_crit(n - 1) * se,
    }


def collect(root: pathlib.Path) -> Tuple[Dict, Dict]:
    """{row: {N: entry}} for success rates, {row: diag} for the offline metrics."""
    rows: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    diags: Dict[str, Dict] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rel = path.relative_to(root)
        if path.name in ("dno_eval_log.json", "eval_log.json"):
            if "mean_success_rate_mean" not in data:
                continue
            if data.get("use_dno"):
                continue          # this table is the no-DNO few-step sweep
            n = int(data.get("num_inference_steps", -1))
            row = str(rel.parent.parent)                       # <arm>/<seed>
            entry = {
                "path": str(path), "sr": float(data["mean_success_rate_mean"]),
                "per_task": per_task(data), "n_test": data.get("n_test"),
                "checkpoint": data.get("checkpoint"), "variant": rel.parent.name,
            }
            prev = rows[row].get(n)
            if prev is not None and prev["path"] != entry["path"]:
                # more episodes = less read noise, so that one wins; ties keep the first
                keep, drop = ((entry, prev) if (entry.get("n_test") or 0) > (prev.get("n_test") or 0)
                              else (prev, entry))
                print(f"!! two no-DNO evals at N={n} for row {row}: keeping "
                      f"{keep['variant']} (n_test={keep.get('n_test')}), ignoring "
                      f"{drop['variant']} (n_test={drop.get('n_test')})")
                rows[row][n] = keep
                continue
            rows[row][n] = entry
        elif path.name.startswith("diag") and "straightness" in data:
            diags[str(rel.parent)] = data
    return rows, diags


def fmt_sr(e: Optional[Dict]) -> str:
    if e is None:
        return "--"
    n = e.get("n_test")
    # stderr of a mean over n_test Bernoulli rollouts; the paired column below is the one
    # to quote for arm-vs-arm, this is only the within-row read noise
    if isinstance(n, int) and n > 0:
        se = (e["sr"] * (1 - e["sr"]) / n) ** 0.5
        return f"{e['sr']:.3f} ±{se:.3f}"
    return f"{e['sr']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", default="output/exp")
    ap.add_argument("-o", "--output", default="output/exp/FEWSTEP.md")
    ap.add_argument("--ref", default=None,
                    help="row key used as the paired reference, e.g. P1_curve/seed42")
    ap.add_argument("--label", action="append", default=[],
                    help="row=Label, repeatable; renames a row in the table")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    labels = dict(l.split("=", 1) for l in args.label)
    rows, diags = collect(root)
    if not rows:
        raise SystemExit(f"no eval logs with a success rate under {root}")

    ref = args.ref
    if ref is not None and ref not in rows:
        raise SystemExit(f"--ref {ref!r} not among the discovered rows: {sorted(rows)}")

    out: List[str] = [
        "# Few-step coupling -- success rate vs sampler steps", "",
        f"root: `{root}`" + (f"  ·  paired reference: `{ref}`" if ref else ""), "",
        "SR cells are `mean ±rollout-stderr` (within-row read noise). Arm-vs-arm claims must",
        "use the paired block below it, whose stderr carries the task-to-task variance.", "",
    ]

    head = "| arm | " + " | ".join(f"SR@N{n}" for n in STEPS) + " | " + \
           " | ".join(k[1] for k in DIAG_KEYS) + " |"
    out += [head, "|" + "---|" * (1 + len(STEPS) + len(DIAG_KEYS))]
    for row in sorted(rows):
        cells = [fmt_sr(rows[row].get(n)) for n in STEPS]
        d = diags.get(row, {})
        dcells = [f"{d[k]:.{p}f}" if k in d else "--" for k, _, p in DIAG_KEYS]
        out.append(f"| `{labels.get(row, row)}` | " + " | ".join(cells + dcells) + " |")
    out.append("")

    if ref:
        out += ["### Paired ΔSR vs `" + labels.get(ref, ref) + "`, at each N", "",
                "| arm | N | ΔSR | paired stderr | t | p (t) | p (sign) | MDE p<0.05 |",
                "|---|---|---|---|---|---|---|---|"]
        for row in sorted(rows):
            if row == ref:
                continue
            for n in STEPS:
                a, b = rows[ref].get(n), rows[row].get(n)
                if a is None or b is None:
                    continue
                st = paired(a["per_task"], b["per_task"])
                if st is None:
                    continue
                out.append(
                    f"| `{labels.get(row, row)}` | {n} | {st['mean_delta']:+.3f} | "
                    f"±{st['stderr']:.3f} | {st['t']:+.2f} | {st['p_ttest']:.3f} | "
                    f"{st['p_sign']:.3f} | {st['mde_p05']:.3f} |"
                )
        out.append("")

        # The plan's two pre-registered success criteria, evaluated rather than eyeballed.
        ref10 = rows[ref].get(10)
        out += ["### Pre-registered criteria (S5)", ""]
        if ref10 is None:
            out.append("- reference has no N=10 eval, so the "
                       "'recovers full-sampler quality at 2 steps' test cannot be read.")
        else:
            for row in sorted(rows):
                if row == ref:
                    continue
                b2 = rows[row].get(2)
                if b2 is None:
                    continue
                gap = b2["sr"] - (ref10["sr"] - 0.065)
                st = paired(rows[ref][2]["per_task"], b2["per_task"]) if 2 in rows[ref] else None
                d = st["mean_delta"] if st else float("nan")
                verdict = ("STRONG" if (gap >= 0 and d >= 0.15)
                           else "USEFUL" if d >= 0.065 else "NULL")
                out.append(
                    f"- `{labels.get(row, row)}`: SR@N2 {b2['sr']:.3f} vs "
                    f"SR@N10(ref) - 0.065 = {ref10['sr'] - 0.065:.3f}; paired ΔSR@N2 = "
                    f"{d:+.3f} → **{verdict}**"
                )
        out += ["", "`STRONG` = claimable at one seed. `USEFUL` = suggestive, add 2 seeds "
                "before claiming. `NULL` = below the 1-seed MDE.", ""]

    text = "\n".join(out)
    p = pathlib.Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
