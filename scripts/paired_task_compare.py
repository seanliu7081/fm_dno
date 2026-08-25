#!/usr/bin/env python3
"""
Paired per-task comparison of two eval_log.json files.

WHY THIS EXISTS
---------------
`eval_policy_sim.py` reports `mean_success_rate_stderr`, which is the standard error over
the `-n` repeats of the *same* checkpoint. That number measures rollout stochasticity at
fixed weights. It does **not** carry the task-to-task variance, which on LIBERO-10
dominates: Phase 2 measured an aggregate delta of +0.0153 alongside a mean per-task delta
of 0.158, a reshuffle 10x larger than the effect being quoted.

Quoting the repeat stderr as the uncertainty on an arm-vs-arm difference is therefore
wrong by roughly 5x, and it will manufacture significant-looking differences that a second
seed absorbs.

The right object is the **paired** comparison: LIBERO-10's aggregate is a mean over 10
tasks, both arms are evaluated on the same 10 tasks, so the per-task differences are paired
samples and the uncertainty on their mean is `std(delta) / sqrt(n_tasks)`.

Reports:
  paired mean delta      the aggregate difference (identical to the difference of means)
  paired stderr          std(delta)/sqrt(n) -- THE resolvable effect size
  paired t / p           two-sided one-sample t-test on the deltas, n-1 dof
  sign test p            distribution-free companion, robust to one task dominating
  mean |delta|           the reshuffle magnitude
  reshuffle ratio        mean|delta| / |mean delta| -- how much the aggregate hides
  MDE                    minimum detectable effect at this n, and projected for 3 seeds

Usage:
    python scripts/paired_task_compare.py \
        -a output/exp/P0_baseline/seed42/eval/eval_log.json \
        -b output/exp/P1_norm_only/seed42/eval/eval_log.json \
        --name-a P0 --name-b P1 -o output/exp/paired_P0_vs_P1.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, List, Tuple

SUFFIX = "/mean_success_rate_mean"


def load_per_task(path: str) -> Tuple[Dict[str, float], float]:
    """Return ({task: SR}, aggregate SR) from an eval_log.json."""
    d = json.load(open(path))
    per = {k[: -len(SUFFIX)]: float(v) for k, v in d.items() if k.endswith(SUFFIX)}
    if not per:
        raise SystemExit(f"{path}: no per-task '*{SUFFIX}' keys found")
    agg = float(d.get("mean_success_rate_mean", sum(per.values()) / len(per)))
    return per, agg


def _t_sf(t: float, df: int) -> float:
    """Two-sided survival function of Student's t, via the regularized incomplete beta."""
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b) by continued fraction (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x >= (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def _t_crit(df: int, alpha: float = 0.05) -> float:
    """Two-sided critical t by bisection on the survival function."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _binom_sign_p(n_pos: int, n: int) -> float:
    """Two-sided exact sign test p-value (ties excluded before calling)."""
    if n == 0:
        return float("nan")
    k = min(n_pos, n - n_pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--arm-a", required=True, help="eval_log.json for arm A (reference)")
    ap.add_argument("-b", "--arm-b", required=True, help="eval_log.json for arm B")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="seeds behind each arm; used only to project the MDE")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    pa, agg_a = load_per_task(args.arm_a)
    pb, agg_b = load_per_task(args.arm_b)

    shared = sorted(set(pa) & set(pb))
    if not shared:
        raise SystemExit("the two eval logs share no task names")
    missing = (set(pa) ^ set(pb))
    if missing:
        print(f"!! {len(missing)} task(s) present in only one arm, excluded: {sorted(missing)}\n")

    deltas: List[float] = [pb[t] - pa[t] for t in shared]
    n = len(deltas)
    mean_d = sum(deltas) / n
    var = sum((d - mean_d) ** 2 for d in deltas) / (n - 1) if n > 1 else float("nan")
    sd = math.sqrt(var) if var == var else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = mean_d / se if se and se == se and se > 0 else float("nan")
    df = n - 1
    p_t = _t_sf(abs(t), df) if t == t else float("nan")

    nz = [d for d in deltas if d != 0.0]
    p_sign = _binom_sign_p(sum(1 for d in nz if d > 0), len(nz))

    mean_abs = sum(abs(d) for d in deltas) / n
    ratio = mean_abs / abs(mean_d) if mean_d != 0 else float("inf")

    tc = _t_crit(df)
    mde_1 = tc * se
    mde_k = tc * se / math.sqrt(max(args.n_seeds + 2, 1)) if args.n_seeds == 1 else tc * se

    A, B = args.name_a, args.name_b
    w = max(len(t_) for t_ in shared)
    print(f"{'task':<{w}}  {A:>8}  {B:>8}  {'delta':>8}")
    print("-" * (w + 30))
    for t_, d in sorted(zip(shared, deltas), key=lambda z: z[1]):
        print(f"{t_:<{w}}  {pa[t_]:8.3f}  {pb[t_]:8.3f}  {d:+8.3f}")

    print(f"\n{'aggregate ' + A:<34}{agg_a:.4f}")
    print(f"{'aggregate ' + B:<34}{agg_b:.4f}")
    print(f"{'paired mean delta':<34}{mean_d:+.4f}")
    print(f"{'paired sd over tasks':<34}{sd:.4f}")
    print(f"{'paired stderr  <-- RESOLUTION':<34}±{se:.4f}")
    print(f"{'paired t (' + str(df) + ' dof)':<34}{t:+.2f}")
    print(f"{'two-sided p (t-test)':<34}{p_t:.3f}")
    print(f"{'two-sided p (sign test)':<34}{p_sign:.3f}")
    print(f"{'mean |per-task delta|':<34}{mean_abs:.4f}")
    print(f"{'reshuffle ratio':<34}{ratio:.1f}x")
    print(f"{'MDE at p<0.05, this n':<34}{mde_1:.4f}")
    if args.n_seeds == 1:
        print(f"{'MDE projected at 3 seeds':<34}{mde_k:.4f}")

    sig = (p_t < 0.05) if p_t == p_t else False
    print("\nVERDICT: difference is "
          + ("SIGNIFICANT" if sig else "NOT significant")
          + f" when paired across {n} tasks.")
    if not sig:
        print(f"  The honest statement is: |delta| < {mde_1:.3f} at this resolution;")
        print(f"  '{B} differs from {A}' is not supported by this data.")
    if ratio > 3:
        print(f"  Per-task reshuffle is {ratio:.1f}x the aggregate: the two arms behave")
        print(f"  differently even though the means nearly agree. Report per-task rates.")

    out = {
        "arm_a": {"name": A, "path": args.arm_a, "aggregate": agg_a, "per_task": pa},
        "arm_b": {"name": B, "path": args.arm_b, "aggregate": agg_b, "per_task": pb},
        "n_tasks": n,
        "paired_mean_delta": mean_d,
        "paired_sd": sd,
        "paired_stderr": se,
        "paired_t": t,
        "p_ttest_two_sided": p_t,
        "p_sign_two_sided": p_sign,
        "mean_abs_delta": mean_abs,
        "reshuffle_ratio": ratio,
        "mde_p05_this_n": mde_1,
        "significant_p05": bool(sig),
        "per_task_delta": dict(zip(shared, deltas)),
    }
    if args.output:
        p = pathlib.Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
