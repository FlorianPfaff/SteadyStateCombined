#!/usr/bin/env python3
"""Run fixed-gain experiments for the steady-state set-membership paper.

Suggested command:

    python examples/run_fixed_gain_evaluation.py --out results --random-systems 200 --grid 121 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

# Allow running from a source checkout without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from steady_state_combined import (  # noqa: E402
    bounded_error_check,
    criterion_value,
    grid_optimize_weights,
    line_search_to_nonmyopic,
    nonmyopic_weights,
    stepwise_trace_steady_state,
)
from steady_state_combined.examples import deterministic_fixed_gain_problem, random_fixed_gain_problem  # noqa: E402

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_line_search(path: Path, rows: list[dict]) -> None:
    if plt is None or not rows:
        return
    eps = np.array([row["epsilon"] for row in rows], dtype=float)
    values = np.array([row["value"] for row in rows], dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.semilogx(eps[finite], values[finite], marker="o")
    ax.set_xlabel("line-search $\\epsilon$")
    ax.set_ylabel("steady-state trace")
    ax.set_title("Adjoint-weighted direction from stepwise weights")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_random_scatter(path: Path, rows: list[dict]) -> None:
    if plt is None or not rows:
        return
    x = np.array([row["trace_step"] for row in rows], dtype=float)
    y = np.array([row["trace_ss"] for row in rows], dtype=float)
    if len(x) == 0:
        return
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(x, y, s=16, alpha=0.75)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("trace, stepwise steady state")
    ax.set_ylabel("trace, optimized steady state")
    ax.set_title("Random fixed-gain systems")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    problem = deterministic_fixed_gain_problem()
    step = stepwise_trace_steady_state(problem)
    opt = grid_optimize_weights(problem, resolution=args.grid, criterion=args.criterion)
    if step is None or opt is None:
        raise RuntimeError("deterministic example failed")
    alpha_step, P_step = step
    alpha_opt, P_opt, _ = opt
    alpha_nm, _, _, contributions = nonmyopic_weights(problem, alpha_step, P_step, criterion=args.criterion)

    deterministic_rows = [
        {
            "method": "stepwise_trace_fixed_gain",
            "alpha0": alpha_step[0],
            "alphaw": alpha_step[1],
            "alphav": alpha_step[2],
            "trace": criterion_value(P_step, "trace"),
            "logdet": criterion_value(P_step, "logdet"),
        },
        {
            "method": "grid_optimized_fixed_gain",
            "alpha0": alpha_opt[0],
            "alphaw": alpha_opt[1],
            "alphav": alpha_opt[2],
            "trace": criterion_value(P_opt, "trace"),
            "logdet": criterion_value(P_opt, "logdet"),
        },
        {
            "method": "adjoint_weight_at_stepwise",
            "alpha0": alpha_nm[0],
            "alphaw": alpha_nm[1],
            "alphav": alpha_nm[2],
            "trace": float("nan"),
            "logdet": float("nan"),
        },
    ]
    write_csv(out / "deterministic_summary.csv", deterministic_rows)

    line_rows = line_search_to_nonmyopic(problem, alpha_step, criterion=args.criterion)
    write_csv(out / "line_search_curve.csv", line_rows)
    plot_line_search(out / "line_search_curve.png", line_rows)

    check = bounded_error_check(
        problem,
        P_opt,
        rng,
        trajectories=args.bound_trajectories,
        horizon=args.bound_horizon,
    )
    write_csv(out / "bound_check.csv", [check])

    random_rows: list[dict] = []
    attempts = 0
    while len(random_rows) < args.random_systems and attempts < 25 * args.random_systems:
        attempts += 1
        random_problem = random_fixed_gain_problem(rng)
        if random_problem is None:
            continue
        random_step = stepwise_trace_steady_state(random_problem)
        random_opt = grid_optimize_weights(random_problem, resolution=max(41, args.grid // 2), criterion=args.criterion)
        if random_step is None or random_opt is None:
            continue
        alpha_s, P_s = random_step
        alpha_o, P_o, _ = random_opt
        trace_step = criterion_value(P_s, "trace")
        trace_ss = criterion_value(P_o, "trace")
        logdet_step = criterion_value(P_s, "logdet")
        logdet_ss = criterion_value(P_o, "logdet")
        random_rows.append(
            {
                "idx": len(random_rows),
                "trace_step": trace_step,
                "trace_ss": trace_ss,
                "ratio_trace": trace_step / trace_ss,
                "logdet_step": logdet_step,
                "logdet_ss": logdet_ss,
                "ratio_det_radius": math.exp((logdet_step - logdet_ss) / random_problem.n),
                "alpha_step_0": alpha_s[0],
                "alpha_step_w": alpha_s[1],
                "alpha_step_v": alpha_s[2],
                "alpha_ss_0": alpha_o[0],
                "alpha_ss_w": alpha_o[1],
                "alpha_ss_v": alpha_o[2],
            }
        )

    write_csv(out / "random_benchmark.csv", random_rows)
    plot_random_scatter(out / "random_scatter.png", random_rows)

    if random_rows:
        ratios = np.array([row["ratio_trace"] for row in random_rows], dtype=float)
        summary = {
            "random_systems": len(random_rows),
            "median_ratio_trace": float(np.median(ratios)),
            "mean_ratio_trace": float(np.mean(ratios)),
            "p25_ratio_trace": float(np.percentile(ratios, 25)),
            "p75_ratio_trace": float(np.percentile(ratios, 75)),
            "max_ratio_trace": float(np.max(ratios)),
            "fraction_improved": float(np.mean(ratios > 1.0 + 1e-8)),
        }
        write_csv(out / "random_summary.csv", [summary])

    print(f"Deterministic stepwise trace:  {criterion_value(P_step, 'trace'):.8g}, alpha={alpha_step}")
    print(f"Deterministic optimized trace: {criterion_value(P_opt, 'trace'):.8g}, alpha={alpha_opt}")
    print(f"Improvement ratio:             {criterion_value(P_step, 'trace') / criterion_value(P_opt, 'trace'):.6g}")
    print(f"Adjoint contributions:         {contributions}")
    print(f"Bound check:                   {check}")
    if random_rows:
        print(f"Random systems:                {len(random_rows)}")
        print(f"Median trace ratio:            {summary['median_ratio_trace']:.6g}")
        print(f"Fraction improved:             {summary['fraction_improved']:.3f}")
    print(f"Wrote results to:              {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--grid", type=int, default=121, help="Simplex grid resolution")
    parser.add_argument("--random-systems", type=int, default=200, help="Number of random systems")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--criterion", choices=["trace", "logdet"], default="trace")
    parser.add_argument("--bound-trajectories", type=int, default=200)
    parser.add_argument("--bound-horizon", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
