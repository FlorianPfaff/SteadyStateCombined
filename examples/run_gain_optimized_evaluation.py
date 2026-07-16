#!/usr/bin/env python3
"""Run gain-reoptimized Riccati-sweep experiments.

This script answers the practical follow-up question to the fixed-gain theorem:
does the greedy-vs-steady-state gap survive when the gain is reoptimized for
fixed approximation weights?

Suggested command:

    python examples/run_gain_optimized_evaluation.py --out results_riccati --random-systems 100 --grid 121 --step-grid 61 --seed 13
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from steady_state_combined.examples import random_fixed_gain_problem  # noqa: E402
from steady_state_combined.riccati import (  # noqa: E402
    GainOptimizedProblem,
    deterministic_gain_optimized_problem,
    optimize_riccati_weights,
    stepwise_gain_trace_recursion,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(path: Path, rows: list[dict]) -> None:
    if plt is None or not rows:
        return
    x = np.array([row["trace_stepwise_gain"] for row in rows], dtype=float)
    y = np.array([row["trace_steady_state_gain"] for row in rows], dtype=float)
    if len(x) == 0:
        return
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.scatter(x, y, s=8, alpha=0.55, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="0.25", linestyle="--", linewidth=0.8)
    ax.set_xlabel("greedy $\\operatorname{tr}(P)$", fontsize=8)
    ax.set_ylabel("steady-state-optimized $\\operatorname{tr}(P)$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_improvement_cdf(path: Path, rows: list[dict]) -> None:
    if plt is None or not rows:
        return
    reduction = np.sort(
        100.0
        * (
            1.0
            - np.array([row["trace_steady_state_gain"] for row in rows], dtype=float)
            / np.array([row["trace_stepwise_gain"] for row in rows], dtype=float)
        )
    )
    cumulative = np.arange(1, len(reduction) + 1, dtype=float) / len(reduction)
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.plot(reduction, cumulative, linewidth=1.1)
    ax.axvline(float(np.median(reduction)), color="0.35", linestyle="--", linewidth=0.8)
    ax.set_xlabel("trace reduction (%)", fontsize=8)
    ax.set_ylabel("empirical CDF", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def to_gain_problem(fixed_problem) -> GainOptimizedProblem:
    return GainOptimizedProblem(
        A=fixed_problem.A,
        H=fixed_problem.H,
        Q=fixed_problem.Q,
        R=fixed_problem.R,
        name=fixed_problem.name.replace("fixed", "gain_optimized"),
    )


def evaluate_random_problem(task: tuple[object, int, int, str]) -> dict | None:
    """Evaluate one random problem; kept top-level for process pickling."""

    fixed, step_resolution, steady_state_resolution, criterion = task
    random_problem = to_gain_problem(fixed)
    random_greedy = stepwise_gain_trace_recursion(
        random_problem,
        resolution=step_resolution,
        criterion=criterion,
    )
    if random_greedy is None:
        return None
    random_ss = optimize_riccati_weights(
        random_problem,
        resolution=steady_state_resolution,
        criterion=criterion,
        initial_alphas=[random_greedy.alpha],
    )
    if random_ss is None:
        return None

    trace_ss = float(np.trace(random_ss.P))
    trace_greedy = float(np.trace(random_greedy.P))
    logdet_ss = float(np.linalg.slogdet(random_ss.P)[1])
    logdet_greedy = float(np.linalg.slogdet(random_greedy.P)[1])
    return {
        "trace_stepwise_gain": trace_greedy,
        "trace_steady_state_gain": trace_ss,
        "ratio_trace": trace_greedy / trace_ss,
        "logdet_stepwise_gain": logdet_greedy,
        "logdet_steady_state_gain": logdet_ss,
        "ratio_det_radius": math.exp((logdet_greedy - logdet_ss) / random_problem.n),
        "alpha_step_0": random_greedy.alpha[0],
        "alpha_step_w": random_greedy.alpha[1],
        "alpha_step_v": random_greedy.alpha[2],
        "alpha_ss_0": random_ss.alpha[0],
        "alpha_ss_w": random_ss.alpha[1],
        "alpha_ss_v": random_ss.alpha[2],
    }


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    problem = deterministic_gain_optimized_problem()
    greedy = stepwise_gain_trace_recursion(
        problem,
        resolution=args.step_grid,
        criterion=args.criterion,
    )
    ss = optimize_riccati_weights(
        problem,
        resolution=args.grid,
        criterion=args.criterion,
        initial_alphas=[] if greedy is None else [greedy.alpha],
    )
    if ss is None or greedy is None:
        raise RuntimeError("deterministic gain-optimized example failed")

    deterministic_rows = [
        {
            "method": "stepwise_gain_reoptimized",
            "alpha0": greedy.alpha[0],
            "alphaw": greedy.alpha[1],
            "alphav": greedy.alpha[2],
            "trace": float(np.trace(greedy.P)),
            "logdet": float(np.linalg.slogdet(greedy.P)[1]),
            "iterations": greedy.iterations,
        },
        {
            "method": "steady_state_riccati_grid",
            "alpha0": ss.alpha[0],
            "alphaw": ss.alpha[1],
            "alphav": ss.alpha[2],
            "trace": float(np.trace(ss.P)),
            "logdet": float(np.linalg.slogdet(ss.P)[1]),
            "iterations": ss.iterations,
        },
    ]
    write_csv(out / "riccati_deterministic_summary.csv", deterministic_rows)

    random_rows: list[dict] = []
    attempts = 0
    max_attempts = 30 * args.random_systems
    step_resolution = max(31, args.step_grid // 2)
    steady_state_resolution = max(41, args.grid // 2)
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        while len(random_rows) < args.random_systems and attempts < max_attempts:
            batch_size = min(
                args.random_systems - len(random_rows),
                4 * args.workers,
                max_attempts - attempts,
            )
            fixed_batch = []
            for _ in range(batch_size):
                attempts += 1
                fixed = random_fixed_gain_problem(rng)
                if fixed is not None:
                    fixed_batch.append(fixed)

            tasks = [
                (fixed, step_resolution, steady_state_resolution, args.criterion)
                for fixed in fixed_batch
            ]
            evaluated = map(evaluate_random_problem, tasks) if executor is None else executor.map(
                evaluate_random_problem,
                tasks,
                chunksize=1,
            )
            for row in evaluated:
                if row is not None:
                    random_rows.append({"idx": len(random_rows), **row})
            print(f"Completed random systems: {len(random_rows)}/{args.random_systems}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    if len(random_rows) != args.random_systems:
        raise RuntimeError(f"generated {len(random_rows)} of {args.random_systems} requested random systems")

    write_csv(out / "riccati_random_benchmark.csv", random_rows)
    plot_scatter(out / "riccati_random_scatter.png", random_rows)
    plot_improvement_cdf(out / "riccati_improvement_cdf.png", random_rows)

    if random_rows:
        ratios = np.array([row["ratio_trace"] for row in random_rows], dtype=float)
        summary = {
            "random_systems": len(random_rows),
            "median_ratio_trace": float(np.median(ratios)),
            "mean_ratio_trace": float(np.mean(ratios)),
            "p25_ratio_trace": float(np.percentile(ratios, 25)),
            "p75_ratio_trace": float(np.percentile(ratios, 75)),
            "max_ratio_trace": float(np.max(ratios)),
            "min_ratio_trace": float(np.min(ratios)),
            "fraction_improved": float(np.mean(ratios > 1.0 + 1e-8)),
            "median_trace_reduction_percent": float(np.median(100.0 * (1.0 - 1.0 / ratios))),
        }
        write_csv(out / "riccati_random_summary.csv", [summary])

    print(f"Deterministic greedy trace:       {np.trace(greedy.P):.8g}, alpha={greedy.alpha}")
    print(f"Deterministic Riccati-sweep trace:{np.trace(ss.P):.8g}, alpha={ss.alpha}")
    print(f"Improvement ratio:                {np.trace(greedy.P) / np.trace(ss.P):.6g}")
    if random_rows:
        print(f"Random systems:                   {len(random_rows)}")
        print(f"Median trace ratio:               {summary['median_ratio_trace']:.6g}")
        print(f"Fraction improved:                {summary['fraction_improved']:.3f}")
    print(f"Wrote results to:                 {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_riccati", help="Output directory")
    parser.add_argument("--grid", type=int, default=121, help="Simplex grid for steady-state Riccati sweep")
    parser.add_argument("--step-grid", type=int, default=61, help="Simplex grid for each greedy step")
    parser.add_argument("--random-systems", type=int, default=100, help="Number of random systems")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--criterion", choices=["trace", "logdet"], default="trace")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for independent random systems")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


if __name__ == "__main__":
    run(parse_args())
