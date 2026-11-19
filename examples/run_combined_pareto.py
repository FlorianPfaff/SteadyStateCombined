#!/usr/bin/env python3
"""Run a deterministic combined stochastic/set-membership Pareto grid search.

Suggested command:

    python examples/run_combined_pareto.py --out results_combined --alpha-grid 31 --gain-grid 25

The script performs an explicit 2D grid search over gains and approximation
weights, then selects the best candidate for each scalarization value after
normalizing both traces by their independently attainable minima. This is meant
to generate a paper figure and check whether the combined problem has a
meaningful Pareto trade-off.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from steady_state_combined.combined_pareto import (  # noqa: E402
    deterministic_combined_pareto_problem,
    enumerate_combined_candidates_2d,
    nondominated_candidates,
    pareto_from_candidates,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


def parse_lambdas(spec: str) -> list[float]:
    if spec == "default":
        return [i / 10 for i in range(11)]
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_pareto(path: Path, rows: list[dict], frontier_rows: list[dict]) -> None:
    if plt is None or not rows or not frontier_rows:
        return
    trace_sigma = np.array([row["normalized_trace_sigma"] for row in rows], dtype=float)
    trace_P = np.array([row["normalized_trace_P"] for row in rows], dtype=float)
    lambdas = np.array([row["lambda_stochastic"] for row in rows], dtype=float)
    frontier_sigma = np.array([row["normalized_trace_sigma"] for row in frontier_rows], dtype=float)
    frontier_P = np.array([row["normalized_trace_P"] for row in frontier_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.plot(frontier_sigma, frontier_P, color="0.55", linewidth=0.8, label="grid frontier")
    ax.scatter(trace_sigma, trace_P, s=14, zorder=3, label="scalarized designs")
    for x, y, lam in zip(trace_sigma, trace_P, lambdas):
        if lam in (0.0, 0.5, 0.9, 1.0):
            offset = (3, -9) if lam == 1.0 else (3, 3)
            ax.annotate(f"{lam:.1f}", (x, y), textcoords="offset points", xytext=offset, fontsize=6)
    ax.set_yscale("log")
    ax.set_xlabel("normalized $\\operatorname{tr}(\\Sigma)$", fontsize=8)
    ax.set_ylabel("normalized $\\operatorname{tr}(P)$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problem = deterministic_combined_pareto_problem()
    lambdas = parse_lambdas(args.lambdas)

    candidates = enumerate_combined_candidates_2d(
        problem,
        alpha_resolution=args.alpha_grid,
        gain_resolution=args.gain_grid,
        k1_range=(args.k1_min, args.k1_max),
        k2_range=(args.k2_min, args.k2_max),
    )
    frontier = nondominated_candidates(candidates)
    results = pareto_from_candidates(candidates, lambdas)
    if not results:
        raise RuntimeError("no feasible combined candidates found")

    sigma_scale = min(candidate.trace_sigma for candidate in candidates)
    P_scale = min(candidate.trace_P for candidate in candidates)
    rows = []
    for result in results:
        c = result.candidate
        rows.append(
            {
                "lambda_stochastic": result.lambda_stochastic,
                "objective": result.objective,
                "trace_sigma": c.trace_sigma,
                "trace_P": c.trace_P,
                "normalized_trace_sigma": c.trace_sigma / sigma_scale,
                "normalized_trace_P": c.trace_P / P_scale,
                "alpha0": c.alpha[0],
                "alphaw": c.alpha[1],
                "alphav": c.alpha[2],
                "K0": c.K[0, 0],
                "K1": c.K[1, 0],
            }
        )
    write_csv(out / "combined_pareto.csv", rows)
    frontier_rows = [
        {
            "trace_sigma": candidate.trace_sigma,
            "trace_P": candidate.trace_P,
            "normalized_trace_sigma": candidate.trace_sigma / sigma_scale,
            "normalized_trace_P": candidate.trace_P / P_scale,
            "alpha0": candidate.alpha[0],
            "alphaw": candidate.alpha[1],
            "alphav": candidate.alpha[2],
            "K0": candidate.K[0, 0],
            "K1": candidate.K[1, 0],
        }
        for candidate in frontier
    ]
    write_csv(out / "combined_frontier.csv", frontier_rows)
    plot_pareto(out / "combined_pareto.png", rows, frontier_rows)

    summary = [
        {
            "candidate_count": len(candidates),
            "nondominated_count": len(frontier),
            "lambda_count": len(lambdas),
            "alpha_grid": args.alpha_grid,
            "gain_grid": args.gain_grid,
            "min_trace_sigma": min(c.trace_sigma for c in candidates),
            "max_trace_sigma": max(c.trace_sigma for c in candidates),
            "min_trace_P": min(c.trace_P for c in candidates),
            "max_trace_P": max(c.trace_P for c in candidates),
            "trace_sigma_scale": sigma_scale,
            "trace_P_scale": P_scale,
        }
    ]
    write_csv(out / "combined_candidate_summary.csv", summary)

    print(f"Feasible candidates: {len(candidates)}")
    print(f"Wrote Pareto rows:   {len(rows)}")
    print(f"trace(Sigma) range:  {summary[0]['min_trace_sigma']:.8g} .. {summary[0]['max_trace_sigma']:.8g}")
    print(f"trace(P) range:      {summary[0]['min_trace_P']:.8g} .. {summary[0]['max_trace_P']:.8g}")
    print(f"Wrote results to:    {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_combined")
    parser.add_argument("--alpha-grid", type=int, default=31)
    parser.add_argument("--gain-grid", type=int, default=25)
    parser.add_argument("--lambdas", default="default", help="default or comma-separated values in [0,1]")
    parser.add_argument("--k1-min", type=float, default=-0.2)
    parser.add_argument("--k1-max", type=float, default=1.6)
    parser.add_argument("--k2-min", type=float, default=-1.0)
    parser.add_argument("--k2-max", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
