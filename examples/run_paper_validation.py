#!/usr/bin/env python3
"""Matched-accuracy baseline audit and dimension/method validation.

Run with the research extra and BLAS threads set to one when using workers.
Failures are recorded and cause a nonzero exit; no successful cases replace
failed draws. Every generated plant is saved in systems.npz for exact replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from steady_state_combined.ellipsoidal import FixedGainProblem, stepwise_trace_steady_state  # noqa: E402
from steady_state_combined.examples import deterministic_fixed_gain_problem, random_fixed_gain_problem  # noqa: E402
from steady_state_combined.research import (  # noqa: E402
    adjoint_fixed_trace,
    continuous_greedy_trace,
    continuous_steady_trace,
    global_fixed_trace,
    solve_dare,
)
from steady_state_combined.riccati import GainOptimizedProblem  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def to_gain_problem(fixed: FixedGainProblem) -> GainOptimizedProblem:
    return GainOptimizedProblem(fixed.A, fixed.H, fixed.Q, fixed.R, fixed.name)


def random_dimension_problem(rng: np.random.Generator, n: int) -> GainOptimizedProblem:
    basis = np.linalg.qr(rng.normal(size=(n, n)))[0]
    dynamics = np.diag(rng.uniform(0.65, 0.98, n))
    dynamics += np.triu(rng.normal(scale=0.08, size=(n, n)), 1)
    A = basis @ dynamics @ basis.T
    H = np.linalg.qr(rng.normal(size=(n, n)))[0][: n // 2]
    noise_basis = np.linalg.qr(rng.normal(size=(n, n)))[0]
    Q = noise_basis @ np.diag(10.0 ** rng.uniform(-4, -1, n)) @ noise_basis.T
    R = np.diag(10.0 ** rng.uniform(-4, -1, n // 2))
    return GainOptimizedProblem(A, H, Q, R, f"random_{n}d")


def tracking_problem() -> GainOptimizedProblem:
    """Six normalized position/velocity states with bounded acceleration."""
    dt = 0.1
    A = np.block([[np.eye(3), np.eye(3)], [np.zeros((3, 3)), 0.98 * np.eye(3)]])
    H = np.hstack((np.eye(3), np.zeros((3, 3))))
    acceleration = dt**2 * np.vstack((0.5 * np.eye(3), np.eye(3)))
    Q = acceleration @ np.diag([0.2**2, 0.3**2, 0.5**2]) @ acceleration.T + 1e-8 * np.eye(6)
    R = np.diag([0.05**2, 0.10**2, 0.20**2])
    return GainOptimizedProblem(A, H, Q, R, "tracking_6d")


def evaluate_case(task: tuple[int, str, GainOptimizedProblem, np.ndarray | None, dict | None]) -> dict:
    index, group, problem, fixed_gain, historical = task
    try:
        reference_old = float("nan")
        historical_start = []
        if historical is not None:
            old_alpha = np.array([float(historical[f"alpha_step_{key}"]) for key in ("0", "w", "v")])
            old_P, _ = solve_dare(problem, old_alpha)
            reference_old = float(historical["trace_stepwise_gain"])
            if not np.isclose(np.trace(old_P), reference_old, rtol=2e-7, atol=1e-10):
                raise RuntimeError("historical plant replay failed")
            historical_start = [np.array([float(historical[f"alpha_ss_{key}"]) for key in ("0", "w", "v")])]

        greedy_initial = continuous_greedy_trace(problem)
        greedy = continuous_greedy_trace(problem, initial_P=greedy_initial.P, grid_resolution=21, top_grid_starts=8)
        steady = continuous_steady_trace(
            problem, [greedy.alpha, *historical_start], grid_resolution=21, top_grid_starts=8
        )
        greedy_seconds = greedy_initial.seconds + greedy.seconds
        greedy_evaluations = greedy_initial.evaluations + greedy.evaluations
        steady_seconds = steady.seconds
        steady_evaluations = steady.evaluations
        boundary = bool(min(greedy.alpha.min(), steady.alpha.min()) <= 1e-7)
        floor_sensitivity = 0.0
        if boundary:
            lower_greedy = continuous_greedy_trace(
                problem, initial_P=greedy.P, grid_resolution=21, top_grid_starts=8, alpha_floor=1e-10
            )
            lower_steady = continuous_steady_trace(
                problem,
                [steady.alpha, lower_greedy.alpha, *historical_start],
                grid_resolution=21,
                top_grid_starts=8,
                alpha_floor=1e-10,
            )
            floor_sensitivity = max(
                abs(lower_greedy.value / greedy.value - 1.0), abs(lower_steady.value / steady.value - 1.0)
            )
            if floor_sensitivity > 2e-6:
                raise RuntimeError(f"weight-floor sensitivity is too large: {floor_sensitivity:g}")
            greedy_seconds += lower_greedy.seconds
            greedy_evaluations += lower_greedy.evaluations
            steady_seconds += lower_steady.seconds
            steady_evaluations += lower_steady.evaluations
            greedy, steady = lower_greedy, lower_steady
        if max(greedy.stationarity, steady.stationarity) > 2e-5:
            raise RuntimeError(f"continuous KKT residual too large: {greedy.stationarity:g}, {steady.stationarity:g}")
        if steady.value > greedy.value * (1.0 + 1e-8):
            raise RuntimeError("steady optimizer did not retain the baseline")

        if fixed_gain is None:
            _, fixed_gain = solve_dare(problem, np.full(3, 1.0 / 3.0))
        fixed = FixedGainProblem(problem.A, problem.H, fixed_gain, problem.Q, problem.R, problem.name)
        baseline = stepwise_trace_steady_state(fixed)
        if baseline is None:
            raise RuntimeError("fixed-gain baseline did not converge")
        global_trace = global_fixed_trace(fixed)
        adjoint = adjoint_fixed_trace(fixed, baseline[0])
        gap = (adjoint.value - global_trace.value) / global_trace.value
        if gap < -1e-7:
            raise RuntimeError("adjoint value is below independently solved global reference")

        row = {
            "case": index,
            "group": group,
            "dimension": problem.n,
            "status": "ok",
            "error": "",
            "historical_greedy_trace": reference_old,
            "continuous_greedy_trace": greedy.value,
            "continuous_steady_trace": steady.value,
            "gain_trace_reduction_percent": 100.0 * (1.0 - steady.value / greedy.value),
            "baseline_change_percent": 100.0 * (1.0 - greedy.value / reference_old),
            "greedy_kkt_relative": greedy.stationarity,
            "greedy_one_step_convex_gap_relative": greedy.simplex_first_order_gap,
            "steady_kkt_relative": steady.stationarity,
            "boundary_case": boundary,
            "weight_floor_relative_sensitivity": floor_sensitivity,
            "greedy_seconds": greedy_seconds,
            "steady_seconds": steady_seconds,
            "greedy_evaluations": greedy_evaluations,
            "steady_evaluations": steady_evaluations,
            "fixed_greedy_trace": float(np.trace(baseline[1])),
            "fixed_global_trace": global_trace.value,
            "fixed_adjoint_trace": adjoint.value,
            "fixed_trace_reduction_percent": 100.0 * (1.0 - global_trace.value / np.trace(baseline[1])),
            "adjoint_global_relative_gap": gap,
            "adjoint_kkt_relative": adjoint.stationarity,
            "scalar_kkt_relative": global_trace.stationarity,
            "adjoint_seconds": adjoint.seconds,
            "scalar_seconds": global_trace.seconds,
            "adjoint_evaluations": adjoint.evaluations,
            "scalar_evaluations": global_trace.evaluations,
        }
        for prefix, result in (("greedy", greedy), ("steady", steady), ("scalar", global_trace), ("adjoint", adjoint)):
            for key, value in zip(("0", "w", "v"), result.alpha, strict=True):
                row[f"{prefix}_alpha_{key}"] = float(value)
        return row
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return {
            "case": index,
            "group": group,
            "dimension": problem.n,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for group in dict.fromkeys(row["group"] for row in rows):
        selected = [row for row in rows if row["group"] == group]
        successes = [row for row in selected if row["status"] == "ok"]
        if not successes:
            continue
        gain = np.array([row["gain_trace_reduction_percent"] for row in successes])
        fixed = np.array([row["fixed_trace_reduction_percent"] for row in successes])
        summaries.append(
            {
                "group": group,
                "requested": len(selected),
                "successful": len(successes),
                "failed": len(selected) - len(successes),
                "boundary_cases": sum(row["boundary_case"] for row in successes),
                "weight_floor_sensitivity_max": max(row["weight_floor_relative_sensitivity"] for row in successes),
                "gain_reduction_median_percent": float(np.median(gain)),
                "gain_reduction_p25_percent": float(np.percentile(gain, 25)),
                "gain_reduction_p75_percent": float(np.percentile(gain, 75)),
                "gain_reduction_min_percent": float(gain.min()),
                "gain_reduction_max_percent": float(gain.max()),
                "gain_fraction_at_least_5_percent": float(np.mean(gain >= 5.0)),
                "fixed_reduction_median_percent": float(np.median(fixed)),
                "adjoint_global_gap_max": max(row["adjoint_global_relative_gap"] for row in successes),
                "greedy_kkt_max": max(row["greedy_kkt_relative"] for row in successes),
                "greedy_one_step_convex_gap_max": max(row["greedy_one_step_convex_gap_relative"] for row in successes),
                "steady_kkt_max": max(row["steady_kkt_relative"] for row in successes),
                "adjoint_kkt_max": max(row["adjoint_kkt_relative"] for row in successes),
                "scalar_seconds_median": float(np.median([row["scalar_seconds"] for row in successes])),
                "adjoint_seconds_median": float(np.median([row["adjoint_seconds"] for row in successes])),
                "greedy_seconds_median": float(np.median([row["greedy_seconds"] for row in successes])),
                "steady_seconds_median": float(np.median([row["steady_seconds"] for row in successes])),
            }
        )
    return summaries


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    historical_rows = []
    if args.historical_csv:
        with Path(args.historical_csv).open(newline="") as stream:
            historical_rows = list(csv.DictReader(stream))
        if len(historical_rows) < args.random_systems:
            raise ValueError("historical CSV does not contain enough systems")
    tasks = []
    deterministic = deterministic_fixed_gain_problem()
    tasks.append((0, "deterministic", to_gain_problem(deterministic), deterministic.K, None))
    replay_rows = []
    draw = 0
    for index in range(args.random_systems):
        historical = historical_rows[index] if historical_rows else None
        for _ in range(200):
            problem = random_fixed_gain_problem(rng)
            draw += 1
            if problem is None:
                continue
            if historical is not None:
                old_alpha = np.array([float(historical[f"alpha_step_{key}"]) for key in ("0", "w", "v")])
                old_P, _ = solve_dare(to_gain_problem(problem), old_alpha)
                if not np.isclose(np.trace(old_P), float(historical["trace_stepwise_gain"]), rtol=2e-7, atol=1e-10):
                    replay_rows.append({"draw": draw - 1, "historical_index": "", "selected": False})
                    continue
            replay_rows.append({"draw": draw - 1, "historical_index": index, "selected": True})
            tasks.append((len(tasks), "original_2d", to_gain_problem(problem), problem.K, historical))
            break
        else:
            raise RuntimeError(f"could not recover historical plant {index}")
    write_csv(out / "historical_replay.csv", replay_rows)
    for dimension in args.dimensions:
        dimension_rng = np.random.default_rng(args.dimension_seed + dimension)
        for _ in range(args.per_dimension):
            problem = random_dimension_problem(dimension_rng, dimension)
            tasks.append((len(tasks), f"random_{dimension}d", problem, None, None))
    tasks.append((len(tasks), "tracking_6d", tracking_problem(), None, None))

    arrays = {}
    for index, _, problem, gain, _ in tasks:
        for field in ("A", "H", "Q", "R"):
            arrays[f"case_{index}_{field}"] = getattr(problem, field)
        if gain is not None:
            arrays[f"case_{index}_fixed_gain"] = gain
    np.savez_compressed(out / "systems.npz", **arrays)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(evaluate_case, tasks, chunksize=1):
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['group']} {row['status']} {row['error']}", flush=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    rows = [{key: row.get(key, "") for key in fieldnames} for row in rows]
    write_csv(out / "validation_cases.csv", rows)
    summaries = summarize(rows)
    write_csv(out / "validation_summary.csv", summaries)
    manifest = {
        "protocol": "continuous matched-accuracy trace validation; failed plants are retained",
        "arguments": vars(args),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "inputs_sha256": {},
        "outputs_sha256": {},
    }
    sources = [
        Path(__file__),
        REPO_ROOT / "src/steady_state_combined/research.py",
        REPO_ROOT / "src/steady_state_combined/examples.py",
    ]
    if args.historical_csv:
        sources.append(Path(args.historical_csv))
    for path in sources:
        manifest["inputs_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in out.iterdir():
        if path.name != "validation_manifest.json" and path.is_file():
            manifest["outputs_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    failed = sum(row["status"] != "ok" for row in rows)
    if failed:
        raise RuntimeError(f"{failed} of {len(rows)} cases failed; inspect validation_cases.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results_validation")
    parser.add_argument("--random-systems", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--historical-csv")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--per-dimension", type=int, default=50)
    parser.add_argument("--dimension-seed", type=int, default=20261002)
    parser.add_argument("--workers", type=int, default=1)
    run(parser.parse_args())
