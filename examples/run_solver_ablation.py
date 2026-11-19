#!/usr/bin/env python3
"""Compare weight solvers on archived plants using one Lyapunov backend."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.linalg import solve_discrete_lyapunov

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from steady_state_combined.ellipsoidal import (  # noqa: E402
    FixedGainProblem,
    simplex_grid,
    spectral_radius,
    stepwise_trace_steady_state,
    sym,
)
from steady_state_combined.research import adjoint_fixed_trace, global_fixed_trace, solve_dare  # noqa: E402
from steady_state_combined.riccati import GainOptimizedProblem  # noqa: E402


def evaluate(task: tuple[dict, dict, int]) -> dict:
    case, arrays, resolution = task
    index = int(case["case"])
    problem = GainOptimizedProblem(*(arrays[f"case_{index}_{key}"] for key in ("A", "H", "Q", "R")))
    K = arrays.get(f"case_{index}_fixed_gain")
    if K is None:
        _, K = solve_dare(problem, np.full(3, 1.0 / 3.0))
    fixed = FixedGainProblem(problem.A, problem.H, K, problem.Q, problem.R)
    baseline = stepwise_trace_steady_state(fixed)
    if baseline is None:
        raise RuntimeError(f"baseline did not converge for case {index}")
    np.testing.assert_allclose(np.trace(baseline[1]), float(case["fixed_greedy_trace"]), rtol=1e-8)
    reference = global_fixed_trace(fixed)
    adjoint = adjoint_fixed_trace(fixed, baseline[0])
    rho2 = spectral_radius(fixed.F) ** 2
    started = perf_counter()
    grid_best = None
    evaluations = 0
    for alpha in simplex_grid(resolution):
        if alpha[0] <= rho2 + 1e-9:
            continue
        P = sym(solve_discrete_lyapunov(fixed.F / np.sqrt(alpha[0]), fixed.S_w / alpha[1] + fixed.S_v / alpha[2]))
        value = float(np.trace(P))
        evaluations += 1
        if grid_best is None or value < grid_best[1]:
            grid_best = (alpha, value)
    if grid_best is None:
        raise RuntimeError(f"no feasible grid point for case {index}")
    grid_seconds = perf_counter() - started
    refined = adjoint_fixed_trace(fixed, grid_best[0])
    best_value = min(refined.value, adjoint.value)
    relative = (best_value - reference.value) / reference.value
    if abs(relative) > 1e-7:
        raise RuntimeError(f"ablation accuracy failure for case {index}: {relative:g}")
    return {
        "case": index,
        "group": case["group"],
        "dimension": problem.n,
        "global_trace": reference.value,
        "adjoint_trace": adjoint.value,
        "grid_trace": grid_best[1],
        "grid_refined_trace": best_value,
        "adjoint_global_relative_gap": (adjoint.value - reference.value) / reference.value,
        "grid_global_relative_gap": (grid_best[1] - reference.value) / reference.value,
        "grid_refined_global_relative_gap": relative,
        "scalar_seconds": reference.seconds,
        "adjoint_seconds": adjoint.seconds,
        "grid_seconds": grid_seconds,
        "grid_refined_seconds": grid_seconds + refined.seconds + adjoint.seconds,
        "scalar_evaluations": reference.evaluations,
        "adjoint_evaluations": adjoint.evaluations,
        "grid_evaluations": evaluations,
        "grid_refined_evaluations": evaluations + refined.evaluations + adjoint.evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--grid", type=int, default=41)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.validation_dir)
    with (root / "validation_cases.csv").open(newline="") as stream:
        cases = list(csv.DictReader(stream))
    if any(case["status"] != "ok" for case in cases):
        raise RuntimeError("validation dataset contains failed cases")
    with np.load(root / "systems.npz") as archive:
        tasks = []
        for case in cases:
            prefix = f"case_{case['case']}_"
            arrays = {key: archive[key] for key in archive.files if key.startswith(prefix)}
            tasks.append((case, arrays, args.grid))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = []
        for row in executor.map(evaluate, tasks, chunksize=1):
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['group']}", flush=True)
    path = root / "solver_ablation.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for group in dict.fromkeys(row["group"] for row in rows):
        selected = [row for row in rows if row["group"] == group]
        summary.append(
            {
                "group": group,
                "cases": len(selected),
                "scalar_seconds_median": float(np.median([row["scalar_seconds"] for row in selected])),
                "adjoint_seconds_median": float(np.median([row["adjoint_seconds"] for row in selected])),
                "grid_refined_seconds_median": float(np.median([row["grid_refined_seconds"] for row in selected])),
                "adjoint_global_gap_max": max(row["adjoint_global_relative_gap"] for row in selected),
                "grid_refined_global_gap_max": max(row["grid_refined_global_relative_gap"] for row in selected),
            }
        )
    (root / "solver_ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "protocol": "fixed-gain solver ablation; shared stepwise initialization excluded from timings",
        "arguments": vars(args),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "inputs_sha256": {
            "validation_cases.csv": hashlib.sha256((root / "validation_cases.csv").read_bytes()).hexdigest(),
            "systems.npz": hashlib.sha256((root / "systems.npz").read_bytes()).hexdigest(),
            "run_solver_ablation.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "research.py": hashlib.sha256(
                (REPO_ROOT / "src/steady_state_combined/research.py").read_bytes()
            ).hexdigest(),
        },
        "outputs_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("solver_ablation.csv", "solver_ablation_summary.json")
        },
    }
    (root / "solver_ablation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
