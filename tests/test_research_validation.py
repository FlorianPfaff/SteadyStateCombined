from __future__ import annotations

import csv
import hashlib
import json
import os
import runpy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

from steady_state_combined.ellipsoidal import solve_fixed_gain_steady_state, stepwise_trace_steady_state
from steady_state_combined.examples import deterministic_fixed_gain_problem
from steady_state_combined.research import (
    adjoint_fixed_trace,
    continuous_greedy_trace,
    continuous_steady_trace,
    dare_trace_value_gradient,
    fixed_trace_value_gradient,
    global_fixed_trace,
    one_step_trace_value_gradient,
    solve_dare,
)
from steady_state_combined.riccati import (
    GainOptimizedProblem,
    deterministic_gain_optimized_problem,
    solve_fixed_alpha_riccati,
)


def test_scipy_dare_matches_independent_iteration() -> None:
    problem = deterministic_gain_optimized_problem()
    alpha = np.array([0.7, 0.2, 0.1])
    P, K = solve_dare(problem, alpha)
    reference = solve_fixed_alpha_riccati(problem, alpha)
    assert reference is not None
    np.testing.assert_allclose(P, reference.P, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(K, reference.K, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("steady", [False, True])
def test_gain_envelope_gradient_matches_simplex_finite_difference(steady: bool) -> None:
    problem = deterministic_gain_optimized_problem()
    alpha = np.array([0.7, 0.2, 0.1])
    evaluate = (
        (lambda a: dare_trace_value_gradient(problem, a))
        if steady
        else (lambda a: one_step_trace_value_gradient(problem, np.eye(problem.n), a))
    )
    gradient = evaluate(alpha)[3]
    for direction in (np.array([1.0, -1.0, 0.0]), np.array([0.0, 1.0, -1.0])):
        h = 1e-6
        difference = (evaluate(alpha + h * direction)[2] - evaluate(alpha - h * direction)[2]) / (2 * h)
        np.testing.assert_allclose(difference, gradient @ direction, rtol=1e-6, atol=1e-7)


def test_gain_eliminated_one_step_trace_is_convex_in_weights() -> None:
    problem = deterministic_gain_optimized_problem()
    rng = np.random.default_rng(28)
    for _ in range(25):
        a, b = rng.dirichlet(np.ones(3), size=2)
        values = [one_step_trace_value_gradient(problem, np.eye(problem.n), alpha)[2] for alpha in (a, b)]
        middle = one_step_trace_value_gradient(problem, np.eye(problem.n), (a + b) / 2)[2]
        assert middle <= np.mean(values) + 1e-10


def test_global_trace_reference_has_kkt_solution_and_matches_adjoint() -> None:
    problem = deterministic_fixed_gain_problem()
    reference = global_fixed_trace(problem)
    baseline = stepwise_trace_steady_state(problem)
    assert baseline is not None
    adjoint = adjoint_fixed_trace(problem, baseline[0])
    assert reference.stationarity < 1e-8
    np.testing.assert_allclose(adjoint.value, reference.value, rtol=1e-8)
    assert reference.value < np.trace(baseline[1]) * 0.7
    np.testing.assert_allclose(reference.P, solve_fixed_gain_steady_state(problem, reference.alpha), rtol=1e-10)


def test_fixed_trace_convexity_inequality_on_feasible_segments() -> None:
    problem = deterministic_fixed_gain_problem()
    rng = np.random.default_rng(18)
    for _ in range(20):
        noise = rng.dirichlet(np.ones(2), size=2)
        a0 = rng.uniform(0.73, 0.91, size=2)
        endpoints = np.column_stack((a0, noise * (1 - a0[:, None])))
        values = [fixed_trace_value_gradient(problem, a)[1] for a in endpoints]
        midpoint = fixed_trace_value_gradient(problem, endpoints.mean(axis=0))[1]
        assert midpoint <= np.mean(values) + 1e-10


def test_continuous_baseline_is_stationary_and_gap_survives() -> None:
    problem = deterministic_gain_optimized_problem()
    greedy = continuous_greedy_trace(problem)
    steady = continuous_steady_trace(problem, [greedy.alpha])
    assert greedy.stationarity < 2e-5
    assert greedy.simplex_first_order_gap < 2e-5
    assert steady.stationarity < 2e-5
    assert steady.value < 0.9 * greedy.value


def test_boundary_design_has_constrained_kkt_and_stable_floor_limit() -> None:
    problem = GainOptimizedProblem(np.array([[0.8]]), np.ones((1, 1)), np.array([[0.01]]), np.array([[100.0]]))
    first = continuous_steady_trace(problem, alpha_floor=1e-8)
    second = continuous_steady_trace(problem, [first.alpha], alpha_floor=1e-10)
    assert first.alpha[2] <= 5e-8
    assert second.alpha[2] <= 5e-10
    assert max(first.stationarity, second.stationarity) < 2e-5
    np.testing.assert_allclose([first.value, second.value], 0.25, rtol=1e-6)


def test_adjoint_stops_at_roundoff_without_thousands_of_tiny_steps() -> None:
    from steady_state_combined.ellipsoidal import FixedGainProblem

    script = Path(__file__).resolve().parents[1] / "examples/run_paper_validation.py"
    generate = runpy.run_path(str(script))["random_dimension_problem"]
    rng = np.random.default_rng(20261010)
    for _ in range(19):
        problem = generate(rng, 8)
    _, K = solve_dare(problem, np.full(3, 1 / 3))
    fixed = FixedGainProblem(problem.A, problem.H, K, problem.Q, problem.R)
    baseline = stepwise_trace_steady_state(fixed)
    assert baseline is not None
    adjoint = adjoint_fixed_trace(fixed, baseline[0])
    reference = global_fixed_trace(fixed)
    assert adjoint.evaluations < 2000
    np.testing.assert_allclose(adjoint.value, reference.value, rtol=1e-8)


def test_validation_cli_writes_complete_hashed_artifacts(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/run_paper_validation.py"
    subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path), "--random-systems", "0", "--per-dimension", "0"],
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    with (tmp_path / "validation_cases.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)
    manifest = json.loads((tmp_path / "validation_manifest.json").read_text())
    for name, digest in manifest["outputs_sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    subprocess.run(
        [sys.executable, str(script.with_name("run_solver_ablation.py")), "--validation-dir", str(tmp_path)],
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    ablation = json.loads((tmp_path / "solver_ablation_manifest.json").read_text())
    for name, digest in ablation["outputs_sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
