from __future__ import annotations

import numpy as np

from steady_state_combined import FixedGainProblem, solve_fixed_gain_steady_state
from steady_state_combined.riccati import (
    deterministic_gain_optimized_problem,
    grid_optimize_riccati,
    optimize_riccati_weights,
    solve_fixed_alpha_riccati,
    stepwise_gain_trace_recursion,
)


def test_fixed_alpha_riccati_matches_fixed_gain_solution_for_its_gain() -> None:
    problem = deterministic_gain_optimized_problem()
    alpha = np.array([0.70, 0.20, 0.10])
    result = solve_fixed_alpha_riccati(problem, alpha)
    assert result is not None

    fixed_gain_problem = FixedGainProblem(A=problem.A, H=problem.H, K=result.K, Q=problem.Q, R=problem.R)
    P_fixed = solve_fixed_gain_steady_state(fixed_gain_problem, alpha)
    assert P_fixed is not None
    assert np.linalg.norm(P_fixed - result.P, ord="fro") < 1e-8


def test_grid_optimized_riccati_returns_valid_solution() -> None:
    problem = deterministic_gain_optimized_problem()
    result = grid_optimize_riccati(problem, resolution=61)
    assert result is not None
    assert result.converged
    assert np.min(np.linalg.eigvalsh(result.P)) > -1e-8
    assert result.K.shape == (problem.n, problem.m)


def test_gain_reoptimized_greedy_baseline_converges() -> None:
    problem = deterministic_gain_optimized_problem()
    result = stepwise_gain_trace_recursion(problem, resolution=61)
    assert result is not None
    assert result.converged
    assert np.min(np.linalg.eigvalsh(result.P)) > -1e-8


def test_refined_riccati_optimization_is_not_worse_than_greedy() -> None:
    problem = deterministic_gain_optimized_problem()
    greedy = stepwise_gain_trace_recursion(problem, resolution=31)
    assert greedy is not None
    result = optimize_riccati_weights(problem, resolution=31, initial_alphas=[greedy.alpha])
    assert result is not None
    assert result.value <= greedy.value + 1e-10
