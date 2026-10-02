from __future__ import annotations

import numpy as np

from steady_state_combined.combined_pareto import (
    deterministic_combined_pareto_problem,
    enumerate_combined_candidates_2d,
    nondominated_candidates,
    pareto_from_candidates,
    solve_combined_fixed_gain_alpha,
)


def test_fixed_gain_alpha_combined_solution_is_valid() -> None:
    problem = deterministic_combined_pareto_problem()
    K = np.array([[1.0], [-0.5]])
    alpha = np.array([0.7, 0.2, 0.1])
    candidate = solve_combined_fixed_gain_alpha(problem, K, alpha)
    assert candidate is not None
    assert candidate.trace_sigma > 0.0
    assert candidate.trace_P > 0.0
    assert np.min(np.linalg.eigvalsh(candidate.Sigma)) > -1e-8
    assert np.min(np.linalg.eigvalsh(candidate.P)) > -1e-8


def test_combined_pareto_sweep_returns_one_result_per_lambda() -> None:
    problem = deterministic_combined_pareto_problem()
    candidates = enumerate_combined_candidates_2d(problem, alpha_resolution=9, gain_resolution=7)
    assert candidates
    lambdas = [0.0, 0.5, 1.0]
    results = pareto_from_candidates(candidates, lambdas)
    assert len(results) == len(lambdas)
    for result in results:
        assert 0.0 <= result.lambda_stochastic <= 1.0
        assert result.objective > 0.0
        assert result.candidate.trace_sigma > 0.0
        assert result.candidate.trace_P > 0.0

    frontier = nondominated_candidates(candidates)
    assert frontier
    assert all(
        left.trace_sigma <= right.trace_sigma and left.trace_P > right.trace_P
        for left, right in zip(frontier, frontier[1:])
    )
