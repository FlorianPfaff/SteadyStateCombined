import numpy as np
import pytest

pytest.importorskip("scipy")

from steady_state_combined.diagnostics import greedy_optimality_diagnostics
from steady_state_combined.ellipsoidal import FixedGainProblem, stepwise_trace_steady_state
from steady_state_combined.examples import deterministic_fixed_gain_problem
from steady_state_combined.research import global_fixed_trace


def test_scaled_orthogonal_dynamics_make_greedy_globally_optimal():
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    problem = FixedGainProblem(
        1.2 * rotation, np.eye(2), 0.5 * np.eye(2), np.diag([0.1, 1.0]), np.diag([2.0, 0.3])
    )
    alpha, _ = stepwise_trace_steady_state(problem, tol=1e-13)
    result = greedy_optimality_diagnostics(problem, alpha)
    optimum = global_fixed_trace(problem)
    np.testing.assert_allclose(result.amplification_ratios, result.amplification_ratios[0], rtol=1e-12)
    np.testing.assert_allclose(result.value, optimum.value, rtol=1e-11)
    assert result.greedy_kkt_residual < 1e-9
    assert result.relative_gap < 1e-9


def test_different_amplifications_explain_strict_improvement():
    problem = deterministic_fixed_gain_problem()
    alpha, _ = stepwise_trace_steady_state(problem, tol=1e-13)
    result = greedy_optimality_diagnostics(problem, alpha)
    optimum = global_fixed_trace(problem)
    assert result.greedy_kkt_residual < 1e-8
    assert np.ptp(result.amplification_ratios) > 0.1
    assert result.steady_kkt_residual > 0.1
    assert optimum.value < result.value * 0.7
    assert result.lower_bound <= optimum.value <= result.value
    assert result.value - optimum.value <= result.first_order_gap


def test_stable_domain_gap_bounds_independent_global_solution():
    problem = deterministic_fixed_gain_problem()
    optimum = global_fixed_trace(problem)
    rng = np.random.default_rng(20261003)
    rho2 = max(abs(np.linalg.eigvals(problem.F))) ** 2
    for _ in range(40):
        propagation = rng.uniform(rho2 + 0.02 * (1 - rho2), 0.98)
        noise = rng.dirichlet(np.ones(2)) * (1 - propagation)
        result = greedy_optimality_diagnostics(problem, np.r_[propagation, noise])
        assert result.lower_bound <= optimum.value + 1e-10
        assert result.value - optimum.value <= result.first_order_gap + 1e-10


def test_floor_and_vanishing_propagation_have_defined_gap():
    problem = FixedGainProblem(np.zeros((2, 2)), np.eye(2), 0.5 * np.eye(2), np.eye(2), np.eye(2))
    alpha = np.array([0.01, 0.495, 0.495])
    result = greedy_optimality_diagnostics(problem, alpha, alpha_floor=0.01)
    assert np.isnan(result.amplification_ratios[0])
    assert result.relative_gap < 1e-14
    assert result.steady_kkt_residual < 1e-14
    with pytest.raises(ValueError, match="floor"):
        greedy_optimality_diagnostics(problem, alpha, alpha_floor=0.02)


def test_scalar_reference_brackets_nonnormal_design_away_from_singular_limit():
    n = 8
    rng = np.random.default_rng(np.random.SeedSequence([20261003, n, 0]))
    N = np.triu(rng.normal(size=(n, n)), 1)
    N /= np.linalg.norm(N)
    F = 0.95 * np.eye(n) + 0.5 * N
    problem = FixedGainProblem(2 * F, np.eye(n), 0.5 * np.eye(n), 4 * np.eye(n) / n, 4 * np.eye(n) / n)
    optimum = global_fixed_trace(problem)
    baseline = stepwise_trace_steady_state(problem, max_iter=20_000)
    assert baseline is not None
    assert optimum.stationarity < 1e-5
    assert optimum.value <= np.trace(baseline[1]) * (1 + 1e-9)
