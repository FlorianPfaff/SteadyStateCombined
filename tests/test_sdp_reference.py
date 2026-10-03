import numpy as np
import pytest

pytest.importorskip("cvxpy")
pytest.importorskip("clarabel")

from steady_state_combined.combined_pareto import CombinedParetoProblem
from steady_state_combined.research import solve_dare
from steady_state_combined.riccati import deterministic_gain_optimized_problem
from steady_state_combined.sdp_reference import OneStepCombinedSdp, fixed_weight_lmi_reference, greedy_combined_mse


@pytest.mark.parametrize("balanced", [False, True])
def test_fixed_weight_sdp_agrees_with_dare(balanced):
    problem = deterministic_gain_optimized_problem()
    alpha = np.array([0.75, 0.1, 0.15])
    reference = fixed_weight_lmi_reference(problem, alpha, balanced=balanced)
    P, _ = solve_dare(problem, alpha)
    np.testing.assert_allclose(reference.value, np.trace(P), rtol=2e-6)
    np.testing.assert_allclose(reference.P, P, rtol=2e-4, atol=2e-6)


def test_sdp_state_coordinates_preserve_physical_trace():
    problem = deterministic_gain_optimized_problem()
    alpha = np.array([0.75, 0.1, 0.15])
    transform = np.array([[0.5, 0.1], [0, 0.2]])
    reference = fixed_weight_lmi_reference(problem, alpha, balanced=True, state_scale=transform)
    P, _ = solve_dare(problem, alpha)
    np.testing.assert_allclose(reference.value, np.trace(P), rtol=2e-6)
    np.testing.assert_allclose(reference.P, P, rtol=3e-4, atol=2e-6)


def test_one_step_spectral_sdp_matches_scalar_kalman_limit():
    problem = CombinedParetoProblem(np.array([[0.8]]), np.ones((1, 1)), np.array([[0.1]]),
                                   np.array([[0.2]]), np.zeros((1, 1)), np.zeros((1, 1)))
    result = OneStepCombinedSdp(problem).solve(np.array([[0.3]]), np.zeros((1, 1)))
    prior = 0.8**2 * 0.3 + 0.1
    np.testing.assert_allclose(result.K, [[prior / (prior + 0.2)]], rtol=1e-5)
    np.testing.assert_allclose(result.value, prior * 0.2 / (prior + 0.2), rtol=1e-7)


def test_greedy_combined_limit_solves_fixed_point():
    problem = CombinedParetoProblem(np.array([[0.4]]), np.ones((1, 1)), np.array([[0.1]]),
                                   np.array([[0.2]]), np.array([[0.02]]), np.array([[0.03]]))
    result = greedy_combined_mse(problem, tolerance=1e-7)
    assert result.value > 0
    assert min(result.alpha) > 0
    np.testing.assert_allclose(sum(result.alpha), 1)
    assert max(result.fixed_point_residual, result.one_step_residual, result.one_step_risk_gap) <= 1e-6


def test_greedy_combined_high_variance_regression():
    bounded = deterministic_gain_optimized_problem()
    rng = np.random.default_rng(np.random.SeedSequence([20261003, 0, 731]))
    U, _ = np.linalg.qr(rng.normal(size=bounded.Q.shape))
    V, _ = np.linalg.qr(rng.normal(size=bounded.R.shape))
    problem = CombinedParetoProblem(bounded.A, bounded.H,
        10 * (U * np.linalg.eigvalsh(bounded.Q)) @ U.T,
        10 * (V * np.linalg.eigvalsh(bounded.R)) @ V.T, bounded.Q, bounded.R)
    result = greedy_combined_mse(problem)
    assert max(result.fixed_point_residual, result.one_step_residual, result.one_step_risk_gap) <= 1e-6
