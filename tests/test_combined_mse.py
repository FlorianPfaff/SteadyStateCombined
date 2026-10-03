import numpy as np
import pytest

pytest.importorskip("scipy")

from steady_state_combined.combined_mse import (
    combined_mse_value_gradient,
    default_combined_starts,
    optimize_combined_mse,
)
from steady_state_combined.combined_pareto import CombinedParetoProblem, deterministic_combined_pareto_problem


def test_two_adjoint_gradients_match_directional_finite_differences():
    problem = deterministic_combined_pareto_problem()
    K = np.array([[0.8], [-0.2]])
    alpha = np.array([0.8, 0.1, 0.1])
    for tau in (0.0, 0.03):
        evaluated = combined_mse_value_gradient(problem, K, alpha, tau)
        for direction in (np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])):
            h = 1e-6
            finite = (combined_mse_value_gradient(problem, K + h * direction, alpha, tau).smooth_value
                      - combined_mse_value_gradient(problem, K - h * direction, alpha, tau).smooth_value) / (2 * h)
            np.testing.assert_allclose(finite, np.sum(evaluated.gradient_K * direction), rtol=2e-6, atol=1e-7)
        for direction in (np.array([1., -1., 0.]), np.array([0., 1., -1.])):
            h = 1e-6
            finite = (combined_mse_value_gradient(problem, K, alpha + h * direction, tau).smooth_value
                      - combined_mse_value_gradient(problem, K, alpha - h * direction, tau).smooth_value) / (2 * h)
            np.testing.assert_allclose(finite, evaluated.gradient_alpha @ direction, rtol=2e-6, atol=1e-7)


def test_spectral_smoothing_handles_repeated_eigenvalues():
    problem = CombinedParetoProblem(0.8 * np.eye(2), np.eye(2), np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    K = 0.5 * np.eye(2)
    alpha = np.array([0.6, 0.2, 0.2])
    result = combined_mse_value_gradient(problem, K, alpha, tau=0.1)
    np.testing.assert_allclose(result.P, result.P[0, 0] * np.eye(2))
    np.testing.assert_allclose(result.smooth_value - result.value, 0.1 * np.log(2))
    direction = np.array([[0.3, 0.5], [-0.7, 0.8]])
    h = 1e-6
    finite = (combined_mse_value_gradient(problem, K + h * direction, alpha, 0.1).smooth_value
              - combined_mse_value_gradient(problem, K - h * direction, alpha, 0.1).smooth_value) / (2 * h)
    np.testing.assert_allclose(finite, np.sum(result.gradient_K * direction), rtol=1e-6)


@pytest.mark.parametrize("zero", ["bounded", "stochastic"])
def test_zero_uncertainty_limits(zero):
    Qs, Rs, Qb, Rb = [np.eye(2) for _ in range(4)]
    if zero == "bounded":
        Qb, Rb = np.zeros((2, 2)), np.zeros((2, 2))
    else:
        Qs, Rs = np.zeros((2, 2)), np.zeros((2, 2))
    problem = CombinedParetoProblem(0.6 * np.eye(2), np.eye(2), Qs, Rs, Qb, Rb)
    result = combined_mse_value_gradient(problem, 0.5 * np.eye(2), [0.6, 0.2, 0.2], tau=0.01)
    if zero == "bounded":
        np.testing.assert_array_equal(result.P, np.zeros((2, 2)))
        np.testing.assert_array_equal(result.gradient_alpha, np.zeros(3))
        assert result.value == np.trace(result.Sigma)
    else:
        np.testing.assert_array_equal(result.Sigma, np.zeros((2, 2)))
        assert result.value == np.linalg.eigvalsh(result.P)[-1]


def test_optimizer_retains_or_improves_feasible_seed():
    problem = deterministic_combined_pareto_problem()
    starts = default_combined_starts(problem)
    result = optimize_combined_mse(problem, starts, max_iter_per_stage=80)
    assert result.evaluation.value <= result.initial_value + 1e-12
    assert result.evaluation.stability_margin > 0
    assert np.min(result.evaluation.alpha) >= result.alpha_floor * (1 - 1e-8)
    assert result.evaluation.smoothing_bound <= result.initial_value * 1e-6 * (1 + 1e-12)
    assert result.evaluation.smooth_value >= result.evaluation.value
    assert np.isfinite(result.stationarity)
    with pytest.raises(ValueError, match="stable"):
        combined_mse_value_gradient(problem, np.zeros((2, 1)), [0.1, 0.45, 0.45])
    with pytest.raises(ValueError, match="no feasible"):
        optimize_combined_mse(problem, [(np.array([[100.], [100.]]), np.full(3, 1 / 3))])


def test_smoothing_schedule_is_explicit_and_validated():
    problem = deterministic_combined_pareto_problem()
    result = optimize_combined_mse(problem, max_iter_per_stage=5, smoothing_exponents=(3, 5))
    assert result.evaluation.smoothing_bound <= result.initial_value * 1e-5 * (1 + 1e-12)
    for invalid in ((), (3, 2), (2, 2), (0,), (1.5,)):
        with pytest.raises(ValueError, match="smoothing"):
            optimize_combined_mse(problem, smoothing_exponents=invalid)
