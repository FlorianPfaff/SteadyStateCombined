import numpy as np
import pytest

pytest.importorskip("scipy")

from steady_state_combined.combined_mse import combined_mse_value_gradient
from steady_state_combined.combined_pareto import CombinedParetoProblem
from steady_state_combined.tracking_validation import directional_bias_sequence, bounded_profiles, simulate_profile


def test_support_sequence_attains_directional_support():
    problem = CombinedParetoProblem(0.5 * np.eye(2), np.eye(2), np.eye(2), np.eye(2), np.eye(2), 2 * np.eye(2))
    K = 0.3 * np.eye(2)
    direction = np.array([0.6, 0.8])
    w, v = directional_bias_sequence(problem, K, direction, 10)
    bias = np.zeros(2)
    for wt, vt in zip(w, v):
        bias = 0.35 * bias + 0.7 * wt - K @ vt
    support = sum(0.35**j * (0.7 + 0.3 * np.sqrt(2)) for j in range(10))
    np.testing.assert_allclose(direction @ bias, support)
    np.testing.assert_allclose(np.sum(w**2, axis=1), 1)
    np.testing.assert_allclose(np.sum(v**2, axis=1) / 2, 1)


def test_tracking_analytic_and_paired_statistics():
    problem = CombinedParetoProblem(0.5 * np.eye(6), np.eye(6), 0.01 * np.eye(6),
                                   0.02 * np.eye(6), 0.01 * np.eye(6), 0.02 * np.eye(6))
    design = combined_mse_value_gradient(problem, 0.5 * np.eye(6), [0.5, 0.25, 0.25])
    designs = {"greedy": design, "steady": design}
    profiles = bounded_profiles(problem, designs, 80, episode=20)
    for w, v in profiles.values():
        result = simulate_profile(problem, designs, w, v, trials=400, burn_in=10)
        assert result.paired_rows[0]["mse_difference"] == 0
        assert result.paired_rows[0]["difference_se"] == 0
        assert result.rows[0]["maximum_bias_ellipsoid_ratio"] <= 1 + 1e-12
        assert result.rows[0]["analytic_peak_mse"] <= design.value
        assert abs(result.rows[0]["monte_carlo_z"]) < 5


def test_tracking_rejects_outside_profile():
    problem = CombinedParetoProblem(0.5 * np.eye(6), np.eye(6), np.eye(6), np.eye(6), np.eye(6), np.eye(6))
    with pytest.raises(ValueError, match="leaves"):
        simulate_profile(problem, {}, 2 * np.ones((10, 6)), np.zeros((10, 6)), burn_in=0)
