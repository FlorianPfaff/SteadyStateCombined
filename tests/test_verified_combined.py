from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("scipy")

from steady_state_combined.combined_mse import combined_mse_value_gradient
from steady_state_combined.combined_pareto import CombinedParetoProblem, deterministic_combined_pareto_problem
from steady_state_combined.verified_combined import _exact_psd, certify_combined_risk, verify_combined_risk_certificate


def test_exact_psd_covariance_validation_includes_singular_cases():
    assert _exact_psd([["1", "1"], ["1", "1"]])
    assert _exact_psd([["0", "0"], ["0", "0"]])
    assert not _exact_psd([["1", "2"], ["2", "1"]])
    assert not _exact_psd([["1", "1/1000000000000"], ["1/1000000000000", "0"]])


def test_verified_combined_bound_and_corruption_rejection():
    problem = deterministic_combined_pareto_problem()
    evaluation = combined_mse_value_gradient(problem, np.array([[0.8], [-0.2]]), [0.8, 0.1, 0.1])
    certificate = certify_combined_risk(problem, evaluation)
    checked = verify_combined_risk_certificate(certificate)
    assert checked["valid"] and not checked["global_optimality_claim"]
    from fractions import Fraction
    bound = float(Fraction(checked["risk_upper"]))
    assert evaluation.value <= bound
    np.testing.assert_allclose(bound, evaluation.value, rtol=1e-9)
    altered = deepcopy(certificate)
    altered["risk_upper"] = "0"
    with pytest.raises(ValueError, match="inconsistent"):
        verify_combined_risk_certificate(altered)
    altered = deepcopy(certificate)
    altered["bias_squared_upper"] = "0"
    with pytest.raises(ValueError, match="spectral"):
        verify_combined_risk_certificate(altered)


@pytest.mark.parametrize("zero", ["bounded", "stochastic"])
def test_zero_component_has_a_verified_risk_bound(zero):
    covariance = np.zeros((2, 2)) if zero == "stochastic" else np.eye(2)
    bounded = np.zeros((2, 2)) if zero == "bounded" else np.eye(2)
    problem = CombinedParetoProblem(0.6 * np.eye(2), np.eye(2), covariance, covariance, bounded, bounded)
    evaluation = combined_mse_value_gradient(problem, 0.5 * np.eye(2), [0.6, 0.2, 0.2])
    certificate = certify_combined_risk(problem, evaluation)
    assert verify_combined_risk_certificate(certificate)["valid"]
