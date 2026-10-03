from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("scipy")

from flint import arb, arb_mat, ctx, fmpq_mat

from steady_state_combined.certification import (
    ROOT,
    certify_steady_trace,
    raw_dare,
    split_triangle,
    upper_corner,
)
from steady_state_combined.research import continuous_steady_trace, solve_dare
from steady_state_combined.riccati import GainOptimizedProblem, deterministic_gain_optimized_problem
from steady_state_combined.verified import (
    ball,
    enclose_candidate,
    endpoint,
    finite_riccati_trace,
    positive_definite,
    problem_record,
    rational,
    rounded_riccati_trace,
    validate_problem,
    verify_invariant_candidate,
)
from steady_state_combined.verify_certificate import verify_trace_certificate
from steady_state_combined.verified_riccati import verified_riccati_lower


def scalar_problem():
    return GainOptimizedProblem(np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)), np.array([[4.0]]))


def test_arb_endpoint_and_rational_serialization_preserve_enclosures():
    with ctx.workprec(128):
        value = ball(Fraction(1, 3))
        assert endpoint(value) <= Fraction(1, 3) <= endpoint(value, upper=True)
        original = np.nextafter(0.1, 1.0)
        assert rational(original) == Fraction.from_float(float(original))
        assert endpoint(ball(rational(original))) == rational(original)


def test_verified_ldl_matches_exact_sylvester_examples():
    rng = np.random.default_rng(991)
    for n in range(1, 6):
        B = rng.integers(-5, 6, size=(n, n))
        matrix = B @ B.T + np.eye(n, dtype=int)
        exact = fmpq_mat(matrix.tolist())
        for k in range(1, n + 1):
            assert fmpq_mat([[exact[i, j] for j in range(k)] for i in range(k)]).det() > 0
        assert positive_definite(arb_mat(matrix.tolist()))
    assert not positive_definite(arb_mat([[1, 2], [2, 1]]))
    assert not positive_definite(arb_mat([[1, 1], [1, 1]]))
    assert not positive_definite(arb_mat([[arb("0 +/- 1")]]))


def test_unnormalized_corner_has_exact_scalar_lower_bound():
    problem = scalar_problem()
    corner = (Fraction(1), Fraction(1), Fraction(1))
    enclosed = finite_riccati_trace(problem_record(problem), corner, 3)
    assert endpoint(enclosed) <= Fraction(4, 5) <= endpoint(enclosed, upper=True)
    normalized, _ = solve_dare(problem, np.full(3, 1 / 3))
    raw, _ = raw_dare(problem, corner)
    np.testing.assert_allclose(raw, [[0.8]])
    np.testing.assert_allclose(normalized, [[2.4]])


def test_regional_lower_bound_orders_independent_dare_solutions():
    problem = deterministic_gain_optimized_problem()
    record = problem_record(problem)
    rng = np.random.default_rng(847)
    triangles = list(split_triangle(ROOT))
    for _ in range(2):
        triangles = [child for triangle in triangles for child in split_triangle(triangle)]
    for triangle in triangles:
        lower = float(endpoint(finite_riccati_trace(record, upper_corner(triangle), 24, precision=256)))
        for _ in range(3):
            alpha = rng.dirichlet(np.ones(3)) @ np.array(triangle, dtype=float)
            P, _ = solve_dare(problem, alpha)
            assert lower <= np.trace(P) + 1e-10


def test_downward_rounded_iterations_are_independently_verified():
    problem = deterministic_gain_optimized_problem()
    weights = [Fraction(4, 5), Fraction(1, 10), Fraction(1, 5)]
    record = problem_record(problem)
    direct = finite_riccati_trace(record, weights, 50, precision=512)
    rounded = rounded_riccati_trace(record, weights, 50, precision=128, verify_steps=True)
    assert float(endpoint(rounded)) <= float(endpoint(direct, upper=True)) + 1e-14
    np.testing.assert_allclose(float(rounded), float(direct), rtol=1e-12)


def test_verified_riccati_subsolution_orders_scalar_and_matrix_limits():
    scalar_lower, _ = verified_riccati_lower(problem_record(scalar_problem()), [1, 1, 1])
    assert scalar_lower < Fraction(4, 5)
    np.testing.assert_allclose(float(scalar_lower), 0.8, rtol=1e-9)
    problem = deterministic_gain_optimized_problem()
    weights = [Fraction(4, 5), Fraction(1, 10), Fraction(1, 5)]
    first, _ = verified_riccati_lower(problem_record(problem), weights)
    replay, _ = verified_riccati_lower(problem_record(problem), weights, schur_check=True)
    reference = np.trace(raw_dare(problem, weights)[0])
    assert float(first) <= reference + 1e-14
    np.testing.assert_allclose([float(first), float(replay)], reference, rtol=1e-9)


def test_invariant_upper_bound_rejects_undersized_shape_and_asymmetry():
    problem = deterministic_gain_optimized_problem()
    result = continuous_steady_trace(problem)
    record = problem_record(problem)
    candidate = enclose_candidate(problem, result.alpha, result.K, result.P)
    assert float(verify_invariant_candidate(record, candidate)) >= result.value - 1e-12
    broken = deepcopy(candidate)
    broken["P"] = [[str(rational(x) / 2) for x in row] for row in candidate["P"]]
    with pytest.raises(ValueError, match="not verified"):
        verify_invariant_candidate(record, broken)
    broken = deepcopy(candidate)
    broken["P"][0][1] = str(rational(broken["P"][0][1]) + Fraction(1, 100))
    with pytest.raises(ValueError, match="symmetric"):
        verify_invariant_candidate(record, broken)


def test_boundary_infimum_is_certified_and_independently_replayed():
    problem = scalar_problem()
    result = certify_steady_trace(problem, [1e-8, 1 - 2e-8, 1e-8], rtol=1e-3, max_nodes=300, max_seconds=60)
    assert result.status == "certified_gap"
    assert result.lower_bound <= 1 <= result.upper_bound
    assert result.relative_gap <= 1e-3
    verification = verify_trace_certificate(result.certificate)
    assert verification["valid"] and verification["gap_achieved"]


def test_checkpoint_resume_and_corrupted_certificate_rejection(tmp_path):
    problem = scalar_problem()
    checkpoint = tmp_path / "checkpoint.json"
    first = certify_steady_trace(problem, [0.01, 0.98, 0.01], rtol=0.05, max_nodes=2, checkpoint_path=checkpoint)
    assert first.status == "budget_limit"
    assert verify_trace_certificate(first.certificate)["valid"]
    second = certify_steady_trace(problem, rtol=0.05, max_nodes=100, checkpoint_path=checkpoint)
    assert second.status == "certified_gap"
    assert second.certificate["subdivisions"] > first.certificate["subdivisions"]
    assert verify_trace_certificate(second.certificate)["valid"]
    for corruption in ("lower", "upper", "cover", "overlap", "problem"):
        broken = deepcopy(second.certificate)
        if corruption == "lower":
            broken["leaves"][0]["lower"] = "1000"
        elif corruption == "upper":
            broken["upper"] = "0"
        elif corruption == "cover":
            broken["leaves"].pop()
        elif corruption == "overlap":
            broken["leaves"].append(deepcopy(broken["leaves"][0]))
        else:
            broken["problem"]["Q"][0][0] = "2"
        with pytest.raises(ValueError):
            verify_trace_certificate(broken)


def test_invalid_covariances_and_inconclusive_bounds_are_not_certified():
    record = problem_record(scalar_problem())
    record["Q"] = [["-1"]]
    with pytest.raises(ValueError, match="positive definite"):
        validate_problem(record)
    result = certify_steady_trace(scalar_problem(), [0.1, 0.8, 0.1], max_nodes=0)
    assert result.status == "budget_limit"
    assert result.lower_bound == 0
    assert verify_trace_certificate(result.certificate)["valid"]
    with pytest.raises(ValueError, match="weights"):
        finite_riccati_trace(problem_record(scalar_problem()), [0, 1, 1], 5)
