from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("scipy")

from flint import arb, arb_mat, ctx, fmpq_mat

from steady_state_combined.certification import (
    ROOT,
    _reaching_horizon,
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


def slow_unobservable_problem():
    """Detectable, but the 0.99 mode is unobservable: the invariant trace is unbounded for alpha_0 <= 0.99**2."""
    return GainOptimizedProblem(np.diag([0.99, 0.5]), np.array([[0.0, 1.0]]), np.eye(2), np.eye(1))


def test_reaching_horizon_matches_direct_iteration_and_respects_the_cap():
    problem = slow_unobservable_problem()
    for weights in ([0.97, 0.5, 0.5], [Fraction(98, 100), Fraction(1, 50), 1], [0.5, 1, 1]):
        a = np.array(list(map(float, weights)))
        information = problem.H.T @ np.linalg.solve(problem.R, problem.H) * a[2]
        P, traces = np.zeros((2, 2)), []
        for _ in range(300):
            prior = problem.A @ P @ problem.A.T / a[0] + problem.Q / a[1]
            P = np.linalg.inv(np.linalg.inv(prior) + information)
            traces.append(np.trace(P))
        assert _reaching_horizon(problem, weights, 0.5 * traces[0], 4096) == 1
        for steps in (2, 3, 17, 64, 65, 200, 300):
            target = 0.5 * (traces[steps - 2] + traces[steps - 1])
            assert _reaching_horizon(problem, weights, target, 4096) == steps
            assert _reaching_horizon(problem, weights, target, steps) == steps
            assert _reaching_horizon(problem, weights, target, steps - 1) is None
    # With a stabilizing solution the iterate converges: a target above its limit is never reached.
    limit = np.trace(raw_dare(problem, [0.99, 0.5, 0.5])[0])
    assert _reaching_horizon(problem, [0.99, 0.5, 0.5], 1.01 * limit, 4096) is None
    assert _reaching_horizon(problem, [0.99, 0.5, 0.5], 0.5 * limit, 4096) is not None


def test_slow_unobservable_mode_is_closed_by_a_longer_horizon():
    # A 64-step iterate at the corner of an infeasible sliver stays below the incumbent however small the sliver
    # gets, so the search used to bisect it until the budget ran out (gap above 200 % after 300 subdivisions).
    result = certify_steady_trace(slow_unobservable_problem(), [0.99, 0.0099, 0.0001], rtol=0.05, max_nodes=300)
    assert result.status == "certified_gap"
    assert result.relative_gap <= 0.05
    horizons = [leaf["iterations"] for leaf in result.certificate["leaves"] if leaf.get("mode") == "finite"]
    assert horizons and max(horizons) > 64
    verification = verify_trace_certificate(result.certificate)
    assert verification["valid"] and verification["gap_achieved"]


def coupled_slow_problem(n=4, slow=0.99):
    """Like slow_unobservable_problem, but the unobserved slow state is driven by the others."""
    rng = np.random.default_rng(1)
    A, H = np.zeros((n, n)), np.zeros((1, n))
    A[0, 0], A[0, 1:], A[1:, 1:] = slow, 0.1, 0.5 * np.eye(n - 1) + 0.05 * rng.normal(size=(n - 1, n - 1))
    H[:, 1:] = rng.normal(size=(1, n - 1))
    return GainOptimizedProblem(A, H, np.eye(n), np.eye(1))


def test_extended_leaves_replay_and_resume_on_a_coupled_plant(tmp_path):
    problem, checkpoint = coupled_slow_problem(), tmp_path / "checkpoint.json"
    alpha = [0.99**2 + 0.6 * (1 - 0.99**2), 0.39 * (1 - 0.99**2), 0.01 * (1 - 0.99**2)]
    result = certify_steady_trace(problem, alpha, rtol=0.05, max_nodes=300, checkpoint_path=checkpoint)
    assert result.status == "certified_gap"
    assert max(leaf["iterations"] for leaf in result.certificate["leaves"]) > 64
    assert verify_trace_certificate(result.certificate)["gap_achieved"]
    assert certify_steady_trace(problem, rtol=0.05, max_nodes=0, checkpoint_path=checkpoint).status == "certified_gap"
    # A cap below the horizon that the blocking leaves need changes nothing: they keep the fallback horizon.
    capped = certify_steady_trace(problem, alpha, rtol=0.05, max_nodes=40, max_horizon=80)
    assert capped.status == "budget_limit"
    assert max(leaf["iterations"] for leaf in capped.certificate["leaves"]) == 64
    assert verify_trace_certificate(capped.certificate)["valid"]


def test_long_horizon_bounds_are_recorded_only_in_a_form_the_replay_accepts(monkeypatch):
    from steady_state_combined import certification

    problem = coupled_slow_problem()
    record, weights = problem_record(problem), (Fraction(39, 40), Fraction(1, 50), Fraction(1, 50))
    iterations = _reaching_horizon(problem, weights, 1e4, 4096)
    assert iterations > 64
    leaf = certification._replayable_bound(record, "", weights, iterations, 1.0, 128, 1024, float("inf"))
    assert leaf["mode"] == "finite" and float(rational(leaf["lower"])) > 0.99e4
    assert rational(leaf["lower"]) <= endpoint(finite_riccati_trace(record, weights, iterations, 128, schur=True))
    # The Schur recursion of the replay breaks down on long diverging iterates of larger plants. Then only the
    # rounded kernel, with every transition verified as in the replay, may supply the record.
    information_form = certification.finite_riccati_trace

    def schur_breaks_down(record, weights, iterations, precision=128, *, schur=False):
        if schur:
            raise ZeroDivisionError("singular matrix in solve()")
        return information_form(record, weights, iterations, precision)

    monkeypatch.setattr(certification, "finite_riccati_trace", schur_breaks_down)
    leaf = certification._replayable_bound(record, "", weights, iterations, 1.0, 128, 1024, float("inf"))
    assert leaf["mode"] == "rounded" and leaf["iterations"] == iterations and float(rational(leaf["lower"])) > 0.99e4
    replayed = rounded_riccati_trace(record, weights, iterations, leaf["precision"], verify_steps=True)
    assert rational(leaf["lower"]) <= endpoint(replayed)
    # Out of time, nothing is recorded rather than something unchecked.
    assert certification._replayable_bound(record, "", weights, iterations, 1.0, 128, 1024, 0.0) is None
