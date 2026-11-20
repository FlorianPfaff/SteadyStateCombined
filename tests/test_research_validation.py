from __future__ import annotations

import csv
import dataclasses
from fractions import Fraction
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
    posynomial_gap_bound,
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
    assert not adjoint.stagnated
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
    assert not adjoint.stagnated
    np.testing.assert_allclose(adjoint.value, reference.value, rtol=1e-8)


def test_adjoint_returns_optimum_when_descent_slope_is_rounding_noise() -> None:
    from steady_state_combined.ellipsoidal import FixedGainProblem

    # rho(F) = 0.97 makes the gradient about 100 times the value: the second-order slope drops below its rounding
    # error before the first-order KKT test is met, and half of these runs used to compute a non-negative slope.
    for coupling in (0.1, 0.2, 0.3, 0.4):
        F = np.array([[0.97, coupling], [0.0, 0.97]])
        problem = FixedGainProblem(2 * F, np.eye(2), 0.5 * np.eye(2), 2 * np.eye(2), 2 * np.eye(2))
        reference = global_fixed_trace(problem)
        for start in ([0.96, 0.02, 0.02], [0.97, 0.02, 0.01], [0.98, 0.01, 0.01]):
            adjoint = adjoint_fixed_trace(problem, np.array(start))
            assert adjoint.evaluations < 100
            np.testing.assert_allclose(adjoint.value, reference.value, rtol=1e-12)
            # The first-order KKT test is still unmet; the second-order bound confirms the optimum instead.
            assert np.ptp(adjoint.gradient) / adjoint.value > 1e-6 and not adjoint.stagnated


def test_adjoint_reports_stagnation_instead_of_a_point_far_from_the_optimum() -> None:
    from steady_state_combined.ellipsoidal import FixedGainProblem

    # The trace falls from about 16/15 to its minimum 1 when alpha_0 moves by d / 4 = 2^-52: every descent slope
    # computed at the start is rounding noise. The solver returned the start after zero iterations as if optimal.
    d = 2.0**-50
    problem = FixedGainProblem(np.array([[2 * (1 - d)]]), np.ones((1, 1)), np.array([[0.5]]), np.array([[d * d]]),
                               np.array([[d * d]]))

    def exact_trace(alpha) -> Fraction:
        # F = 1 - d and S_w = S_v = d^2 / 4 exactly, so P = (S_w / alpha_w + S_v / alpha_v) / (1 - F^2 / alpha_0).
        a0, aw, av = map(Fraction, alpha)
        return (Fraction(d) ** 2 / 4 * (1 / aw + 1 / av)) / (1 - (1 - Fraction(d)) ** 2 / a0)

    assert exact_trace([1 - d, d / 2, d / 2]) == 1
    start = np.array([1 - 3 * d / 4, 3 * d / 8, 3 * d / 8])
    result = adjoint_fixed_trace(problem, start)
    assert result.stagnated
    assert exact_trace(result.alpha) > Fraction(106, 100)
    assert posynomial_gap_bound(result.alpha, result.value, result.gradient) > 0.06
    # Running out of iterations is no confirmation either.
    fixed = deterministic_fixed_gain_problem()
    assert adjoint_fixed_trace(fixed, np.array([0.8, 0.1, 0.1]), max_iter=2).stagnated
    assert not adjoint_fixed_trace(fixed, np.array([0.8, 0.1, 0.1])).stagnated


def test_adjoint_flags_points_whose_gap_hides_below_rounding() -> None:
    from steady_state_combined.ellipsoidal import FixedGainProblem

    # Points that an earlier version of the gap bound confirmed far from the minimum. A weight of 1e-140 leaves the
    # other gradient shares below eps, so log1p(ratio - 1) was -inf, which the clamp at zero turned into a bound of 0.
    problem = FixedGainProblem(np.ones((1, 1)), np.ones((1, 1)), np.array([[0.5]]), np.array([[4.0]]), np.array([[4.0]]))
    result = adjoint_fixed_trace(problem, np.array([0.5, 1e-140, 0.5]))  # the minimum is 16
    assert result.value > 1e139 and result.stagnated
    assert posynomial_gap_bound(result.alpha, result.value, result.gradient) == 1.0
    # With 1 - rho(F) = 1.2e-13 the trace moves by about 1e-3 between neighbouring floating-point weights, so the
    # divergence computed from the float gradient was rounding noise; it confirmed a point 7.3e-5 above the minimum
    # (in 256-bit arithmetic). No accuracy beyond that resolution can be confirmed.
    A = np.array([[-0.48335732454404606, 0.12447908132549203, -0.36377500175139504],
                  [-0.14124130610842214, -1.1274568899701494, -0.05473086343030783],
                  [-1.9120770038082664, -0.005064448216694418, -1.5565983911362462]])
    Q = np.array([[0.00017652286786572904, 0.0001595220730012635, -3.60548582191036e-05],
                  [0.0001595220730012635, 0.0004348221778550011, -0.00012597746933564697],
                  [-3.60548582191036e-05, -0.00012597746933564697, 4.77551079594547e-05]])
    R = np.array([[0.013523601909422232, -0.011215904631709973, 0.004557314846429718],
                  [-0.011215904631709973, 0.011170129370801907, -0.008082624261491192],
                  [0.004557314846429718, -0.008082624261491192, 0.012061499765258922]])
    problem = FixedGainProblem(A, np.eye(3), 0.5 * np.eye(3), Q, R)
    result = adjoint_fixed_trace(problem, np.array([0.9999999999999231, 7.545262431120981e-14, 1.485831295313535e-15]))
    elasticity = -float(result.gradient @ result.alpha) / result.value
    assert elasticity * np.finfo(float).eps > 1e-6 and result.stagnated


def test_adjoint_confirms_its_optimum_for_a_singular_semidefinite_weight() -> None:
    # Rounding gives this rank-one weight an eigenvalue of about -1e-17; it is still a semidefinite weight.
    problem = deterministic_fixed_gain_problem()
    x = np.array([-0.24836162209524854, 0.4204452380655215])
    singular = adjoint_fixed_trace(problem, np.array([0.8, 0.1, 0.1]), weight=np.outer(x, x))
    regular = adjoint_fixed_trace(problem, np.array([0.8, 0.1, 0.1]), weight=np.outer(x, x) + 1e-15 * np.eye(2))
    assert not singular.stagnated and not regular.stagnated
    np.testing.assert_allclose(singular.value, regular.value, rtol=1e-12)


def test_validation_requires_the_adjoint_and_global_values_to_agree_from_both_sides() -> None:
    script = Path(__file__).resolve().parents[1] / "examples/run_paper_validation.py"
    evaluate = runpy.run_path(str(script))["evaluate_case"]
    task = (0, "deterministic", deterministic_gain_optimized_problem(), None, None)
    accurate = evaluate(task)
    assert accurate["status"] == "ok"
    solve = evaluate.__globals__["adjoint_fixed_trace"]  # run_path returns a copy of the script's globals
    rows = {}
    for stagnated in (False, True):
        for excess in (0.0, 2e-6):
            def adjoint(*args, stagnated=stagnated, excess=excess):
                result = solve(*args)
                return dataclasses.replace(result, stagnated=stagnated, value=result.value * (1 + excess))

            evaluate.__globals__["adjoint_fixed_trace"] = adjoint
            rows[stagnated, excess] = evaluate(task)
    # Agreement validates the plant, flag or not; a value above the global one fails it, flag or not.
    assert rows[True, 0.0]["status"] == "ok" and rows[True, 0.0]["fixed_adjoint_trace"] == accurate["fixed_adjoint_trace"]
    for stagnated in (False, True):
        assert rows[stagnated, 2e-6]["status"] == "failed"
        assert "above independently solved global reference" in rows[stagnated, 2e-6]["error"]


def test_posynomial_gap_bound_is_an_upper_bound_of_second_order() -> None:
    from steady_state_combined.ellipsoidal import FixedGainProblem, spectral_radius

    rng = np.random.default_rng(20261119)
    checked = 0
    while checked < 12:
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 3))
        A = rng.normal(size=(n, n))
        A *= rng.uniform(0.3, 1.5) / spectral_radius(A)
        H, L, M = rng.normal(size=(m, n)), rng.normal(size=(n, n)), rng.normal(size=(m, m))
        gain_problem = GainOptimizedProblem(A, H, L @ L.T, M @ M.T)
        fixed = FixedGainProblem(A, H, solve_dare(gain_problem, rng.dirichlet(np.ones(3)))[1], L @ L.T, M @ M.T)
        X = rng.normal(size=(n, n))
        weight = None if checked % 2 else X @ X.T
        try:
            reference = global_fixed_trace(fixed, weight)
        except (ValueError, RuntimeError):
            continue
        rho2 = spectral_radius(fixed.F) ** 2
        points = []
        for _ in range(15):
            a0 = rho2 + (1 - rho2) * rng.uniform(0.01, 0.99)
            alpha = np.array([a0, *(1 - a0) * rng.dirichlet(np.ones(2))])
            points.append((alpha, *fixed_trace_value_gradient(fixed, alpha, weight)[1:]))
        best = min(reference.value, *(value for _, value, _ in points))
        for alpha, value, gradient in points:
            # value * (1 - bound) bounds the minimum from below, so it may not exceed any trace that is attained.
            assert value * (1 - posynomial_gap_bound(alpha, value, gradient)) <= best * (1 + 1e-9)
        # Near the minimum the bound is of second order in the distance, like the gap itself.
        bounds = []
        for offset in (1e-2, 1e-3, 1e-4):
            alpha = reference.alpha + offset * min(reference.alpha[1:]) * np.array([0.0, 1.0, -1.0])
            _, value, gradient = fixed_trace_value_gradient(fixed, alpha, weight)
            gap = (value - reference.value) / value
            bounds.append(posynomial_gap_bound(alpha, value, gradient))
            assert 0 < gap <= bounds[-1] <= 20 * gap
        assert 50 < bounds[1] / bounds[2] < 200
        checked += 1
    assert posynomial_gap_bound(np.full(3, 1 / 3), 1.0, np.array([-1.0, -2.0, 0.5])) == np.inf


def test_adjoint_descent_check_still_raises_for_broken_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    from steady_state_combined import research

    problem = deterministic_fixed_gain_problem()
    start = np.array([0.8, 0.1, 0.1])
    # A negative definite weight makes every gradient entry positive: a genuine ascent direction.
    with pytest.raises(RuntimeError, match="descent check"):
        adjoint_fixed_trace(problem, start, weight=-np.eye(problem.n))
    evaluate = research.fixed_trace_value_gradient
    monkeypatch.setattr(
        research, "fixed_trace_value_gradient", lambda *args: (*evaluate(*args)[:2], np.full(3, np.nan))
    )
    with pytest.raises(RuntimeError, match="descent check"):
        adjoint_fixed_trace(problem, start)


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
