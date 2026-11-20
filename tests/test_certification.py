from collections import Counter
from copy import deepcopy
import csv
from fractions import Fraction
import gzip
import hashlib
from itertools import count
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("scipy")

from flint import arb, arb_mat, ctx, fmpq_mat

from steady_state_combined.certification import (
    ROOT,
    CertifiedTraceResult,
    _reaching_horizon,
    certify_steady_trace,
    raw_dare,
    split_triangle,
    upper_corner,
)
from steady_state_combined.combined_mse import combined_mse_value_gradient
from steady_state_combined.combined_pareto import CombinedParetoProblem
from steady_state_combined.research import continuous_steady_trace, solve_dare
from steady_state_combined.riccati import GainOptimizedProblem, deterministic_gain_optimized_problem
from steady_state_combined.verified_combined import certify_combined_risk
from steady_state_combined.verified import (
    ball,
    decode_matrix,
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
from steady_state_combined.verify_certificate import main as replay_main, verify_trace_certificate
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


def test_rounded_iterates_are_exactly_symmetric_even_if_mirrored_midpoints_disagree(monkeypatch):
    problem = deterministic_gain_optimized_problem()
    weights = [Fraction(4, 5), Fraction(1, 10), Fraction(1, 5)]
    record = problem_record(problem)
    iterates, calls = [], count()

    def recording_decode(matrix):
        if isinstance(matrix[0][0], Fraction):
            iterates.append(matrix)
        return decode_matrix(matrix)

    def skewed_endpoint(value, upper=False):
        # Every midpoint extraction is displaced differently; the rigorous upper bounds are left alone.
        return endpoint(value, upper) + (0 if upper else Fraction(next(calls) % 5, 2**70))

    monkeypatch.setattr("steady_state_combined.verified.decode_matrix", recording_decode)
    monkeypatch.setattr("steady_state_combined.verified.endpoint", skewed_endpoint)
    rounded = rounded_riccati_trace(record, weights, 50, precision=128, verify_steps=True)
    assert next(calls) and iterates
    assert all(m[i][j] == m[j][i] for m in iterates for i in range(problem.n) for j in range(i))
    direct = finite_riccati_trace(record, weights, 50, precision=512)
    assert endpoint(rounded) <= endpoint(direct, upper=True)
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


PAPER_TARGET = Fraction(1, 1000)


def budget_limited_certificate():
    result = certify_steady_trace(scalar_problem(), [0.1, 0.8, 0.1], rtol=1e-3, max_nodes=4)
    assert result.status == "budget_limit" and 0.2 < result.relative_gap < 0.3
    return result.certificate


def test_required_gap_is_decided_by_the_replay_not_by_the_certificate_labels(tmp_path):
    genuine = certify_steady_trace(scalar_problem(), [1e-8, 1 - 2e-8, 1e-8], rtol=1e-3, max_nodes=300).certificate
    report = verify_trace_certificate(genuine, required_rtol=PAPER_TARGET)
    assert report["valid"] and report["gap_achieved"] and report["target_met"] and report["required_rtol"] == "1/1000"
    lower, upper = Fraction(report["lower"]), Fraction(report["upper"])
    assert Fraction(report["relative_gap"]) == (upper - lower) / lower <= PAPER_TARGET
    assert not verify_trace_certificate(genuine, required_rtol=(upper - lower) / lower / 2)["target_met"]
    limited = budget_limited_certificate()
    report = verify_trace_certificate(limited, required_rtol=PAPER_TARGET)
    assert report["valid"] and not report["gap_achieved"] and not report["target_met"]
    assert Fraction(report["relative_gap"]) > PAPER_TARGET
    assert "target_met" not in verify_trace_certificate(limited)
    # Relabelling by hand, or by resuming the checkpoint with a loose tolerance, satisfies only the certificate itself.
    checkpoint = tmp_path / "checkpoint.json"
    certify_steady_trace(scalar_problem(), [0.1, 0.8, 0.1], rtol=1e-3, max_nodes=4, checkpoint_path=checkpoint)
    resumed = certify_steady_trace(scalar_problem(), rtol=1000.0, max_nodes=0, checkpoint_path=checkpoint)
    assert resumed.status == "certified_gap" and resumed.certificate["leaves"] == limited["leaves"]
    for relabelled in (dict(limited, status="certified_gap", rtol="1000"),
                       dict(limited, status="certified_gap", atol="1000"), resumed.certificate):
        assert verify_trace_certificate(relabelled)["gap_achieved"]
        report = verify_trace_certificate(relabelled, required_rtol=PAPER_TARGET)
        assert report["valid"] and not report["target_met"]
        assert verify_trace_certificate(relabelled, required_rtol=0.3)["target_met"]
    unbounded = verify_trace_certificate(
        certify_steady_trace(scalar_problem(), [0.1, 0.8, 0.1], max_nodes=0).certificate, required_rtol=1000)
    assert unbounded["valid"] and unbounded["relative_gap"] is None and not unbounded["target_met"]
    with pytest.raises(ValueError, match="positive"):
        verify_trace_certificate(genuine, required_rtol=0)


def test_replay_rejects_unknown_status_and_forged_subdivision_count():
    certificate = budget_limited_certificate()
    assert verify_trace_certificate(dict(certificate, status="running"))["valid"]
    for status in ("certified", "CERTIFIED_GAP", "no_feasible_incumbent", None):
        with pytest.raises(ValueError, match="status"):
            verify_trace_certificate(dict(certificate, status=status))
    with pytest.raises(ValueError, match="subdivision"):
        verify_trace_certificate(dict(certificate, subdivisions=0))
    with pytest.raises(ValueError, match="subdivision"):
        verify_trace_certificate({key: value for key, value in certificate.items() if key != "subdivisions"})


def test_a_failed_start_does_not_block_a_retry_from_new_weights(tmp_path):
    # The default initialization finds no feasible design for this plant, and the checkpoint records just that. A
    # retry from feasible weights used to replay that checkpoint first and fail: "no verified feasible incumbent".
    problem, checkpoint = slow_unobservable_problem(), tmp_path / "checkpoint.json"
    assert certify_steady_trace(problem, checkpoint_path=checkpoint).status == "no_feasible_incumbent"
    assert certify_steady_trace(problem, checkpoint_path=checkpoint).status == "no_feasible_incumbent"
    alpha = [0.99, 0.0099, 0.0001]
    retried = certify_steady_trace(problem, alpha, max_nodes=2000, checkpoint_path=checkpoint)
    fresh = certify_steady_trace(problem, alpha, max_nodes=2000)
    assert retried.status == "certified_gap"
    assert dict(retried.certificate, seconds=0) == dict(fresh.certificate, seconds=0)
    assert verify_trace_certificate(retried.certificate, required_rtol=PAPER_TARGET)["target_met"]
    assert certify_steady_trace(problem, max_nodes=0, checkpoint_path=checkpoint).status == "certified_gap"
    # Every other checkpoint is still replayed before it is used: one without an incumbent is not a failed start
    # unless it says so, and a failed start of another plant is no checkpoint of this one.
    checkpoint.write_text(json.dumps(dict(json.loads(checkpoint.read_text()), incumbent=None)))
    with pytest.raises(ValueError, match="incumbent"):
        certify_steady_trace(problem, alpha, checkpoint_path=checkpoint)
    other = GainOptimizedProblem(problem.A, problem.H, 2 * problem.Q, problem.R)
    assert certify_steady_trace(other, checkpoint_path=checkpoint, resume=False).status == "no_feasible_incumbent"
    with pytest.raises(ValueError, match="another problem"):
        certify_steady_trace(problem, alpha, checkpoint_path=checkpoint)


def test_certificate_cli_separates_short_of_target_from_invalid(tmp_path, monkeypatch, capsys):
    relabelled = dict(budget_limited_certificate(), status="certified_gap", rtol="1000")
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(relabelled))
    monkeypatch.setattr(sys, "argv", ["verify_certificate", str(path)])
    with pytest.raises(SystemExit) as stopped:
        replay_main()
    assert stopped.value.code == 3
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and report["required_rtol"] == "1/1000" and not report["target_met"]
    monkeypatch.setattr(sys, "argv", ["verify_certificate", str(path), "--rtol", "0.3"])
    replay_main()
    assert json.loads(capsys.readouterr().out)["target_met"]
    path.write_text(json.dumps(dict(relabelled, upper="0")))
    with pytest.raises(ValueError, match="upper bound"):
        replay_main()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_validation(folder, systems, cases):
    """Validation inputs as run_paper_validation.py writes them: the plants by case ID, and the case table."""
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "systems.npz", **systems)
    write_table(folder / "validation_cases.csv", cases)


def write_archive(folder, certificates):
    """Lay out validation inputs, certificates, their case table and the manifest like run_certified_evaluation.py."""
    (folder / "certificates").mkdir(parents=True)
    systems, cases, rows = {}, [], []
    for index, certificate in enumerate(certificates):
        record = certificate["problem"]
        for key in ("A", "H", "Q", "R"):
            systems[f"case_{index}_{key}"] = np.array([[float(Fraction(x)) for x in row] for row in record[key]])
        cases.append(dict(case=index, group="scalar", dimension=len(record["A"]), status="ok"))
        path = folder / "certificates" / f"case_{index}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(certificate, sort_keys=True).encode(), mtime=0))
        result = CertifiedTraceResult(certificate)
        rows.append(dict(case=index, group="scalar", dimension=len(record["A"]), status=result.status, verified=True,
                         lower=result.lower_bound, upper=result.upper_bound, relative_gap=result.relative_gap,
                         subdivisions=certificate["subdivisions"], problem_sha256=certificate["problem_sha256"],
                         certificate_sha256=digest(path)))
    write_validation(folder / "validation", systems, cases)
    write_certification_table(folder, rows)
    return write_manifest(folder, [f"certificates/case_{index}.json.gz" for index in range(len(rows))])


def write_certification_table(folder, rows):
    """The case table and the summary that counts it, as run_certified_evaluation.py writes them."""
    write_table(folder / "certification_cases.csv", rows)
    (folder / "certification_summary.json").write_text(json.dumps({
        "selected": len(rows), "retained": len(rows), "statuses": dict(Counter(str(row["status"]) for row in rows)),
        "verified": sum(str(row["verified"]) == "True" for row in rows)}))


def write_manifest(folder, certificates, cases=None, tables=("certification_cases.csv", "certification_summary.json")):
    """Manifest of a certification run of the selected cases (None: all) on the inputs in folder / "validation"."""
    manifest = folder / "certification_manifest.json"
    manifest.write_text(json.dumps({
        "arguments": {"validation_dir": str(folder / "validation"), "cases": cases},
        "inputs_sha256": {name: digest(folder / "validation" / name) for name in ("systems.npz", "validation_cases.csv")},
        "sources_changed_during_run": False,
        "outputs_sha256": {name: digest(folder / name) for name in [*certificates, *tables]},
    }))
    return manifest


def write_combined_archive(folder, plants, ratios=(0.1, 1.0)):
    """Lay out validation inputs, tasks, risk certificates and their table like run_combined_mse_evaluation.py."""
    (folder / "risk_certificates").mkdir(parents=True)
    systems, cases, combined, tasks, rows = {}, [], {}, [], []
    for index, (A, H, Q, R) in enumerate(plants):
        systems.update({f"case_{index}_{key}": value for key, value in zip("AHQR", (A, H, Q, R))})
        cases.append(dict(case=index, group="random_2d", dimension=A.shape[0], status="ok"))
        for ratio in ratios:
            case = len(tasks)
            problem = CombinedParetoProblem(A, H, ratio * Q, ratio * R, Q, R)
            for key in ("A", "H", "Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded"):
                combined[f"case_{case}_{key}"] = getattr(problem, key)
            alpha = np.full(3, 1 / 3)
            gain = solve_dare(GainOptimizedProblem(A, H, Q, R), alpha)[1]
            certificate = certify_combined_risk(problem, combined_mse_value_gradient(problem, gain, alpha))
            path = folder / "risk_certificates" / f"case_{case}.json.gz"
            path.write_bytes(gzip.compress(json.dumps(certificate, sort_keys=True).encode(), mtime=0))
            tasks.append(dict(case=case, archive_case=index, group="random_2d", dimension=A.shape[0],
                              stochastic_ratio=ratio))
            rows.append(dict(tasks[-1], status="ok", verified_risk_upper=float(Fraction(certificate["risk_upper"])),
                             risk_certificate_sha256=digest(path)))
    write_validation(folder / "validation", systems, cases)
    np.savez_compressed(folder / "combined_systems.npz", **combined)
    (folder / "selected_cases.json").write_text(json.dumps(tasks))
    write_table(folder / "combined_mse_cases.csv", rows)
    (folder / "combined_mse_summary.json").write_text(json.dumps({"selected": len(rows), "retained": len(rows),
                                                                  "failed": 0}))
    return write_combined_manifest(folder, ratios)


def write_combined_manifest(folder, ratios=(0.1, 1.0)):
    manifest = folder / "combined_mse_manifest.json"
    outputs = [path for path in sorted(folder.rglob("*")) if path.is_file() and path != manifest
               and path.relative_to(folder).parts[0] != "validation"]
    manifest.write_text(json.dumps({
        "arguments": {"validation_dir": str(folder / "validation"), "cases": None, "per_dimension": 10,
                      "ratios": list(ratios)},
        "inputs_sha256": {name: digest(folder / "validation" / name) for name in ("systems.npz", "validation_cases.csv")},
        "sources_changed_during_run": False,
        "outputs_sha256": {str(path.relative_to(folder)): digest(path) for path in outputs},
    }))
    return manifest


def run_archive_verifier(manifest, *arguments):
    script = Path(__file__).resolve().parents[1] / "examples" / "verify_expanded_archive.py"
    done = subprocess.run([sys.executable, str(script), str(manifest), "--workers", "1", *arguments],
                          capture_output=True, text=True)
    lines = [json.loads(line) for line in done.stdout.splitlines()]
    return done.returncode, lines, done.stderr


def test_archive_verifier_enforces_the_target_and_replays_every_cited_certificate(tmp_path):
    genuine = certify_steady_trace(scalar_problem(), [1e-8, 1 - 2e-8, 1e-8], rtol=1e-3, max_nodes=300).certificate
    relabelled = dict(budget_limited_certificate(), status="certified_gap", rtol="1000")
    code, lines, _ = run_archive_verifier(write_archive(tmp_path / "genuine", [genuine]))
    assert code == 0 and lines[-1]["trace_certificates"] == lines[-1]["table_rows_backed"] == 1
    assert lines[-1]["required_rtol"] == "1/1000" and lines[-1]["short_of_target"] == []
    # Valid bounds that are short of the target: neither a pass nor an invalid archive.
    manifest = write_archive(tmp_path / "relabelled", [genuine, relabelled])
    code, lines, error = run_archive_verifier(manifest)
    assert code == 3 and "do not prove a relative gap of 1/1000" in error
    assert all(line["valid"] for line in lines[:-1]) and lines[-1]["short_of_target"] == ["case_1.json.gz"]
    assert run_archive_verifier(manifest, "--rtol", "0.3")[0] == 0
    # A certificate that the table cites cannot be kept out of the replay by its file name.
    (manifest.parent / "certificates" / "case_1.json.gz").rename(manifest.parent / "certificates" / "case_1.gz")
    code, _, error = run_archive_verifier(write_manifest(manifest.parent, ["certificates/case_0.json.gz",
                                                                           "certificates/case_1.gz"]))
    assert code == 1 and "case 1 is not backed by a replayed certificate" in error
    # A table that reports something else than its certificate proves is rejected, whatever the number format.
    manifest = write_archive(tmp_path / "edited", [genuine])
    table = manifest.parent / "certification_cases.csv"
    row = next(csv.DictReader(table.open(newline="")))
    table.write_text(table.read_text().replace(row["lower"], f"{float(row['lower']):.17e}"))
    assert run_archive_verifier(write_manifest(manifest.parent, ["certificates/case_0.json.gz"]))[0] == 0
    table.write_text(table.read_text().replace(f"{float(row['lower']):.17e}", row["upper"]))
    code, _, error = run_archive_verifier(write_manifest(manifest.parent, ["certificates/case_0.json.gz"]))
    assert code == 1 and "case 0 disagrees with its certificate in ['lower']" in error




def two_plant_certificates():
    """Certificates of two plants that share group and dimension, so that only the plants tell their rows apart."""
    noisy = GainOptimizedProblem(np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)), np.array([[9.0]]))
    results = [certify_steady_trace(problem, [0.1, 0.8, 0.1], rtol=0.05, max_nodes=300) for problem in
               (scalar_problem(), noisy)]
    assert [result.status for result in results] == ["certified_gap"] * 2
    return [result.certificate for result in results]


def test_archive_verifier_binds_every_case_id_to_its_original_plant(tmp_path):
    manifest = write_archive(tmp_path / "archive", two_plant_certificates())
    folder, certificates = manifest.parent, [f"certificates/case_{index}.json.gz" for index in range(2)]
    code, lines, _ = run_archive_verifier(manifest, "--rtol", "0.05")
    assert code == 0 and lines[-1]["table_rows_backed"] == 2
    rows = list(csv.DictReader((folder / "certification_cases.csv").open(newline="")))

    def verify_table(rows, cases=None, tables=("certification_cases.csv", "certification_summary.json")):
        write_certification_table(folder, rows)
        return run_archive_verifier(write_manifest(folder, certificates, cases, tables), "--rtol", "0.05")

    # Swapped case IDs: every row still says what the certificate it cites proves, but for the other plant.
    code, _, error = verify_table([dict(rows[0], case="1"), dict(rows[1], case="0")])
    assert code == 1 and "case 1 cites a certificate of another plant" in error
    # One row for every selected case, and none for another.
    for table, cases, message in (([rows[0]], None, "missing [1]"), ([rows[0], rows[1], rows[1]], None, "duplicated [1]"),
                                  (rows, [1], "unexpected [0]")):
        code, _, error = verify_table(table, cases)
        assert code == 1 and message in error
    assert verify_table([rows[1]], [1])[0] == 0
    # A row describes its case as the validation archive does.
    code, _, error = verify_table([dict(rows[0], dimension="2"), rows[1]])
    assert code == 1 and "case 0 does not describe that case in ['dimension']" in error
    # Renaming the hash column does not take the table out of the check, and leaving the table out of the
    # manifest does not either.
    renamed = [{("hash" if key == "certificate_sha256" else key): value for key, value in row.items()} for row in rows]
    code, _, error = verify_table(renamed)
    assert code == 1 and "must cite its certificates in the column certificate_sha256" in error
    code, _, error = verify_table(rows, tables=())
    assert code == 1 and "no result table cites the trace certificates" in error
    # An error row reports no result, and the summary counts the rows of its table.
    code, _, error = verify_table([dict(rows[0], status="error", error="RuntimeError: lost"), rows[1]])
    assert code == 1 and "case 0 is an error row but reports" in error
    erased = dict.fromkeys(("lower", "upper", "relative_gap", "subdivisions", "problem_sha256", "certificate_sha256"), "")
    assert verify_table([dict(rows[0], status="error", error="RuntimeError: lost", verified="False", **erased),
                         rows[1]])[0] == 0
    write_certification_table(folder, rows)
    (folder / "certification_summary.json").write_text(json.dumps(
        {"selected": 2, "retained": 2, "statuses": {"certified_gap": 1, "error": 1}, "verified": 1}))
    code, _, error = run_archive_verifier(write_manifest(folder, certificates), "--rtol", "0.05")
    assert code == 1 and "certification_summary.json does not count the rows" in error
    # The plants come from the validation inputs that the run recorded, found where the manifest says or given.
    write_certification_table(folder, rows)
    manifest = write_manifest(folder, certificates)
    moved = (folder / "validation").rename(tmp_path / "elsewhere")
    code, _, error = run_archive_verifier(manifest, "--rtol", "0.05")
    assert code == 1 and "--validation-dir" in error
    assert run_archive_verifier(manifest, "--rtol", "0.05", "--validation-dir", str(moved))[0] == 0
    with np.load(moved / "systems.npz") as archive:
        systems = dict(archive)
    np.savez_compressed(moved / "systems.npz", **dict(systems, case_0_R=2 * systems["case_0_R"]))
    code, _, error = run_archive_verifier(manifest, "--rtol", "0.05", "--validation-dir", str(moved))
    assert code == 1 and "not the input that the run recorded" in error


def test_archive_verifier_binds_combined_rows_to_the_plants_of_their_tasks(tmp_path):
    plants = [(0.6 * np.eye(2), np.eye(2), np.eye(2), np.eye(2)),
              (np.array([[0.5, 0.1], [0.0, 0.7]]), np.array([[1.0, 0.0]]), np.diag([1.0, 2.0]), np.array([[0.5]]))]
    # Tasks 0 and 1 reuse archive case 0 at the ratios 0.1 and 1, tasks 2 and 3 archive case 1.
    manifest = write_combined_archive(tmp_path / "combined", plants)
    folder = manifest.parent
    code, lines, _ = run_archive_verifier(manifest)
    assert code == 0 and lines[-1]["table_rows_backed"] == 4 and lines[-1]["trace_certificates"] == 0
    rows = list(csv.DictReader((folder / "combined_mse_cases.csv").open(newline="")))
    tasks = json.loads((folder / "selected_cases.json").read_text())
    with np.load(folder / "combined_systems.npz") as archive:
        systems = dict(archive)

    def verify(rows=rows, tasks=tasks, systems=systems):
        write_table(folder / "combined_mse_cases.csv", rows)
        (folder / "selected_cases.json").write_text(json.dumps(tasks))
        np.savez_compressed(folder / "combined_systems.npz", **systems)
        return run_archive_verifier(write_combined_manifest(folder))

    # A row that cites the risk certificate of another task, with that task's bound, or that claims its case ID.
    cited = {key: rows[2][key] for key in ("verified_risk_upper", "risk_certificate_sha256")}
    code, _, error = verify([dict(rows[0], **cited), *rows[1:]])
    assert code == 1 and "case 0 cites a certificate of another plant" in error
    code, _, error = verify([dict(rows[0], case="2"), rows[1], dict(rows[2], case="0"), rows[3]])
    assert code == 1 and "does not describe that case in ['archive_case']" in error
    # The ratio labels belong to the tasks, whether swapped in the table, also in the task list, or a task dropped.
    swapped = [dict(rows[0], stochastic_ratio=rows[1]["stochastic_ratio"]),
               dict(rows[1], stochastic_ratio=rows[0]["stochastic_ratio"]), *rows[2:]]
    code, _, error = verify(swapped)
    assert code == 1 and "case 0 does not describe that case in ['stochastic_ratio']" in error
    relabelled = [dict(tasks[0], stochastic_ratio=1.0), dict(tasks[1], stochastic_ratio=0.1), *tasks[2:]]
    for table, listed in ((swapped, relabelled), (rows[:3], tasks[:3])):
        code, _, error = verify(table, listed)
        assert code == 1 and "selected_cases.json does not list the tasks that the recorded arguments select" in error
    # Every task reuses the original set-membership plant of its archive case, with descriptors scaled by its ratio.
    exchanged = dict(systems)
    for key in ("Q_stochastic", "R_stochastic"):
        exchanged[f"case_0_{key}"], exchanged[f"case_1_{key}"] = systems[f"case_1_{key}"], systems[f"case_0_{key}"]
    code, _, error = verify(systems=exchanged)
    assert code == 1 and "task 0 does not scale the Q of archive case 0 by 0.1" in error
    code, _, error = verify(systems=dict(systems, case_0_A=0.5 * systems["case_0_A"]))
    assert code == 1 and "task 0 does not use the A of archive case 0" in error
    # The result table must cite the risk certificates, and must not be missing.
    code, _, error = verify([{("hash" if key == "risk_certificate_sha256" else key): value for key, value in row.items()}
                             for row in rows])
    assert code == 1 and "must cite its certificates in the column risk_certificate_sha256" in error
    verify()
    (folder / "combined_mse_cases.csv").unlink()
    code, _, error = run_archive_verifier(write_combined_manifest(folder))
    assert code == 1 and "no result table that cites its risk certificates" in error
