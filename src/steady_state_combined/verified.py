"""Small verified-arithmetic kernels, with exact rational serialization.

Arb operations enclose the exact operations on the stored binary matrices.
No decimal display string is used as a numerical endpoint.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from flint import arb, arb_mat, ctx, fmpq
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .ellipsoidal import sym
from .riccati import GainOptimizedProblem


def rational(value) -> Fraction:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("finite input required")
        return Fraction.from_float(float(value))
    return Fraction(value)


def ball(value) -> arb:
    q = rational(value)
    return arb(fmpq(q.numerator, q.denominator))


def endpoint(value: arb, upper: bool = False) -> Fraction:
    bound = value.upper() if upper else value.lower()
    mantissa, exponent = bound.man_exp()
    return Fraction(int(mantissa)) * Fraction(2) ** int(exponent)


def encode_matrix(matrix) -> list[list[str]]:
    return [[str(rational(x)) for x in row] for row in matrix]


def decode_matrix(matrix) -> arb_mat:
    return arb_mat([[ball(x) for x in row] for row in matrix])


def problem_record(problem: GainOptimizedProblem) -> dict:
    return {name: encode_matrix(getattr(problem, name)) for name in ("A", "H", "Q", "R")}


def record_hash(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def trace(matrix: arb_mat) -> arb:
    return sum((matrix[i, i] for i in range(matrix.nrows())), arb(0))


def identity(n: int) -> arb_mat:
    return arb_mat([[int(i == j) for j in range(n)] for i in range(n)])


def positive_definite(matrix: arb_mat) -> bool:
    """Verified LDL pivots for a known symmetric matrix enclosure.

The caller must establish exact symmetry independently. An inconclusive
interval pivot returns False; it never certifies indefiniteness.
"""
    n = matrix.nrows()
    if matrix.ncols() != n:
        return False
    lower = [[arb(0) for _ in range(n)] for _ in range(n)]
    diagonal = []
    for i in range(n):
        pivot = matrix[i, i] - sum((lower[i][k] ** 2 * diagonal[k] for k in range(i)), arb(0))
        if not pivot > 0:
            return False
        diagonal.append(pivot)
        for j in range(i + 1, n):
            numerator = matrix[j, i] - sum(
                (lower[j][k] * lower[i][k] * diagonal[k] for k in range(i)), arb(0)
            )
            lower[j][i] = numerator / pivot
    return True


def validate_problem(record: dict, precision: int = 128) -> None:
    arrays = {name: [[rational(x) for x in row] for row in record[name]] for name in ("A", "H", "Q", "R")}
    n = len(arrays["A"])
    m = len(arrays["H"])
    if n < 1 or m < 1:
        raise ValueError("empty system")
    for name, shape in (("A", (n, n)), ("H", (m, n)), ("Q", (n, n)), ("R", (m, m))):
        a = arrays[name]
        if len(a) != shape[0] or any(len(row) != shape[1] for row in a):
            raise ValueError(f"invalid {name} dimensions")
    with ctx.workprec(precision):
        for name in ("Q", "R"):
            a = arrays[name]
            if any(a[i][j] != a[j][i] for i in range(len(a)) for j in range(i)):
                raise ValueError(f"{name} must be exactly symmetric")
            if not positive_definite(decode_matrix(record[name])):
                raise ValueError(f"{name} not verified positive definite at {precision} bits")


def finite_riccati_trace(record: dict, weights, iterations: int, precision: int = 128, *, schur: bool = False) -> arb:
    """Enclose a finite Riccati iterate from zero, without normalizing weights.

The information form is used by the search; the independent verifier uses
the Schur form. Both are valid for arbitrary positive weights.
"""
    if iterations < 0 or any(rational(a) <= 0 for a in weights) or len(weights) != 3:
        raise ValueError("invalid weights or iteration count")
    with ctx.workprec(precision):
        A, H, Q, R = (decode_matrix(record[k]) for k in ("A", "H", "Q", "R"))
        a0, aw, av = map(ball, weights)
        P = arb_mat(A.nrows(), A.nrows())
        information = H.transpose() * R.inv() * H * av if not schur else None
        for _ in range(iterations):
            prior = A * P * A.transpose() / a0 + Q / aw
            if schur:
                cross = prior * H.transpose()
                P = prior - cross * (H * cross + R / av).solve(cross.transpose())
            else:
                P = (prior.inv() + information).inv()
            P = (P + P.transpose()) / 2
        result = trace(P)
        if not result.is_finite():
            raise ValueError("nonfinite Riccati enclosure")
        return result


def rounded_riccati_trace(record: dict, weights, iterations: int, precision: int = 128, *, verify_steps=False) -> arb:
    """Enclose trace of a PSD Riccati subiteration with verified downward rounding.

    Resetting interval widths prevents dependency overestimation from
    accumulating over long horizons. The independent replay additionally
    verifies every transition with the Schur update at higher precision.
    """
    if iterations < 0 or len(weights) != 3 or any(rational(a) <= 0 for a in weights):
        raise ValueError("invalid weights or iteration count")
    with ctx.workprec(precision):
        A, H, Q, R = (decode_matrix(record[k]) for k in ("A", "H", "Q", "R"))
        a0, aw, av = map(ball, weights)
        n = A.nrows()
        previous = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        information = H.transpose() * R.inv() * H * av
        for _ in range(iterations):
            P = decode_matrix(previous)
            prior = A * P * A.transpose() / a0 + Q / aw
            enclosure = (prior.inv() + information).inv()
            enclosure = (enclosure + enclosure.transpose()) / 2
            midpoint = [[endpoint(enclosure[i, j].mid()) for j in range(n)] for i in range(n)]
            radius = max(sum(endpoint(abs(enclosure[i, j] - ball(midpoint[i][j])), upper=True)
                             for j in range(n)) for i in range(n))
            scale = max(abs(midpoint[i][i]) for i in range(n))
            shift = 4 * radius + scale * Fraction(2) ** (-precision // 2)
            following = [[midpoint[i][j] - (shift if i == j else 0) for j in range(n)] for i in range(n)]
            next_matrix = decode_matrix(following)
            if not positive_definite(next_matrix):
                raise ValueError("rounded Riccati subiterate not verified positive definite")
            if verify_steps:
                with ctx.workprec(precision * 2):
                    exact_A, exact_H, exact_Q, exact_R = (decode_matrix(record[k]) for k in ("A", "H", "Q", "R"))
                    check_prior = exact_A * decode_matrix(previous) * exact_A.transpose() / ball(weights[0])
                    check_prior += exact_Q / ball(weights[1])
                    cross = check_prior * exact_H.transpose()
                    check = check_prior - cross * (exact_H * cross + exact_R / ball(weights[2])).solve(cross.transpose())
                    if not positive_definite(check - decode_matrix(following)):
                        raise ValueError("rounded Riccati transition not independently verified")
            previous = following
        return ball(sum((previous[i][i] for i in range(n)), Fraction(0)))


def invariant_residual(record: dict, candidate: dict) -> tuple[arb_mat, arb_mat]:
    a = [rational(x) for x in candidate["alpha"]]
    if len(a) != 3 or min(a) <= 0 or sum(a) != 1:
        raise ValueError("candidate weights must lie exactly on the open simplex")
    A, H, Q, R = (decode_matrix(record[k]) for k in ("A", "H", "Q", "R"))
    K, P = (decode_matrix(candidate[k]) for k in ("K", "P"))
    n, m = A.nrows(), H.nrows()
    if K.nrows() != n or K.ncols() != m or P.nrows() != n or P.ncols() != n:
        raise ValueError("candidate matrix dimensions are invalid")
    exact_P = [[rational(x) for x in row] for row in candidate["P"]]
    if any(exact_P[i][j] != exact_P[j][i] for i in range(n) for j in range(i)):
        raise ValueError("candidate shape must be exactly symmetric")
    L = identity(n) - K * H
    F = L * A
    residual = P - F * P * F.transpose() / ball(a[0]) - L * Q * L.transpose() / ball(a[1])
    residual -= K * R * K.transpose() / ball(a[2])
    return P, residual


def verify_invariant_candidate(record: dict, candidate: dict, precision: int = 128) -> Fraction:
    with ctx.workprec(precision):
        P, residual = invariant_residual(record, candidate)
        if not positive_definite(P) or not positive_definite(residual):
            raise ValueError("invariant shape/residual not verified positive definite")
    return sum((rational(candidate["P"][i][i]) for i in range(len(candidate["P"]))), Fraction(0))


def enclose_candidate(problem: GainOptimizedProblem, alpha, K, P, precision: int = 128) -> dict:
    """Inflate a floating-point solution, then verify the exact candidate.

    A numerical Lyapunov solution with identity forcing proposes an
    inflation direction. Only the enclosed residual decides feasibility.
    """
    a = [rational(alpha[0]), rational(alpha[1])]
    a.append(1 - sum(a))
    if min(a) <= 0:
        raise ValueError("candidate is not interior")
    F = (np.eye(problem.n) - K @ problem.H) @ problem.A
    direction = sym(solve_discrete_lyapunov(F / np.sqrt(float(a[0])), np.eye(problem.n)))
    scale = max(float(np.linalg.norm(P, ord=2)), np.finfo(float).tiny)
    record = problem_record(problem)
    for exponent in range(-13, 2):
        enlarged = sym(P + scale * 10.0**exponent * direction)
        candidate = {"alpha": list(map(str, a)), "K": encode_matrix(K), "P": encode_matrix(enlarged)}
        try:
            upper = verify_invariant_candidate(record, candidate, precision)
        except ValueError:
            continue
        candidate.update(upper=str(upper), precision=precision)
        return candidate
    raise ValueError("unable to verify an inflated invariant candidate")
