"""Verified feasible combined-risk bounds, without a global optimality claim."""

from fractions import Fraction

from flint import ctx, fmpq, fmpq_mat
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .ellipsoidal import sym
from .riccati import GainOptimizedProblem
from .verified import (
    ball, decode_matrix, encode_matrix, enclose_candidate, identity, positive_definite,
    rational, record_hash, verify_invariant_candidate,
)


def _exact_psd(matrix):
    """Symmetric rational PSD test via coefficients of det(t I + M)."""
    values = [[rational(x) for x in row] for row in matrix]
    n = len(values)
    if not n or any(len(row) != n for row in values):
        return False
    if any(values[i][j] != values[j][i] for i in range(n) for j in range(i)):
        return False
    exact = fmpq_mat([[fmpq(x.numerator, x.denominator) for x in row] for row in values])
    coefficients = exact.charpoly()
    return all(coefficients[k] * (-1) ** (n - k) >= 0 for k in range(n + 1))


def verify_combined_risk_certificate(certificate):
    if certificate["schema"] != "steady_state_combined.feasible_risk.v1":
        raise ValueError("unknown combined-risk certificate")
    record = certificate["problem"]
    if record_hash(record) != certificate["problem_sha256"]:
        raise ValueError("combined problem hash mismatch")
    for name in ("Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded"):
        if not _exact_psd(record[name]):
            raise ValueError(f"{name} is not exactly positive semidefinite")
    bounded = {"A": record["A"], "H": record["H"], "Q": record["Q_bounded"], "R": record["R_bounded"]}
    candidate = certificate["bounded_candidate"]
    precision = int(certificate["precision"])
    verify_invariant_candidate(bounded, candidate, precision)
    with ctx.workprec(precision):
        A, H = decode_matrix(record["A"]), decode_matrix(record["H"])
        K = decode_matrix(candidate["K"])
        L = identity(A.nrows()) - K * H
        F = L * A
        sigma_record = certificate["Sigma"]
        if any(rational(sigma_record[i][j]) != rational(sigma_record[j][i])
               for i in range(A.nrows()) for j in range(i)):
            raise ValueError("covariance bound is not symmetric")
        Sigma = decode_matrix(sigma_record)
        residual = Sigma - F * Sigma * F.transpose()
        residual -= L * decode_matrix(record["Q_stochastic"]) * L.transpose()
        residual -= K * decode_matrix(record["R_stochastic"]) * K.transpose()
        if not positive_definite(Sigma) or not positive_definite(residual):
            raise ValueError("covariance supersolution not verified")
        spectral_upper = rational(certificate["bias_squared_upper"])
        if not positive_definite(identity(A.nrows()) * ball(spectral_upper) - decode_matrix(candidate["P"])):
            raise ValueError("spectral bias bound not verified")
    risk = spectral_upper + sum((rational(sigma_record[i][i]) for i in range(len(sigma_record))), Fraction(0))
    if risk != rational(certificate["risk_upper"]):
        raise ValueError("combined-risk upper bound is inconsistent")
    return {"valid": True, "risk_upper": str(risk), "global_optimality_claim": False}


def certify_combined_risk(problem, evaluation, precision=128):
    record = {name: encode_matrix(getattr(problem, name)) for name in
              ("A", "H", "Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded")}
    bounded = GainOptimizedProblem(problem.A, problem.H, problem.Q_bounded, problem.R_bounded)
    candidate = enclose_candidate(bounded, evaluation.alpha, evaluation.K, evaluation.P, precision)
    F = (np.eye(problem.n) - evaluation.K @ problem.H) @ problem.A
    direction = sym(solve_discrete_lyapunov(F, np.eye(problem.n)))
    covariance_scale = max(np.linalg.norm(evaluation.Sigma, ord=2), np.finfo(float).tiny)
    P_upper = np.array([[float(rational(x)) for x in row] for row in candidate["P"]])
    spectral = float(np.linalg.eigvalsh(P_upper)[-1])
    for exponent in range(-13, 1):
        Sigma = sym(evaluation.Sigma + covariance_scale * 10.0**exponent * direction)
        sigma_record = encode_matrix(Sigma)
        spectral_upper = rational(spectral + max(spectral, np.finfo(float).tiny) * 10.0**exponent)
        risk = spectral_upper + sum((rational(sigma_record[i][i]) for i in range(problem.n)), Fraction(0))
        certificate = {
            "schema": "steady_state_combined.feasible_risk.v1", "problem": record,
            "problem_sha256": record_hash(record), "bounded_candidate": candidate,
            "Sigma": sigma_record, "bias_squared_upper": str(spectral_upper),
            "risk_upper": str(risk), "precision": precision,
        }
        try:
            verify_combined_risk_certificate(certificate)
            return certificate
        except ValueError:
            continue
    raise ValueError("unable to verify combined-risk supersolutions")
