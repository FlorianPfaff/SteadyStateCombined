"""Verified Riccati subsolutions, with a feasible supersolution witness."""

from fractions import Fraction

from flint import ctx
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from .ellipsoidal import sym
from .verified import ball, decode_matrix, encode_matrix, identity, positive_definite, rational


def verified_riccati_lower(record, weights, precision=128, *, schur_check=False):
    """Return a rigorous trace lower bound at arbitrary positive weights.

    SciPy only proposes witnesses. Arb verifies B > 0, B <= R(B), and
    U >= Phi_K(U), U > 0. These imply B <= the minimal Riccati fixed point.
    The replay constructs its own witnesses and uses the Schur update.
    """
    if len(weights) != 3 or min(map(rational, weights)) <= 0:
        raise ValueError("strictly positive weights required")
    a = np.array(list(map(float, weights)))
    A, H, Q, R = (np.array([[float(rational(x)) for x in row] for row in record[name]])
                  for name in ("A", "H", "Q", "R"))
    n = len(A)
    prior = solve_discrete_are(A.T / np.sqrt(a[0]), H.T, Q / a[1], R / a[2])
    K = np.linalg.solve(H @ prior @ H.T + R / a[2], H @ prior).T
    L = np.eye(n) - K @ H
    P = sym(L @ prior @ L.T + K @ R @ K.T / a[2])
    direction = sym(solve_discrete_lyapunov(L @ A / np.sqrt(a[0]), np.eye(n)))
    scale = max(np.linalg.norm(P, ord=2), np.finfo(float).tiny)
    with ctx.workprec(precision):
        exact_A, exact_H, exact_Q, exact_R = (decode_matrix(record[name]) for name in ("A", "H", "Q", "R"))
        exact_K = decode_matrix(encode_matrix(K))
        exact_L = identity(n) - exact_K * exact_H
        exact_F = exact_L * exact_A
        C = exact_L * exact_Q * exact_L.transpose() / ball(weights[1])
        C += exact_K * exact_R * exact_K.transpose() / ball(weights[2])
        for exponent in range(-13, -2):
            lower_record = encode_matrix(sym(P - scale * 10.0**exponent * direction))
            upper_record = encode_matrix(sym(P + scale * 10.0**exponent * direction))
            B, U = decode_matrix(lower_record), decode_matrix(upper_record)
            if not positive_definite(B) or not positive_definite(U):
                continue
            if not positive_definite(U - exact_F * U * exact_F.transpose() / ball(weights[0]) - C):
                continue
            before = exact_A * B * exact_A.transpose() / ball(weights[0]) + exact_Q / ball(weights[1])
            if schur_check:
                cross = before * exact_H.transpose()
                after = before - cross * (exact_H * cross + exact_R / ball(weights[2])).solve(cross.transpose())
            else:
                after = (before.inv() + exact_H.transpose() * exact_R.inv() * exact_H * ball(weights[2])).inv()
            if not positive_definite(after - B):
                continue
            lower = sum((rational(lower_record[i][i]) for i in range(n)), Fraction(0))
            return lower, exponent
    raise ValueError("no verified Riccati subsolution/supersolution bracket")
