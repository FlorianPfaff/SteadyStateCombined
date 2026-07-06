from __future__ import annotations

import numpy as np

from steady_state_combined import FixedGainProblem, solve_fixed_gain_steady_state, spectral_radius
from steady_state_combined.ellipsoidal import sym


def random_spd(rng: np.random.Generator, n: int, shift: float = 0.5) -> np.ndarray:
    M = rng.normal(size=(n, n))
    return sym(M @ M.T + shift * np.eye(n))


def block_lmi_matrix(A: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, X: np.ndarray, Y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Block LMI from the invariant-ellipsoid formulation.

    X=P^{-1}, Y=XK, and mu contains the three positive multipliers.
    """

    n = A.shape[0]
    m = H.shape[0]
    Z_nn = np.zeros((n, n))
    Z_nm = np.zeros((n, m))
    Z_mn = np.zeros((m, n))
    M = X - Y @ H
    return sym(
        np.block(
            [
                [mu[0] * X, Z_nn, Z_nm, A.T @ M.T],
                [Z_nn, mu[1] * np.linalg.inv(Q), Z_nm, M.T],
                [Z_mn, Z_mn, mu[2] * np.linalg.inv(R), -Y.T],
                [M @ A, M, -Y, X],
            ]
        )
    )


def direct_invariance_residual(
    A: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Return P - Phi_K(P) for the direct matrix inequality."""

    P = np.linalg.inv(X)
    K = P @ Y
    L = np.eye(A.shape[0]) - K @ H
    rhs = (L @ A @ P @ A.T @ L.T) / mu[0] + (L @ Q @ L.T) / mu[1] + (K @ R @ K.T) / mu[2]
    return sym(P - rhs)


def is_psd(M: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.min(np.linalg.eigvalsh(sym(M))) >= -tol)


def test_block_lmi_matches_direct_invariance_inequality() -> None:
    rng = np.random.default_rng(12)
    n = 3
    m = 2
    mu = np.array([0.55, 0.25, 0.20])
    for _ in range(80):
        A = 0.8 * rng.normal(size=(n, n)) / np.sqrt(n)
        H = rng.normal(size=(m, n))
        Q = random_spd(rng, n)
        R = random_spd(rng, m)
        X = random_spd(rng, n)
        Y = rng.normal(size=(n, m))
        lmi_ok = is_psd(block_lmi_matrix(A, H, Q, R, X, Y, mu))
        direct_ok = is_psd(direct_invariance_residual(A, H, Q, R, X, Y, mu))
        assert lmi_ok == direct_ok


def solve_scaled_dare(A: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the fixed-alpha Riccati equation by monotone iteration."""

    a0, aw, av = 1.0 / alpha[0], 1.0 / alpha[1], 1.0 / alpha[2]
    P = np.zeros_like(Q)
    for _ in range(50_000):
        S = a0 * A @ P @ A.T + aw * Q
        innovation = H @ S @ H.T + av * R
        P_next = sym(S - S @ H.T @ np.linalg.solve(innovation, H @ S))
        if np.linalg.norm(P_next - P, ord="fro") <= 1e-12 * (1.0 + np.linalg.norm(P, ord="fro")):
            P = P_next
            break
        P = P_next
    S = a0 * A @ P @ A.T + aw * Q
    K = S @ H.T @ np.linalg.inv(H @ S @ H.T + av * R)
    return sym(P), K


def test_scaled_dare_gain_matches_fixed_gain_steady_state_and_dominates_other_gain() -> None:
    A = np.array([[0.89, -0.11], [0.11, 0.89]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.0051, -0.0052], [-0.0052, 0.0061]])
    R = np.array([[0.077]])
    alpha = np.array([0.70, 0.20, 0.10])

    P_star, K_star = solve_scaled_dare(A, H, Q, R, alpha)
    optimal_problem = FixedGainProblem(A=A, H=H, K=K_star, Q=Q, R=R)
    P_from_fixed_gain = solve_fixed_gain_steady_state(optimal_problem, alpha)
    assert P_from_fixed_gain is not None
    assert np.linalg.norm(P_from_fixed_gain - P_star, ord="fro") < 1e-8

    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(200):
        K = K_star + 0.35 * rng.normal(size=K_star.shape)
        problem = FixedGainProblem(A=A, H=H, K=K, Q=Q, R=R)
        if spectral_radius(problem.F) / np.sqrt(alpha[0]) >= 1.0:
            continue
        P_K = solve_fixed_gain_steady_state(problem, alpha)
        if P_K is None:
            continue
        checked += 1
        assert is_psd(P_K - P_star, tol=2e-8)
    assert checked >= 20
