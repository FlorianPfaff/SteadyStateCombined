"""Gain-reoptimized Riccati sweeps for fixed approximation weights.

For fixed approximation weights alpha, the ellipsoidal steady-state update can
be written as the Kalman/Riccati equation of the scaled artificial system

    S = A P A^T / alpha_0 + Q / alpha_w,
    P = S - S H^T (H S H^T + R / alpha_v)^{-1} H S.

This module evaluates that fixed-alpha Riccati equation and searches over the
2D simplex of alpha values. It complements the fixed-gain theorem implemented
in :mod:`steady_state_combined.ellipsoidal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np

from .ellipsoidal import criterion_value, simplex_grid, spectral_radius, sym

Criterion = Literal["trace", "logdet"]


@dataclass(frozen=True)
class GainOptimizedProblem:
    """LTI ellipsoidal set-membership problem without a prescribed gain."""

    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    name: str = "gain_optimized_problem"

    def __post_init__(self) -> None:
        object.__setattr__(self, "A", np.asarray(self.A, dtype=float))
        object.__setattr__(self, "H", np.asarray(self.H, dtype=float))
        object.__setattr__(self, "Q", sym(np.asarray(self.Q, dtype=float)))
        object.__setattr__(self, "R", sym(np.asarray(self.R, dtype=float)))
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if self.H.shape[1] != n:
            raise ValueError("H must have n columns")
        if self.Q.shape != (n, n):
            raise ValueError("Q must have shape (n, n)")
        if self.R.shape != (self.H.shape[0], self.H.shape[0]):
            raise ValueError("R must have shape (m, m)")

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])


@dataclass(frozen=True)
class RiccatiResult:
    """Result of a fixed-alpha Riccati solve."""

    alpha: np.ndarray
    P: np.ndarray
    K: np.ndarray
    value: float
    iterations: int
    converged: bool


def _validate_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float).reshape(3)
    if np.any(alpha <= 0.0) or not np.all(np.isfinite(alpha)):
        raise ValueError("alpha must be finite and strictly positive")
    return alpha / float(np.sum(alpha))


def riccati_update(problem: GainOptimizedProblem, P: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One posterior Riccati update for a fixed approximation-weight vector."""

    alpha = _validate_alpha(alpha)
    S = sym(problem.A @ P @ problem.A.T / alpha[0] + problem.Q / alpha[1])
    innovation = sym(problem.H @ S @ problem.H.T + problem.R / alpha[2])
    try:
        K = np.linalg.solve(innovation, problem.H @ S).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("innovation matrix is singular") from exc
    P_next = sym(S - S @ problem.H.T @ np.linalg.solve(innovation, problem.H @ S))
    return P_next, K


def solve_fixed_alpha_riccati(
    problem: GainOptimizedProblem,
    alpha: np.ndarray,
    criterion: Criterion = "trace",
    max_iter: int = 50_000,
    tol: float = 1e-12,
    reject_unstable: bool = True,
) -> Optional[RiccatiResult]:
    """Solve the fixed-alpha Riccati equation by monotone iteration.

    The function returns ``None`` if the iteration diverges, the innovation
    matrix becomes singular, or the resulting closed loop fails the fixed-alpha
    stability check ``rho((I-KH)A)/sqrt(alpha_0)<1``.
    """

    alpha = _validate_alpha(alpha)
    P = np.zeros((problem.n, problem.n), dtype=float)
    converged = False
    K = np.zeros((problem.n, problem.m), dtype=float)
    for iteration in range(1, max_iter + 1):
        try:
            P_next, K = riccati_update(problem, P, alpha)
        except ValueError:
            return None
        if not np.all(np.isfinite(P_next)) or np.linalg.norm(P_next, ord="fro") > 1e14:
            return None
        rel = np.linalg.norm(P_next - P, ord="fro") / (1.0 + np.linalg.norm(P, ord="fro"))
        P = P_next
        if rel <= tol:
            converged = True
            break
    if not converged:
        return None
    if np.min(np.linalg.eigvalsh(P)) <= -1e-8:
        return None
    F = (np.eye(problem.n) - K @ problem.H) @ problem.A
    if reject_unstable and spectral_radius(F) / np.sqrt(alpha[0]) >= 1.0 - 1e-9:
        return None
    return RiccatiResult(
        alpha=alpha,
        P=P,
        K=K,
        value=criterion_value(P, criterion),
        iterations=iteration,
        converged=True,
    )


def grid_optimize_riccati(
    problem: GainOptimizedProblem,
    resolution: int = 121,
    criterion: Criterion = "trace",
) -> Optional[RiccatiResult]:
    """Search the open simplex for the best fixed-alpha Riccati solution."""

    best: Optional[RiccatiResult] = None
    for alpha in simplex_grid(resolution):
        result = solve_fixed_alpha_riccati(problem, alpha, criterion=criterion)
        if result is None:
            continue
        if best is None or result.value < best.value:
            best = result
    return best


def refine_riccati_weights(
    problem: GainOptimizedProblem,
    alpha_start: np.ndarray,
    criterion: Criterion = "trace",
    initial_step: float = 0.02,
    min_step: float = 1e-7,
    max_evaluations: int = 2000,
) -> Optional[RiccatiResult]:
    """Refine a fixed-alpha DARE solution by simplex pattern search.

    The six directions transfer mass between pairs of simplex coordinates.
    Only improving feasible points are accepted, which makes this suitable for
    seeding with the converged greedy baseline as well as with a grid point.
    """

    best = solve_fixed_alpha_riccati(problem, alpha_start, criterion=criterion)
    if best is None:
        return None

    directions = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    step = float(initial_step)
    evaluations = 0
    while step >= min_step and evaluations < max_evaluations:
        candidates: list[RiccatiResult] = []
        for direction in directions:
            alpha = best.alpha + step * direction
            if np.min(alpha) <= 1e-10:
                continue
            result = solve_fixed_alpha_riccati(problem, alpha, criterion=criterion)
            evaluations += 1
            if result is not None:
                candidates.append(result)
            if evaluations >= max_evaluations:
                break

        candidate = min(candidates, key=lambda result: result.value, default=None)
        decrease_tol = 1e-13 * (1.0 + abs(best.value))
        if candidate is not None and candidate.value < best.value - decrease_tol:
            best = candidate
        else:
            step *= 0.5
    return best


def optimize_riccati_weights(
    problem: GainOptimizedProblem,
    resolution: int = 121,
    criterion: Criterion = "trace",
    initial_alphas: Iterable[np.ndarray] = (),
) -> Optional[RiccatiResult]:
    """Combine a simplex grid with baseline-seeded local DARE refinement."""

    starts: list[np.ndarray] = []
    grid_result = grid_optimize_riccati(problem, resolution=resolution, criterion=criterion)
    if grid_result is not None:
        starts.append(grid_result.alpha)
    starts.extend(_validate_alpha(alpha) for alpha in initial_alphas)

    best: Optional[RiccatiResult] = None
    initial_step = max(1.0 / resolution, 1e-3)
    for alpha in starts:
        result = refine_riccati_weights(
            problem,
            alpha,
            criterion=criterion,
            initial_step=initial_step,
        )
        if result is not None and (best is None or result.value < best.value):
            best = result
    return best


def stepwise_gain_trace_recursion(
    problem: GainOptimizedProblem,
    resolution: int = 81,
    criterion: Criterion = "trace",
    initial_P: Optional[np.ndarray] = None,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> Optional[RiccatiResult]:
    """Recursive myopic baseline with gain reoptimization in every step.

    At each iteration, this function searches over alpha for the one-step
    posterior shape obtained from ``riccati_update(problem, P_k, alpha)`` and
    selects the alpha/gain minimizing the current-step criterion. The resulting
    limiting value is the gain-reoptimized analogue of the greedy baseline.
    """

    P = np.eye(problem.n) if initial_P is None else sym(initial_P)
    best_alpha = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    best_K = np.zeros((problem.n, problem.m), dtype=float)
    best_value = float("inf")
    for iteration in range(1, max_iter + 1):
        best_step: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = None
        for alpha in simplex_grid(resolution):
            try:
                P_next, K = riccati_update(problem, P, alpha)
            except ValueError:
                continue
            if not np.all(np.isfinite(P_next)):
                continue
            value = criterion_value(P_next, criterion)
            if best_step is None or value < best_step[3]:
                best_step = (alpha, P_next, K, value)
        if best_step is None:
            return None
        best_alpha, P_next, best_K, best_value = best_step
        rel = np.linalg.norm(P_next - P, ord="fro") / (1.0 + np.linalg.norm(P, ord="fro"))
        P = sym(P_next)
        if rel <= tol:
            F = (np.eye(problem.n) - best_K @ problem.H) @ problem.A
            if spectral_radius(F) / np.sqrt(best_alpha[0]) >= 1.0 - 1e-9:
                return None
            return RiccatiResult(
                alpha=best_alpha,
                P=P,
                K=best_K,
                value=best_value,
                iterations=iteration,
                converged=True,
            )
    return None


def deterministic_gain_optimized_problem() -> GainOptimizedProblem:
    """Use the same matrices as the fixed-gain deterministic example, without K."""

    A = np.array([[0.87808946, -0.10955969], [0.10955969, 0.87808946]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)
    Q = np.array([[0.00510331, -0.00519052], [-0.00519052, 0.00607184]], dtype=float)
    R = np.array([[0.07701328]], dtype=float)
    return GainOptimizedProblem(A=A, H=H, Q=Q, R=R, name="deterministic_gain_optimized_2d")
