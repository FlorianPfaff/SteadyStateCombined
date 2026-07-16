"""Grid-search Pareto experiment for combined stochastic/set-membership filtering.

This module deliberately uses a low-dimensional deterministic grid search.
It is intended as an evaluation tool for the paper, not as a claim that the
combined gain/weight design problem is convex.

For a fixed gain K and approximation weights alpha, it solves

    Sigma = F Sigma F^T + G_w Q_s G_w^T + G_v R_s G_v^T,
    P     = F P F^T / alpha_0 + G_w Q_b G_w^T / alpha_w
            + G_v R_b G_v^T / alpha_v,

and evaluates a dimensionless weighted sum after normalizing each trace by its
independently attainable minimum on the finite candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .ellipsoidal import simplex_grid, spectral_radius, sym


@dataclass(frozen=True)
class CombinedParetoProblem:
    """Two-descriptor combined stochastic/set-membership problem."""

    A: np.ndarray
    H: np.ndarray
    Q_stochastic: np.ndarray
    R_stochastic: np.ndarray
    Q_bounded: np.ndarray
    R_bounded: np.ndarray
    name: str = "combined_pareto_problem"

    def __post_init__(self) -> None:
        object.__setattr__(self, "A", np.asarray(self.A, dtype=float))
        object.__setattr__(self, "H", np.asarray(self.H, dtype=float))
        object.__setattr__(self, "Q_stochastic", sym(np.asarray(self.Q_stochastic, dtype=float)))
        object.__setattr__(self, "R_stochastic", sym(np.asarray(self.R_stochastic, dtype=float)))
        object.__setattr__(self, "Q_bounded", sym(np.asarray(self.Q_bounded, dtype=float)))
        object.__setattr__(self, "R_bounded", sym(np.asarray(self.R_bounded, dtype=float)))
        n = self.A.shape[0]
        m = self.H.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if self.H.shape[1] != n:
            raise ValueError("H must have n columns")
        if self.Q_stochastic.shape != (n, n) or self.Q_bounded.shape != (n, n):
            raise ValueError("Q matrices must have shape (n, n)")
        if self.R_stochastic.shape != (m, m) or self.R_bounded.shape != (m, m):
            raise ValueError("R matrices must have shape (m, m)")

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])


@dataclass(frozen=True)
class CombinedCandidate:
    """Feasible fixed-gain/fixed-alpha combined solution."""

    alpha: np.ndarray
    K: np.ndarray
    Sigma: np.ndarray
    P: np.ndarray
    trace_sigma: float
    trace_P: float


@dataclass(frozen=True)
class CombinedParetoResult:
    """Best candidate for one scalarization value."""

    lambda_stochastic: float
    objective: float
    candidate: CombinedCandidate


def _validate_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float).reshape(3)
    if np.any(alpha <= 0.0) or not np.all(np.isfinite(alpha)):
        raise ValueError("alpha must be finite and strictly positive")
    return alpha / float(np.sum(alpha))


def solve_discrete_lyapunov(F: np.ndarray, C: np.ndarray, scale: float = 1.0) -> Optional[np.ndarray]:
    """Solve X = scale F X F^T + C by vectorization."""

    F = np.asarray(F, dtype=float)
    C = sym(C)
    n = F.shape[0]
    if spectral_radius(F) * np.sqrt(scale) >= 1.0 - 1e-10:
        return None
    system = np.eye(n * n) - scale * np.kron(F, F)
    try:
        vec_X = np.linalg.solve(system, C.reshape(-1, order="F"))
    except np.linalg.LinAlgError:
        return None
    X = sym(vec_X.reshape(n, n, order="F"))
    if not np.all(np.isfinite(X)) or np.min(np.linalg.eigvalsh(X)) <= -1e-8:
        return None
    return X


def solve_combined_fixed_gain_alpha(
    problem: CombinedParetoProblem,
    K: np.ndarray,
    alpha: np.ndarray,
) -> Optional[CombinedCandidate]:
    """Solve the two steady-state descriptors for fixed K and alpha."""

    alpha = _validate_alpha(alpha)
    K = np.asarray(K, dtype=float).reshape(problem.n, problem.m)
    L = np.eye(problem.n) - K @ problem.H
    F = L @ problem.A
    G_w = L
    G_v = -K

    C_sigma = G_w @ problem.Q_stochastic @ G_w.T + G_v @ problem.R_stochastic @ G_v.T
    Sigma = solve_discrete_lyapunov(F, C_sigma, scale=1.0)
    if Sigma is None:
        return None

    C_P = G_w @ problem.Q_bounded @ G_w.T / alpha[1] + G_v @ problem.R_bounded @ G_v.T / alpha[2]
    P = solve_discrete_lyapunov(F, C_P, scale=1.0 / alpha[0])
    if P is None:
        return None

    return CombinedCandidate(
        alpha=alpha,
        K=K,
        Sigma=Sigma,
        P=P,
        trace_sigma=float(np.trace(Sigma)),
        trace_P=float(np.trace(P)),
    )


def two_dimensional_gain_grid(
    k1_min: float,
    k1_max: float,
    k2_min: float,
    k2_max: float,
    resolution: int,
) -> Iterable[np.ndarray]:
    """Yield 2x1 gains on a rectangular grid."""

    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    for k1 in np.linspace(k1_min, k1_max, resolution):
        for k2 in np.linspace(k2_min, k2_max, resolution):
            yield np.array([[k1], [k2]], dtype=float)


def enumerate_combined_candidates_2d(
    problem: CombinedParetoProblem,
    alpha_resolution: int = 31,
    gain_resolution: int = 25,
    k1_range: tuple[float, float] = (-0.2, 1.6),
    k2_range: tuple[float, float] = (-1.0, 1.0),
) -> list[CombinedCandidate]:
    """Enumerate feasible candidates for a 2D/single-output problem."""

    if problem.n != 2 or problem.m != 1:
        raise ValueError("the built-in gain grid is only for n=2, m=1")
    candidates: list[CombinedCandidate] = []
    alphas = list(simplex_grid(alpha_resolution))
    for K in two_dimensional_gain_grid(k1_range[0], k1_range[1], k2_range[0], k2_range[1], gain_resolution):
        for alpha in alphas:
            candidate = solve_combined_fixed_gain_alpha(problem, K, alpha)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def pareto_from_candidates(
    candidates: list[CombinedCandidate],
    lambdas: Iterable[float],
    normalize: bool = True,
) -> list[CombinedParetoResult]:
    """Choose the best candidate for each scalarization value.

    By default, each trace is divided by its independently attainable minimum.
    This removes arbitrary unit/scale effects from the scalarization while
    leaving the underlying Pareto ordering unchanged.
    """

    if not candidates:
        return []
    sigma_scale = min(candidate.trace_sigma for candidate in candidates) if normalize else 1.0
    P_scale = min(candidate.trace_P for candidate in candidates) if normalize else 1.0
    if sigma_scale <= 0.0 or P_scale <= 0.0:
        raise ValueError("trace normalization requires strictly positive minima")
    results: list[CombinedParetoResult] = []
    for lam in lambdas:
        lam = float(lam)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lambda values must lie in [0, 1]")
        def scalarized(candidate: CombinedCandidate) -> float:
            return lam * candidate.trace_sigma / sigma_scale + (1.0 - lam) * candidate.trace_P / P_scale

        best_candidate = min(candidates, key=scalarized)
        objective = scalarized(best_candidate)
        results.append(CombinedParetoResult(lambda_stochastic=lam, objective=float(objective), candidate=best_candidate))
    return results


def nondominated_candidates(candidates: list[CombinedCandidate]) -> list[CombinedCandidate]:
    """Return the trace-Pareto frontier of a finite candidate set."""

    ordered = sorted(candidates, key=lambda candidate: (candidate.trace_sigma, candidate.trace_P))
    frontier: list[CombinedCandidate] = []
    best_trace_P = float("inf")
    for candidate in ordered:
        if not frontier or candidate.trace_P < best_trace_P - 1e-12 * (1.0 + abs(best_trace_P)):
            frontier.append(candidate)
            best_trace_P = candidate.trace_P
    return frontier


def deterministic_combined_pareto_problem() -> CombinedParetoProblem:
    """Deterministic 2D example for combined stochastic/bounded uncertainty."""

    A = np.array([[0.87808946, -0.10955969], [0.10955969, 0.87808946]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)
    Q_stochastic = np.array([[0.0030, 0.0004], [0.0004, 0.0015]], dtype=float)
    R_stochastic = np.array([[0.0200]], dtype=float)
    Q_bounded = np.array([[0.00510331, -0.00519052], [-0.00519052, 0.00607184]], dtype=float)
    R_bounded = np.array([[0.07701328]], dtype=float)
    return CombinedParetoProblem(
        A=A,
        H=H,
        Q_stochastic=Q_stochastic,
        R_stochastic=R_stochastic,
        Q_bounded=Q_bounded,
        R_bounded=R_bounded,
        name="deterministic_combined_pareto_2d",
    )
