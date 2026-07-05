"""Utilities for the combined stochastic/set-membership fixed-gain case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ellipsoidal import FixedGainProblem, criterion_value, grid_optimize_weights, spectral_radius, sym


@dataclass(frozen=True)
class CombinedFixedGainProblem:
    """Fixed-gain problem with stochastic and bounded uncertainty components."""

    set_problem: FixedGainProblem
    Q_stochastic: np.ndarray
    R_stochastic: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "Q_stochastic", sym(np.asarray(self.Q_stochastic, dtype=float)))
        object.__setattr__(self, "R_stochastic", sym(np.asarray(self.R_stochastic, dtype=float)))
        if self.Q_stochastic.shape != (self.set_problem.n, self.set_problem.n):
            raise ValueError("Q_stochastic has incompatible shape")
        if self.R_stochastic.shape != (self.set_problem.m, self.set_problem.m):
            raise ValueError("R_stochastic has incompatible shape")


def solve_stochastic_covariance(problem: CombinedFixedGainProblem) -> Optional[np.ndarray]:
    """Solve Sigma = F Sigma F^T + G_w Q_s G_w^T + G_v R_s G_v^T."""

    base = problem.set_problem
    if spectral_radius(base.F) >= 1.0 - 1e-10:
        return None
    n = base.n
    C = base.G_w @ problem.Q_stochastic @ base.G_w.T + base.G_v @ problem.R_stochastic @ base.G_v.T
    system = np.eye(n * n) - np.kron(base.F, base.F)
    try:
        vec_sigma = np.linalg.solve(system, C.reshape(-1, order="F"))
    except np.linalg.LinAlgError:
        return None
    sigma = sym(vec_sigma.reshape(n, n, order="F"))
    if not np.all(np.isfinite(sigma)):
        return None
    return sigma


def combined_trace_objective(sigma: np.ndarray, P: np.ndarray, lambda_stochastic: float) -> float:
    """Weighted trace objective lambda tr(Sigma)+(1-lambda)tr(P)."""

    lam = float(lambda_stochastic)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda_stochastic must be in [0, 1]")
    return lam * criterion_value(sigma, "trace") + (1.0 - lam) * criterion_value(P, "trace")


def optimize_set_membership_part(
    problem: CombinedFixedGainProblem,
    lambda_stochastic: float,
    resolution: int = 121,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Optimize alpha for a fixed-gain combined problem.

    For fixed gain, the stochastic covariance is independent of alpha. The
    alpha optimization therefore acts on the bounded component, but the returned
    value is the weighted combined criterion.
    """

    sigma = solve_stochastic_covariance(problem)
    if sigma is None:
        return None
    best = grid_optimize_weights(problem.set_problem, resolution=resolution, criterion="trace")
    if best is None:
        return None
    alpha, P, _ = best
    value = combined_trace_objective(sigma, P, lambda_stochastic)
    return alpha, sigma, P, value
