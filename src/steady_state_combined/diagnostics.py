"""Explanatory diagnostics for convex fixed-gain weight design.

These floating-point diagnostics are not verified-arithmetic certificates.
The stable-domain first-order bound is valid in exact arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ellipsoidal import FixedGainProblem, normalize_alpha, spectral_radius
from .research import fixed_trace_value_gradient


@dataclass(frozen=True)
class GreedyOptimalityDiagnostics:
    alpha: np.ndarray
    value: float
    gradient: np.ndarray
    component_traces: np.ndarray
    amplification_ratios: np.ndarray
    greedy_kkt_residual: float
    steady_kkt_residual: float
    first_order_gap: float
    lower_bound: float
    stability_margin: float

    @property
    def relative_gap(self) -> float:
        return self.first_order_gap / max(self.value, np.finfo(float).tiny)


def _simplex_kkt(gradient: np.ndarray, alpha: np.ndarray, floor: float) -> float:
    free = alpha > floor + 1e-9 * max(1.0, floor)
    if not np.any(free):
        return float("inf")
    level = float(np.mean(gradient[free]))
    residual = float(np.max(np.abs(gradient[free] - level)))
    if np.any(~free):
        residual = max(residual, float(np.max(np.maximum(level - gradient[~free], 0.0))))
    return residual


def greedy_optimality_diagnostics(
    problem: FixedGainProblem,
    alpha: np.ndarray,
    alpha_floor: float = 0.0,
) -> GreedyOptimalityDiagnostics:
    """Compare one-step and invariant trace stationarity at a fixed point.

At an interior greedy fixed point with positive component traces, the
invariant design is optimal iff all amplification ratios are equal.
For arbitrary stable weights, ``first_order_gap`` bounds the remaining
improvement over weights with the supplied floor. Zero components have
undefined (NaN) ratios; the gradient and gap remain applicable.
"""
    if not np.isfinite(alpha_floor) or not 0 <= alpha_floor < 1 / 3:
        raise ValueError("alpha_floor must lie in [0, 1/3)")
    alpha = normalize_alpha(alpha)
    if np.min(alpha) < alpha_floor:
        raise ValueError("weights violate alpha_floor")
    rho2 = spectral_radius(problem.F) ** 2
    lower = np.array([max(rho2, alpha_floor), alpha_floor, alpha_floor])
    if lower.sum() >= 1:
        raise ValueError("stable weight domain is empty")
    P, value, gradient = fixed_trace_value_gradient(problem, alpha)
    traces = np.array([np.trace(problem.F @ P @ problem.F.T), np.trace(problem.S_w), np.trace(problem.S_v)])
    contributions = -gradient * alpha**2
    ratios = np.full(3, np.nan)
    np.divide(contributions, traces, out=ratios, where=traces > 0)
    step_gradient = -traces / alpha**2
    linear_minimum = float(lower @ gradient + (1 - lower.sum()) * np.min(gradient))
    gap = max(0.0, float(gradient @ alpha) - linear_minimum)
    scale = max(value, np.finfo(float).tiny)
    return GreedyOptimalityDiagnostics(
        alpha=alpha,
        value=value,
        gradient=gradient,
        component_traces=traces,
        amplification_ratios=ratios,
        greedy_kkt_residual=_simplex_kkt(step_gradient, alpha, alpha_floor) / scale,
        steady_kkt_residual=_simplex_kkt(gradient, alpha, alpha_floor) / scale,
        first_order_gap=gap,
        lower_bound=max(0.0, value - gap),
        stability_margin=float(alpha[0] - rho2),
    )
