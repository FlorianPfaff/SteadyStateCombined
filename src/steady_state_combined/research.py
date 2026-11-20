"""Continuous, independently checked solvers for paper validation.

SciPy is optional so the original NumPy evaluation remains reproducible.
One-step and gain-optimized objectives use multistart SLSQP; only the fixed-
gain weighted-trace problem has the global convexity guarantee used here.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import LinAlgWarning, solve_discrete_are, solve_discrete_lyapunov
from scipy.optimize import brentq, minimize

from .ellipsoidal import FixedGainProblem, normalize_alpha, simplex_grid, spectral_radius, sym
from .riccati import GainOptimizedProblem, riccati_update


@dataclass(frozen=True)
class ContinuousResult:
    alpha: np.ndarray
    P: np.ndarray
    K: np.ndarray
    value: float
    gradient: np.ndarray
    evaluations: int
    iterations: int
    seconds: float
    alpha_floor: float = 0.0
    # Set by adjoint_fixed_trace only: it stopped before it could confirm that value is within kkt_tol of the minimum.
    stagnated: bool = False

    @property
    def stationarity(self) -> float:
        """Relative simplex KKT residual, including active positivity bounds."""
        free = self.alpha > 5.0 * self.alpha_floor
        if not np.any(free):
            return float("inf")
        level = float(np.mean(self.gradient[free]))
        residual = float(np.max(np.abs(self.gradient[free] - level)))
        if np.any(~free):
            residual = max(residual, float(np.max(np.maximum(level - self.gradient[~free], 0.0))))
        return residual / max(abs(self.value), np.finfo(float).tiny)

    @property
    def simplex_first_order_gap(self) -> float:
        """Relative linearization gap; an optimality bound for convex objectives."""
        minimum = self.alpha_floor * self.gradient.sum() + (1.0 - 3.0 * self.alpha_floor) * self.gradient.min()
        gap = float(self.gradient @ self.alpha - minimum)
        return max(gap, 0.0) / max(abs(self.value), np.finfo(float).tiny)


def fixed_trace_value_gradient(
    problem: FixedGainProblem,
    alpha: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    alpha = normalize_alpha(alpha)
    F = problem.F
    scaled = F / np.sqrt(alpha[0])
    if spectral_radius(scaled) >= 1.0:
        raise ValueError("fixed-gain weights are not stable")
    W = np.eye(problem.n) if weight is None else sym(weight)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            P = sym(solve_discrete_lyapunov(scaled, problem.S_w / alpha[1] + problem.S_v / alpha[2]))
            adjoint = sym(solve_discrete_lyapunov(scaled.T, W))
    except LinAlgWarning as exc:
        raise ValueError("ill-conditioned Lyapunov system") from exc
    components = (sym(F @ P @ F.T), problem.S_w, problem.S_v)
    gradient = -np.array([np.trace(adjoint @ component) for component in components]) / alpha**2
    return P, float(np.trace(W @ P)), gradient


def solve_dare(problem: GainOptimizedProblem, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the prior DARE and independently check its posterior residual."""
    alpha = normalize_alpha(alpha)
    prior = solve_discrete_are(problem.A.T / np.sqrt(alpha[0]), problem.H.T, problem.Q / alpha[1], problem.R / alpha[2])
    innovation = problem.H @ prior @ problem.H.T + problem.R / alpha[2]
    K = np.linalg.solve(innovation, problem.H @ prior).T
    G = np.eye(problem.n) - K @ problem.H
    P = sym(G @ prior @ G.T + K @ problem.R @ K.T / alpha[2])
    check, _ = riccati_update(problem, P, alpha)
    residual = np.linalg.norm(P - check, ord="fro") / max(np.linalg.norm(P, ord="fro"), 1e-12)
    if residual > 1e-8 or spectral_radius(G @ problem.A) / np.sqrt(alpha[0]) >= 1.0:
        raise ValueError(f"DARE residual/stability check failed: {residual:g}")
    return P, K


def dare_trace_value_gradient(
    problem: GainOptimizedProblem,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    P, K = solve_dare(problem, alpha)
    fixed = FixedGainProblem(problem.A, problem.H, K, problem.Q, problem.R)
    _, value, gradient = fixed_trace_value_gradient(fixed, alpha)
    return P, K, value, gradient


def one_step_trace_value_gradient(
    problem: GainOptimizedProblem,
    P: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    alpha = normalize_alpha(alpha)
    next_P, K = riccati_update(problem, P, alpha)
    G = np.eye(problem.n) - K @ problem.H
    components = (G @ problem.A @ P @ problem.A.T @ G.T, G @ problem.Q @ G.T, K @ problem.R @ K.T)
    gradient = -np.array([np.trace(component) for component in components]) / alpha**2
    return next_P, K, float(np.trace(next_P)), gradient


def _multistart_trace(
    evaluate,
    starts: Iterable[np.ndarray],
    grid_resolution: int = 9,
    top_grid_starts: int = 3,
    alpha_floor: float = 1e-8,
) -> tuple[np.ndarray, tuple, int, int]:
    """Use the same normalized objective, bounds, and tolerance for both designs."""
    evaluations = 0
    ranked = []
    for alpha in simplex_grid(grid_resolution):
        try:
            result = evaluate(alpha)
        except (ValueError, np.linalg.LinAlgError):
            continue
        evaluations += 1
        if np.isfinite(result[2]):
            ranked.append((result[2], alpha, result))
    if not ranked:
        raise RuntimeError("no admissible multistart points")
    ranked.sort(key=lambda item: item[0])
    scale = max(abs(ranked[0][0]), np.finfo(float).tiny)
    best = ranked[0]
    seeds = [normalize_alpha(alpha) for alpha in starts]
    seeds.extend(item[1] for item in ranked[:top_grid_starts])
    seeds.append(np.full(3, 1.0 / 3.0))
    iterations = 0
    floor = alpha_floor

    def objective(x):
        nonlocal evaluations
        alpha = np.array([x[0], x[1], 1.0 - x[0] - x[1]])
        if np.min(alpha) < floor * 0.5:
            return 1e20, np.zeros(2)
        try:
            result = evaluate(alpha)
        except (ValueError, np.linalg.LinAlgError):
            return 1e20, np.zeros(2)
        evaluations += 1
        return result[2] / scale, (result[3][:2] - result[3][2]) / scale

    for alpha in seeds:
        optimum = minimize(
            objective,
            alpha[:2],
            method="SLSQP",
            jac=True,
            bounds=[(floor, 1.0 - 2 * floor)] * 2,
            constraints=[{"type": "ineq", "fun": lambda x: 1.0 - floor - x.sum(), "jac": lambda x: -np.ones(2)}],
            options={"ftol": 1e-12, "maxiter": 300},
        )
        iterations += optimum.nit
        candidate_alpha = np.array([optimum.x[0], optimum.x[1], 1.0 - optimum.x.sum()])
        if np.min(candidate_alpha) < floor * 0.5:
            continue
        try:
            candidate = evaluate(candidate_alpha)
        except (ValueError, np.linalg.LinAlgError):
            continue
        evaluations += 1
        if np.isfinite(candidate[2]) and candidate[2] < best[0]:
            best = (candidate[2], candidate_alpha, candidate)
    return best[1], best[2], evaluations, iterations


def continuous_greedy_trace(
    problem: GainOptimizedProblem,
    initial_P: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-10,
    grid_resolution: int = 9,
    top_grid_starts: int = 3,
    alpha_floor: float = 1e-8,
) -> ContinuousResult:
    started = perf_counter()
    P = np.eye(problem.n) if initial_P is None else sym(initial_P)
    alpha = np.full(3, 1.0 / 3.0)
    total_evaluations = 0
    for iteration in range(1, max_iter + 1):
        alpha, result, evaluations, _ = _multistart_trace(
            lambda candidate, current=P: one_step_trace_value_gradient(problem, current, candidate),
            [alpha],
            grid_resolution=grid_resolution,
            top_grid_starts=top_grid_starts,
            alpha_floor=alpha_floor,
        )
        next_P = result[0]
        total_evaluations += evaluations
        relative = np.linalg.norm(next_P - P, ord="fro") / max(np.linalg.norm(P, ord="fro"), 1e-12)
        P = next_P
        if relative <= tol:
            fixed_P, fixed_K = solve_dare(problem, alpha)
            residual = np.linalg.norm(P - fixed_P, ord="fro") / max(np.linalg.norm(P, ord="fro"), 1e-12)
            if residual > 2e-7:
                raise RuntimeError(f"greedy fixed-point residual is {residual:g}")
            gradient = one_step_trace_value_gradient(problem, fixed_P, alpha)[3]
            return ContinuousResult(
                alpha,
                fixed_P,
                fixed_K,
                float(np.trace(fixed_P)),
                gradient,
                total_evaluations,
                iteration,
                perf_counter() - started,
                alpha_floor=alpha_floor,
            )
    raise RuntimeError("continuous greedy recursion did not converge")


def continuous_steady_trace(
    problem: GainOptimizedProblem,
    initial_alphas: Iterable[np.ndarray] = (),
    grid_resolution: int = 9,
    top_grid_starts: int = 3,
    alpha_floor: float = 1e-8,
) -> ContinuousResult:
    started = perf_counter()
    alpha, result, evaluations, iterations = _multistart_trace(
        lambda candidate: dare_trace_value_gradient(problem, candidate),
        initial_alphas,
        grid_resolution=grid_resolution,
        top_grid_starts=top_grid_starts,
        alpha_floor=alpha_floor,
    )
    P, K, value, gradient = result
    return ContinuousResult(
        alpha, P, K, value, gradient, evaluations, iterations, perf_counter() - started, alpha_floor=alpha_floor
    )


def global_fixed_trace(problem: FixedGainProblem, weight: np.ndarray | None = None) -> ContinuousResult:
    """Reduce the convex fixed-gain weighted-trace design to one scalar.

    The implementation targets nondegenerate interior optima: both noise sources
    contribute positive weighted trace and the propagation matrix is nonzero.
    """
    started = perf_counter()
    W = np.eye(problem.n) if weight is None else sym(weight)
    if W.shape != (problem.n, problem.n) or np.min(np.linalg.eigvalsh(W)) <= 0.0:
        raise ValueError("weight must be positive definite")
    if np.linalg.norm(problem.F) == 0 or min(np.trace(W @ problem.S_w), np.trace(W @ problem.S_v)) <= 0:
        raise ValueError("global scalar solver requires nondegenerate interior design")
    rho2 = spectral_radius(problem.F) ** 2
    if rho2 >= 1:
        raise ValueError("fixed gain is unstable")
    evaluations = 0

    def profile(s):
        nonlocal evaluations
        evaluations += 1
        scaled = problem.F / np.sqrt(s)
        zw = sym(solve_discrete_lyapunov(scaled, problem.S_w))
        zv = sym(solve_discrete_lyapunov(scaled, problem.S_v))
        traces = np.array([np.trace(W @ zw), np.trace(W @ zv)])
        if not np.all(np.isfinite(traces)) or np.min(traces) <= 0:
            raise ValueError("invalid component Lyapunov solution")
        roots = np.sqrt(traces)
        noise = (1.0 - s) * roots / roots.sum()
        alpha = np.array([s, noise[0], noise[1]])
        P, value, gradient = fixed_trace_value_gradient(problem, alpha, W)
        return alpha, P, value, gradient

    def derivative(s):
        gradient = profile(s)[3]
        return gradient[0] - gradient[1]

    brackets = []
    for side in ("lower", "upper"):
        for exponent in range(25):
            # Search outward from the midpoint before approaching singular limits.
            margin = 0.5 ** (exponent + 1) * (1.0 - rho2)
            endpoint = rho2 + margin if side == "lower" else 1.0 - margin
            slope = derivative(endpoint)
            if (side == "lower" and slope < 0) or (side == "upper" and slope > 0):
                brackets.append(endpoint)
                break
        else:
            raise RuntimeError("interior optimum is not bracketed")
    lower, upper = brackets
    scalar = brentq(derivative, lower, upper, xtol=1e-13, rtol=1e-13)
    alpha, P, value, gradient = profile(scalar)
    return ContinuousResult(alpha, P, problem.K, value, gradient, evaluations, evaluations, perf_counter() - started)


def posynomial_gap_bound(alpha: np.ndarray, value: float, gradient: np.ndarray) -> float:
    """Bound on (value - minimum) / value over the simplex for a fixed-gain trace with semidefinite W, Q and R.

    Such a trace is a posynomial in the weights, the sum over k of tr(W F^k S_w F^kT) / (alpha_0^k alpha_w) and
    tr(W F^k S_v F^kT) / (alpha_0^k alpha_v), so in log coordinates it lies above its tangent, the monomial
    value * prod (alpha_i / beta_i)^e_i with e_i = -alpha_i gradient_i / value. The minimum of that monomial on the
    simplex gives the bound 1 - exp(-E KL(p || alpha)), where E = sum(e) and p = e / E. Unlike the linearization
    gap it is of second order in the distance to the optimum, so it can confirm a minimum that rounding hides from
    a descent test. The divergence carries an allowance of 4 eps for its own rounding, so the bound never drops
    below about E eps, the relative change of the trace between neighbouring floating-point weights. The result is
    inf when the gradient does not have the signs of such a trace.
    """
    alpha = np.asarray(alpha, dtype=float) / np.sum(alpha)
    gradient = np.asarray(gradient, dtype=float)
    mean = float(gradient @ alpha)
    if not (np.isfinite(value) and value > 0 and mean < 0 and np.all(gradient <= 0)):
        return float("inf")
    ratio = gradient / mean  # p / alpha, kept apart: 1 + (ratio - 1) loses ratios below eps
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        divergence = float(np.sum(np.where(ratio > 0, alpha * ratio * np.log(ratio), 0.0)))
    if not np.isfinite(divergence):
        return float("inf")
    return float(-np.expm1(mean / value * (max(divergence, 0.0) + 4 * np.finfo(float).eps)))


def _semidefinite(weight: np.ndarray | None) -> bool:
    """Whether a trace weight is positive semidefinite up to the rounding of its eigenvalues."""
    if weight is None:
        return True
    eigenvalues = np.linalg.eigvalsh(sym(np.asarray(weight, dtype=float)))
    return bool(eigenvalues[0] >= -8 * len(eigenvalues) * np.finfo(float).eps * np.max(np.abs(eigenvalues)))


def adjoint_fixed_trace(
    problem: FixedGainProblem,
    alpha_start: np.ndarray,
    weight: np.ndarray | None = None,
    max_iter: int = 1000,
    kkt_tol: float = 1e-6,
) -> ContinuousResult:
    """Minimize the weighted fixed-gain trace over the simplex by adjoint fixed-point directions and Armijo steps.

    The iteration stops when the relative KKT residual ptp(gradient) / value is at most ``kkt_tol``, which bounds
    the relative suboptimality of the convex trace by ``kkt_tol``. Rounding can stop it before: the computed
    descent slope loses its sign, no step passes the line search, or a step gains no more than rounding. The point
    reached then is reported as converged only if :func:`posynomial_gap_bound` confirms the same accuracy.
    Otherwise, and when ``max_iter`` runs out, the result has ``stagnated=True``: its value may be far from the
    minimum, for instance where the trace changes by a few percent within the last bits of the weights. So does a
    result whose trace moves by more than ``kkt_tol`` between neighbouring floating-point weights, KKT test or not.
    """
    started = perf_counter()
    alpha = normalize_alpha(alpha_start)
    P, value, gradient = fixed_trace_value_gradient(problem, alpha, weight)
    evaluations = 1
    roundoff_stagnation = False
    for iteration in range(max_iter):
        if np.ptp(gradient) / max(abs(value), np.finfo(float).tiny) <= kkt_tol:
            break
        contributions = np.maximum(-gradient * alpha**2, np.finfo(float).tiny)
        target = np.sqrt(contributions)
        target /= target.sum()
        direction = target - alpha
        slope = float(gradient @ direction)
        if not slope < 0:
            # With a negative gradient, slope <= (alpha.sum() - 1) * G exactly, G = -gradient @ alpha (Cauchy-Schwarz),
            # and rounding adds at most 9 eps * G. Such a slope only says that rounding hides any further descent;
            # whether the point is optimal is decided after the loop.
            roundoff = (alpha.sum() - 1.0 + 9 * np.finfo(float).eps) * -float(gradient @ alpha)
            if contributions.min() > np.finfo(float).tiny and slope <= roundoff:
                break
            raise RuntimeError("adjoint direction failed descent check")
        for halving in range(45):
            step = 2.0 ** (-halving)
            candidate_alpha = alpha + step * direction
            try:
                candidate_P, candidate_value, candidate_gradient = fixed_trace_value_gradient(
                    problem, candidate_alpha, weight
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            evaluations += 1
            if candidate_value <= value + 1e-4 * step * slope:
                improvement = (value - candidate_value) / max(abs(value), np.finfo(float).tiny)
                roundoff_stagnation = improvement <= 64 * np.finfo(float).eps
                alpha, P, value, gradient = candidate_alpha, candidate_P, candidate_value, candidate_gradient
                break
        else:
            break
        if roundoff_stagnation:
            break
    # E = -gradient @ alpha / value is the sum of the elasticities of the trace in the weights, so neighbouring
    # floating-point weights differ by about E eps in trace: no result is confirmed beyond that resolution.
    elasticity = -float(gradient @ (alpha / alpha.sum())) / value if value > 0 else 0.0
    confirmed = (np.ptp(gradient) / max(abs(value), np.finfo(float).tiny) <= kkt_tol
                 and elasticity * 4 * np.finfo(float).eps <= kkt_tol)
    if not confirmed and _semidefinite(weight):
        confirmed = posynomial_gap_bound(alpha, value, gradient) <= kkt_tol
    return ContinuousResult(
        alpha, P, problem.K, value, gradient, evaluations, iteration, perf_counter() - started, stagnated=not confirmed
    )
