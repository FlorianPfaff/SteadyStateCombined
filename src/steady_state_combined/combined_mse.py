"""Local steady-state minimax-MSE design with two Lyapunov adjoints."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from .combined_pareto import CombinedParetoProblem
from .ellipsoidal import normalize_alpha, spectral_radius, sym
from .research import continuous_steady_trace, solve_dare
from .riccati import GainOptimizedProblem


@dataclass(frozen=True)
class CombinedMseEvaluation:
    K: np.ndarray
    alpha: np.ndarray
    Sigma: np.ndarray
    P: np.ndarray
    value: float
    smooth_value: float
    smoothing_bound: float
    gradient_K: np.ndarray
    gradient_alpha: np.ndarray
    tau: float
    stability_margin: float


@dataclass(frozen=True)
class CombinedMseResult:
    evaluation: CombinedMseEvaluation
    initial_value: float
    status: str
    stationarity: float
    iterations: int
    evaluations: int
    seconds: float
    starts: int
    alpha_floor: float


def validate_combined_problem(problem: CombinedParetoProblem):
    for name in ("Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded"):
        matrix = getattr(problem, name)
        if not np.all(np.isfinite(matrix)) or np.min(np.linalg.eigvalsh(matrix)) < -1e-12 * max(np.linalg.norm(matrix), 1e-300):
            raise ValueError(f"{name} must be positive semidefinite and finite")


def combined_mse_value_gradient(
    problem: CombinedParetoProblem, K: np.ndarray, alpha: np.ndarray, tau: float = 0.0
) -> CombinedMseEvaluation:
    """Evaluate the exact risk and a smooth upper approximation with gradients.

    At tau=0 an eigenprojector gives a subgradient at a repeated maximum;
    smooth optimization uses positive tau. The stochastic error is assumed
    zero mean, with deterministic bounded bias independent of that error.
    """
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    K = np.asarray(K, dtype=float)
    if K.shape != (problem.n, problem.m) or not np.all(np.isfinite(K)):
        raise ValueError("invalid gain")
    alpha = normalize_alpha(alpha)
    L = np.eye(problem.n) - K @ problem.H
    F = L @ problem.A
    rho2 = spectral_radius(F) ** 2
    if rho2 >= alpha[0]:
        raise ValueError("gain/weights do not have a stable invariant shape")
    Qs, Rs = problem.Q_stochastic, problem.R_stochastic
    Qb, Rb = problem.Q_bounded, problem.R_bounded
    Sigma = sym(solve_discrete_lyapunov(F, L @ Qs @ L.T + K @ Rs @ K.T))
    components = [None, sym(L @ Qb @ L.T), sym(K @ Rb @ K.T)]
    scaled = F / np.sqrt(alpha[0])
    P = sym(solve_discrete_lyapunov(scaled, components[1] / alpha[1] + components[2] / alpha[2]))
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Sigma)):
        raise ValueError("nonfinite descriptor solution")
    for descriptor in (Sigma, P):
        if np.min(np.linalg.eigvalsh(descriptor)) < -1e-10 * max(np.linalg.norm(descriptor), 1e-300):
            raise ValueError("descriptor solution is not numerically positive semidefinite")
    components[0] = sym(F @ P @ F.T)
    eigenvalues, vectors = np.linalg.eigh(P)
    largest = float(eigenvalues[-1])
    if tau > 0:
        probability = np.exp((eigenvalues - largest) / tau)
        normalizer = probability.sum()
        probability /= normalizer
        smooth_bias = largest + tau * np.log(normalizer)
    else:
        probability = np.zeros(problem.n)
        probability[-1] = 1
        smooth_bias = largest
    spectral_gradient = (vectors * probability) @ vectors.T
    adjoint_s = sym(solve_discrete_lyapunov(F.T, np.eye(problem.n)))
    adjoint_b = sym(solve_discrete_lyapunov(scaled.T, spectral_gradient))
    prior_s = problem.A @ Sigma @ problem.A.T + Qs
    prior_b = problem.A @ P @ problem.A.T / alpha[0] + Qb / alpha[1]
    gradient_K = 2 * adjoint_s @ (K @ Rs - L @ prior_s @ problem.H.T)
    gradient_K += 2 * adjoint_b @ (K @ Rb / alpha[2] - L @ prior_b @ problem.H.T)
    gradient_alpha = -np.array([np.trace(adjoint_b @ component) for component in components]) / alpha**2
    variance = float(np.trace(Sigma))
    return CombinedMseEvaluation(
        K.copy(), alpha.copy(), Sigma, P, variance + largest, variance + float(smooth_bias),
        float(tau * np.log(problem.n)), gradient_K, gradient_alpha, tau, float(alpha[0] - rho2),
    )


def _weight_target(contributions, floor):
    roots = np.sqrt(np.maximum(contributions, 0))
    if roots.sum() == 0:
        return None
    active = np.zeros(3, dtype=bool)
    target = np.full(3, floor)
    while True:
        free = ~active
        target[free] = (1 - floor * active.sum()) * roots[free] / roots[free].sum()
        newly_active = free & (target < floor)
        if not np.any(newly_active):
            return target
        target[newly_active] = floor
        active |= newly_active


def combined_stationarity(evaluation, floor):
    scale = max(abs(evaluation.value), np.finfo(float).tiny)
    gain_residual = np.linalg.norm(evaluation.gradient_K) * max(1.0, np.linalg.norm(evaluation.K)) / scale
    free = evaluation.alpha > 5 * floor
    gradient = evaluation.gradient_alpha
    if not np.any(free):
        return float("inf")
    level = gradient[free].mean()
    residual = float(np.max(np.abs(gradient[free] - level)))
    if np.any(~free):
        residual = max(residual, float(np.max(np.maximum(level - gradient[~free], 0))))
    return float(max(gain_residual, residual / scale))


def default_combined_starts(problem, floor=1e-8):
    bounded = GainOptimizedProblem(problem.A, problem.H, problem.Q_bounded, problem.R_bounded)
    starts = []
    try:
        optimized = continuous_steady_trace(bounded, alpha_floor=floor)
        starts.append((optimized.K, optimized.alpha))
        _, equal_gain = solve_dare(bounded, np.full(3, 1 / 3))
        starts.append((equal_gain, np.full(3, 1 / 3)))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    try:
        prior = solve_discrete_are(problem.A.T, problem.H.T, problem.Q_stochastic, problem.R_stochastic)
        gain = np.linalg.solve(problem.H @ prior @ problem.H.T + problem.R_stochastic, problem.H @ prior).T
        rho2 = spectral_radius((np.eye(problem.n) - gain @ problem.H) @ problem.A) ** 2
        if rho2 < 1 - 2 * floor:
            a0 = (rho2 + 1 - 2 * floor) / 2
            starts.append((gain, np.array([a0, (1 - a0) / 2, (1 - a0) / 2])))
    except (ValueError, np.linalg.LinAlgError):
        pass
    return starts


def optimize_combined_mse(
    problem: CombinedParetoProblem,
    initial_designs=(),
    *,
    alpha_floor: float = 1e-8,
    max_iter_per_stage: int = 400,
    stationarity_tol: float = 1e-6,
    smoothing_exponents: tuple[int, ...] = (2, 3, 4, 5, 6),
) -> CombinedMseResult:
    """Local multistart optimization with feasible Armijo block steps.

    Source weights use the adjoint square-root target with active floors.
    The gain block uses safeguarded inverse-BFGS directions. All line-search
    evaluations solve both descriptors and enforce scaled stability.
    """
    if not 0 < alpha_floor < 1 / 3 or max_iter_per_stage < 1 or stationarity_tol <= 0:
        raise ValueError("invalid optimizer settings")
    if (not smoothing_exponents or any(not isinstance(x, int) or not 1 <= x <= 12 for x in smoothing_exponents)
            or tuple(sorted(set(smoothing_exponents))) != tuple(smoothing_exponents)):
        raise ValueError("smoothing exponents must be strictly increasing integers in [1, 12]")
    validate_combined_problem(problem)
    started = perf_counter()
    starts = list(initial_designs) or default_combined_starts(problem, alpha_floor)
    feasible = []
    evaluations = 0

    def evaluate(K, alpha, tau):
        nonlocal evaluations
        evaluations += 1
        if min(alpha) < alpha_floor * (1 - 1e-8):
            raise ValueError("weights below floor")
        return combined_mse_value_gradient(problem, K, alpha, tau)

    for K, alpha in starts:
        try:
            alpha = normalize_alpha(alpha)
            if np.min(alpha) < alpha_floor:
                alpha = alpha_floor + (1 - 3 * alpha_floor) * alpha
            rho2 = spectral_radius((np.eye(problem.n) - K @ problem.H) @ problem.A) ** 2
            if alpha[0] <= rho2:
                if rho2 >= 1 - 2 * alpha_floor:
                    continue
                a0 = (rho2 + 1 - 2 * alpha_floor) / 2
                alpha = np.array([a0, (1 - a0) / 2, (1 - a0) / 2])
            feasible.append(evaluate(K, alpha, 0))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not feasible:
        raise ValueError("no feasible combined initial design")
    initial_value = min(item.value for item in feasible)
    scale = max(initial_value, np.finfo(float).tiny)
    final_tau = scale * 10.0**(-smoothing_exponents[-1]) / max(1.0, np.log(problem.n))
    best = evaluate(min(feasible, key=lambda item: item.value).K,
                    min(feasible, key=lambda item: item.value).alpha, final_tau)
    best_status = "initial_design_retained"
    iterations = 0

    def line_search(current, gain_direction, weight_direction):
        slope = float(np.sum(current.gradient_K * gain_direction) + current.gradient_alpha @ weight_direction)
        if slope >= 0 or not np.isfinite(slope):
            return current, False
        for halving in range(50):
            step = 0.5**halving
            try:
                trial = evaluate(current.K + step * gain_direction, current.alpha + step * weight_direction, current.tau)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if trial.smooth_value <= current.smooth_value + 1e-4 * step * slope:
                changed = current.smooth_value - trial.smooth_value > 64 * np.finfo(float).eps * scale
                return (trial, True) if changed else (current, False)
        return current, False

    for start in feasible:
        current = start
        for exponent in smoothing_exponents:
            tau = scale * 10.0**(-exponent) / max(1.0, np.log(problem.n))
            current = evaluate(current.K, current.alpha, tau)
            size = current.K.size
            inverse = np.eye(size) * max(1.0, np.linalg.norm(current.K)) / max(np.linalg.norm(current.gradient_K / scale), 1e-6)
            status = "iteration_limit"
            for _ in range(max_iter_per_stage):
                iterations += 1
                if combined_stationarity(current, alpha_floor) <= max(stationarity_tol, 10.0**(-exponent - 1)):
                    status = "stationary"
                    break
                target = _weight_target(-current.gradient_alpha * current.alpha**2, alpha_floor)
                changed_alpha = False
                if target is not None:
                    current, changed_alpha = line_search(current, np.zeros_like(current.K), target - current.alpha)
                gradient = current.gradient_K.ravel() / scale
                direction = -inverse @ gradient
                if gradient @ direction >= 0 or not np.all(np.isfinite(direction)):
                    inverse = np.eye(size) / max(np.linalg.norm(gradient), 1e-6)
                    direction = -inverse @ gradient
                updated, changed_gain = line_search(current, direction.reshape(current.K.shape), np.zeros(3))
                if changed_gain:
                    displacement = (updated.K - current.K).ravel()
                    difference = (updated.gradient_K - current.gradient_K).ravel() / scale
                    curvature = float(displacement @ difference)
                    if curvature > 1e-10 * np.linalg.norm(displacement) * np.linalg.norm(difference):
                        transform = np.eye(size) - np.outer(displacement, difference) / curvature
                        inverse = sym(transform @ inverse @ transform.T + np.outer(displacement, displacement) / curvature)
                        if np.linalg.norm(inverse) > 1e10:
                            inverse = np.eye(size)
                    current = updated
                if not changed_alpha and not changed_gain:
                    status = "roundoff_or_line_search_stagnation"
                    break
            if exponent == smoothing_exponents[-1] and current.value <= best.value:
                best, best_status = current, status
    return CombinedMseResult(best, initial_value, best_status, combined_stationarity(best, alpha_floor),
                             iterations, evaluations, perf_counter() - started, len(feasible), alpha_floor)
