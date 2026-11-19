"""Independent optional SDP references for invariant and one-step design."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .combined_mse import combined_mse_value_gradient, validate_combined_problem
from .combined_pareto import CombinedParetoProblem
from .ellipsoidal import normalize_alpha, sym
from .riccati import GainOptimizedProblem

SOLVER_OPTIONS = {"solver": "CLARABEL", "tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10,
                  "tol_feas": 1e-10, "max_iter": 300}


@dataclass(frozen=True)
class SdpDesign:
    K: np.ndarray
    alpha: np.ndarray
    P: np.ndarray
    Sigma: np.ndarray | None
    value: float
    status: str
    iterations: int
    fixed_point_residual: float = 0.0
    one_step_residual: float = 0.0
    one_step_risk_gap: float = 0.0


def fixed_weight_lmi_reference(problem: GainOptimizedProblem, alpha, *, balanced=False, state_scale=None) -> SdpDesign:
    """Minimize invariant trace via the information-form block LMI."""
    alpha = normalize_alpha(alpha)
    n, m = problem.n, problem.m
    transform = np.eye(n) if state_scale is None else np.asarray(state_scale, dtype=float)
    if transform.shape != (n, n) or not np.all(np.isfinite(transform)):
        raise ValueError("invalid state-coordinate scaling")
    weight = np.eye(n)
    if state_scale is not None:
        inverse = np.linalg.inv(transform)
        problem = GainOptimizedProblem(inverse @ problem.A @ transform, problem.H @ transform,
                                       sym(inverse @ problem.Q @ inverse.T), problem.R)
        weight = transform.T @ transform
        weight /= np.trace(weight)
    scale = 1.0 if state_scale is not None else (np.trace(problem.Q) / n + np.trace(problem.R) / m) / 2
    X, T = cp.Variable((n, n), symmetric=True), cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((n, m))
    L = X - Y @ problem.H
    nn, nm = np.zeros((n, n)), np.zeros((n, m))
    if balanced:
        # Block congruence removes inverse noise matrices and small diagonal
        # weights without changing the feasible set or using a DARE solution.
        propagation = L @ problem.A / np.sqrt(alpha[0])
        process = L @ _sqrt(problem.Q / (scale * alpha[1]))
        measurement = -Y @ _sqrt(problem.R / (scale * alpha[2]))
        block = cp.bmat([
            [X, nn, nm, propagation.T],
            [nn, np.eye(n), nm, process.T],
            [nm.T, nm.T, np.eye(m), measurement.T],
            [propagation, process, measurement, X],
        ])
    else:
        block = cp.bmat([
            [alpha[0] * X, nn, nm, problem.A.T @ L.T],
            [nn, alpha[1] * np.linalg.inv(problem.Q / scale), nm, L.T],
            [nm.T, nm.T, alpha[2] * np.linalg.inv(problem.R / scale), -Y.T],
            [L @ problem.A, L, -Y, X],
        ])
    optimization = cp.Problem(cp.Minimize(cp.trace(weight @ T)),
                              [block >> 0, cp.bmat([[T, np.eye(n)], [np.eye(n), X]]) >> 0])
    optimization.solve(**SOLVER_OPTIONS)
    if optimization.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or X.value is None:
        raise RuntimeError(f"fixed-weight reference SDP failed: {optimization.status}")
    P = sym(transform @ np.linalg.inv(X.value) @ transform.T) * scale
    K = transform @ np.linalg.solve(X.value, Y.value)
    return SdpDesign(K, alpha, P, None, float(np.trace(P)), optimization.status, optimization.solver_stats.num_iters)


def _sqrt(matrix):
    values, vectors = np.linalg.eigh(sym(matrix))
    if values[0] < -1e-10 * max(np.linalg.norm(matrix), 1e-300):
        raise ValueError("square root needs a positive semidefinite matrix")
    return vectors * np.sqrt(np.maximum(values, 0))


class OneStepCombinedSdp:
    """Reusable convex one-step exact spectral-risk comparator."""

    def __init__(self, problem: CombinedParetoProblem, alpha_floor=1e-8):
        validate_combined_problem(problem)
        self.problem = problem
        self.scale = max(sum(np.trace(getattr(problem, name)) for name in
                             ("Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded")) / (problem.n + problem.m), 1e-12)
        n, m = problem.n, problem.m
        self.K, self.alpha, self.t = cp.Variable((n, m)), cp.Variable(3), cp.Variable()
        self.propagation_root, self.stochastic_root = cp.Parameter((n, n)), cp.Parameter((n, n))
        L = np.eye(n) - self.K @ problem.H
        B0 = L @ self.propagation_root
        Bw = L @ _sqrt(problem.Q_bounded / self.scale)
        Bv = self.K @ _sqrt(problem.R_bounded / self.scale)
        nn, nm = np.zeros((n, n)), np.zeros((n, m))
        block = cp.bmat([
            [self.t * np.eye(n), B0, Bw, Bv],
            [B0.T, self.alpha[0] * np.eye(n), nn, nm],
            [Bw.T, nn, self.alpha[1] * np.eye(n), nm],
            [Bv.T, nm.T, nm.T, self.alpha[2] * np.eye(m)],
        ])
        variance = cp.sum_squares(L @ self.stochastic_root)
        variance += cp.sum_squares(self.K @ _sqrt(problem.R_stochastic / self.scale))
        self.optimization = cp.Problem(cp.Minimize(variance + self.t),
                                       [block >> 0, self.alpha >= alpha_floor, cp.sum(self.alpha) == 1])

    def solve(self, Sigma, P) -> SdpDesign:
        problem = self.problem
        self.propagation_root.value = problem.A @ _sqrt(P / self.scale)
        self.stochastic_root.value = _sqrt((problem.A @ Sigma @ problem.A.T + problem.Q_stochastic) / self.scale)
        self.optimization.solve(**SOLVER_OPTIONS, warm_start=True)
        status = self.optimization.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.K.value is None:
            raise RuntimeError(f"combined one-step SDP failed: {status}")
        K, alpha = self.K.value, normalize_alpha(self.alpha.value)
        L = np.eye(problem.n) - K @ problem.H
        F = L @ problem.A
        next_Sigma = sym(F @ Sigma @ F.T + L @ problem.Q_stochastic @ L.T + K @ problem.R_stochastic @ K.T)
        next_P = sym(F @ P @ F.T / alpha[0] + L @ problem.Q_bounded @ L.T / alpha[1]
                     + K @ problem.R_bounded @ K.T / alpha[2])
        value = float(np.trace(next_Sigma) + np.linalg.eigvalsh(next_P)[-1])
        return SdpDesign(K, alpha, next_P, next_Sigma, value, status, self.optimization.solver_stats.num_iters)


def greedy_combined_mse(problem, alpha_floor=1e-8, max_iter=1000, tolerance=1e-7,
                        verification_tolerance=1e-6) -> SdpDesign:
    """Iterate globally convex one-step SDPs to a checked steady-state design."""
    if max_iter < 1 or not 0 < tolerance <= verification_tolerance:
        raise ValueError("invalid greedy convergence settings")
    solver = OneStepCombinedSdp(problem, alpha_floor)
    Sigma, P = np.eye(problem.n) * solver.scale, np.eye(problem.n) * solver.scale
    total_iterations = 0
    inaccurate = False
    minimum_residual = float("inf")
    for _ in range(max_iter):
        result = solver.solve(Sigma, P)
        total_iterations += result.iterations
        inaccurate |= result.status == cp.OPTIMAL_INACCURATE
        residual = max(np.linalg.norm(result.Sigma - Sigma) / max(np.linalg.norm(Sigma), 1e-12),
                       np.linalg.norm(result.P - P) / max(np.linalg.norm(P), 1e-12))
        minimum_residual = min(minimum_residual, residual)
        Sigma, P = result.Sigma, result.P
        if residual <= tolerance:
            fixed = combined_mse_value_gradient(problem, result.K, result.alpha)
            checked = max(np.linalg.norm(fixed.Sigma - Sigma) / max(np.linalg.norm(Sigma), 1e-12),
                          np.linalg.norm(fixed.P - P) / max(np.linalg.norm(P), 1e-12))
            if checked > verification_tolerance:
                continue
            # Re-solve at independently computed steady descriptors, not the
            # recursively propagated approximations used by the stopping test.
            response = solver.solve(fixed.Sigma, fixed.P)
            total_iterations += response.iterations
            inaccurate |= response.status == cp.OPTIMAL_INACCURATE
            response_residual = max(
                np.linalg.norm(response.Sigma - fixed.Sigma) / max(np.linalg.norm(fixed.Sigma), 1e-12),
                np.linalg.norm(response.P - fixed.P) / max(np.linalg.norm(fixed.P), 1e-12))
            risk_gap = abs(response.value - fixed.value) / max(abs(fixed.value), 1e-12)
            if max(response_residual, risk_gap) > verification_tolerance:
                continue
            return SdpDesign(result.K, result.alpha, fixed.P, fixed.Sigma, fixed.value,
                             "optimal_inaccurate" if inaccurate else "optimal", total_iterations,
                             checked, response_residual, risk_gap)
    raise RuntimeError(f"combined greedy fixed point did not converge; minimum step residual {minimum_residual:g}")
