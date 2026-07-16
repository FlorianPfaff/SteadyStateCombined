"""Ellipsoidal steady-state set-membership filtering utilities.

The functions in this module implement the fixed-gain part of the theory in
the companion paper repository. The focus is the approximation weights of the
outer ellipsoidal bound

    E(S_0) + E(S_w) + E(S_v) subset E(S_0/a_0 + S_w/a_w + S_v/a_v).

The clean theorem currently applies to the fixed-gain recursion

    P = F P F^T / alpha_0 + S_w / alpha_w + S_v / alpha_v.

The code is intentionally explicit and NumPy-only, so that results are easy to
inspect and reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np

Criterion = Literal["trace", "logdet"]

_EPS = 1e-12


@dataclass(frozen=True)
class FixedGainProblem:
    """Linear set-membership filtering problem with a fixed observer gain.

    The system is

        x_{k+1} = A x_k + B u_k + w_k,
        y_{k+1} = H x_{k+1} + v_{k+1},

    with ``w_k in E(Q)`` and ``v_k in E(R)``. The fixed gain ``K`` induces the
    posterior error dynamics

        e_{k+1} = F e_k + G_w w_k + G_v v_k.
    """

    A: np.ndarray
    H: np.ndarray
    K: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    name: str = "fixed_gain_problem"

    def __post_init__(self) -> None:
        object.__setattr__(self, "A", np.asarray(self.A, dtype=float))
        object.__setattr__(self, "H", np.asarray(self.H, dtype=float))
        object.__setattr__(self, "K", np.asarray(self.K, dtype=float))
        object.__setattr__(self, "Q", sym(np.asarray(self.Q, dtype=float)))
        object.__setattr__(self, "R", sym(np.asarray(self.R, dtype=float)))
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if self.H.shape[1] != n:
            raise ValueError("H must have n columns")
        if self.K.shape[0] != n or self.K.shape[1] != self.H.shape[0]:
            raise ValueError("K must have shape (n, m), where H has shape (m, n)")
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

    @property
    def F(self) -> np.ndarray:
        return (np.eye(self.n) - self.K @ self.H) @ self.A

    @property
    def G_w(self) -> np.ndarray:
        return np.eye(self.n) - self.K @ self.H

    @property
    def G_v(self) -> np.ndarray:
        return -self.K

    @property
    def S_w(self) -> np.ndarray:
        return sym(self.G_w @ self.Q @ self.G_w.T)

    @property
    def S_v(self) -> np.ndarray:
        return sym(self.G_v @ self.R @ self.G_v.T)


def sym(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def matrix_inner(left: np.ndarray, right: np.ndarray) -> float:
    """Frobenius inner product ``tr(left^T right)``."""

    return float(np.trace(left.T @ right))


def spectral_radius(matrix: np.ndarray) -> float:
    """Spectral radius of a square matrix."""

    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def simplex_grid(resolution: int) -> Iterable[np.ndarray]:
    """Yield strictly positive three-component simplex points.

    ``resolution=101`` produces points with coordinates multiples of 1/101,
    excluding the boundary.
    """

    if resolution < 4:
        raise ValueError("resolution must be at least 4")
    for i in range(1, resolution):
        alpha_0 = i / resolution
        for j in range(1, resolution - i):
            alpha_w = j / resolution
            alpha_v = 1.0 - alpha_0 - alpha_w
            if alpha_v > 0.0:
                yield np.array([alpha_0, alpha_w, alpha_v], dtype=float)


def normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    """Validate and normalize approximation weights."""

    alpha = np.asarray(alpha, dtype=float).reshape(3)
    if not np.all(np.isfinite(alpha)):
        raise ValueError("alpha contains non-finite entries")
    if np.any(alpha <= 0.0):
        raise ValueError("alpha entries must be strictly positive")
    total = float(np.sum(alpha))
    if total <= 0.0:
        raise ValueError("alpha must have positive sum")
    return alpha / total


def criterion_value(P: np.ndarray, criterion: Criterion = "trace") -> float:
    """Evaluate an ellipsoid quality criterion."""

    P = sym(P)
    if criterion == "trace":
        return float(np.trace(P))
    if criterion == "logdet":
        sign, logdet = np.linalg.slogdet(P)
        if sign <= 0:
            return float("inf")
        return float(logdet)
    raise ValueError(f"Unknown criterion: {criterion}")


def is_feasible_fixed_gain(problem: FixedGainProblem, alpha: np.ndarray, margin: float = 1e-10) -> bool:
    """Check the fixed-gain stability condition rho(F)/sqrt(alpha_0)<1."""

    alpha = normalize_alpha(alpha)
    return spectral_radius(problem.F) / np.sqrt(alpha[0]) < 1.0 - margin


def solve_fixed_gain_steady_state(problem: FixedGainProblem, alpha: np.ndarray) -> Optional[np.ndarray]:
    """Solve the fixed-gain steady-state shape equation.

    Returns ``None`` if the fixed-gain feasibility condition fails or the
    resulting linear system is singular/numerically invalid.
    """

    alpha = normalize_alpha(alpha)
    if not is_feasible_fixed_gain(problem, alpha):
        return None

    F = problem.F
    n = problem.n
    C = problem.S_w / alpha[1] + problem.S_v / alpha[2]
    system = np.eye(n * n) - np.kron(F, F) / alpha[0]
    try:
        vec_P = np.linalg.solve(system, C.reshape(-1, order="F"))
    except np.linalg.LinAlgError:
        return None
    P = sym(vec_P.reshape(n, n, order="F"))
    if not np.all(np.isfinite(P)):
        return None
    if np.min(np.linalg.eigvalsh(P)) <= -1e-8:
        return None
    return P


def component_matrices(problem: FixedGainProblem, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``S_0=FPF^T``, ``S_w``, and ``S_v``."""

    return sym(problem.F @ P @ problem.F.T), problem.S_w, problem.S_v


def stepwise_trace_weights(problem: FixedGainProblem, P: np.ndarray) -> np.ndarray:
    """Trace-minimal one-step weights for the current component matrices."""

    components = component_matrices(problem, P)
    traces = np.maximum(np.array([np.trace(S) for S in components], dtype=float), _EPS)
    alpha = np.sqrt(traces)
    return alpha / float(np.sum(alpha))


def stepwise_trace_steady_state(
    problem: FixedGainProblem,
    initial_P: Optional[np.ndarray] = None,
    max_iter: int = 10_000,
    tol: float = 1e-11,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Iterate the stepwise trace-minimal approximation until convergence.

    Returns ``(alpha, P)`` where ``alpha`` is the limiting current-step weight
    vector and ``P`` the limiting shape matrix.
    """

    P = np.eye(problem.n) if initial_P is None else sym(initial_P)
    alpha = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    for _ in range(max_iter):
        alpha = stepwise_trace_weights(problem, P)
        S0, Sw, Sv = component_matrices(problem, P)
        P_next = sym(S0 / alpha[0] + Sw / alpha[1] + Sv / alpha[2])
        if not np.all(np.isfinite(P_next)) or np.linalg.norm(P_next, ord="fro") > 1e14:
            return None
        rel = np.linalg.norm(P_next - P, ord="fro") / (1.0 + np.linalg.norm(P, ord="fro"))
        P = P_next
        if rel < tol:
            return alpha, P
    return None


def grid_optimize_weights(
    problem: FixedGainProblem,
    resolution: int = 121,
    criterion: Criterion = "trace",
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Optimize fixed-gain approximation weights on a dense simplex grid."""

    best: Optional[tuple[np.ndarray, np.ndarray, float]] = None
    rho2 = spectral_radius(problem.F) ** 2
    for alpha in simplex_grid(resolution):
        if alpha[0] <= rho2 + 1e-10:
            continue
        P = solve_fixed_gain_steady_state(problem, alpha)
        if P is None:
            continue
        value = criterion_value(P, criterion)
        if best is None or value < best[2]:
            best = (alpha, P, value)
    return best


def refine_fixed_gain_weights(
    problem: FixedGainProblem,
    alpha_start: np.ndarray,
    criterion: Criterion = "trace",
    max_iter: int = 200,
    tol: float = 1e-10,
    max_halvings: int = 30,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Refine feasible weights with repeated adjoint-weighted line searches.

    Each update points toward the minimizer of the local adjoint surrogate.
    The backtracking line search accepts only strict objective decrease, so the
    returned solution is never worse than ``alpha_start``.
    """

    alpha = normalize_alpha(alpha_start)
    P = solve_fixed_gain_steady_state(problem, alpha)
    if P is None:
        return None
    value = criterion_value(P, criterion)

    for _ in range(max_iter):
        alpha_target, _, _, _ = nonmyopic_weights(problem, alpha, P, criterion)
        direction = alpha_target - alpha
        if np.linalg.norm(direction) <= tol:
            break

        accepted = False
        for halving in range(max_halvings + 1):
            step = 1.0 / (2**halving)
            candidate_alpha = normalize_alpha(alpha + step * direction)
            candidate_P = solve_fixed_gain_steady_state(problem, candidate_alpha)
            if candidate_P is None:
                continue
            candidate_value = criterion_value(candidate_P, criterion)
            decrease_tol = 1e-13 * (1.0 + abs(value))
            if candidate_value < value - decrease_tol:
                alpha = candidate_alpha
                P = candidate_P
                value = candidate_value
                accepted = True
                break
        if not accepted:
            break

    return alpha, P, value


def optimize_fixed_gain_weights(
    problem: FixedGainProblem,
    resolution: int = 121,
    criterion: Criterion = "trace",
    initial_alphas: Iterable[np.ndarray] = (),
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Combine a deterministic simplex grid with local adjoint refinement.

    ``initial_alphas`` can include a baseline design. This makes finite-grid
    benchmark comparisons conservative: the reported optimized result cannot
    be worse merely because the grid omitted the baseline weight vector.
    """

    starts: list[np.ndarray] = []
    grid_result = grid_optimize_weights(problem, resolution=resolution, criterion=criterion)
    if grid_result is not None:
        starts.append(grid_result[0])
    starts.extend(normalize_alpha(alpha) for alpha in initial_alphas)

    best: Optional[tuple[np.ndarray, np.ndarray, float]] = None
    for alpha in starts:
        result = refine_fixed_gain_weights(problem, alpha, criterion=criterion)
        if result is not None and (best is None or result[2] < best[2]):
            best = result
    return best


def solve_adjoint(problem: FixedGainProblem, alpha: np.ndarray, P: np.ndarray, criterion: Criterion = "trace") -> np.ndarray:
    """Solve Lambda = grad J(P) + F^T Lambda F / alpha_0."""

    alpha = normalize_alpha(alpha)
    F = problem.F
    n = problem.n
    if criterion == "trace":
        gradient = np.eye(n)
    elif criterion == "logdet":
        gradient = np.linalg.inv(sym(P))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    system = np.eye(n * n) - np.kron(F.T, F.T) / alpha[0]
    vec_L = np.linalg.solve(system, gradient.reshape(-1, order="F"))
    return sym(vec_L.reshape(n, n, order="F"))


def nonmyopic_weights(
    problem: FixedGainProblem,
    alpha: np.ndarray,
    P: Optional[np.ndarray] = None,
    criterion: Criterion = "trace",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute adjoint-weighted non-myopic approximation weights.

    Returns ``(alpha_nm, P, Lambda, contributions)``.
    """

    alpha = normalize_alpha(alpha)
    if P is None:
        P = solve_fixed_gain_steady_state(problem, alpha)
        if P is None:
            raise ValueError("alpha is not feasible for the fixed-gain steady-state problem")
    Lambda = solve_adjoint(problem, alpha, P, criterion)
    contributions = np.array([matrix_inner(Lambda, S) for S in component_matrices(problem, P)], dtype=float)
    contributions = np.maximum(contributions, _EPS)
    alpha_nm = np.sqrt(contributions)
    alpha_nm /= float(np.sum(alpha_nm))
    return alpha_nm, P, Lambda, contributions


def line_search_to_nonmyopic(
    problem: FixedGainProblem,
    alpha_start: np.ndarray,
    criterion: Criterion = "trace",
    max_halvings: int = 30,
) -> list[dict[str, float | int]]:
    """Evaluate objective values from ``alpha_start`` toward non-myopic weights."""

    alpha_start = normalize_alpha(alpha_start)
    P_start = solve_fixed_gain_steady_state(problem, alpha_start)
    if P_start is None:
        raise ValueError("alpha_start is infeasible")
    start_value = criterion_value(P_start, criterion)
    alpha_nm, _, _, _ = nonmyopic_weights(problem, alpha_start, P_start, criterion)

    rows: list[dict[str, float | int]] = []
    for halving in range(max_halvings + 1):
        epsilon = 1.0 / (2**halving)
        alpha = normalize_alpha((1.0 - epsilon) * alpha_start + epsilon * alpha_nm)
        P = solve_fixed_gain_steady_state(problem, alpha)
        value = float("inf") if P is None else criterion_value(P, criterion)
        rows.append(
            {
                "epsilon": float(epsilon),
                "alpha0": float(alpha[0]),
                "alphaw": float(alpha[1]),
                "alphav": float(alpha[2]),
                "value": float(value),
                "improved": int(value < start_value - 1e-10),
            }
        )
    return rows


def sample_from_ellipsoid(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from the interior of E(P), up to numerical precision."""

    P = sym(P)
    n = P.shape[0]
    direction = rng.normal(size=n)
    norm = max(float(np.linalg.norm(direction)), _EPS)
    direction /= norm
    radius = rng.uniform(0.0, 1.0) ** (1.0 / n)
    L = np.linalg.cholesky(P + 1e-14 * np.eye(n))
    return L @ (radius * direction)


def bounded_error_check(
    problem: FixedGainProblem,
    P: np.ndarray,
    rng: np.random.Generator,
    trajectories: int = 200,
    horizon: int = 50,
) -> dict[str, float | int]:
    """Monte-Carlo containment check for bounded disturbances.

    This is not a proof of invariance; it is a regression test and a useful
    sanity check for numerical implementation.
    """

    inv_P = np.linalg.inv(sym(P))
    max_normalized_error = -float("inf")
    violations = 0
    for _ in range(trajectories):
        e = sample_from_ellipsoid(P, rng)
        for _ in range(horizon):
            w = sample_from_ellipsoid(problem.Q, rng)
            v = sample_from_ellipsoid(problem.R, rng)
            e = problem.F @ e + problem.G_w @ w + problem.G_v @ v
            normalized = float(e.T @ inv_P @ e)
            max_normalized_error = max(max_normalized_error, normalized)
            if normalized > 1.0 + 1e-8:
                violations += 1
    return {
        "trajectories": trajectories,
        "horizon": horizon,
        "max_normalized_error": float(max_normalized_error),
        "violations": int(violations),
    }
