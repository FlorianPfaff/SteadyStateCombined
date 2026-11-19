"""Controlled tracking validation with noise-independent bounded disturbances."""

from dataclasses import dataclass

import numpy as np

from .combined_mse import CombinedMseEvaluation
from .combined_pareto import CombinedParetoProblem
from .ellipsoidal import sym


def psd_root(matrix):
    values, vectors = np.linalg.eigh(sym(matrix))
    if values[0] < -1e-12 * max(np.linalg.norm(matrix), 1e-300):
        raise ValueError("matrix must be positive semidefinite")
    return vectors * np.sqrt(np.maximum(values, 0))


def directional_bias_sequence(problem, gain, direction, horizon):
    """Exactly maximize terminal directional bias over independent ellipsoids.

    The direction and observer are fixed before Gaussian noise is generated.
    This is a directional support maximizer, not a global norm maximizer.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    L = np.eye(problem.n) - gain @ problem.H
    F = L @ problem.A
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (problem.n,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        raise ValueError("invalid support direction")
    w, v = np.zeros((horizon, problem.n)), np.zeros((horizon, problem.m))
    costate = direction / np.linalg.norm(direction)
    for t in range(horizon - 1, -1, -1):
        for destination, shape, gradient in ((w, problem.Q_bounded, L.T @ costate),
                                             (v, problem.R_bounded, -gain.T @ costate)):
            denominator = np.sqrt(max(float(gradient @ shape @ gradient), 0))
            if denominator > np.finfo(float).tiny:
                destination[t] = shape @ gradient / denominator
        costate = F.T @ costate
    return w, v


def bounded_profiles(problem, designs, horizon, seed=20261003, episode=200):
    """Constant, switching, and cross-evaluated support-maximizing profiles."""
    if horizon < 1 or episode < 1:
        raise ValueError("positive horizons required")
    rng = np.random.default_rng(seed)
    roots = (psd_root(problem.Q_bounded), psd_root(problem.R_bounded))
    unit = []
    for size in (problem.n, problem.m):
        direction = rng.normal(size=size)
        unit.append(direction / np.linalg.norm(direction))
    profiles = {"constant": tuple(np.tile(root @ u, (horizon, 1)) for root, u in zip(roots, unit))}
    segments = (horizon + 49) // 50
    switching = []
    for root in roots:
        directions = rng.normal(size=(segments, root.shape[1]))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        switching.append(np.repeat(directions @ root.T, 50, axis=0)[:horizon])
    profiles["switching"] = tuple(switching)
    for name, design in designs.items():
        direction = np.linalg.eigh(design.P)[1][:, -1]
        one = directional_bias_sequence(problem, design.K, direction, episode)
        profiles[f"support_{name}"] = tuple(np.tile(array, ((horizon + episode - 1) // episode, 1))[:horizon]
                                             for array in one)
    return profiles


@dataclass(frozen=True)
class TrackingSimulation:
    rows: list[dict]
    paired_rows: list[dict]
    trajectories: dict[str, np.ndarray]


def simulate_profile(
    problem: CombinedParetoProblem,
    designs: dict[str, CombinedMseEvaluation],
    w_bias, v_bias, *, trials=2000, burn_in=200, seed=431, dt=0.1,
) -> TrackingSimulation:
    """Paired Monte Carlo plus analytic finite-time mean squared error.

    Confidence intervals use independent trajectory-level averages, preserving
    temporal dependence. No claim of bounded total Gaussian error is made.
    """
    w_bias, v_bias = np.asarray(w_bias), np.asarray(v_bias)
    horizon = len(w_bias)
    if trials < 2 or not 0 <= burn_in < horizon or dt <= 0:
        raise ValueError("invalid simulation settings")
    if w_bias.shape != (horizon, problem.n) or v_bias.shape != (horizon, problem.m):
        raise ValueError("disturbance profile dimensions disagree")
    if problem.n != 6:
        raise ValueError("physical tracking metrics require six position/velocity states")
    # This physical model has positive-definite disturbance shapes.
    disturbance_ratio = max(np.max(np.sum(w_bias * np.linalg.solve(problem.Q_bounded, w_bias.T).T, axis=1)),
                            np.max(np.sum(v_bias * np.linalg.solve(problem.R_bounded, v_bias.T).T, axis=1)))
    if disturbance_ratio > 1 + 1e-8:
        raise ValueError("profile leaves its prescribed disturbance ellipsoid")
    names = list(designs)
    count, n = len(names), problem.n
    gains = np.array([designs[name].K for name in names])
    L = np.eye(n) - gains @ problem.H
    F = L @ problem.A
    invP = np.array([np.linalg.inv(designs[name].P) for name in names])
    z, bias, covariance = np.zeros((count, trials, n)), np.zeros((count, n)), np.zeros((count, n, n))
    force = L @ problem.Q_stochastic @ L.transpose(0, 2, 1) + gains @ problem.R_stochastic @ gains.transpose(0, 2, 1)
    qroot, rroot = psd_root(problem.Q_stochastic), psd_root(problem.R_stochastic)
    rng = np.random.default_rng(seed)
    cluster = np.zeros((count, trials))
    path = np.zeros((horizon, count, 9))
    first_error = np.zeros((horizon, count, n))
    for t in range(horizon):
        ws = rng.normal(size=(trials, n)) @ qroot.T
        vs = rng.normal(size=(trials, problem.m)) @ rroot.T
        z = z @ F.transpose(0, 2, 1) + ws @ L.transpose(0, 2, 1) - vs @ gains.transpose(0, 2, 1)
        bias = np.einsum("dij,dj->di", F, bias) + np.einsum("dij,j->di", L, w_bias[t]) - np.einsum("dij,j->di", gains, v_bias[t])
        covariance = F @ covariance @ F.transpose(0, 2, 1) + force
        error = z + bias[:, None, :]
        squared = np.sum(error**2, axis=2)
        expected = np.trace(covariance, axis1=1, axis2=2) + np.sum(bias**2, axis=1)
        variance_position = np.trace(covariance[:, :3, :3], axis1=1, axis2=2)
        variance_velocity = np.trace(covariance[:, 3:, 3:], axis1=1, axis2=2) / dt**2
        bias_ratio = np.einsum("di,dij,dj->d", bias, invP, bias)
        path[t] = np.column_stack((squared.mean(axis=1), expected, np.sum(bias**2, axis=1),
                                  np.trace(covariance, axis1=1, axis2=2), bias_ratio,
                                  np.mean(np.sum(error[:, :, :3]**2, axis=2), axis=1),
                                  np.mean(np.sum(error[:, :, 3:]**2, axis=2), axis=1) / dt**2,
                                  variance_position + np.sum(bias[:, :3]**2, axis=1),
                                  variance_velocity + np.sum(bias[:, 3:]**2, axis=1) / dt**2))
        first_error[t] = error[:, 0]
        if t >= burn_in:
            cluster += squared / (horizon - burn_in)
    rows = []
    for index, name in enumerate(names):
        stderr = float(np.std(cluster[index], ddof=1) / np.sqrt(trials))
        observed = float(np.mean(cluster[index]))
        expected = float(np.mean(path[burn_in:, index, 1]))
        rows.append(dict(method=name, mse=observed, mse_se=stderr, mse_ci_low=observed - 1.96 * stderr,
                         mse_ci_high=observed + 1.96 * stderr, analytic_mse=expected,
                         monte_carlo_z=(observed - expected) / max(stderr, 1e-300),
                         analytic_peak_mse=float(np.max(path[:, index, 1])),
                         position_rmse=float(np.sqrt(np.mean(path[burn_in:, index, 5]))),
                         velocity_rmse=float(np.sqrt(np.mean(path[burn_in:, index, 6]))),
                         analytic_position_rmse=float(np.sqrt(np.mean(path[burn_in:, index, 7]))),
                         analytic_velocity_rmse=float(np.sqrt(np.mean(path[burn_in:, index, 8]))),
                         maximum_bias_ellipsoid_ratio=float(np.max(path[:, index, 4])),
                         maximum_disturbance_ellipsoid_ratio=float(disturbance_ratio),
                         steady_risk_bound=designs[name].value))
    paired = []
    for index, name in enumerate(names):
        if name == "greedy" or "greedy" not in names:
            continue
        greedy = names.index("greedy")
        difference = cluster[index] - cluster[greedy]
        mean, se = float(np.mean(difference)), float(np.std(difference, ddof=1) / np.sqrt(trials))
        paired.append(dict(method=name, comparator="greedy", mse_difference=mean, difference_se=se,
                           difference_ci_low=mean - 1.96 * se, difference_ci_high=mean + 1.96 * se,
                           analytic_difference=float(np.mean(path[burn_in:, index, 1] - path[burn_in:, greedy, 1]))))
    return TrackingSimulation(rows, paired, {"metrics": path, "first_error": first_error,
                                             "cluster_mse": cluster, "method_names": np.array(names)})
