"""Built-in example problems for evaluation."""

from __future__ import annotations

import math

import numpy as np

from .ellipsoidal import FixedGainProblem, sym


def rotation(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def deterministic_fixed_gain_problem() -> FixedGainProblem:
    """A 2D orientation-sensitive fixed-gain benchmark.

    The example is chosen so the steady-state approximation weights noticeably
    improve over the repeated stepwise trace-minimal weights.
    """

    A = np.array(
        [
            [0.87808946, -0.10955969],
            [0.10955969, 0.87808946],
        ],
        dtype=float,
    )
    H = np.array([[1.0, 0.0]], dtype=float)
    K = np.array([[1.02894003], [-0.49389588]], dtype=float)
    Q = np.array(
        [
            [0.00510331, -0.00519052],
            [-0.00519052, 0.00607184],
        ],
        dtype=float,
    )
    R = np.array([[0.07701328]], dtype=float)
    return FixedGainProblem(A=A, H=H, K=K, Q=Q, R=R, name="deterministic_2d")


def random_spd_2d(rng: np.random.Generator, scale_low: float = -4.0, scale_high: float = -1.0) -> np.ndarray:
    q1 = 10 ** rng.uniform(scale_low, scale_high)
    q2 = q1 * 10 ** rng.uniform(-2.0, -0.2)
    phi = rng.uniform(0.0, math.pi)
    return sym(rotation(phi) @ np.diag([q1, q2]) @ rotation(phi).T)


def random_fixed_gain_problem(rng: np.random.Generator) -> FixedGainProblem | None:
    """Generate a random stable 2D fixed-gain problem."""

    from .ellipsoidal import spectral_radius

    H = np.array([[1.0, 0.0]], dtype=float)
    for _ in range(200):
        rho = rng.uniform(0.72, 0.98)
        theta = rng.uniform(math.radians(3.0), math.radians(55.0))
        A = rho * rotation(theta)
        K = np.array([[rng.uniform(0.25, 1.20)], [rng.uniform(-0.65, 0.65)]], dtype=float)
        Q = random_spd_2d(rng)
        R = np.array([[10 ** rng.uniform(-4.0, -1.0)]], dtype=float)
        problem = FixedGainProblem(A=A, H=H, K=K, Q=Q, R=R, name="random_2d")
        if spectral_radius(problem.F) < 0.90:
            return problem
    return None
