from __future__ import annotations

import numpy as np

from steady_state_combined import (
    criterion_value,
    grid_optimize_weights,
    line_search_to_nonmyopic,
    solve_fixed_gain_steady_state,
    stepwise_trace_steady_state,
)
from steady_state_combined.examples import deterministic_fixed_gain_problem


def test_deterministic_example_improves_trace() -> None:
    problem = deterministic_fixed_gain_problem()
    step = stepwise_trace_steady_state(problem)
    assert step is not None
    _, P_step = step
    opt = grid_optimize_weights(problem, resolution=81, criterion="trace")
    assert opt is not None
    _, P_opt, _ = opt
    assert criterion_value(P_opt, "trace") < criterion_value(P_step, "trace")


def test_line_search_has_improving_step() -> None:
    problem = deterministic_fixed_gain_problem()
    step = stepwise_trace_steady_state(problem)
    assert step is not None
    alpha_step, _ = step
    rows = line_search_to_nonmyopic(problem, alpha_step)
    assert any(row["improved"] for row in rows)


def test_fixed_alpha_solution_is_positive_semidefinite() -> None:
    problem = deterministic_fixed_gain_problem()
    alpha = np.array([0.7, 0.2, 0.1])
    P = solve_fixed_gain_steady_state(problem, alpha)
    assert P is not None
    assert np.min(np.linalg.eigvalsh(P)) > -1e-8
