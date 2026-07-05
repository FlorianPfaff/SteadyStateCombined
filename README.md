# SteadyStateCombined

Code for evaluating steady-state approximation design in ellipsoidal set-membership filtering.

The companion paper/notes/results repository is separate:

- `FlorianPfaff/2026-07-SteadyStateCombined-Paper`

This repository contains reusable Python code and runnable experiments.

## Installation

From a fresh checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,plot]
```

For the minimal evaluation, only NumPy is required. Matplotlib is optional and only needed for PNG figures.

## First evaluation

Run:

```bash
python examples/run_fixed_gain_evaluation.py --out results --random-systems 200 --grid 121 --seed 7
```

This creates:

- `results/deterministic_summary.csv`
- `results/line_search_curve.csv`
- `results/line_search_curve.png` if Matplotlib is available
- `results/random_benchmark.csv`
- `results/random_summary.csv`
- `results/random_scatter.png` if Matplotlib is available
- `results/bound_check.csv`

## What the current code evaluates

The current implementation focuses on the fixed-gain theorem:

```text
P = F P F^T / alpha_0 + S_w / alpha_w + S_v / alpha_v.
```

It compares:

1. repeated stepwise trace-minimal ellipsoidal approximation;
2. steady-state approximation-weight optimization over the simplex;
3. the adjoint-weighted non-myopic direction from the stepwise weights.

This is the cleanest match to the local improvement theorem in the paper draft.

## Suggested stronger run

```bash
python examples/run_fixed_gain_evaluation.py --out results_grid201 --random-systems 500 --grid 201 --seed 11
```

Use this for stronger paper figures and tables.

## Tests

```bash
pytest
```

## Repository layout

```text
src/steady_state_combined/
  ellipsoidal.py     fixed-gain ellipsoidal recursions, optimization, adjoint rule
  combined.py        fixed-gain combined stochastic/set-membership helpers
  examples.py        deterministic and random benchmark systems
examples/
  run_fixed_gain_evaluation.py
```

## Next code steps

- Add gain-reoptimized Riccati experiments.
- Add a combined stochastic/set-membership Pareto experiment.
- Add paper-figure export helpers that write directly into the paper repository's expected `figures/` and `tables/` folders.
