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

## First evaluation: fixed-gain theorem

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

## Second evaluation: gain-reoptimized Riccati sweep

Run:

```bash
python examples/run_gain_optimized_evaluation.py --out results_riccati --random-systems 100 --grid 121 --step-grid 61 --seed 13
```

This compares:

1. a greedy baseline that reoptimizes the gain and approximation weights at every step;
2. a steady-state Riccati sweep that solves the scaled DARE for each fixed approximation-weight vector and optimizes the invariant ellipsoid over the simplex.

Generated outputs include:

- `results_riccati/riccati_deterministic_summary.csv`
- `results_riccati/riccati_random_benchmark.csv`
- `results_riccati/riccati_random_summary.csv`
- `results_riccati/riccati_random_scatter.png` if Matplotlib is available

## What the current code evaluates

The fixed-gain implementation focuses on the theorem:

```text
P = F P F^T / alpha_0 + S_w / alpha_w + S_v / alpha_v.
```

It compares:

1. repeated stepwise trace-minimal ellipsoidal approximation;
2. steady-state approximation-weight optimization over the simplex;
3. the adjoint-weighted non-myopic direction from the stepwise weights.

The gain-reoptimized implementation evaluates the practical fixed-alpha Riccati equation:

```text
S = A P A^T / alpha_0 + Q / alpha_w
P = S - S H^T (H S H^T + R / alpha_v)^(-1) H S
```

and compares a steady-state simplex sweep against a recursive greedy gain/weight baseline.

## Suggested stronger runs

```bash
python examples/run_fixed_gain_evaluation.py --out results_grid201 --random-systems 500 --grid 201 --seed 11
python examples/run_gain_optimized_evaluation.py --out results_riccati_grid201 --random-systems 300 --grid 201 --step-grid 101 --seed 17
```

Use these for stronger paper figures and tables.

## Tests

```bash
pytest
```

## Repository layout

```text
src/steady_state_combined/
  ellipsoidal.py     fixed-gain ellipsoidal recursions, optimization, adjoint rule
  riccati.py         gain-reoptimized fixed-alpha Riccati sweeps
  combined.py        fixed-gain combined stochastic/set-membership helpers
  examples.py        deterministic and random benchmark systems
examples/
  run_fixed_gain_evaluation.py
  run_gain_optimized_evaluation.py
```

## Next code steps

- Add a combined stochastic/set-membership Pareto experiment.
- Add paper-figure export helpers that write directly into the paper repository's expected `figures/` and `tables/` folders.
