# SteadyStateCombined

Code for evaluating steady-state approximation design in ellipsoidal set-membership filtering.

The companion paper/notes/results repository is separate:

- `FlorianPfaff/2026-07-SteadyStateCombined-Paper`

This repository contains reusable Python code and runnable experiments.

## Continuous integration

GitHub Actions runs the Python test suite and three small smoke evaluations on each push and pull request:

- fixed-gain evaluation;
- gain-reoptimized Riccati evaluation;
- combined stochastic/set-membership Pareto evaluation.

The smoke-result folders are uploaded as a workflow artifact named `smoke-evaluation-results`.

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

## Third evaluation: combined stochastic/set-membership Pareto sweep

Run:

```bash
python examples/run_combined_pareto.py --out results_combined --alpha-grid 31 --gain-grid 25
```

This performs a deterministic 2D grid search over gains and approximation weights, then selects the best candidate for each scalarization

```text
lambda tr(Sigma) + (1-lambda) tr(P).
```

Generated outputs include:

- `results_combined/combined_pareto.csv`
- `results_combined/combined_pareto.png` if Matplotlib is available
- `results_combined/combined_candidate_summary.csv`

This experiment is intentionally presented as a low-dimensional diagnostic/Pareto study, not as a claim that the full combined gain-and-weight problem is convex.

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

The combined implementation evaluates fixed-gain/fixed-alpha steady-state descriptors

```text
Sigma = F Sigma F^T + G_w Q_s G_w^T + G_v R_s G_v^T
P     = F P F^T / alpha_0 + G_w Q_b G_w^T / alpha_w + G_v R_b G_v^T / alpha_v
```

and constructs a Pareto curve over stochastic covariance size and bounded-error ellipsoid size.

## Suggested stronger runs

```bash
python examples/run_fixed_gain_evaluation.py --out results_grid201 --random-systems 500 --grid 201 --seed 11
python examples/run_gain_optimized_evaluation.py --out results_riccati_grid201 --random-systems 300 --grid 201 --step-grid 101 --seed 17
python examples/run_combined_pareto.py --out results_combined_grid41 --alpha-grid 41 --gain-grid 41
```

Use these for stronger paper figures and tables.

With both repositories checked out as siblings, export generated artifacts into the paper repository with:

```bash
python scripts/export_results_to_paper.py --paper-root ../2026-07-SteadyStateCombined-Paper
```

or simply:

```bash
make export-paper
```

Then generate LaTeX table fragments in the paper repository with:

```bash
python scripts/generate_latex_tables.py --paper-root ../2026-07-SteadyStateCombined-Paper
```

or:

```bash
make tables-paper
```

The combined target

```bash
make paper-artifacts
```

exports CSV/figure files and generates table fragments.

The exporter copies CSV files into `../2026-07-SteadyStateCombined-Paper/results/`, figures into `../2026-07-SteadyStateCombined-Paper/figures/`, and writes `results/export_manifest.json`. The table generator writes LaTeX fragments into `../2026-07-SteadyStateCombined-Paper/tables/`.

## Tests

```bash
pytest
```

## Make targets

```bash
make test
make eval-fixed
make eval-riccati
make eval-combined
make eval-all
make export-paper
make tables-paper
make paper-artifacts
```

## Repository layout

```text
src/steady_state_combined/
  ellipsoidal.py        fixed-gain ellipsoidal recursions, optimization, adjoint rule
  riccati.py            gain-reoptimized fixed-alpha Riccati sweeps
  combined.py           fixed-gain combined stochastic/set-membership helpers
  combined_pareto.py    deterministic combined Pareto grid search
  examples.py           deterministic and random benchmark systems
examples/
  run_fixed_gain_evaluation.py
  run_gain_optimized_evaluation.py
  run_combined_pareto.py
scripts/
  export_results_to_paper.py
  generate_latex_tables.py
```

## Next code steps

- Insert generated result tables and figures into `main.tex` once the evaluations have been run.
