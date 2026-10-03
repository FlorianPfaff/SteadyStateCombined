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

A separate manually triggered workflow, **Paper evaluation**, runs larger paper-oriented evaluations and uploads the result bundle. Trigger it from the GitHub Actions tab and choose one of:

- `smoke`: fast sanity check;
- `standard`: moderate-size evaluation suitable for draft iteration;
- `strong`: larger paper-strength evaluation.

The uploaded artifact is named `paper-evaluation-<profile>` and contains `results_grid201/`, `results_riccati_grid201/`, `results_combined_grid41/`, `results_validation/`, and `results_report.md`. The validation folder contains the continuous comparator, higher-dimensional cases, archived plants, and solver ablation. The report from the older grid evaluations is retained for historical comparison; use the continuous validation for gain-reoptimized scientific claims.

## Installation

From a fresh checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,plot]
```

For the minimal evaluation, only NumPy is required. Matplotlib is optional and only needed for PNG figures.

## Authoritative matched-accuracy paper validation

The earlier grid-based gain comparison used different numerical accuracy for the two designs. It overstated the median reduction on the original 300 plants (4.01% versus 2.23% after continuous refinement). The revised paper uses the following protocol, not that legacy number.

Install the optional SciPy solvers and run against the committed historical cohort:

```bash
python -m pip install -e '.[dev,plot,research]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/run_paper_validation.py \
  --out results_validation \
  --historical-csv ../2026-07-SteadyStateCombined-Paper/results/riccati/riccati_random_benchmark.csv \
  --random-systems 300 --per-dimension 50 --workers 16
python examples/run_solver_ablation.py --validation-dir results_validation --workers 16
```

The primary archive has 452 cases: one deterministic plant, 300 historical two-state plants, 50 each in dimensions 4, 8, and 12, and one six-state synthetic tracking model. No failures are discarded or replaced. The historical cohort was selected by the older evaluator; its acceptance indices are recorded, and the new ensembles are unconditioned predetermined draws. `systems.npz` contains every selected plant at full precision. CSV files and SHA-256 manifests retain numerical diagnostics, active-bound cases, timings, and source provenance.

The research module provides:

- globally solved fixed-gain weighted trace via convex scalar reduction;
- adjoint-only feasible Armijo iteration, with roundoff-level stagnation detection;
- independent SciPy DARE/Lyapunov solutions and residual checks;
- analytic one-step and DARE envelope gradients;
- continuous greedy and steady-state searches with matched tolerances and positivity floors.

The fixed-gain trace problem and gain-eliminated one-step trace problem are convex. The gain-reoptimized steady-state outer search is multistart and is **not globally certified**. Boundary cases are repeated at a smaller positivity floor. Reductions describe outer-bound trace, not observed MSE.

After a complete run, generate the authoritative paper tables and figure:

```bash
python ../2026-07-SteadyStateCombined-Paper/scripts/generate_validation_artifacts.py \
  --import-from results_validation
```

The generator verifies the archive hashes and derives values from individual cases. It intentionally requires the full 452-case paper profile. GitHub's smoke and standard profiles use smaller cohorts and cannot replace the paper archive. A freshly generated legacy cohort can differ from the committed historical cohort; pass the exact historical CSV above when reproducing the reported numbers.

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
2. steady-state approximation-weight optimization using a simplex grid followed by adjoint line-search refinement;
3. the adjoint-weighted non-myopic direction from the stepwise weights.

The gain-reoptimized implementation evaluates the practical fixed-alpha Riccati equation:

```text
S = A P A^T / alpha_0 + Q / alpha_w
P = S - S H^T (H S H^T + R / alpha_v)^(-1) H S
```

and compares a steady-state simplex sweep plus baseline-seeded pattern refinement against a recursive greedy gain/weight baseline. Including the baseline weight in the refinement prevents a finite grid from creating spurious ratios below one.

The combined implementation evaluates fixed-gain/fixed-alpha steady-state descriptors

```text
Sigma = F Sigma F^T + G_w Q_s G_w^T + G_v R_s G_v^T
P     = F P F^T / alpha_0 + G_w Q_b G_w^T / alpha_w + G_v R_b G_v^T / alpha_v
```

and constructs a Pareto curve over stochastic covariance size and bounded-error ellipsoid size. The scalarization normalizes each trace by its independently attainable minimum so the result is not dominated by units or scale.

## Suggested stronger runs

```bash
python examples/run_fixed_gain_evaluation.py --out results_grid201 --random-systems 500 --grid 201 --seed 11
python examples/run_gain_optimized_evaluation.py --out results_riccati_grid201 --random-systems 300 --grid 201 --step-grid 101 --seed 17 --workers 16
python examples/run_combined_pareto.py --out results_combined_grid41 --alpha-grid 41 --gain-grid 41
```

These reproduce the legacy grid studies. Use the matched-accuracy protocol above for the revised paper's primary gain comparisons and method ablation.

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

The expanded research modules add greedy-optimality diagnostics, verified
joint gain/weight trace certificates, and local combined worst-case MSE
optimization. The expanded manuscript and frozen evaluation archives live
in the separate paper repository.
The original conference evaluation remains available above.

Install the additional verification and independent SDP backends with:

```bash
python -m pip install -e '.[dev,plot,research,certification,reference]'
```

The certified evaluation reuses the exact archived plants in the paper
repository. It retains all selected cases, checkpoints each search, and
independently replays every returned certificate:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/run_certified_evaluation.py \
  --validation-dir ../2026-07-SteadyStateCombined-Paper/results/validation \
  --out results_certified --workers 16
python -m steady_state_combined.verify_certificate results_certified/certificates/case_0.json.gz
```

The default target is a verified relative gap of 0.001, with 100,000
subdivisions or 1,800 seconds per plant. A budget-limited run keeps its
verified bounds without claiming that the target was reached. Certificates
bound the infimum on the open simplex and use the exact stored binary input
matrices. They do not certify physical-model uncertainty or boundary
attainment. The combined MSE optimizer remains local; its smoothing error
and stationarity diagnostics are distinct from global trace certificates.

Run the paired mechanism study and original-archive diagnostics with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/run_greedy_mechanisms.py \
  --validation-dir ../2026-07-SteadyStateCombined-Paper/results/validation \
  --out results_mechanisms --workers 16
```

The expanded study also uses a matched one-step spectral-risk SDP, a local
two-adjoint combined-MSE solver, and paired controlled tracking:

```bash
python -m pip install -e '.[dev,plot,research,certification,reference]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/run_combined_mse_evaluation.py \
  --validation-dir ../2026-07-SteadyStateCombined-Paper/results/validation \
  --out results_combined_mse --workers 12
python examples/run_controlled_tracking.py \
  --combined-dir results_combined_mse --out results_tracking --workers 8
python examples/run_combined_sensitivity.py \
  --combined-dir results_combined_mse --out results_sensitivity --workers 12
python examples/run_sdp_crosscheck.py \
  --validation-dir ../2026-07-SteadyStateCombined-Paper/results/validation \
  --out results_sdp --workers 12
python examples/verify_expanded_archive.py \
  results_combined_mse/combined_mse_manifest.json --workers 8
```

`verify_expanded_archive.py` also accepts a certification or tracking
manifest. It verifies every recorded output hash and replays each included
certificate, without invoking the optimizer. The paper archive contains
frozen source bundles for the recorded runs, since development continued
between experiments.

The recorded expansion certifies all 452 archived trace designs within a
0.1% global gap. The combined study has 119 successful matched comparisons
and seven retained greedy-baseline convergence failures; its median
risk-bound reduction among those 119 cases is 17.79%. Only 25 combined
designs meet the smooth stationarity tolerance. The remaining outputs are
feasible improvements with explicit stagnation or iteration-limit status,
not certified combined optima. Tracking uses deterministic bias sequences
independent of Gaussian noise, with trajectory-level paired confidence
intervals. It never treats Gaussian total error as deterministically bounded.

The SDP cross-check uses the DARE shape only as a coordinate preconditioner
and preserves the physical trace objective. Its independent formulation
is an implementation check, not a theorem or an interval certificate.

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
make analyze-results
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
  research.py           optional SciPy continuous solvers and independent validation
examples/
  run_fixed_gain_evaluation.py
  run_gain_optimized_evaluation.py
  run_combined_pareto.py
  run_paper_validation.py
  run_solver_ablation.py
scripts/
  analyze_results.py
  export_results_to_paper.py
  generate_latex_tables.py
```
