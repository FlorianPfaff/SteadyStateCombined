.PHONY: test eval-fixed eval-riccati eval-combined eval-all export-paper tables-paper paper-artifacts

test:
	pytest

eval-fixed:
	python examples/run_fixed_gain_evaluation.py --out results_grid201 --random-systems 500 --grid 201 --seed 11

eval-riccati:
	python examples/run_gain_optimized_evaluation.py --out results_riccati_grid201 --random-systems 300 --grid 201 --step-grid 101 --seed 17

eval-combined:
	python examples/run_combined_pareto.py --out results_combined_grid41 --alpha-grid 41 --gain-grid 41

eval-all: eval-fixed eval-riccati eval-combined

export-paper:
	python scripts/export_results_to_paper.py --paper-root ../2026-07-SteadyStateCombined-Paper

tables-paper:
	python scripts/generate_latex_tables.py --paper-root ../2026-07-SteadyStateCombined-Paper

paper-artifacts: export-paper tables-paper
