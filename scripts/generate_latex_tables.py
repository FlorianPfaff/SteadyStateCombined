#!/usr/bin/env python3
"""Generate LaTeX table fragments from exported evaluation CSV files.

The expected workflow is:

    make eval-all
    make export-paper
    python scripts/generate_latex_tables.py --paper-root ../2026-07-SteadyStateCombined-Paper

The generated files are standalone table environments that can be included with
``\input{tables/<name>.tex}`` from the paper. Missing CSV files are skipped by
default so partial evaluations can still produce partial paper artifacts; use
``--strict`` to fail on missing inputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TableSpec:
    source: Path
    destination: Path
    kind: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_float(value: str | float, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if x != x:
        return "--"
    if abs(x) >= 1e3 or (abs(x) < 1e-2 and x != 0.0):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def tex_escape(text: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_table(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content.strip() + "\n", encoding="utf-8")


def row_for_method(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    for row in rows:
        if row.get("method") == method:
            return row
    raise ValueError(f"Missing method row: {method}")


def trace_reduction(step: dict[str, str], optimized: dict[str, str]) -> float:
    step_trace = float(step["trace"])
    optimized_trace = float(optimized["trace"])
    return 100.0 * (1.0 - optimized_trace / step_trace)


def generate_result_macros(paper_root: Path, strict: bool) -> bool:
    sources = {
        "fixed_det": paper_root / "results" / "fixed_gain" / "deterministic_summary.csv",
        "fixed_random": paper_root / "results" / "fixed_gain" / "random_summary.csv",
        "bound": paper_root / "results" / "fixed_gain" / "bound_check.csv",
        "riccati_det": paper_root / "results" / "riccati" / "riccati_deterministic_summary.csv",
        "riccati_random": paper_root / "results" / "riccati" / "riccati_random_summary.csv",
        "combined": paper_root / "results" / "combined" / "combined_candidate_summary.csv",
    }
    missing = [path for path in sources.values() if not path.exists()]
    if missing:
        if strict:
            raise FileNotFoundError(missing[0])
        return False

    fixed_det = read_csv(sources["fixed_det"])
    fixed_step = row_for_method(fixed_det, "stepwise_trace_fixed_gain")
    fixed_opt = row_for_method(fixed_det, "grid_optimized_fixed_gain")
    fixed_random = read_csv(sources["fixed_random"])[0]
    bound = read_csv(sources["bound"])[0]
    riccati_det = read_csv(sources["riccati_det"])
    riccati_step = row_for_method(riccati_det, "stepwise_gain_reoptimized")
    riccati_opt = row_for_method(riccati_det, "steady_state_riccati_grid")
    riccati_random = read_csv(sources["riccati_random"])[0]
    combined = read_csv(sources["combined"])[0]

    def ratio(step: dict[str, str], optimized: dict[str, str]) -> float:
        return float(step["trace"]) / float(optimized["trace"])

    def max_reduction(row: dict[str, str]) -> float:
        value = float(row["max_ratio_trace"])
        return 100.0 * (1.0 - 1.0 / value)

    values = {
        "FixedStepTrace": fmt_float(fixed_step["trace"], 5),
        "FixedOptTrace": fmt_float(fixed_opt["trace"], 5),
        "FixedDetRatio": fmt_float(ratio(fixed_step, fixed_opt), 5),
        "FixedDetReduction": fmt_float(trace_reduction(fixed_step, fixed_opt), 4),
        "FixedStepAlphaZero": fmt_float(fixed_step["alpha0"], 4),
        "FixedStepAlphaW": fmt_float(fixed_step["alphaw"], 4),
        "FixedStepAlphaV": fmt_float(fixed_step["alphav"], 4),
        "FixedOptAlphaZero": fmt_float(fixed_opt["alpha0"], 4),
        "FixedOptAlphaW": fmt_float(fixed_opt["alphaw"], 4),
        "FixedOptAlphaV": fmt_float(fixed_opt["alphav"], 4),
        "FixedRandomN": str(int(float(fixed_random["random_systems"]))),
        "FixedRandomMedianRatio": fmt_float(fixed_random["median_ratio_trace"], 4),
        "FixedRandomMedianReduction": fmt_float(fixed_random["median_trace_reduction_percent"], 4),
        "FixedRandomPtf": fmt_float(fixed_random["p25_ratio_trace"], 4),
        "FixedRandomPsf": fmt_float(fixed_random["p75_ratio_trace"], 4),
        "FixedRandomMaxRatio": fmt_float(fixed_random["max_ratio_trace"], 4),
        "FixedRandomMaxReduction": fmt_float(max_reduction(fixed_random), 4),
        "FixedRandomImproved": fmt_float(100.0 * float(fixed_random["fraction_improved"]), 4),
        "RiccatiStepTrace": fmt_float(riccati_step["trace"], 5),
        "RiccatiOptTrace": fmt_float(riccati_opt["trace"], 5),
        "RiccatiDetRatio": fmt_float(ratio(riccati_step, riccati_opt), 5),
        "RiccatiDetReduction": fmt_float(trace_reduction(riccati_step, riccati_opt), 4),
        "RiccatiStepAlphaZero": fmt_float(riccati_step["alpha0"], 4),
        "RiccatiStepAlphaW": fmt_float(riccati_step["alphaw"], 4),
        "RiccatiStepAlphaV": fmt_float(riccati_step["alphav"], 4),
        "RiccatiOptAlphaZero": fmt_float(riccati_opt["alpha0"], 4),
        "RiccatiOptAlphaW": fmt_float(riccati_opt["alphaw"], 4),
        "RiccatiOptAlphaV": fmt_float(riccati_opt["alphav"], 4),
        "RiccatiRandomN": str(int(float(riccati_random["random_systems"]))),
        "RiccatiRandomMedianRatio": fmt_float(riccati_random["median_ratio_trace"], 4),
        "RiccatiRandomMedianReduction": fmt_float(riccati_random["median_trace_reduction_percent"], 4),
        "RiccatiRandomPtf": fmt_float(riccati_random["p25_ratio_trace"], 4),
        "RiccatiRandomPsf": fmt_float(riccati_random["p75_ratio_trace"], 4),
        "RiccatiRandomMaxRatio": fmt_float(riccati_random["max_ratio_trace"], 4),
        "RiccatiRandomMaxReduction": fmt_float(max_reduction(riccati_random), 4),
        "RiccatiRandomImproved": fmt_float(100.0 * float(riccati_random["fraction_improved"]), 4),
        "BoundTrajectories": str(int(float(bound["trajectories"]))),
        "BoundHorizon": str(int(float(bound["horizon"]))),
        "BoundMaxNormalized": fmt_float(bound["max_normalized_error"], 4),
        "BoundViolations": str(int(float(bound["violations"]))),
        "CombinedCandidates": str(int(float(combined["candidate_count"]))),
        "CombinedFrontier": str(int(float(combined["nondominated_count"]))),
        "CombinedMinSigma": fmt_float(combined["min_trace_sigma"], 4),
        "CombinedMinP": fmt_float(combined["min_trace_P"], 4),
    }
    content = "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in values.items())
    write_table(paper_root / "tables" / "results_macros.tex", content)
    return True


def deterministic_table(rows: list[dict[str, str]], caption: str, label: str) -> str:
    body = []
    for row in rows:
        body.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                tex_escape(row.get("method", "")),
                fmt_float(row.get("alpha0", "")),
                fmt_float(row.get("alphaw", "")),
                fmt_float(row.get("alphav", "")),
                fmt_float(row.get("trace", "")),
                fmt_float(row.get("logdet", "")),
            )
        )
    return rf"""
\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{lccccc}}
\toprule
Method & $\alpha_0$ & $\alpha_w$ & $\alpha_v$ & $\tr(P)$ & $\log\det(P)$ \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def random_summary_table(rows: list[dict[str, str]], caption: str, label: str) -> str:
    row = rows[0]
    return rf"""
\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{lccccc}}
\toprule
Systems & Median & Mean & 25\% & 75\% & Max \\
\midrule
{fmt_float(row.get('random_systems', ''))} & {fmt_float(row.get('median_ratio_trace', ''))} & {fmt_float(row.get('mean_ratio_trace', ''))} & {fmt_float(row.get('p25_ratio_trace', ''))} & {fmt_float(row.get('p75_ratio_trace', ''))} & {fmt_float(row.get('max_ratio_trace', ''))} \\
\bottomrule
\end{{tabular}}
\vspace{{0.3em}}
\begin{{tabular}}{{lc}}
\toprule
Fraction improved & {fmt_float(row.get('fraction_improved', ''))} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def bound_check_table(rows: list[dict[str, str]]) -> str:
    row = rows[0]
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Monte-Carlo containment check for bounded disturbances.}}
\label{{tab:bound-check}}
\begin{{tabular}}{{lccc}}
\toprule
Trajectories & Horizon & Max. normalized error & Violations \\
\midrule
{fmt_float(row.get('trajectories', ''))} & {fmt_float(row.get('horizon', ''))} & {fmt_float(row.get('max_normalized_error', ''))} & {fmt_float(row.get('violations', ''))} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def combined_summary_table(rows: list[dict[str, str]]) -> str:
    row = rows[0]
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Combined stochastic/set-membership candidate-set summary.}}
\label{{tab:combined-candidate-summary}}
\begin{{tabular}}{{lcccc}}
\toprule
Candidates & $\min\tr(\Sigma)$ & $\max\tr(\Sigma)$ & $\min\tr(P)$ & $\max\tr(P)$ \\
\midrule
{fmt_float(row.get('candidate_count', ''))} & {fmt_float(row.get('min_trace_sigma', ''))} & {fmt_float(row.get('max_trace_sigma', ''))} & {fmt_float(row.get('min_trace_P', ''))} & {fmt_float(row.get('max_trace_P', ''))} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def combined_pareto_table(rows: list[dict[str, str]]) -> str:
    selected = rows
    if len(rows) > 6:
        # Include endpoints and a few representative middle points.
        indices = sorted(set([0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, len(rows) - 1]))
        selected = [rows[i] for i in indices]
    body = []
    for row in selected:
        body.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                fmt_float(row.get("lambda_stochastic", "")),
                fmt_float(row.get("trace_sigma", "")),
                fmt_float(row.get("trace_P", "")),
                fmt_float(row.get("alpha0", "")),
                fmt_float(row.get("K0", "")),
                fmt_float(row.get("K1", "")),
            )
        )
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Representative points from the combined Pareto sweep.}}
\label{{tab:combined-pareto}}
\begin{{tabular}}{{cccccc}}
\toprule
$\lambda$ & $\tr(\Sigma)$ & $\tr(P)$ & $\alpha_0$ & $K_1$ & $K_2$ \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def render_table(kind: str, rows: list[dict[str, str]]) -> str:
    if kind == "fixed_deterministic":
        return deterministic_table(rows, "Fixed-gain deterministic example.", "tab:fixed-deterministic")
    if kind == "riccati_deterministic":
        return deterministic_table(rows, "Gain-reoptimized deterministic example.", "tab:riccati-deterministic")
    if kind == "fixed_random_summary":
        return random_summary_table(rows, "Random fixed-gain benchmark: trace improvement ratios.", "tab:fixed-random-summary")
    if kind == "riccati_random_summary":
        return random_summary_table(rows, "Gain-reoptimized random benchmark: trace improvement ratios.", "tab:riccati-random-summary")
    if kind == "bound_check":
        return bound_check_table(rows)
    if kind == "combined_summary":
        return combined_summary_table(rows)
    if kind == "combined_pareto":
        return combined_pareto_table(rows)
    raise ValueError(f"Unknown table kind: {kind}")


def default_specs(paper_root: Path) -> list[TableSpec]:
    return [
        TableSpec(paper_root / "results" / "fixed_gain" / "deterministic_summary.csv", paper_root / "tables" / "fixed_gain_deterministic.tex", "fixed_deterministic"),
        TableSpec(paper_root / "results" / "fixed_gain" / "random_summary.csv", paper_root / "tables" / "fixed_gain_random_summary.tex", "fixed_random_summary"),
        TableSpec(paper_root / "results" / "fixed_gain" / "bound_check.csv", paper_root / "tables" / "bound_check.tex", "bound_check"),
        TableSpec(paper_root / "results" / "riccati" / "riccati_deterministic_summary.csv", paper_root / "tables" / "riccati_deterministic.tex", "riccati_deterministic"),
        TableSpec(paper_root / "results" / "riccati" / "riccati_random_summary.csv", paper_root / "tables" / "riccati_random_summary.tex", "riccati_random_summary"),
        TableSpec(paper_root / "results" / "combined" / "combined_candidate_summary.csv", paper_root / "tables" / "combined_candidate_summary.tex", "combined_summary"),
        TableSpec(paper_root / "results" / "combined" / "combined_pareto.csv", paper_root / "tables" / "combined_pareto.tex", "combined_pareto"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", default="../2026-07-SteadyStateCombined-Paper", help="Paper repository root")
    parser.add_argument("--strict", action="store_true", help="Fail if any input CSV is missing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_root = Path(args.paper_root).resolve()
    generated = []
    missing = []
    for spec in default_specs(paper_root):
        if not spec.source.exists():
            missing.append(str(spec.source))
            if args.strict:
                raise FileNotFoundError(spec.source)
            continue
        rows = read_csv(spec.source)
        if not rows:
            missing.append(str(spec.source))
            if args.strict:
                raise ValueError(f"No rows in {spec.source}")
            continue
        write_table(spec.destination, render_table(spec.kind, rows))
        generated.append(str(spec.destination))
    if generate_result_macros(paper_root, strict=args.strict):
        generated.append(str(paper_root / "tables" / "results_macros.tex"))
    print(f"Generated tables: {len(generated)}")
    print(f"Missing inputs:    {len(missing)}")
    for path in generated:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
