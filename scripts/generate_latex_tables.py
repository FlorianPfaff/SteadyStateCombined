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
    print(f"Generated tables: {len(generated)}")
    print(f"Missing inputs:    {len(missing)}")
    for path in generated:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
