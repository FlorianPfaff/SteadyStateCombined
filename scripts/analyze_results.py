#!/usr/bin/env python3
"""Summarize generated evaluation results for paper-framing decisions.

This script reads the CSV files produced by the evaluation scripts and writes a
Markdown report with the key numbers. It is intentionally lightweight: it does
not replace the paper text, but it gives a quick decision aid after a workflow
run.

Example:

    python scripts/analyze_results.py --results-root . --out results_report.md
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Summary:
    name: str
    rows: list[dict[str, str]]


def read_csv_optional(path: Path) -> Optional[list[dict[str, str]]]:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def fmt(x: float, digits: int = 4) -> str:
    if x != x:
        return "n/a"
    if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def ratio_to_percent(ratio: float) -> float:
    return 100.0 * (ratio - 1.0)


def classify_improvement(median_ratio: float, max_ratio: float, fraction_improved: float) -> str:
    if median_ratio != median_ratio:
        return "insufficient data"
    median_pct = ratio_to_percent(median_ratio)
    max_pct = ratio_to_percent(max_ratio)
    if median_pct >= 5.0 and fraction_improved >= 0.6:
        return "strong broad improvement"
    if median_pct >= 1.0 and fraction_improved >= 0.5:
        return "moderate broad improvement"
    if max_pct >= 10.0:
        return "rare but potentially large improvement"
    return "weak or inconclusive improvement"


def deterministic_improvement(rows: list[dict[str, str]]) -> Optional[float]:
    traces: dict[str, float] = {}
    for row in rows:
        method = row.get("method", "")
        traces[method] = as_float(row, "trace")
    step = traces.get("stepwise_trace_fixed_gain", traces.get("stepwise_gain_reoptimized"))
    ss = traces.get("grid_optimized_fixed_gain", traces.get("steady_state_riccati_grid"))
    if step is None or ss is None or ss <= 0:
        return None
    return step / ss


def summarize_random(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    row = rows[0]
    return {
        "systems": as_float(row, "random_systems"),
        "median": as_float(row, "median_ratio_trace"),
        "mean": as_float(row, "mean_ratio_trace"),
        "p25": as_float(row, "p25_ratio_trace"),
        "p75": as_float(row, "p75_ratio_trace"),
        "max": as_float(row, "max_ratio_trace"),
        "fraction": as_float(row, "fraction_improved"),
    }


def combined_ranges(rows: list[dict[str, str]]) -> Optional[dict[str, float]]:
    if not rows:
        return None
    row = rows[0]
    return {
        "candidates": as_float(row, "candidate_count"),
        "min_sigma": as_float(row, "min_trace_sigma"),
        "max_sigma": as_float(row, "max_trace_sigma"),
        "min_P": as_float(row, "min_trace_P"),
        "max_P": as_float(row, "max_trace_P"),
    }


def framing_recommendation(fixed: dict[str, float], riccati: dict[str, float], combined: Optional[dict[str, float]]) -> str:
    fixed_class = classify_improvement(fixed.get("median", float("nan")), fixed.get("max", float("nan")), fixed.get("fraction", float("nan")))
    riccati_class = classify_improvement(riccati.get("median", float("nan")), riccati.get("max", float("nan")), riccati.get("fraction", float("nan")))

    if fixed_class.startswith("strong") and riccati_class.startswith(("strong", "moderate")):
        return (
            "Lead with the non-myopic steady-state approximation rule. The fixed-gain result is broad, "
            "and the gain-reoptimized experiment supports practical relevance."
        )
    if riccati_class.startswith("strong") or riccati_class.startswith("moderate"):
        return (
            "Lead with the Riccati-sweep practical result, using the fixed-gain theorem as the clean "
            "analytical foundation."
        )
    if fixed_class.startswith("rare") or riccati_class.startswith("rare"):
        return (
            "Frame the contribution as a principled failure mode of greedy ellipsoidal approximation: "
            "the median effect may be small, but some systems exhibit a large avoidable gap."
        )
    if combined is not None:
        return (
            "Pure set-membership improvements look weak in aggregate. Consider moving the combined "
            "stochastic/set-membership Pareto story closer to the center of the paper."
        )
    return "More data are needed before choosing the framing."


def make_report(results_root: Path) -> str:
    fixed_det = read_csv_optional(results_root / "results_grid201" / "deterministic_summary.csv")
    fixed_rand = read_csv_optional(results_root / "results_grid201" / "random_summary.csv")
    bound_check = read_csv_optional(results_root / "results_grid201" / "bound_check.csv")
    riccati_det = read_csv_optional(results_root / "results_riccati_grid201" / "riccati_deterministic_summary.csv")
    riccati_rand = read_csv_optional(results_root / "results_riccati_grid201" / "riccati_random_summary.csv")
    combined_summary = read_csv_optional(results_root / "results_combined_grid41" / "combined_candidate_summary.csv")

    fixed_summary = summarize_random(fixed_rand or [])
    riccati_summary = summarize_random(riccati_rand or [])
    combined_summary_values = combined_ranges(combined_summary or [])

    lines: list[str] = []
    lines.append("# Steady-state combined evaluation report")
    lines.append("")
    lines.append("## Fixed-gain experiment")
    lines.append("")
    if fixed_det:
        ratio = deterministic_improvement(fixed_det)
        lines.append(f"- Deterministic trace ratio `stepwise / steady-state`: **{fmt(ratio or float('nan'))}**.")
    else:
        lines.append("- Deterministic summary not found.")
    if fixed_summary:
        lines.append(f"- Random systems: **{fmt(fixed_summary['systems'], 0)}**.")
        lines.append(f"- Median trace ratio: **{fmt(fixed_summary['median'])}** ({fmt(ratio_to_percent(fixed_summary['median']))}% improvement).")
        lines.append(f"- Mean trace ratio: **{fmt(fixed_summary['mean'])}**.")
        lines.append(f"- IQR trace ratio: **{fmt(fixed_summary['p25'])} -- {fmt(fixed_summary['p75'])}**.")
        lines.append(f"- Max trace ratio: **{fmt(fixed_summary['max'])}** ({fmt(ratio_to_percent(fixed_summary['max']))}% improvement).")
        lines.append(f"- Fraction improved: **{fmt(fixed_summary['fraction'])}**.")
        lines.append(f"- Classification: **{classify_improvement(fixed_summary['median'], fixed_summary['max'], fixed_summary['fraction'])}**.")
    else:
        lines.append("- Random summary not found.")

    lines.append("")
    lines.append("## Gain-reoptimized Riccati experiment")
    lines.append("")
    if riccati_det:
        ratio = deterministic_improvement(riccati_det)
        lines.append(f"- Deterministic trace ratio `greedy / steady-state`: **{fmt(ratio or float('nan'))}**.")
    else:
        lines.append("- Deterministic Riccati summary not found.")
    if riccati_summary:
        lines.append(f"- Random systems: **{fmt(riccati_summary['systems'], 0)}**.")
        lines.append(f"- Median trace ratio: **{fmt(riccati_summary['median'])}** ({fmt(ratio_to_percent(riccati_summary['median']))}% improvement).")
        lines.append(f"- Mean trace ratio: **{fmt(riccati_summary['mean'])}**.")
        lines.append(f"- IQR trace ratio: **{fmt(riccati_summary['p25'])} -- {fmt(riccati_summary['p75'])}**.")
        lines.append(f"- Max trace ratio: **{fmt(riccati_summary['max'])}** ({fmt(ratio_to_percent(riccati_summary['max']))}% improvement).")
        lines.append(f"- Fraction improved: **{fmt(riccati_summary['fraction'])}**.")
        lines.append(f"- Classification: **{classify_improvement(riccati_summary['median'], riccati_summary['max'], riccati_summary['fraction'])}**.")
    else:
        lines.append("- Random Riccati summary not found.")

    lines.append("")
    lines.append("## Bound check")
    lines.append("")
    if bound_check:
        row = bound_check[0]
        lines.append(f"- Trajectories: **{row.get('trajectories', 'n/a')}**, horizon: **{row.get('horizon', 'n/a')}**.")
        lines.append(f"- Max normalized error: **{fmt(as_float(row, 'max_normalized_error'))}**.")
        lines.append(f"- Violations: **{row.get('violations', 'n/a')}**.")
    else:
        lines.append("- Bound check not found.")

    lines.append("")
    lines.append("## Combined stochastic/set-membership Pareto sweep")
    lines.append("")
    if combined_summary_values:
        lines.append(f"- Feasible candidates: **{fmt(combined_summary_values['candidates'], 0)}**.")
        lines.append(f"- `tr(Sigma)` range: **{fmt(combined_summary_values['min_sigma'])} -- {fmt(combined_summary_values['max_sigma'])}**.")
        lines.append(f"- `tr(P)` range: **{fmt(combined_summary_values['min_P'])} -- {fmt(combined_summary_values['max_P'])}**.")
    else:
        lines.append("- Combined candidate summary not found.")

    lines.append("")
    lines.append("## Framing recommendation")
    lines.append("")
    lines.append(framing_recommendation(fixed_summary, riccati_summary, combined_summary_values))
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=".", help="Directory containing results_* folders")
    parser.add_argument("--out", default="results_report.md", help="Markdown report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    out = Path(args.out).resolve()
    out.write_text(make_report(results_root), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
