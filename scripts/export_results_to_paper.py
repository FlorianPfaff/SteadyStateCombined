#!/usr/bin/env python3
"""Copy generated evaluation artifacts into the paper repository.

Run this from the code repository after generating results, for example:

    python scripts/export_results_to_paper.py --paper-root ../2026-07-SteadyStateCombined-Paper

The script copies CSV files into ``<paper-root>/results`` and image files into
``<paper-root>/figures``. Missing files are skipped by default so that partial
runs can still be exported; use ``--strict`` to fail on missing artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    source: Path
    destination: Path
    description: str


def default_artifacts(results_root: Path, paper_root: Path) -> list[Artifact]:
    """Return the conventional artifact mapping used by the paper draft."""

    return [
        Artifact(
            results_root / "results_grid201" / "deterministic_summary.csv",
            paper_root / "results" / "fixed_gain" / "deterministic_summary.csv",
            "fixed-gain deterministic summary",
        ),
        Artifact(
            results_root / "results_grid201" / "line_search_curve.csv",
            paper_root / "results" / "fixed_gain" / "line_search_curve.csv",
            "fixed-gain non-myopic line-search data",
        ),
        Artifact(
            results_root / "results_grid201" / "line_search_curve.png",
            paper_root / "figures" / "line_search_curve.png",
            "fixed-gain non-myopic line-search figure",
        ),
        Artifact(
            results_root / "results_grid201" / "deterministic_ellipsoids.png",
            paper_root / "figures" / "deterministic_ellipsoids.png",
            "fixed-gain deterministic ellipsoid comparison",
        ),
        Artifact(
            results_root / "results_grid201" / "random_benchmark.csv",
            paper_root / "results" / "fixed_gain" / "random_benchmark.csv",
            "fixed-gain random benchmark",
        ),
        Artifact(
            results_root / "results_grid201" / "random_summary.csv",
            paper_root / "results" / "fixed_gain" / "random_summary.csv",
            "fixed-gain random summary",
        ),
        Artifact(
            results_root / "results_grid201" / "random_scatter.png",
            paper_root / "figures" / "random_scatter.png",
            "fixed-gain random scatter figure",
        ),
        Artifact(
            results_root / "results_grid201" / "random_improvement_cdf.png",
            paper_root / "figures" / "fixed_gain_improvement_cdf.png",
            "fixed-gain trace-reduction empirical CDF",
        ),
        Artifact(
            results_root / "results_grid201" / "bound_check.csv",
            paper_root / "results" / "fixed_gain" / "bound_check.csv",
            "fixed-gain bounded-error containment check",
        ),
        Artifact(
            results_root / "results_riccati_grid201" / "riccati_deterministic_summary.csv",
            paper_root / "results" / "riccati" / "riccati_deterministic_summary.csv",
            "gain-reoptimized deterministic summary",
        ),
        Artifact(
            results_root / "results_riccati_grid201" / "riccati_random_benchmark.csv",
            paper_root / "results" / "riccati" / "riccati_random_benchmark.csv",
            "gain-reoptimized random benchmark",
        ),
        Artifact(
            results_root / "results_riccati_grid201" / "riccati_random_summary.csv",
            paper_root / "results" / "riccati" / "riccati_random_summary.csv",
            "gain-reoptimized random summary",
        ),
        Artifact(
            results_root / "results_riccati_grid201" / "riccati_random_scatter.png",
            paper_root / "figures" / "riccati_random_scatter.png",
            "gain-reoptimized random scatter figure",
        ),
        Artifact(
            results_root / "results_riccati_grid201" / "riccati_improvement_cdf.png",
            paper_root / "figures" / "riccati_improvement_cdf.png",
            "gain-reoptimized trace-reduction empirical CDF",
        ),
        Artifact(
            results_root / "results_combined_grid41" / "combined_pareto.csv",
            paper_root / "results" / "combined" / "combined_pareto.csv",
            "combined Pareto curve data",
        ),
        Artifact(
            results_root / "results_combined_grid41" / "combined_candidate_summary.csv",
            paper_root / "results" / "combined" / "combined_candidate_summary.csv",
            "combined candidate summary",
        ),
        Artifact(
            results_root / "results_combined_grid41" / "combined_frontier.csv",
            paper_root / "results" / "combined" / "combined_frontier.csv",
            "combined nondominated candidate frontier",
        ),
        Artifact(
            results_root / "results_combined_grid41" / "combined_pareto.png",
            paper_root / "figures" / "combined_pareto.png",
            "combined Pareto figure",
        ),
    ]


def copy_artifacts(
    artifacts: list[Artifact],
    results_root: Path,
    paper_root: Path,
    strict: bool,
) -> dict:
    copied: list[dict] = []
    missing: list[dict] = []
    for artifact in artifacts:
        item = {
            "source": str(artifact.source.relative_to(results_root)),
            "destination": str(artifact.destination.relative_to(paper_root)),
            "description": artifact.description,
        }
        if not artifact.source.exists():
            missing.append(item)
            if strict:
                raise FileNotFoundError(f"Missing artifact: {artifact.source}")
            continue
        artifact.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact.source, artifact.destination)
        item["bytes"] = artifact.source.stat().st_size
        item["sha256"] = hashlib.sha256(artifact.source.read_bytes()).hexdigest()
        copied.append(item)
    return {"copied": copied, "missing": missing}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=".", help="Code repository root containing results_* folders")
    parser.add_argument("--paper-root", default="../2026-07-SteadyStateCombined-Paper", help="Paper repository root")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected artifact is missing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    paper_root = Path(args.paper_root).resolve()
    artifacts = default_artifacts(results_root, paper_root)
    manifest = copy_artifacts(artifacts, results_root, paper_root, strict=args.strict)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["path_format"] = "repository-relative"

    manifest_path = paper_root / "results" / "export_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Copied artifacts: {len(manifest['copied'])}")
    print(f"Missing artifacts: {len(manifest['missing'])}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
