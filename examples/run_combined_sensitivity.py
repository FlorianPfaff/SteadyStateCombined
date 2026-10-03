#!/usr/bin/env python3
"""Predetermined anchor sensitivity to floors, smoothing, and initialization."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.combined_mse import default_combined_starts, optimize_combined_mse  # noqa: E402
from steady_state_combined.combined_pareto import CombinedParetoProblem  # noqa: E402


def evaluate(task):
    metadata, arrays, starts = task
    row = dict(metadata)
    try:
        problem = CombinedParetoProblem(*arrays)
        if row["initialization"] == "default_multistart":
            starts = default_combined_starts(problem, row["floor"])
        result = optimize_combined_mse(problem, starts, alpha_floor=row["floor"],
            smoothing_exponents=tuple(range(2, row["final_exponent"] + 1)))
        row.update(status=result.status, value=result.evaluation.value,
                   relative_change=result.evaluation.value / row["archived_value"] - 1,
                   stationarity=result.stationarity, smoothing_bound=result.evaluation.smoothing_bound,
                   iterations=result.iterations, seconds=result.seconds, min_alpha=float(min(result.evaluation.alpha)), error="")
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]}
    manifest_path = args.combined_dir / "combined_mse_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["outputs_sha256"].items():
        if hashlib.sha256((args.combined_dir / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"hash mismatch: {name}")
    with (args.combined_dir / "combined_mse_cases.csv").open(newline="") as stream:
        cases = [r for r in csv.DictReader(stream) if int(r["archive_case"]) in (0, 301, 351, 401, 451)
                 and float(r["stochastic_ratio"]) == 1]
    tasks = []
    with np.load(args.combined_dir / "combined_systems.npz") as archive:
        for case in cases:
            index = int(case["case"])
            arrays = tuple(archive[f"case_{index}_{key}"] for key in
                           ("A", "H", "Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded"))
            with np.load(args.combined_dir / f"design_{index}.npz") as designs:
                for floor in (1e-6, 1e-8, 1e-10):
                    for exponent in (4, 6, 8):
                        meta = dict(case=index, archive_case=int(case["archive_case"]), dimension=int(case["dimension"]),
                                    floor=floor, final_exponent=exponent, initialization="archived_solution",
                                    archived_value=float(case["optimized_mse"]))
                        tasks.append((meta, arrays, [(designs["K"], designs["alpha"])]))
                for initialization in ("greedy_only", "default_multistart"):
                    meta = dict(case=index, archive_case=int(case["archive_case"]), dimension=int(case["dimension"]),
                                floor=1e-8, final_exponent=6, initialization=initialization,
                                archived_value=float(case["optimized_mse"]))
                    tasks.append((meta, arrays, [(designs["greedy_K"], designs["greedy_alpha"])]))
    (args.out / "sensitivity_selection.json").write_text(json.dumps([t[0] for t in tasks], indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(evaluate, t) for t in tasks]):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with (args.out / "sensitivity_cases.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    changed = any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value for name, value in sources.items())
    report = dict(selected=len(tasks), retained=len(rows), failed=sum(r["status"] == "error" for r in rows),
                  sources_sha256=sources, sources_changed_during_run=changed,
                  combined_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  outputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir()
                                  if p.is_file() and p.name != "sensitivity_manifest.json"})
    (args.out / "sensitivity_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    if changed:
        raise RuntimeError("source changed during sensitivity run")


if __name__ == "__main__":
    main()
