#!/usr/bin/env python3
"""Certify every selected archived plant and retain unresolved cases."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import flint
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.certification import certify_steady_trace  # noqa: E402
from steady_state_combined.riccati import GainOptimizedProblem  # noqa: E402
from steady_state_combined.verify_certificate import verify_trace_certificate  # noqa: E402


def evaluate(task):
    case, arrays, output, options = task
    index = int(case["case"])
    row = {"case": index, "group": case["group"], "dimension": int(case["dimension"])}
    problem = GainOptimizedProblem(*(arrays[key] for key in ("A", "H", "Q", "R")))
    alpha = np.array([float(case[f"steady_alpha_{key}"]) for key in ("0", "w", "v")])
    checkpoint = Path(output) / "checkpoints" / f"case_{index}.json"
    try:
        result = certify_steady_trace(problem, alpha, checkpoint_path=checkpoint, **options)
        certificate = result.certificate
        if certificate["incumbent"] is not None:
            verified = verify_trace_certificate(certificate)
        else:
            verified = {"valid": False, "gap_achieved": False}
        destination = Path(output) / "certificates" / f"case_{index}.json.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(certificate, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        destination.write_bytes(gzip.compress(encoded, mtime=0))
        row.update(
            status=result.status, verified=verified["valid"], lower=result.lower_bound,
            upper=result.upper_bound, relative_gap=result.relative_gap,
            subdivisions=certificate["subdivisions"], seconds=certificate["seconds"],
            previous_trace=float(case["continuous_steady_trace"]),
            incumbent_improvement_percent=100 * (1 - result.upper_bound / float(case["continuous_steady_trace"])),
            problem_sha256=certificate["problem_sha256"],
            certificate_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), error="",
        )
    except Exception as exc:
        row.update(status="error", verified=False, error=f"{type(exc).__name__}: {exc}")
    return row


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["case"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    inputs = args.validation_dir
    manifest = json.loads((inputs / "validation_manifest.json").read_text())
    for name in ("systems.npz", "validation_cases.csv"):
        actual = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        if actual != manifest["outputs_sha256"][name]:
            raise RuntimeError(f"archive hash mismatch: {name}")
    with (inputs / "validation_cases.csv").open(newline="") as stream:
        cases = [row for row in csv.DictReader(stream) if args.cases is None or int(row["case"]) in args.cases]
    if args.cases is not None and set(args.cases) != {int(row["case"]) for row in cases}:
        raise ValueError("requested case absent from archive")
    options = {key: getattr(args, key) for key in ("rtol", "max_seconds", "max_nodes")}
    with np.load(inputs / "systems.npz") as archive:
        tasks = [(case, {key: archive[f"case_{case['case']}_{key}"] for key in ("A", "H", "Q", "R")},
                  str(args.out), options) for case in cases]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_csv(args.out / "certification_cases.csv", rows)
            print(json.dumps(row, sort_keys=True), flush=True)
    counts = {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}
    summary = {"selected": len(cases), "retained": len(rows), "statuses": counts,
               "verified": sum(row["verified"] for row in rows)}
    (args.out / "certification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    output_manifest = {
        "protocol": "verified open-simplex infimum; all selected cases retained",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "python_flint": flint.__version__},
        "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "inputs_sha256": {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
                          for name in ("systems.npz", "validation_cases.csv", "validation_manifest.json")},
        "sources_sha256": source_hashes,
        "sources_changed_during_run": any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                                          for name, digest in source_hashes.items()),
        "outputs_sha256": {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(args.out.glob("certificates/*.gz"))
                           + [args.out / "certification_cases.csv", args.out / "certification_summary.json"]},
    }
    (args.out / "certification_manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if output_manifest["sources_changed_during_run"]:
        raise RuntimeError("source changed during evaluation; this run is not a publication archive")


if __name__ == "__main__":
    main()
