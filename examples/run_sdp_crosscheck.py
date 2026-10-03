#!/usr/bin/env python3
"""Audit the fixed-weight LMI in DARE-scaled coordinates on the full archive."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys

import cvxpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.ellipsoidal import sym  # noqa: E402
from steady_state_combined.research import solve_dare  # noqa: E402
from steady_state_combined.riccati import GainOptimizedProblem  # noqa: E402
from steady_state_combined.sdp_reference import fixed_weight_lmi_reference  # noqa: E402


def evaluate(task):
    row, arrays = task
    result = {key: row[key] for key in ("case", "group", "dimension")}
    try:
        problem = GainOptimizedProblem(*(arrays[k] for k in ("A", "H", "Q", "R")))
        alpha = np.array([float(row[f"steady_alpha_{k}"]) for k in ("0", "w", "v")])
        P, _ = solve_dare(problem, alpha)
        reference = fixed_weight_lmi_reference(problem, alpha, balanced=True, state_scale=np.linalg.cholesky(P))
        L = np.eye(problem.n) - reference.K @ problem.H
        F = L @ problem.A
        residual = sym(reference.P - F @ reference.P @ F.T / alpha[0]
                       - L @ problem.Q @ L.T / alpha[1] - reference.K @ problem.R @ reference.K.T / alpha[2])
        result.update(status="ok", solver_status=reference.status,
                      relative_trace_gap=reference.value / np.trace(P) - 1,
                      relative_matrix_gap=float(np.linalg.norm(reference.P-P) / np.linalg.norm(P)),
                      normalized_minimum_residual_eigenvalue=float(np.linalg.eigvalsh(residual)[0] / np.linalg.norm(P)),
                      error="")
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]}
    source = json.loads((args.validation_dir / "validation_manifest.json").read_text())
    for name in ("systems.npz", "validation_cases.csv"):
        if hashlib.sha256((args.validation_dir / name).read_bytes()).hexdigest() != source["outputs_sha256"][name]:
            raise ValueError(f"input hash mismatch: {name}")
    with (args.validation_dir / "validation_cases.csv").open(newline="") as stream:
        cases = list(csv.DictReader(stream))
    with np.load(args.validation_dir / "systems.npz") as archive:
        tasks = [(r, {k: archive[f"case_{r['case']}_{k}"] for k in ("A", "H", "Q", "R")}) for r in cases]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(evaluate, task) for task in tasks]):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (args.out / "sdp_crosscheck_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["case"])))
    good = [r for r in rows if r["status"] == "ok"]
    summary = dict(selected=len(cases), retained=len(rows), failed=len(rows)-len(good),
                   max_absolute_trace_gap=max(abs(r["relative_trace_gap"]) for r in good),
                   max_relative_matrix_gap=max(r["relative_matrix_gap"] for r in good),
                   minimum_normalized_residual=min(r["normalized_minimum_residual_eigenvalue"] for r in good))
    (args.out / "sdp_crosscheck_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    changed = any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected for name, expected in sources.items())
    manifest = dict(protocol="independent SDP formulation; DARE shape used only for coordinate conditioning; no SDP warm start; physical trace objective preserved",
                    sources_sha256=sources, sources_changed_during_run=changed, cvxpy_version=cvxpy.__version__,
                    inputs_sha256={name: source["outputs_sha256"][name] for name in ("systems.npz", "validation_cases.csv")},
                    outputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir()
                                    if p.is_file() and p.name != "sdp_crosscheck_manifest.json"})
    (args.out / "sdp_crosscheck_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    if changed:
        raise RuntimeError("source changed during cross-check")


if __name__ == "__main__":
    main()
