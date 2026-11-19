#!/usr/bin/env python3
"""Matched combined-MSE comparison on a preselected subset of the archive."""

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
from time import perf_counter

import cvxpy
import flint
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.combined_mse import default_combined_starts, optimize_combined_mse  # noqa: E402
from steady_state_combined.combined_pareto import CombinedParetoProblem  # noqa: E402
from steady_state_combined.ellipsoidal import sym  # noqa: E402
from steady_state_combined.research import solve_dare  # noqa: E402
from steady_state_combined.riccati import GainOptimizedProblem  # noqa: E402
from steady_state_combined.sdp_reference import fixed_weight_lmi_reference, greedy_combined_mse  # noqa: E402
from steady_state_combined.verified_combined import certify_combined_risk  # noqa: E402


def evaluate(task):
    metadata, arrays, output, max_iter = task
    row = dict(metadata)
    started = perf_counter()
    try:
        problem = CombinedParetoProblem(*(arrays[k] for k in
            ("A", "H", "Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded")))
        greedy_started = perf_counter()
        greedy = greedy_combined_mse(problem)
        greedy_seconds = perf_counter() - greedy_started
        starts = [(greedy.K, greedy.alpha)] + default_combined_starts(problem)
        optimized = optimize_combined_mse(problem, starts, max_iter_per_stage=max_iter)
        evaluation = optimized.evaluation
        certificate = certify_combined_risk(problem, evaluation)
        path = Path(output) / "risk_certificates" / f"case_{metadata['case']}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
        # The fixed-weight SDP is independent of the combined-risk algorithm.
        bounded = GainOptimizedProblem(problem.A, problem.H, problem.Q_bounded, problem.R_bounded)
        reference_error = ""
        sdp_gap = None
        try:
            reference = fixed_weight_lmi_reference(bounded, greedy.alpha)
            dare_P, _ = solve_dare(bounded, greedy.alpha)
            sdp_gap = (reference.value - np.trace(dare_P)) / np.trace(dare_P)
        except Exception as exc:
            reference_error = f"{type(exc).__name__}: {exc}"
        from fractions import Fraction
        row.update(
            status="ok", error="", greedy_mse=greedy.value, optimized_mse=evaluation.value,
            reduction_percent=100 * (1 - evaluation.value / greedy.value),
            variance=float(np.trace(evaluation.Sigma)), bias_squared=float(np.linalg.eigvalsh(evaluation.P)[-1]),
            trace_shape=float(np.trace(evaluation.P)), verified_risk_upper=float(Fraction(certificate["risk_upper"])),
            greedy_status=greedy.status, optimizer_status=optimized.status, stationarity=optimized.stationarity,
            greedy_fixed_point_residual=greedy.fixed_point_residual,
            greedy_one_step_residual=greedy.one_step_residual, greedy_one_step_risk_gap=greedy.one_step_risk_gap,
            smoothing_bound=evaluation.smoothing_bound, stability_margin=evaluation.stability_margin,
            greedy_seconds=greedy_seconds, optimizer_seconds=optimized.seconds, total_seconds=perf_counter() - started,
            optimizer_iterations=optimized.iterations, optimizer_evaluations=optimized.evaluations,
            fixed_weight_sdp_relative_gap=sdp_gap, fixed_weight_sdp_error=reference_error,
            alpha_0=evaluation.alpha[0], alpha_w=evaluation.alpha[1], alpha_v=evaluation.alpha[2],
            risk_certificate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        np.savez_compressed(Path(output) / f"design_{metadata['case']}.npz", greedy_K=greedy.K,
                            greedy_alpha=greedy.alpha, greedy_P=greedy.P, greedy_Sigma=greedy.Sigma,
                            K=evaluation.K, alpha=evaluation.alpha, P=evaluation.P, Sigma=evaluation.Sigma)
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}", total_seconds=perf_counter() - started)
    return row


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["case"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--per-dimension", type=int, default=10)
    parser.add_argument("--cases", type=int, nargs="+")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--max-iter-per-stage", type=int, default=400)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    root = args.validation_dir
    source_manifest = json.loads((root / "validation_manifest.json").read_text())
    for name in ("systems.npz", "validation_cases.csv"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != source_manifest["outputs_sha256"][name]:
            raise RuntimeError(f"archive hash mismatch: {name}")
    with (root / "validation_cases.csv").open(newline="") as stream:
        archive_cases = list(csv.DictReader(stream))
    selected, counts = [], {}
    for row in archive_cases:
        if args.cases is not None:
            include = int(row["case"]) in args.cases
        elif row["group"] in ("deterministic", "tracking_6d"):
            include = True
        else:
            include = counts.get(row["dimension"], 0) < args.per_dimension
            if include:
                counts[row["dimension"]] = counts.get(row["dimension"], 0) + 1
        if include:
            selected.append(row)
    tasks, systems = [], {}
    with np.load(root / "systems.npz") as archive:
        for original in selected:
            index = int(original["case"])
            A, H, Q, R = (archive[f"case_{index}_{key}"] for key in ("A", "H", "Q", "R"))
            rng = np.random.default_rng(np.random.SeedSequence([20261003, index, 731]))
            U, _ = np.linalg.qr(rng.normal(size=Q.shape))
            V, _ = np.linalg.qr(rng.normal(size=R.shape))
            # Reuse orientations across uncertainty ratios for paired comparisons.
            stochastic_Q = sym((U * np.linalg.eigvalsh(Q)) @ U.T)
            stochastic_R = sym((V * np.linalg.eigvalsh(R)) @ V.T)
            if original["group"] == "tracking_6d":
                stochastic_Q, stochastic_R = Q, R
            for ratio in args.ratios:
                case = len(tasks)
                metadata = {"case": case, "archive_case": index, "group": original["group"],
                            "dimension": int(original["dimension"]), "stochastic_ratio": ratio}
                arrays = {"A": A, "H": H, "Q_bounded": Q, "R_bounded": R,
                          "Q_stochastic": ratio * stochastic_Q, "R_stochastic": ratio * stochastic_R}
                for key, value in arrays.items():
                    systems[f"case_{case}_{key}"] = value
                tasks.append((metadata, arrays, str(args.out), args.max_iter_per_stage))
    np.savez_compressed(args.out / "combined_systems.npz", **systems)
    (args.out / "selected_cases.json").write_text(json.dumps([task[0] for task in tasks], indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(evaluate, task) for task in tasks]):
            row = future.result()
            rows.append(row)
            write_csv(args.out / "combined_mse_cases.csv", rows)
            print(json.dumps(row, sort_keys=True), flush=True)
    summary = {"selected": len(tasks), "retained": len(rows), "failed": sum(row["status"] != "ok" for row in rows)}
    (args.out / "combined_mse_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    outputs = [path for path in args.out.rglob("*") if path.is_file() and path.name != "combined_mse_manifest.json"]
    manifest = {
        "protocol": "paired exact spectral-risk greedy SDP and local steady-state design; all selected cases retained",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "cvxpy": cvxpy.__version__, "python_flint": flint.__version__},
        "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "inputs_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("systems.npz", "validation_cases.csv")},
        "sources_sha256": source_hashes,
        "sources_changed_during_run": any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                                          for name, digest in source_hashes.items()),
        "outputs_sha256": {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
    }
    (args.out / "combined_mse_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if manifest["sources_changed_during_run"]:
        raise RuntimeError("source changed during evaluation")


if __name__ == "__main__":
    main()
