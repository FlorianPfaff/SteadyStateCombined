#!/usr/bin/env python3
"""Paired factorial study of downstream amplification and greedy suboptimality."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.diagnostics import greedy_optimality_diagnostics  # noqa: E402
from steady_state_combined.ellipsoidal import FixedGainProblem, stepwise_trace_steady_state, sym  # noqa: E402
from steady_state_combined.research import global_fixed_trace, solve_dare  # noqa: E402
from steady_state_combined.riccati import GainOptimizedProblem  # noqa: E402


def evaluate(task):
    metadata, problem = task
    row = dict(metadata)
    try:
        baseline = stepwise_trace_steady_state(problem, max_iter=20_000, tol=1e-12)
        if baseline is None:
            raise RuntimeError("greedy fixed point did not converge")
        diagnostic = greedy_optimality_diagnostics(problem, baseline[0])
        optimum = global_fixed_trace(problem)
        if diagnostic.lower_bound > optimum.value + 1e-7 * optimum.value:
            raise RuntimeError("numerical first-order bound exceeds independent optimum")
        row.update(
            status="ok", error="", greedy_trace=diagnostic.value, optimal_trace=optimum.value,
            reduction_percent=100 * (1 - optimum.value / diagnostic.value),
            ratio_0=diagnostic.amplification_ratios[0], ratio_w=diagnostic.amplification_ratios[1],
            ratio_v=diagnostic.amplification_ratios[2],
            relative_ratio_spread=np.ptp(diagnostic.amplification_ratios) / np.mean(diagnostic.amplification_ratios),
            greedy_kkt=diagnostic.greedy_kkt_residual, steady_kkt=diagnostic.steady_kkt_residual,
            first_order_gap=diagnostic.first_order_gap, lower_bound=diagnostic.lower_bound,
            optimum_kkt=optimum.stationarity,
        )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return row


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["case"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--validation-dir", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    tasks, systems = [], {}
    for n in args.dimensions:
        for repetition in range(args.repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, n, repetition]))
            N = np.triu(rng.normal(size=(n, n)), 1)
            N /= np.linalg.norm(N)
            U, _ = np.linalg.qr(rng.normal(size=(n, n)))
            V, _ = np.linalg.qr(rng.normal(size=(n, n)))
            for rho in (0.5, 0.8, 0.95):
                for eta in (0.0, 0.2, 0.5):
                    for anisotropy in (1.0, 10.0, 100.0):
                        spectrum = np.geomspace(1.0, 1.0 / anisotropy, n)
                        spectrum /= spectrum.sum()
                        F = rho * np.eye(n) + eta * N
                        problem = FixedGainProblem(2 * F, np.eye(n), 0.5 * np.eye(n),
                                                   4 * sym((U * spectrum) @ U.T), 4 * sym((V * spectrum) @ V.T))
                        case = len(tasks)
                        metadata = {"case": case, "group": "controlled", "dimension": n, "repetition": repetition,
                                    "spectral_radius": rho, "nonnormality": eta, "anisotropy": anisotropy}
                        tasks.append((metadata, problem))
                        for key in ("A", "H", "K", "Q", "R"):
                            systems[f"case_{case}_{key}"] = getattr(problem, key)
    input_hashes = {}
    if args.validation_dir is not None:
        root = args.validation_dir
        original_manifest = json.loads((root / "validation_manifest.json").read_text())
        for name in ("systems.npz", "validation_cases.csv"):
            digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if digest != original_manifest["outputs_sha256"][name]:
                raise RuntimeError(f"archive hash mismatch: {name}")
            input_hashes[name] = digest
        with (root / "validation_cases.csv").open(newline="") as stream:
            cases = list(csv.DictReader(stream))
        with np.load(root / "systems.npz") as archive:
            for original in cases:
                index = int(original["case"])
                plant = GainOptimizedProblem(*(archive[f"case_{index}_{key}"] for key in ("A", "H", "Q", "R")))
                if f"case_{index}_fixed_gain" in archive:
                    K = archive[f"case_{index}_fixed_gain"]
                else:
                    _, K = solve_dare(plant, np.full(3, 1 / 3))
                fixed = FixedGainProblem(plant.A, plant.H, K, plant.Q, plant.R)
                case = len(tasks)
                tasks.append(({"case": case, "group": original["group"], "dimension": plant.n,
                               "archive_case": index}, fixed))
                for key in ("A", "H", "K", "Q", "R"):
                    systems[f"case_{case}_{key}"] = getattr(fixed, key)
    np.savez_compressed(args.out / "mechanism_systems.npz", **systems)
    # Save the selection before evaluations so no failed case can be replaced.
    (args.out / "selected_cases.json").write_text(json.dumps([task[0] for task in tasks], indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(evaluate, task) for task in tasks]):
            row = future.result()
            rows.append(row)
            write_csv(args.out / "mechanism_cases.csv", rows)
            if len(rows) % 50 == 0 or row["status"] != "ok":
                print(f"{len(rows)}/{len(tasks)} case={row['case']} status={row['status']}", flush=True)
    controls = [row for row in rows if row["status"] == "ok" and row.get("nonnormality") == 0]
    summary = {"selected": len(tasks), "retained": len(rows), "failed": sum(row["status"] != "ok" for row in rows),
               "scaled_identity_controls": len(controls),
               "control_max_absolute_reduction_percent": max((abs(row["reduction_percent"]) for row in controls), default=0),
               "control_max_ratio_spread": max((row["relative_ratio_spread"] for row in controls), default=0)}
    (args.out / "mechanism_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "protocol": "paired factorial fixed-gain study; unit trace of each source; all selected cases retained",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "inputs_sha256": input_hashes, "sources_sha256": source_hashes,
        "sources_changed_during_run": any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                                          for name, digest in source_hashes.items()),
        "outputs_sha256": {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                           for name in ("mechanism_cases.csv", "mechanism_systems.npz", "selected_cases.json", "mechanism_summary.json")},
    }
    (args.out / "mechanism_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if manifest["sources_changed_during_run"]:
        raise RuntimeError("source changed during evaluation")


if __name__ == "__main__":
    main()
