#!/usr/bin/env python3
"""Paired physical tracking validation of archived combined-MSE designs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.combined_mse import combined_mse_value_gradient  # noqa: E402
from steady_state_combined.combined_pareto import CombinedParetoProblem  # noqa: E402
from steady_state_combined.ellipsoidal import FixedGainProblem  # noqa: E402
from steady_state_combined.research import continuous_steady_trace, global_fixed_trace  # noqa: E402
from steady_state_combined.riccati import GainOptimizedProblem  # noqa: E402
from steady_state_combined.tracking_validation import bounded_profiles, simulate_profile  # noqa: E402
from steady_state_combined.verified_combined import certify_combined_risk  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(task):
    case, ratio, profile, problem, designs, disturbances, settings, output = task
    try:
        result = simulate_profile(problem, designs, *disturbances, **settings)
        np.savez_compressed(Path(output) / f"tracking_{case}_{profile}.npz", **result.trajectories,
                            w_bias=disturbances[0], v_bias=disturbances[1])
        metadata = dict(case=case, stochastic_ratio=ratio, profile=profile, status="ok", error="")
        return ([dict(metadata, **row) for row in result.rows],
                [dict(metadata, **row) for row in result.paired_rows])
    except Exception as exc:
        return ([dict(case=case, stochastic_ratio=ratio, profile=profile, status="error",
                      error=f"{type(exc).__name__}: {exc}")], [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--seed", type=int, default=71943)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    archive_manifest = json.loads((args.combined_dir / "combined_mse_manifest.json").read_text())
    for name, expected in archive_manifest["outputs_sha256"].items():
        if digest(args.combined_dir / name) != expected:
            raise RuntimeError(f"combined archive hash mismatch: {name}")
    with (args.combined_dir / "combined_mse_cases.csv").open(newline="") as stream:
        cases = [row for row in csv.DictReader(stream) if row["group"] == "tracking_6d"]
    tasks, design_rows = [], []
    with np.load(args.combined_dir / "combined_systems.npz") as systems:
        for row in cases:
            if row["status"] != "ok":
                design_rows.append(dict(case=int(row["case"]), status="error", error="source design failed"))
                continue
            index = int(row["case"])
            problem = CombinedParetoProblem(*(systems[f"case_{index}_{key}"] for key in
                ("A", "H", "Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded")))
            with np.load(args.combined_dir / f"design_{index}.npz") as archive:
                designs = {"greedy": combined_mse_value_gradient(problem, archive["greedy_K"], archive["greedy_alpha"]),
                           "steady": combined_mse_value_gradient(problem, archive["K"], archive["alpha"])}
            bounded = GainOptimizedProblem(problem.A, problem.H, problem.Q_bounded, problem.R_bounded)
            bounded_design = continuous_steady_trace(bounded)
            designs["bounded_trace"] = combined_mse_value_gradient(problem, bounded_design.K, bounded_design.alpha)
            prior = solve_discrete_are(problem.A.T, problem.H.T, problem.Q_stochastic, problem.R_stochastic)
            gain = np.linalg.solve(problem.H @ prior @ problem.H.T + problem.R_stochastic, problem.H @ prior).T
            fixed = FixedGainProblem(problem.A, problem.H, gain, problem.Q_bounded, problem.R_bounded)
            kalman_weights = global_fixed_trace(fixed)
            designs["kalman"] = combined_mse_value_gradient(problem, gain, kalman_weights.alpha)
            for name, design in designs.items():
                certificate = certify_combined_risk(problem, design)
                (args.out / f"risk_{index}_{name}.json.gz").write_bytes(gzip.compress(
                    json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
                design_rows.append(dict(case=index, stochastic_ratio=float(row["stochastic_ratio"]), method=name,
                    status="ok", risk_bound=design.value, verified_risk_upper=float(Fraction(certificate["risk_upper"])),
                    variance=float(np.trace(design.Sigma)), bias_squared=float(np.linalg.eigvalsh(design.P)[-1]),
                    alpha_0=design.alpha[0], alpha_w=design.alpha[1], alpha_v=design.alpha[2]))
                np.savez_compressed(args.out / f"design_{index}_{name}.npz", K=design.K, alpha=design.alpha,
                                    P=design.P, Sigma=design.Sigma)
            profiles = bounded_profiles(problem, designs, args.horizon)
            for name, disturbances in profiles.items():
                tasks.append((index, float(row["stochastic_ratio"]), name, problem, designs, disturbances,
                              dict(trials=args.trials, burn_in=args.burn_in, seed=args.seed, dt=0.1), str(args.out)))
    write_csv(args.out / "tracking_designs.csv", design_rows)
    protocol = {
        "state": "(p_x,p_y,p_z,dt*v_x,dt*v_y,dt*v_z), position in meters, velocity in meters/second, dt=0.1 second",
        "dynamics": "p_next=p+dt*v; v_next=0.98*v, with independent additive stochastic and bounded acceleration disturbances",
        "initial_errors": "zero stochastic and deterministic errors",
        "profiles": "constant; directions switching every 50 steps; 200-step directional-support sequences for each observer, cross-evaluated on every observer",
        "adversary": "precomputed from observer and shape, independent of all Gaussian noise; no claim of global norm-maximizing reachable disturbance",
        "pairing": "identical Gaussian draws across observers, profiles, and ratios (with covariance scaling)",
        "confidence_intervals": "pointwise normal 95% intervals over independent trial-level time averages; temporal samples are not treated as independent",
        "metrics": "normalized MSE=position squared error+dt^2*velocity squared error; physical position and velocity RMSE reported separately",
        "caveat": "Gaussian total errors are not bounded; deterministic bias inclusion and stochastic covariance are checked separately",
        "selected": [{"case": task[0], "ratio": task[1], "profile": task[2]} for task in tasks],
    }
    (args.out / "tracking_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, paired_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run, task) for task in tasks]):
            new_rows, new_paired = future.result()
            rows.extend(new_rows)
            paired_rows.extend(new_paired)
            print(json.dumps(new_rows), flush=True)
            write_csv(args.out / "tracking_cases.csv", rows)
            write_csv(args.out / "tracking_paired.csv", paired_rows)
    summary = dict(selected_profiles=len(tasks), retained_method_profiles=len(rows),
                   failed=sum(row["status"] != "ok" for row in rows))
    (args.out / "tracking_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = [p for p in args.out.iterdir() if p.is_file() and p.name != "tracking_manifest.json"]
    changed = any(digest(ROOT / name) != expected for name, expected in sources.items())
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    sources_sha256=sources, sources_changed_during_run=changed,
                    combined_manifest_sha256=digest(args.combined_dir / "combined_mse_manifest.json"),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                    outputs_sha256={p.name: digest(p) for p in files})
    (args.out / "tracking_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if changed:
        raise RuntimeError("source changed during evaluation")


if __name__ == "__main__":
    main()
