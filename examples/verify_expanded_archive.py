#!/usr/bin/env python3
"""Independently replay certificates and check an expanded result manifest."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.combined_mse import select_combined_cases  # noqa: E402
from steady_state_combined.combined_pareto import CombinedParetoProblem  # noqa: E402
from steady_state_combined.ellipsoidal import sym  # noqa: E402
from steady_state_combined.riccati import GainOptimizedProblem  # noqa: E402
from steady_state_combined.verified import encode_matrix, problem_record, record_hash  # noqa: E402
from steady_state_combined.verify_certificate import verify_trace_certificate  # noqa: E402
from steady_state_combined.verified_combined import verify_combined_risk_certificate  # noqa: E402

# Column through which a result table cites certificates: the replay result that a citing row needs, and the
# cells in which a row reports what its certificate proves (an error row must leave them empty).
CERTIFICATE_COLUMNS = {
    "certificate_sha256": ("lower", ("problem_sha256", "relative_gap", "subdivisions", "lower", "upper")),
    "risk_certificate_sha256": ("risk_upper", ("verified_risk_upper",)),
}
# Result tables of the evaluation scripts, with the column each must cite certificates through and its summary.
RESULT_TABLES = {"certification_cases.csv": ("certificate_sha256", "certification_summary.json"),
                 "combined_mse_cases.csv": ("risk_certificate_sha256", "combined_mse_summary.json")}
COMBINED_MATRICES = ("A", "H", "Q_stochastic", "R_stochastic", "Q_bounded", "R_bounded")


def verify(path, required_rtol):
    data = path.read_bytes()
    certificate = json.loads(gzip.decompress(data))
    if certificate.get("schema") == "steady_state_combined.feasible_risk.v1":
        result = verify_combined_risk_certificate(certificate)
    else:
        result = verify_trace_certificate(certificate, required_rtol=required_rtol)
        result.update(status=certificate["status"])
    # Both replays check this hash against the problem they were run on.
    result.update(problem_sha256=certificate["problem_sha256"])
    return dict(file=path.name, sha256=hashlib.sha256(data).hexdigest(), **result)


def validation_inputs(manifest, validation_dir):
    """The case table and plants of the validation archive that a run read, checked against its recorded hashes."""
    recorded = manifest.get("inputs_sha256", {})
    folder = validation_dir if validation_dir is not None else manifest.get("arguments", {}).get("validation_dir")
    if folder is None or not {"systems.npz", "validation_cases.csv"} <= set(recorded):
        raise RuntimeError("the manifest does not record its validation inputs, so no case ID can be bound to a plant")
    for name in ("systems.npz", "validation_cases.csv"):
        path = Path(folder) / name
        if not path.is_file():
            raise RuntimeError(f"{path} not found: pass the validation archive of the run with --validation-dir")
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded[name]:
            raise RuntimeError(f"{path} is not the input that the run recorded")
    with (Path(folder) / "validation_cases.csv").open(newline="") as stream:
        cases = {int(row["case"]): row for row in csv.DictReader(stream)}
    with np.load(Path(folder) / "systems.npz") as archive:
        systems = {key: archive[key] for key in archive.files}
    return cases, systems


def trace_plants(manifest, cases, systems):
    """Rows a certification table must have: one per selected archive case, bound to that case's original plant."""
    if "cases" not in manifest.get("arguments", {}):
        raise RuntimeError("the manifest does not record which cases were selected")
    selected = manifest["arguments"]["cases"]
    plants = {}
    for case in sorted(cases) if selected is None else sorted(set(selected)):
        if case not in cases:
            raise RuntimeError(f"selected case {case} is absent from the validation archive")
        problem = GainOptimizedProblem(*(systems[f"case_{case}_{key}"] for key in ("A", "H", "Q", "R")))
        plants[case] = {"cells": {key: cases[case][key] for key in ("group", "dimension")},
                        "problem_sha256": record_hash(problem_record(problem))}
    return plants


def combined_plants(manifest, folder, outputs, cases, systems):
    """Rows a combined table must have: one per task of the run, bound to the plant of the archive case it reuses.

    The tasks are rebuilt from the recorded arguments, as run_combined_mse_evaluation.py builds them, and must be
    those of selected_cases.json. The set-membership plant of a task is the original one of its archive case. Its
    stochastic descriptors, the ratio times a seeded rotation of that plant's own, come from the hashed
    combined_systems.npz; their spectra must be the ratio times the plant's.
    """
    if not {"selected_cases.json", "combined_systems.npz"} <= set(outputs):
        raise RuntimeError("the manifest does not cover selected_cases.json and combined_systems.npz")
    arguments = manifest.get("arguments", {})
    if not {"cases", "per_dimension", "ratios"} <= set(arguments):
        raise RuntimeError("the manifest does not record how the combined cases were selected")
    tasks = []
    for row in select_combined_cases(list(cases.values()), arguments["cases"], arguments["per_dimension"]):
        for ratio in arguments["ratios"]:
            tasks.append({"case": len(tasks), "archive_case": int(row["case"]), "group": row["group"],
                          "dimension": int(row["dimension"]), "stochastic_ratio": ratio})
    if json.loads((folder / "selected_cases.json").read_text()) != tasks:
        raise RuntimeError("selected_cases.json does not list the tasks that the recorded arguments select")
    plants = {}
    with np.load(folder / "combined_systems.npz") as archive:
        for task in tasks:
            case, original, ratio = task["case"], task["archive_case"], task["stochastic_ratio"]
            arrays = {key: archive[f"case_{case}_{key}"] for key in COMBINED_MATRICES}
            for key, source in (("A", "A"), ("H", "H"), ("Q_bounded", "Q"), ("R_bounded", "R")):
                if not np.array_equal(arrays[key], systems[f"case_{original}_{source}"]):
                    raise RuntimeError(f"task {case} does not use the {source} of archive case {original}")
            for key, source in (("Q_stochastic", "Q"), ("R_stochastic", "R")):
                expected = ratio * np.linalg.eigvalsh(systems[f"case_{original}_{source}"])
                scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
                if not np.allclose(np.linalg.eigvalsh(sym(arrays[key])), expected, rtol=0.0, atol=1e-9 * scale):
                    raise RuntimeError(f"task {case} does not scale the {source} of archive case {original} by {ratio}")
            problem = CombinedParetoProblem(*(arrays[key] for key in COMBINED_MATRICES))
            record = {key: encode_matrix(getattr(problem, key)) for key in COMBINED_MATRICES}
            plants[case] = {"cells": {key: str(task[key]) for key in
                                      ("archive_case", "group", "dimension", "stochastic_ratio")},
                            "problem_sha256": record_hash(record)}
    return plants


def check_summary(folder, names, table, rows):
    """A run summary that the manifest covers must count the rows of its table as the evaluation script does."""
    name = RESULT_TABLES.get(table, (None, None))[1]
    if name not in names:
        return
    statuses = Counter(row.get("status") for row in rows)
    if table == "certification_cases.csv":
        expected = {"selected": len(rows), "retained": len(rows), "statuses": dict(statuses),
                    "verified": sum(row.get("verified") == "True" for row in rows)}
    else:
        expected = {"selected": len(rows), "retained": len(rows), "failed": len(rows) - statuses["ok"]}
    if json.loads((folder / name).read_text()) != expected:
        raise RuntimeError(f"{name} does not count the rows of {table}")


def table_cells(result):
    """The cells that the evaluation scripts write into their result table for a replayed certificate."""
    if "risk_upper" in result:
        return {"verified_risk_upper": repr(float(Fraction(result["risk_upper"])))}
    gap = float(Fraction(result["relative_gap"])) if result["relative_gap"] else float("inf")
    return {"status": result["status"], "problem_sha256": result["problem_sha256"], "relative_gap": repr(gap),
            "subdivisions": str(result["leaves_verified"] - 1),
            **{key: repr(float(Fraction(result[key]))) for key in ("lower", "upper")}}


def same_cell(cell, expected):
    """Equal text, or the same number up to how a table writer may have formatted it."""
    if cell == expected:
        return True
    try:
        return math.isclose(float(cell), float(expected), rel_tol=1e-12, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def backed_rows(folder, names, results, plants):
    """Check every table that cites certificates and count the rows backed by one.

    ``plants(column)`` gives the rows that a table citing ``column`` must have, as {case ID: expected cells and
    problem hash}. The table needs exactly one row per expected case, each row must describe its case, and a row
    that is not an error must cite a replayed certificate of the plant of its own case that says what the row
    says; an error row reports no result. The result tables of the evaluation scripts must cite their
    certificates, and their summaries must count their rows. Certificates that no row cites are still replayed,
    but nothing then ties a table to them.
    """
    replayed, count, cited = {result["sha256"]: result for result in results}, 0, set()
    for name in names:
        if not name.endswith(".csv"):
            continue
        with (folder / name).open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns = [column for column in CERTIFICATE_COLUMNS if column in (reader.fieldnames or ())]
            rows = list(reader)
        required = RESULT_TABLES.get(name, (None,))[0]
        if required is not None and columns != [required]:
            raise RuntimeError(f"{name} must cite its certificates in the column {required}")
        if not columns:
            continue
        if len(columns) > 1:
            raise RuntimeError(f"{name} cites certificates of two kinds")
        column = columns[0]
        bound, claims = CERTIFICATE_COLUMNS[column]
        cited.add(column)
        expected = plants(column)
        try:
            listed = Counter(int(row["case"]) for row in rows)
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(f"{name}: every row needs an integer case ID") from None
        duplicated = sorted(case for case, times in listed.items() if times > 1)
        missing, unexpected = sorted(set(expected) - set(listed)), sorted(set(listed) - set(expected))
        if duplicated or missing or unexpected:
            raise RuntimeError(f"{name} needs one row per selected case: duplicated {duplicated}, "
                               f"missing {missing}, unexpected {unexpected}")
        for row in rows:
            case = int(row["case"])
            plant = expected[case]
            wrong = [key for key, value in plant["cells"].items() if not same_cell(row.get(key), value)]
            if wrong:
                raise RuntimeError(f"{name}: case {case} does not describe that case in {wrong}")
            if row.get("status") == "error":
                reported = [key for key in (column, *claims) if row.get(key) not in (None, "")]
                if reported or not row.get("error"):
                    raise RuntimeError(f"{name}: case {case} is an error row but reports {reported or 'no error'}")
                continue
            result = replayed.get(row[column], {})
            if bound not in result:
                raise RuntimeError(f"{name}: case {case} is not backed by a replayed certificate")
            if result["problem_sha256"] != plant["problem_sha256"]:
                raise RuntimeError(f"{name}: case {case} cites a certificate of another plant")
            wrong = [k for k, v in table_cells(result).items() if not same_cell(row.get(k), v)]
            if wrong:
                raise RuntimeError(f"{name}: case {case} disagrees with its certificate in {wrong}")
            count += 1
        check_summary(folder, names, name, rows)
    # Trace certificates and combined runs always come with their result table, so none may be left out.
    if any("target_met" in result for result in results) and "certificate_sha256" not in cited:
        raise RuntimeError("no result table cites the trace certificates")
    if "selected_cases.json" in names and "risk_certificate_sha256" not in cited:
        raise RuntimeError("the combined run has no result table that cites its risk certificates")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rtol", type=Fraction, default=Fraction(1, 1000),
                        help="relative gap every trace certificate must prove, read as an exact rational "
                             "(default: the paper's 0.001)")
    parser.add_argument("--validation-dir", type=Path,
                        help="validation archive that the run read, which binds every case ID to its original plant "
                             "(default: the validation_dir recorded in the manifest)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("sources_changed_during_run"):
        raise RuntimeError("source snapshot was not frozen")
    certificates = []
    for name, expected in manifest["outputs_sha256"].items():
        path = args.manifest.parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"hash mismatch: {name}")
        if path.name.endswith(".json.gz"):
            certificates.append(path)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(verify, path, args.rtol) for path in certificates]):
            result = future.result()
            if not result["valid"]:
                raise RuntimeError(f"certificate replay failed: {result}")
            results.append(result)
            print(json.dumps(result), flush=True)
    inputs = []

    def plants(column):
        if not inputs:
            inputs.append(validation_inputs(manifest, args.validation_dir))
        if column == "certificate_sha256":
            return trace_plants(manifest, *inputs[0])
        return combined_plants(manifest, args.manifest.parent, manifest["outputs_sha256"], *inputs[0])

    backed = backed_rows(args.manifest.parent, manifest["outputs_sha256"], results, plants)
    if results and not backed:
        print("note: no result table cites these certificates, so no table was checked against them", file=sys.stderr)
    # Risk certificates make no gap claim; only trace certificates are held to the required tolerance.
    traces = [row for row in results if "target_met" in row]
    short = sorted(row["file"] for row in traces if not row["target_met"])
    report = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  checked_hashes=len(manifest["outputs_sha256"]), certificates=len(certificates),
                  table_rows_backed=backed, trace_certificates=len(traces), required_rtol=str(args.rtol),
                  short_of_target=short, results=sorted(results, key=lambda row: row["file"]))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    if short:
        print(f"valid bounds, but {len(short)} trace certificate(s) do not prove a relative gap of {args.rtol}",
              file=sys.stderr)
        sys.exit(3)  # an invalid archive raises instead (status 1); 2 is an argparse usage error


if __name__ == "__main__":
    main()
