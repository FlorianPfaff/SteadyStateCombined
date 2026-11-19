#!/usr/bin/env python3
"""Independently replay certificates and check an expanded result manifest."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.verify_certificate import verify_trace_certificate  # noqa: E402
from steady_state_combined.verified_combined import verify_combined_risk_certificate  # noqa: E402


def verify(path, required_rtol):
    data = path.read_bytes()
    certificate = json.loads(gzip.decompress(data))
    if certificate.get("schema") == "steady_state_combined.feasible_risk.v1":
        result = verify_combined_risk_certificate(certificate)
    else:
        result = verify_trace_certificate(certificate, required_rtol=required_rtol)
        result.update(status=certificate["status"], problem_sha256=certificate["problem_sha256"])
    return dict(file=path.name, sha256=hashlib.sha256(data).hexdigest(), **result)


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


def backed_rows(folder, names, results):
    """Count the table rows that cite a certificate: each must cite a replayed one that says what the row says.

    Only rows of .csv tables with a certificate hash column can be checked. Certificates that no row cites are
    still replayed, but nothing then ties a table to them.
    """
    replayed, count = {result["sha256"]: result for result in results}, 0
    for name in names:
        if not name.endswith(".csv"):
            continue
        with (folder / name).open(newline="") as stream:
            for row in csv.DictReader(stream):
                for column, bound in (("certificate_sha256", "lower"), ("risk_certificate_sha256", "risk_upper")):
                    if column not in row or row.get("status") == "error":
                        continue
                    result = replayed.get(row[column], {})
                    if bound not in result:
                        raise RuntimeError(f"{name}: case {row.get('case')} is not backed by a replayed certificate")
                    wrong = [k for k, v in table_cells(result).items() if not same_cell(row.get(k), v)]
                    if wrong:
                        raise RuntimeError(f"{name}: case {row.get('case')} disagrees with its certificate in {wrong}")
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rtol", type=Fraction, default=Fraction(1, 1000),
                        help="relative gap every trace certificate must prove, read as an exact rational "
                             "(default: the paper's 0.001)")
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
    backed = backed_rows(args.manifest.parent, manifest["outputs_sha256"], results)
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
