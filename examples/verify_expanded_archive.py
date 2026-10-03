#!/usr/bin/env python3
"""Independently replay certificates and check an expanded result manifest."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steady_state_combined.verify_certificate import verify_trace_certificate  # noqa: E402
from steady_state_combined.verified_combined import verify_combined_risk_certificate  # noqa: E402


def verify(path):
    certificate = json.loads(gzip.decompress(path.read_bytes()))
    if certificate.get("schema") == "steady_state_combined.feasible_risk.v1":
        result = verify_combined_risk_certificate(certificate)
    else:
        result = verify_trace_certificate(certificate)
    return dict(file=path.name, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
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
        for future in as_completed([executor.submit(verify, path) for path in certificates]):
            result = future.result()
            if not result["valid"]:
                raise RuntimeError(f"certificate replay failed: {result}")
            results.append(result)
            print(json.dumps(result), flush=True)
    report = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  checked_hashes=len(manifest["outputs_sha256"]), certificates=len(certificates),
                  results=sorted(results, key=lambda row: row["file"]))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
