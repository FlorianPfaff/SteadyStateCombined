"""Replay a trace certificate without running or trusting the optimizer."""

from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import json
from pathlib import Path

from flint import ctx

from .verified import (
    endpoint,
    finite_riccati_trace,
    rational,
    record_hash,
    rounded_riccati_trace,
    validate_problem,
    verify_invariant_candidate,
)
from .verified_riccati import verified_riccati_lower


def _regions(leaves):
    """Independently reconstruct the exact binary partition of the simplex."""
    paths = [item["path"] for item in leaves]
    if len(set(paths)) != len(paths) or any(set(path) - {"0", "1"} for path in paths):
        raise ValueError("duplicate or invalid leaf path")
    tree = {}
    for path in paths:
        node = tree
        for digit in path:
            if "leaf" in node:
                raise ValueError("overlapping certificate leaves")
            node = node.setdefault(digit, {})
        if node:
            raise ValueError("overlapping certificate leaves")
        node["leaf"] = True
    stack = [("", tuple(tuple(Fraction(int(i == j)) for i in range(3)) for j in range(3)), tree)]
    while stack:
        prefix, triangle, node = stack.pop()
        if "leaf" in node:
            yield prefix, tuple(max(v[i] for v in triangle) for i in range(3))
            continue
        if set(node) != {"0", "1"}:
            raise ValueError("certificate leaves do not cover the simplex")
        pairs = [(0, 1), (0, 2), (1, 2)]
        squared = [sum((triangle[i][k] - triangle[j][k]) ** 2 for k in range(3)) for i, j in pairs]
        i, j = pairs[squared.index(max(squared))]
        k = 3 - i - j
        middle = tuple((triangle[i][d] + triangle[j][d]) / 2 for d in range(3))
        stack.append((prefix + "1", (middle, triangle[j], triangle[k]), node["1"]))
        stack.append((prefix + "0", (triangle[i], middle, triangle[k]), node["0"]))


def verify_trace_certificate(certificate: dict, max_precision: int = 2048) -> dict:
    if certificate["schema"] != "steady_state_combined.trace_certificate.v1":
        raise ValueError("unsupported certificate schema")
    if certificate["domain"] != "open_simplex_infimum":
        raise ValueError("unsupported certificate domain")
    record = certificate["problem"]
    if record_hash(record) != certificate["problem_sha256"]:
        raise ValueError("problem hash mismatch")
    validate_problem(record, int(certificate["problem_precision"]))
    if certificate["incumbent"] is None:
        raise ValueError("no verified feasible incumbent")
    incumbent = certificate["incumbent"]
    upper = verify_invariant_candidate(record, incumbent, int(incumbent["precision"]))
    if rational(incumbent["upper"]) != upper or rational(certificate["upper"]) != upper:
        raise ValueError("incorrect upper bound")
    leaves = certificate["leaves"]
    if not leaves:
        raise ValueError("empty certificate cover")
    by_path = {item["path"]: item for item in leaves}
    checked = 0
    for path, weights in _regions(leaves):
        item = by_path[path]
        lower = rational(item["lower"])
        iterations = int(item["iterations"])
        if lower < 0 or iterations < 0:
            raise ValueError("negative lower bound or horizon")
        if item.get("mode") == "riccati_subsolution":
            precision = max(128, int(item["precision"]))
            while precision <= max_precision:
                try:
                    replayed, _ = verified_riccati_lower(record, weights, precision, schur_check=True)
                    if lower <= replayed:
                        break
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
                precision *= 2
            else:
                raise ValueError(f"Riccati subsolution not independently verified at leaf {path}")
            checked += 1
            continue
        if iterations == 0:
            if lower != 0:
                raise ValueError("a zero-horizon bound must be zero")
        else:
            precision = max(128, int(item["precision"]))
            mode = item.get("mode", "finite")
            if mode == "rounded":
                enclosed = rounded_riccati_trace(record, weights, iterations, precision, verify_steps=True)
                if lower > endpoint(enclosed):
                    raise ValueError(f"incorrect rounded lower bound at leaf {path}")
                checked += 1
                continue
            if mode != "finite":
                raise ValueError("unknown lower-bound mode")
            while precision <= max_precision:
                try:
                    with ctx.workprec(precision):
                        enclosed = finite_riccati_trace(record, weights, iterations, precision, schur=True)
                        replayed = endpoint(enclosed)
                    if lower <= replayed:
                        break
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
                precision *= 2
            else:
                raise ValueError(f"lower bound not independently verified at leaf {path}")
        checked += 1
    lower = min(rational(item["lower"]) for item in leaves)
    if rational(certificate["lower"]) != lower or lower > upper:
        raise ValueError("incorrect aggregate lower bound")
    rtol, atol = rational(certificate["rtol"]), rational(certificate["atol"])
    if rtol <= 0 or atol < 0:
        raise ValueError("invalid certificate tolerances")
    achieved = upper - lower <= atol + rtol * lower
    if certificate["status"] == "certified_gap" and not achieved:
        raise ValueError("claimed gap is not achieved")
    return {"valid": True, "gap_achieved": achieved, "leaves_verified": checked, "lower": str(lower), "upper": str(upper)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.certificate.suffix == ".gz" else open
    with opener(args.certificate, "rt") as stream:
        certificate = json.load(stream)
    print(json.dumps(verify_trace_certificate(certificate), indent=2))


if __name__ == "__main__":
    main()
