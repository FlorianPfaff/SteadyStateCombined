"""Verified branch-and-bound for joint gain/weight invariant trace design."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
import json
from pathlib import Path
from time import perf_counter

from flint import ctx
import numpy as np
from scipy.linalg import solve_discrete_are

from .ellipsoidal import sym
from .research import continuous_steady_trace, solve_dare
from .riccati import GainOptimizedProblem
from .verified import (
    enclose_candidate,
    endpoint,
    finite_riccati_trace,
    problem_record,
    rational,
    record_hash,
    rounded_riccati_trace,
    validate_problem,
)
from .verified_riccati import verified_riccati_lower

SCHEMA = "steady_state_combined.trace_certificate.v1"
ROOT = tuple(tuple(Fraction(int(i == j)) for i in range(3)) for j in range(3))


def split_triangle(vertices):
    edges = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    i, j, k = max(edges, key=lambda edge: sum((vertices[edge[0]][d] - vertices[edge[1]][d]) ** 2 for d in range(3)))
    midpoint = tuple((a + b) / 2 for a, b in zip(vertices[i], vertices[j]))
    return (vertices[i], midpoint, vertices[k]), (midpoint, vertices[j], vertices[k])


def triangle_from_path(path: str):
    vertices = ROOT
    for digit in path:
        if digit not in "01":
            raise ValueError("invalid triangle path")
        vertices = split_triangle(vertices)[int(digit)]
    return vertices


def upper_corner(vertices):
    return tuple(max(v[i] for v in vertices) for i in range(3))


def raw_dare(problem: GainOptimizedProblem, weights):
    """Numerical proposal only, on arbitrary positive unnormalized weights."""
    a = np.array(list(map(float, weights)))
    prior = solve_discrete_are(problem.A.T / np.sqrt(a[0]), problem.H.T, problem.Q / a[1], problem.R / a[2])
    innovation = problem.H @ prior @ problem.H.T + problem.R / a[2]
    K = np.linalg.solve(innovation, problem.H @ prior).T
    L = np.eye(problem.n) - K @ problem.H
    return sym(L @ prior @ L.T + K @ problem.R @ K.T / a[2]), K


def _choose_horizon(problem, weights, accuracy, max_horizon):
    """Choose work numerically; validity never depends on this heuristic."""
    try:
        limit = float(np.trace(raw_dare(problem, weights)[0]))
        a = np.array(list(map(float, weights)))
        information = problem.H.T @ np.linalg.solve(problem.R, problem.H) * a[2]
        P = np.zeros_like(problem.Q)
        for iteration in range(1, max_horizon + 1):
            prior = problem.A @ P @ problem.A.T / a[0] + problem.Q / a[1]
            P = sym(np.linalg.inv(np.linalg.inv(prior) + information))
            if iteration >= 8 and limit - np.trace(P) <= accuracy:
                return iteration
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return min(64, max_horizon)
    return max_horizon


def _bound_region(problem, record, path, vertices, accuracy, start_precision, max_precision, max_horizon):
    weights = upper_corner(vertices)
    precision = start_precision
    while precision <= max_precision:
        try:
            lower, exponent = verified_riccati_lower(record, weights, precision)
            return {"path": path, "iterations": 0, "precision": precision,
                    "lower": str(lower * (1 - Fraction(1, 2**32))), "mode": "riccati_subsolution",
                    "inflation_exponent": exponent}
        except (ValueError, ZeroDivisionError, OverflowError, np.linalg.LinAlgError):
            precision *= 2
    iterations = _choose_horizon(problem, weights, accuracy, max_horizon)
    for mode in ("finite", "rounded"):
        precision = start_precision
        while precision <= (min(256, max_precision) if mode == "finite" else max_precision):
            try:
                with ctx.workprec(precision):
                    kernel = finite_riccati_trace if mode == "finite" else rounded_riccati_trace
                    enclosed = kernel(record, weights, iterations, precision)
                    lower, upper = endpoint(enclosed), endpoint(enclosed, upper=True)
                    if upper - lower > rational(accuracy):
                        raise ValueError("Riccati enclosure is too wide")
                    lower = max(Fraction(0), lower * (1 - Fraction(1, 2**40)))
                return {"path": path, "iterations": iterations, "precision": precision, "lower": str(lower), "mode": mode}
            except (ValueError, ZeroDivisionError, OverflowError):
                precision *= 2
    return {"path": path, "iterations": 0, "precision": start_precision, "lower": "0", "unresolved": True}


@dataclass(frozen=True)
class CertifiedTraceResult:
    certificate: dict

    @property
    def status(self):
        return self.certificate["status"]

    @property
    def lower_bound(self):
        return float(rational(self.certificate["lower"]))

    @property
    def upper_bound(self):
        value = self.certificate["upper"]
        return float("inf") if value is None else float(rational(value))

    @property
    def relative_gap(self):
        lower = rational(self.certificate["lower"])
        upper = self.certificate["upper"]
        return float("inf") if lower <= 0 or upper is None else float((rational(upper) - lower) / lower)


def _write_checkpoint(path: Path | None, certificate: dict):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(certificate, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def certify_steady_trace(
    problem: GainOptimizedProblem,
    initial_alpha=None,
    *,
    rtol: float = 1e-3,
    atol: float = 0.0,
    max_nodes: int = 100_000,
    max_seconds: float = 1800.0,
    start_precision: int = 128,
    max_precision: int = 1024,
    max_horizon: int = 4096,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> CertifiedTraceResult:
    """Bound the infimum over all positive weights, including boundary limits.

    ``max_nodes`` counts additional subdivisions in this invocation.
    All retained leaves, including pruned ones, form a complete cover and
    are written to the certificate. A checkpoint can be verified and resumed.
    """
    if rtol <= 0 or atol < 0 or not np.isfinite([rtol, atol, max_seconds]).all() or max_seconds < 0:
        raise ValueError("invalid tolerances or time budget")
    if max_nodes < 0 or max_horizon < 1 or start_precision < 32 or max_precision < start_precision:
        raise ValueError("invalid work or precision budget")
    started = perf_counter()
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    record = problem_record(problem)
    precision = start_precision
    while True:
        try:
            validate_problem(record, precision)
            break
        except ValueError:
            precision *= 2
            if precision > max_precision:
                raise
    certificate = {
        "schema": SCHEMA,
        "problem": record,
        "problem_sha256": record_hash(record),
        "domain": "open_simplex_infimum",
        "rtol": str(rational(rtol)),
        "atol": str(rational(atol)),
        "problem_precision": precision,
        "incumbent": None,
        "status": "no_feasible_incumbent",
        "lower": "0",
        "upper": None,
        "leaves": [{"path": "", "iterations": 0, "precision": start_precision, "lower": "0"}],
        "subdivisions": 0,
        "seconds": 0.0,
    }
    if path is not None and path.exists() and resume:
        from .verify_certificate import verify_trace_certificate

        previous = json.loads(path.read_text())
        if previous["problem_sha256"] != certificate["problem_sha256"]:
            raise ValueError("checkpoint belongs to another problem")
        verify_trace_certificate(previous)
        certificate = previous
        certificate.update(rtol=str(rational(rtol)), atol=str(rational(atol)))
    incumbent = certificate["incumbent"]
    if incumbent is None:
        try:
            if initial_alpha is None:
                proposal = continuous_steady_trace(problem)
                alpha, P, K = proposal.alpha, proposal.P, proposal.K
            else:
                alpha = np.asarray(initial_alpha, dtype=float)
                alpha = alpha / alpha.sum()
                P, K = solve_dare(problem, alpha)
            incumbent = enclose_candidate(problem, alpha, K, P, precision)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            _write_checkpoint(path, certificate)
            return CertifiedTraceResult(certificate)
    upper = rational(incumbent["upper"])
    accuracy = max(float(upper) * rtol / 50, np.finfo(float).tiny)
    leaves = {item["path"]: item for item in certificate["leaves"]}
    heap = []
    for key, item in leaves.items():
        lower = rational(item["lower"])
        if lower < upper:
            heapq.heappush(heap, (lower, key))
    previous_nodes = certificate["subdivisions"]
    previous_seconds = certificate["seconds"]
    nodes = 0
    last_checkpoint = perf_counter()

    def snapshot(status):
        lower = min(rational(item["lower"]) for item in leaves.values())
        certificate.update(
            incumbent=incumbent, upper=str(upper), lower=str(lower), status=status,
            leaves=[leaves[key] for key in sorted(leaves)], subdivisions=previous_nodes + nodes,
            seconds=previous_seconds + perf_counter() - started,
        )
        return certificate

    status = "budget_limit"
    while heap:
        lower, key = heap[0]
        if upper - lower <= rational(atol) + rational(rtol) * lower:
            status = "certified_gap"
            break
        if nodes >= max_nodes or perf_counter() - started >= max_seconds:
            break
        heapq.heappop(heap)
        if lower >= upper:
            continue
        vertices = triangle_from_path(key)
        children = split_triangle(vertices)
        child_records = []
        for digit, child in enumerate(children):
            child_key = key + str(digit)
            child_records.append(_bound_region(
                problem, record, child_key, child, accuracy, start_precision, max_precision, max_horizon
            ))
            alpha_exact = [sum(v[i] for v in child) / 3 for i in range(3)]
            alpha = np.array(list(map(float, alpha_exact)))
            try:
                P, K = solve_dare(problem, alpha)
                if np.trace(P) < float(upper):
                    candidate = enclose_candidate(problem, alpha, K, P, precision)
                    candidate_upper = rational(candidate["upper"])
                    if candidate_upper < upper:
                        incumbent, upper = candidate, candidate_upper
            except (ValueError, np.linalg.LinAlgError):
                pass
        # Replace a parent only after both child bounds exist, preserving coverage.
        del leaves[key]
        for item in child_records:
            leaves[item["path"]] = item
            child_lower = rational(item["lower"])
            if child_lower < upper:
                heapq.heappush(heap, (child_lower, item["path"]))
        nodes += 1
        if perf_counter() - last_checkpoint >= 30:
            _write_checkpoint(path, snapshot("running"))
            last_checkpoint = perf_counter()
    else:
        status = "certified_gap"
    snapshot(status)
    if upper - rational(certificate["lower"]) > rational(atol) + rational(rtol) * rational(certificate["lower"]):
        if status == "certified_gap":
            raise RuntimeError("internal certificate gap inconsistency")
    _write_checkpoint(path, certificate)
    return CertifiedTraceResult(certificate)
