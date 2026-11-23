#!/usr/bin/env python3
"""Run the paper's exact PF Algorithm 4 on the saved BOKP instances."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gurobipy as gp

from bokp_ks.instance_generator import InstanceGenerator
from bokp_ks.proportional_fairness import PFOptimizer

INPUT_ROOT = ROOT / "results_30_instances_20261017"
OUTPUT_ROOT = Path(os.environ.get("PF_OUTPUT_ROOT", ROOT / "pf_results_20261017_final2"))
INSTANCE_DIR = INPUT_ROOT / "instances"
NS = list(range(50, 1001, 50))
SEED = 271828
FIELDS = ["n", "instance_id", "seed", "status", "F1_PF", "F2_PF", "solution_path", "F2_max", "epsilon", "f_solves", "g_solves", "search_iterations", "runtime_seconds", "positivity_assumption_holds", "positivity_witness", "error"]


def run_one(instance_path: Path, output_root: Path) -> tuple[dict, dict | None]:
    instance = InstanceGenerator.from_dict(json.loads(instance_path.read_text(encoding="utf-8")))
    log_path = output_root / "logs" / f"{instance.instance_id}.log"
    outcome = PFOptimizer(instance, log_path, SEED).run()
    row = outcome.as_row(instance)
    solution = None
    if outcome.point:
        solution = {"n": instance.n, "instance_id": instance.instance_id, "seed": instance.seed, "status": outcome.status, "objective_point": {"F1": outcome.point.F1, "F2": outcome.point.F2}, "x": list(outcome.point.x), "F2_max": outcome.f2_max, "epsilon": {"numerator": outcome.epsilon.numerator, "denominator": outcome.epsilon.denominator}, "positivity_assumption_holds": outcome.positivity_assumption_holds}
        (output_root / "solutions" / f"{instance.instance_id}.json").write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")
    return row, solution


def main() -> None:
    if OUTPUT_ROOT.exists() and not (OUTPUT_ROOT / "pf_rows.jsonl").exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {OUTPUT_ROOT}")
    paths = sorted(INSTANCE_DIR.glob("*.json"))
    expected = {f"bokp_n{n:04d}_i{i:03d}" for n in NS for i in range(1, 31)}
    actual = {path.stem for path in paths}
    if actual != expected:
        raise SystemExit(f"Saved instance set mismatch: missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}")
    for name in ("logs", "solutions"):
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_ROOT / "pf_rows.jsonl"

    smoke_path = next(path for path in paths if path.stem == "bokp_n0050_i001")
    smoke_row, _ = run_one(smoke_path, OUTPUT_ROOT)
    if smoke_row["status"] not in {"PF_FOUND", "PF_NOT_FOUND", "FAILED"}:
        raise RuntimeError(f"Invalid smoke status: {smoke_row}")
    print(f"PF smoke test passed: {smoke_row['instance_id']} -> {smoke_row['status']}", flush=True)

    started = time.perf_counter()
    rows = []
    if checkpoint.exists():
        rows = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_ids = {row["instance_id"] for row in rows}
    if smoke_path.stem not in completed_ids:
        rows.append(smoke_row)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(smoke_row) + "\n")
        completed_ids.add(smoke_path.stem)
    solutions = []
    for index, path in enumerate(paths, start=1):
        if path.stem in completed_ids:
            row = next(row for row in rows if row["instance_id"] == path.stem)
            solution_path = OUTPUT_ROOT / "solutions" / f"{path.stem}.json"
            solution = json.loads(solution_path.read_text()) if solution_path.exists() else None
        else:
            row, solution = run_one(path, OUTPUT_ROOT)
            rows.append(row)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            completed_ids.add(path.stem)
        if solution:
            solutions.append(solution)
        print(f"[{index}/{len(paths)}] {row['instance_id']}: {row['status']}", flush=True)

    rows.sort(key=lambda row: (int(row["n"]), row["instance_id"]))
    with (OUTPUT_ROOT / "pf_results_per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    solutions.sort(key=lambda item: (item["n"], item["instance_id"]))
    (OUTPUT_ROOT / "pf_solutions.json").write_text(json.dumps(solutions, indent=2) + "\n", encoding="utf-8")

    counts = defaultdict(lambda: {"PF_FOUND": 0, "PF_NOT_FOUND": 0, "FAILED": 0})
    positivity_violations = []
    for row in rows:
        counts[int(row["n"])][row["status"]] += 1
        if not row["positivity_assumption_holds"]:
            positivity_violations.append(row["instance_id"])
    metadata = {
        "paper": "Proportional Fairness for Combinatorial Optimization (Nguyen et al.)",
        "algorithm": "Algorithm 4 with VerifyPFsol, VerifyPFcoeff, and SEARCH Procedures 1-3",
        "input_root": str(INPUT_ROOT), "output_root": str(OUTPUT_ROOT), "instance_count": len(rows),
        "objective_definitions": {"F1": "sum(p_i*x_i)", "F2": "sum(q_i*x_i)", "unshifted": True, "not_KS_PQ": True},
        "epsilon": "1/(F2_max^2), with F2_max=max feasible original F2 per instance",
        "arithmetic": "fractions.Fraction for alpha; scaled integer b*F1+a*F2 for F; scaled G uses 2*min(b*F1,a*F2)",
        "comparison_policy": "exact integer comparisons of returned feasible objective values; non-optimal solver status is FAILED",
        "solver": {"name": "Gurobi", "version": ".".join(map(str, gp.gurobi.version())), "threads": 1, "seed": SEED, "mip_gap": 0.0, "mip_gap_abs": 0.0, "time_limit": "infinite"},
        "counts_by_n": {str(n): counts[n] for n in NS},
        "total_counts": {status: sum(group[status] for group in counts.values()) for status in ("PF_FOUND", "PF_NOT_FOUND", "FAILED")},
        "positivity_assumption": {"holds": False, "reason": "The BOKP capacity model permits x=0, whose original objectives are (0,0); no objective shift was applied.", "affected_instances": positivity_violations},
        "runtime_seconds": time.perf_counter() - started,
        "smoke_test": smoke_path.stem,
    }
    (OUTPUT_ROOT / "pf_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_ROOT), "total_counts": metadata["total_counts"], "positivity_violations": len(positivity_violations)}, indent=2))


if __name__ == "__main__":
    main()
