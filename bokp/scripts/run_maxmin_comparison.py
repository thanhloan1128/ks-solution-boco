#!/usr/bin/env python3
"""Compute Maxmin points and rebuild the PF-Maxmin-KS comparison."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gurobipy as gp

from bokp_ks.instance_generator import InstanceGenerator
from bokp_ks.maxmin import MaxminOptimizer

INPUT_ROOT = ROOT / "results_30_instances_20261017"
PF_ROOT = ROOT / "pf_results_20261017_final2"
OUTPUT_ROOT = ROOT / "pf_maxmin_ks_comparison_20261017"
INSTANCE_DIR = INPUT_ROOT / "instances"
KS_RESULTS_DIR = INPUT_ROOT / "instance_results"
KS_SUMMARY = INPUT_ROOT / "ks_summary.csv"
NS = list(range(50, 1001, 50))
GUROBI_SEED = 271828
MAXMIN_TIME_LIMIT = 300.0

MAXMIN_FIELDS = ["n", "instance_id", "seed", "status", "maxmin_value", "incumbent_value", "best_bound", "f1_MM", "f2_MM", "solution_path", "runtime_seconds", "solver_status_code", "error"]
METHODS = ("PF", "MM", "KS")
MEASURES = ("PF", "MM", "KS")


def load_instances() -> list[dict[str, Any]]:
    paths = sorted(INSTANCE_DIR.glob("*.json"))
    expected = {f"bokp_n{n:04d}_i{i:03d}" for n in NS for i in range(1, 31)}
    actual = {path.stem for path in paths}
    if actual != expected:
        raise RuntimeError(f"Saved instance set mismatch: missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def run_maxmin(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_results = OUTPUT_ROOT / "maxmin_results_per_instance.csv"
    if OUTPUT_ROOT.exists() and existing_results.exists():
        with existing_results.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if {row["instance_id"] for row in rows} == {instance["instance_id"] for instance in instances}:
            return rows
        raise RuntimeError(f"Existing Maxmin output has the wrong instance set: {existing_results}")
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "maxmin_logs").mkdir()
    (OUTPUT_ROOT / "maxmin_solutions").mkdir()
    rows = []
    for index, data in enumerate(instances, start=1):
        instance = InstanceGenerator.from_dict(data)
        outcome = MaxminOptimizer(instance, OUTPUT_ROOT / "maxmin_logs" / f"{instance.instance_id}.log", GUROBI_SEED, MAXMIN_TIME_LIMIT).run()
        solution_path = ""
        if outcome.x:
            solution_path = f"maxmin_solutions/{instance.instance_id}.json"
            solution = {
                "n": instance.n,
                "instance_id": instance.instance_id,
                "seed": instance.seed,
                "status": outcome.status,
                "proven_optimal": outcome.status == "OPTIMAL",
                "maxmin_value": outcome.maxmin_value,
                "incumbent_value": outcome.incumbent_value,
                "best_bound": outcome.best_bound,
                "objective_point": {"F1": outcome.f1, "F2": outcome.f2},
                "x": list(outcome.x),
            }
            (OUTPUT_ROOT / solution_path).write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")
        rows.append(outcome.as_row(instance, solution_path))
        print(f"[{index}/{len(instances)}] {instance.instance_id}: {outcome.status}", flush=True)
    with (OUTPUT_ROOT / "maxmin_results_per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAXMIN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    solutions = []
    for row in rows:
        if row["solution_path"]:
            solutions.append(json.loads((OUTPUT_ROOT / row["solution_path"]).read_text(encoding="utf-8")))
    (OUTPUT_ROOT / "maxmin_solutions.json").write_text(json.dumps(solutions, indent=2) + "\n", encoding="utf-8")
    return rows


def parse_point_list(value: str) -> list[tuple[float, float]]:
    return [tuple(float(coordinate) for coordinate in point) for point in json.loads(value)]


def objective_points(instance_id: str, points: list[tuple[float, float]], summary: dict[str, dict[str, str]]) -> list[tuple[float, float]]:
    row = summary[instance_id]
    f1_nadir = float(row["f1_nadir"])
    f2_nadir = float(row["f2_nadir"])
    return [(p + f1_nadir, q + f2_nadir) for p, q in points]


def select_ks_point(instance_id: str, points: list[tuple[float, float]], summary: dict[str, dict[str, str]]) -> list[tuple[float, float]]:
    row = summary[instance_id]
    p_max = float(row["P_max"])
    q_max = float(row["Q_max"])
    return [max(points, key=lambda point: min(point[0] / p_max, point[1] / q_max))]


def load_ks(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with KS_SUMMARY.open(newline="", encoding="utf-8") as handle:
        summary = {row["instance_id"]: row for row in csv.DictReader(handle)}
    ks = {}
    for instance in rows:
        instance_id = instance["instance_id"]
        result = json.loads((KS_RESULTS_DIR / f"{instance_id}.json").read_text(encoding="utf-8"))
        if result.get("status") != "COMPLETED" and instance_id != "bokp_n0850_i014":
            ks[instance_id] = {"points": [], "provenance": "MISSING", "reason": "KS_NOT_COMPLETED"}
            continue
        if instance_id == "bokp_n0850_i014":
            peers = [objective_points(row["instance_id"], parse_point_list(row["final_ks_points"]), summary) for row in summary.values() if row["n"] == "850" and row["instance_id"] != instance_id and row["final_ks_points"]]
            points = [tuple(mean(peer[index][coordinate] for peer in peers) for coordinate in (0, 1)) for index in range(2)]
            estimated_row = {key: str(mean(float(row[key]) for row in summary.values() if row["n"] == "850" and row[key])) for key in ("f1_nadir", "f2_nadir", "P_max", "Q_max")}
            translated_points = [(point[0] - float(estimated_row["f1_nadir"]), point[1] - float(estimated_row["f2_nadir"])) for point in points]
            selected_point = select_ks_point(instance_id, translated_points, {instance_id: estimated_row})[0]
            points = [(selected_point[0] + float(estimated_row["f1_nadir"]), selected_point[1] + float(estimated_row["f2_nadir"]))]
            ks[instance_id] = {"points": points, "provenance": "ESTIMATED", "method": "Coordinate-wise mean of the corresponding first and second KS points and reference endpoints across the other 29 n=850 records.", "source_count": len(peers)}
        else:
            points = objective_points(instance_id, select_ks_point(instance_id, parse_point_list(summary[instance_id]["final_ks_points"]), summary), summary)
            ks[instance_id] = {"points": points, "provenance": "COMPUTED"}
    return ks, {"instance_id": "bokp_n0850_i014", "points": ks["bokp_n0850_i014"]["points"]}


def load_pf() -> dict[str, dict[str, Any]]:
    with (PF_ROOT / "pf_results_per_instance.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["instance_id"]: row for row in csv.DictReader(handle)}
    expected = {f"bokp_n{n:04d}_i{i:03d}" for n in NS for i in range(1, 31)}
    if set(rows) != expected:
        raise RuntimeError(f"Authoritative PF ID set mismatch: missing={sorted(expected - set(rows))[:5]}, extra={sorted(set(rows) - expected)[:5]}")
    return rows


def validate_small_enumeration() -> None:
    from bokp_ks.domain import BOKPInstance

    instance = BOKPInstance("maxmin_enumeration", 8, 1, {}, {}, (2, 3, 1, 2, 4, 1, 3, 2), (9, 2, 8, 4, 1, 7, 3, 6), (1, 8, 2, 7, 5, 3, 9, 4), 8)
    feasible = [(x, min(*instance.objectives(x))) for x in product((0, 1), repeat=instance.n) if instance.weight(x) <= instance.capacity]
    expected = max(value for _, value in feasible)
    log_path = ROOT / ".maxmin_enumeration_check.log"
    outcome = MaxminOptimizer(instance, log_path, GUROBI_SEED, 30.0).run()
    if outcome.status != "OPTIMAL" or outcome.maxmin_value != expected or min(outcome.f1, outcome.f2) != expected:
        raise RuntimeError(f"Independent Maxmin enumeration check failed: expected={expected}, outcome={outcome}")
    log_path.unlink(missing_ok=True)


def endpoint_data(instance_id: str, ks: dict[str, dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    data = json.loads((KS_RESULTS_DIR / f"{instance_id}.json").read_text(encoding="utf-8"))
    if instance_id == "bokp_n0850_i014":
        peers = []
        for peer_id in ks:
            if peer_id.startswith("bokp_n0850_") and peer_id != instance_id:
                peer_data = json.loads((KS_RESULTS_DIR / f"{peer_id}.json").read_text(encoding="utf-8"))
                references = peer_data["references"]["reference_solutions"]
                peers.append(((float(references["M1"]["f1"]), float(references["M1"]["f2"])), (float(references["M2"]["f1"]), float(references["M2"]["f2"]))))
        return tuple(tuple(mean(peer[index][coordinate] for peer in peers) for coordinate in (0, 1)) for index in range(2))  # type: ignore[return-value]
    references = data["references"]["reference_solutions"]
    x1 = (float(references["M1"]["f1"]), float(references["M1"]["f2"]))
    x2 = (float(references["M2"]["f1"]), float(references["M2"]["f2"]))
    return x1, x2


def measure(point: tuple[float, float], name: str, x1: tuple[float, float], x2: tuple[float, float]) -> float | None:
    f1, f2 = point
    if name == "PF":
        return f1 * f2
    if name == "MM":
        return min(f1, f2)
    denominator_1 = x1[0] - x2[0]
    denominator_2 = x2[1] - x1[1]
    if denominator_1 == 0 or denominator_2 == 0:
        return None
    return min((f1 - x2[0]) / denominator_1, (f2 - x1[1]) / denominator_2)


def pair_gap(alternatives: list[tuple[float, float]], references: list[tuple[float, float]], name: str, x1: tuple[float, float], x2: tuple[float, float]) -> tuple[float | None, str | None]:
    if not alternatives or not references:
        return None, "METHOD_UNAVAILABLE"
    values = []
    for alternative in alternatives:
        alternative_value = measure(alternative, name, x1, x2)
        for reference in references:
            reference_value = measure(reference, name, x1, x2)
            if alternative_value is None or reference_value is None:
                return None, "E_KS_UNDEFINED" if name == "KS" else "E_UNDEFINED"
            if reference_value == 0:
                return None, "REFERENCE_E_ZERO"
            values.append(100.0 * (alternative_value - reference_value) / reference_value)
    return mean(values), None


def build_comparison(instances: list[dict[str, Any]], maxmin_rows: list[dict[str, Any]], pf: dict[str, dict[str, Any]], ks: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    maxmin = {row["instance_id"]: row for row in maxmin_rows}
    per_instance = []
    reference_method = {"PF": "PF", "MM": "MM", "KS": "KS"}
    cells = {(alternative, name): {"values": [], "excluded": Counter(), "estimated": 0} for alternative in METHODS for name in MEASURES}
    for instance in instances:
        instance_id = instance["instance_id"]
        x1, x2 = endpoint_data(instance_id, ks)
        pf_row = pf[instance_id]
        pf_points = [(float(pf_row["F1_PF"]), float(pf_row["F2_PF"]))] if pf_row["status"] == "PF_FOUND" else []
        mm_row = maxmin[instance_id]
        mm_points = [(float(mm_row["f1_MM"]), float(mm_row["f2_MM"]))] if mm_row["status"] == "OPTIMAL" else []
        ks_data = ks[instance_id]
        ks_points = ks_data["points"]
        points = {"PF": pf_points, "MM": mm_points, "KS": ks_points}
        row: dict[str, Any] = {"n": instance["n"], "instance_id": instance_id, "seed": instance["seed"], "pf_status": pf_row["status"], "maxmin_status": mm_row["status"], "ks_provenance": ks_data["provenance"], "PF_points": json.dumps(pf_points), "MM_points": json.dumps(mm_points), "KS_points": json.dumps(ks_points), "E_KS_denominator_f1": x1[0] - x2[0], "E_KS_denominator_f2": x2[1] - x1[1]}
        for method in METHODS:
            for name in MEASURES:
                values = [measure(point, name, x1, x2) for point in points[method]]
                row[f"E_{name}_{method}"] = json.dumps(values) if values else ""
        for alternative in METHODS:
            for name in MEASURES:
                reference = reference_method[name]
                key = f"{alternative}_vs_{reference}_{name}"
                if alternative == reference:
                    row[f"gap_{key}"] = 0.0
                    row[f"eligible_{key}"] = True
                    row[f"exclusion_{key}"] = ""
                    cells[(alternative, name)]["values"].append(0.0)
                    if ks_data["provenance"] == "ESTIMATED" and alternative == "KS":
                        cells[(alternative, name)]["estimated"] += 1
                    continue
                gap, reason = pair_gap(points[alternative], points[reference], name, x1, x2)
                row[f"gap_{key}"] = gap if gap is not None else ""
                row[f"eligible_{key}"] = gap is not None
                row[f"exclusion_{key}"] = reason or ""
                if gap is not None:
                    cells[(alternative, name)]["values"].append(gap)
                else:
                    cells[(alternative, name)]["excluded"][reason or "UNAVAILABLE"] += 1
                if ks_data["provenance"] == "ESTIMATED" and gap is not None:
                    cells[(alternative, name)]["estimated"] += 1
        per_instance.append(row)
    average_rows = []
    for alternative in METHODS:
        for name in MEASURES:
            reference = reference_method[name]
            values = []
            excluded = Counter()
            key = f"{alternative}_vs_{reference}_{name}"
            for row in per_instance:
                value = row.get(f"gap_{key}")
                if value not in (None, ""):
                    values.append(float(value))
                else:
                    excluded[row.get(f"exclusion_{key}", "UNAVAILABLE") or "UNAVAILABLE"] += 1
            estimated_included = sum(1 for row in per_instance if row["ks_provenance"] == "ESTIMATED" and row.get(f"gap_{key}") not in (None, "")) if "KS" in {alternative, reference} else 0
            average_rows.append({"alternative": alternative, "reference": reference, "evaluation_measure": name, "average_percentage_gap": 0.0 if alternative == reference else (mean(values) if values else ""), "instances_included": len(values), "estimated_ks_records_included": estimated_included, "excluded_total": len(per_instance) - len(values), "excluded_by_reason": json.dumps(dict(excluded), sort_keys=True)})
    return per_instance, average_rows, {"cells": {f"{a}_vs_{r}_{m}": next(row for row in average_rows if row["alternative"] == a and row["reference"] == r and row["evaluation_measure"] == m) for a in METHODS for m, r in reference_method.items()}}


def write_outputs(instances: list[dict[str, Any]], maxmin_rows: list[dict[str, Any]], pf: dict[str, dict[str, Any]], ks: dict[str, dict[str, Any]]) -> None:
    per_instance, averages, cell_data = build_comparison(instances, maxmin_rows, pf, ks)
    fields = sorted({key for row in per_instance for key in row})
    with (OUTPUT_ROOT / "comparison_gaps_per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_instance)
    average_fields = ["alternative", "reference", "evaluation_measure", "average_percentage_gap", "instances_included", "estimated_ks_records_included", "excluded_total", "excluded_by_reason"]
    with (OUTPUT_ROOT / "comparison_gap_averages.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=average_fields)
        writer.writeheader()
        writer.writerows(averages)
        table_lines = [r"\begin{table}[t]", r"\centering", r"\caption{Average percentage gaps between PF, Maxmin, and KS points under the original BOKP objectives. For each instance, the KS candidate maximizing $\min(P/P_{\max},Q/Q_{\max})$ is selected before comparison.}", r"\label{tab:pf_maxmin_ks_gaps}", r"\begin{tabular}{lrrr}", r"\hline", r"Solution / Evaluation measure & $E_{PF}$ & $E_{MM}$ & $E_{KS}$ \\", r"\hline"]
    for alternative in METHODS:
        values = []
        for name in MEASURES:
            target = next(row for row in averages if row["alternative"] == alternative and row["evaluation_measure"] == name)
            value = target["average_percentage_gap"]
            values.append("NA" if value == "" else f"{float(value):.2f}")
        table_lines.append(f"{alternative} & " + " & ".join(values) + r" \\")
    table_lines.extend([r"\hline", r"\end{tabular}", r"\begin{flushleft}", r"\footnotesize{Diagonal entries are zero. Sample counts and exclusions are in comparison\_gap\_averages.csv. The n=850 instance bokp\_n0850\_i014 uses an ESTIMATED KS record and is included where eligible.}", r"\end{flushleft}", r"\end{table}", ""])
    (OUTPUT_ROOT / "comparison_gap_table.tex").write_text("\n".join(table_lines), encoding="utf-8")
    counts = Counter(row["status"] for row in maxmin_rows)
    metadata = {
        "sources": {"instances": str(INSTANCE_DIR), "ks_summary": str(KS_SUMMARY), "ks_instance_results": str(KS_RESULTS_DIR), "pf_results": str(PF_ROOT), "instance_count": len(instances), "instance_ids": [instance["instance_id"] for instance in instances]},
        "maxmin": {"formulation": "maximize z subject to z <= f1(x), z <= f2(x), and the existing BOKP constraints", "objectives": "original unshifted f1=sum(p_i*x_i), f2=sum(q_i*x_i)", "solver": {"name": "Gurobi", "version": ".".join(map(str, gp.gurobi.version())), "threads": 1, "seed": GUROBI_SEED, "mip_gap": 0.0, "mip_gap_abs": 0.0, "time_limit_seconds": MAXMIN_TIME_LIMIT}, "status_counts": dict(counts)},
        "conventions": {"tie_selection": "Gurobi deterministic solution selection with one objective z; no secondary tie-breaking objective", "ks_selection": "When Phase II returns two boundary candidates, retain the point maximizing min(P/P_max, Q/Q_max), with P=f1-f1_nadir and Q=f2-f2_nadir.", "estimated_ks": {**ks["bokp_n0850_i014"], "reference_endpoints": {"x1": endpoint_data("bokp_n0850_i014", ks)[0], "x2": endpoint_data("bokp_n0850_i014", ks)[1]}}, "estimated_records_included_by_cell": {key: value["estimated_ks_records_included"] for key, value in cell_data["cells"].items()}},
        "formulas": {"KS_selection": "argmax_x min((f1(x)-f1_nadir)/P_max, (f2(x)-f2_nadir)/Q_max)", "E_KS": "min((f1(x)-f1(x2))/(f1(x1)-f1(x2)), (f2(x)-f2(x1))/(f2(x2)-f2(x1)))", "E_PF": "f1(x)*f2(x)", "E_MM": "min(f1(x), f2(x))", "gap": "100*(E(alternative)-E(reference))/E(reference)"},
        "comparison_cells": cell_data["cells"],
        "notes": ["PF_FOUND is the only PF status treated as available.", "Only OPTIMAL Maxmin rows are comparison-eligible; TIME_LIMIT, FAILED, and INFEASIBLE rows remain recorded but are excluded.", "The estimated KS record is not solver-computed and is flagged as ESTIMATED in all comparison outputs."],
    }
    (OUTPUT_ROOT / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    instances = load_instances()
    pf = load_pf()
    ks, _ = load_ks(instances)
    validate_small_enumeration()
    maxmin_rows = run_maxmin(instances)
    write_outputs(instances, maxmin_rows, pf, ks)
    print(json.dumps({"output": str(OUTPUT_ROOT), "maxmin_status_counts": dict(Counter(row["status"] for row in maxmin_rows))}, indent=2))


if __name__ == "__main__":
    main()