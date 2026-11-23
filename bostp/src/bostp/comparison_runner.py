from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from .instance import BOSTPInstance
from .reference_points import ReferencePoints
from .spanning_tree_model import MSTSolver, TreeSolution


STAGES = {
    1: (
        Path("results_corrected_stage1/20261031T194058Z"),
        (80, 100),
    ),
    2: (
        Path("results_corrected_stage2_seed20261101/20261101T084002Z"),
        (120, 140),
    ),
    3: (
        Path("results_corrected_stage3_seed20261102/20261101T102448Z"),
        (160, 180),
    ),
}


@dataclass(frozen=True)
class ComparisonPoint:
    f1: int
    f2: int
    edges: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {"f1": self.f1, "f2": self.f2, "edges": self.edges}


def _point_from_dict(data: dict[str, object]) -> ComparisonPoint:
    return ComparisonPoint(
        int(data["f1"]),
        int(data["f2"]),
        tuple(tuple(edge) for edge in data.get("edges", [])),
    )


def _reference(data: dict[str, object]) -> ReferencePoints:
    def tree(value):
        return TreeSolution(
            tuple(tuple(edge) for edge in value["edges"]),
            int(value["f1"]),
            int(value["f2"]),
        )

    return ReferencePoints(
        tree(data["x1"]),
        tree(data["x2"]),
        int(data["f1_u"]),
        int(data["f2_u"]),
        int(data["f1_n"]),
        int(data["f2_n"]),
        int(data["m1"]),
        int(data["m2"]),
        tuple(data["w_x1"]),
        tuple(data["w_x2"]),
    )


def nf_point(instance: BOSTPInstance, reference: ReferencePoints) -> dict[str, object]:
    solver = MSTSolver(instance)
    current = ComparisonPoint(reference.x1.f1, reference.x1.f2, reference.x1.edges)
    verification = []
    replacements = 0
    started = perf_counter()
    while True:
        a, b = current.f1, current.f2
        optimizer = solver.solve(b, a)
        weighted_value = b * optimizer.f1 + a * optimizer.f2
        expected_value = 2 * a * b
        verification.append({
            "a": a,
            "b": b,
            "weighted_optimum": weighted_value,
            "twice_product": expected_value,
            "satisfies_nf": weighted_value == expected_value,
        })
        if weighted_value == expected_value:
            break
        if optimizer.f1 * optimizer.f2 >= a * b:
            raise RuntimeError("NF replacement did not strictly decrease the positive product")
        current = ComparisonPoint(optimizer.f1, optimizer.f2, optimizer.edges)
        replacements += 1
    return {
        "point": current.as_dict(),
        "replacements": replacements,
        "iterations": len(verification),
        "verification": verification,
        "runtime_seconds": perf_counter() - started,
    }


def _add_mtz_constraints(model, instance, gp):
    edges = instance.edges
    root = instance.nodes[0]
    x = {i: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}") for i in range(len(edges))}
    arcs = {
        (u, v, i): model.addVar(vtype=gp.GRB.BINARY, name=f"y_{u}_{v}_{i}")
        for i, edge in enumerate(edges)
        for u, v in ((edge.u, edge.v), (edge.v, edge.u))
    }
    order = {
        node: model.addVar(
            lb=0 if node == root else 1,
            ub=0 if node == root else instance.n - 1,
            vtype=gp.GRB.CONTINUOUS,
            name=f"u_{node}",
        )
        for node in instance.nodes
    }
    model.addConstr(gp.quicksum(x.values()) == instance.n - 1)
    for i, edge in enumerate(edges):
        model.addConstr(arcs[edge.u, edge.v, i] + arcs[edge.v, edge.u, i] == x[i])
    model.addConstr(gp.quicksum(
        arcs[u, root, i]
        for i, edge in enumerate(edges)
        for u, v in ((edge.u, edge.v), (edge.v, edge.u))
        if v == root
    ) == 0)
    for node in instance.nodes:
        if node == root:
            continue
        model.addConstr(gp.quicksum(
            arcs[u, node, i]
            for i, edge in enumerate(edges)
            for u, v in ((edge.u, edge.v), (edge.v, edge.u))
            if v == node
        ) == 1)
    for i, edge in enumerate(edges):
        for u, v in ((edge.u, edge.v), (edge.v, edge.u)):
            if v != root:
                model.addConstr(
                    order[v] >= order[u] + 1 - instance.n * (1 - arcs[u, v, i])
                )
    model.update()
    return x


def minmax_point(instance: BOSTPInstance) -> dict[str, object]:
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError("gurobipy is required for Minmax") from exc

    started = perf_counter()
    model = gp.Model("bostp_minmax")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-5
    model.Params.MIPGapAbs = 0
    model.Params.IntFeasTol = 1e-9
    model.Params.FeasibilityTol = 1e-9
    x = _add_mtz_constraints(model, instance, gp)
    z = model.addVar(vtype=gp.GRB.INTEGER, lb=0, name="z")
    f1 = gp.quicksum(edge.cost * x[i] for i, edge in enumerate(instance.edges))
    f2 = gp.quicksum(edge.time * x[i] for i, edge in enumerate(instance.edges))
    model.addConstr(f1 <= z)
    model.addConstr(f2 <= z)
    model.setObjective(z, gp.GRB.MINIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        model.dispose()
        raise RuntimeError(f"Minmax primary solve failed with Gurobi status {model.Status}")
    optimal_z = int(round(model.ObjVal))
    model.addConstr(z <= optimal_z)
    model.setObjective(f1 + f2, gp.GRB.MINIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        model.dispose()
        raise RuntimeError(f"Minmax tie-break solve failed with Gurobi status {model.Status}")
    selected = tuple(sorted(
        (min(instance.edges[i].u, instance.edges[i].v),
         max(instance.edges[i].u, instance.edges[i].v))
        for i in x if x[i].X > 0.5
    ))
    objective = instance.objective(list(selected))
    solution = ComparisonPoint(objective[0], objective[1], selected)
    MSTSolver(instance).verify(TreeSolution(selected, *objective))
    model.dispose()
    if max(solution.f1, solution.f2) != optimal_z:
        raise RuntimeError("Minmax extracted point does not attain the optimal z")
    return {
        "point": solution.as_dict(),
        "optimal_z": optimal_z,
        "tie_break_sum": solution.f1 + solution.f2,
        "runtime_seconds": perf_counter() - started,
    }


def evaluations(point: ComparisonPoint, reference: ReferencePoints) -> dict[str, str]:
    if reference.p_max == 0 or reference.q_max == 0:
        raise ValueError("cannot evaluate KS score with a degenerate reference range")
    ks = max(
        Fraction(point.f1 - reference.f1_u, reference.p_max),
        Fraction(point.f2 - reference.f2_u, reference.q_max),
    )
    return {
        "E_NF": str(point.f1 * point.f2),
        "E_MM": str(max(point.f1, point.f2)),
        "E_KS": str(ks),
    }


def _percentage_gap(alternative: str, reference: str) -> Fraction | None:
    denominator = Fraction(reference)
    if denominator == 0:
        return None
    return Fraction(alternative) - denominator, denominator


def _load_instances_and_results(source: Path):
    instances = {
        instance.instance_id: instance
        for path in (source / "data" / "instances").glob("*.json")
        for instance in [BOSTPInstance.load(path)]
    }
    results = {
        result["instance_id"]: result
        for path in (source / "checkpoints").glob("*.json")
        for result in [json.loads(path.read_text())]
    }
    if set(instances) != set(results):
        raise ValueError(f"instance/checkpoint mismatch in {source}")
    return instances, results


def _comparison_result(instance, ks_result):
    reference = _reference(ks_result["reference"])
    nf = nf_point(instance, reference)
    minmax = minmax_point(instance)
    ks_points = [_point_from_dict(point) for point in ks_result.get("final_ks_points", [])]
    if not ks_points:
        raise RuntimeError(f"KS result has no final point: {instance.instance_id}")
    ks_records = [
        {"point": point.as_dict(), "evaluations": evaluations(point, reference)}
        for point in ks_points
    ]
    representative = min(
        ks_records,
        key=lambda item: (Fraction(item["evaluations"]["E_KS"]), item["point"]["f1"], item["point"]["f2"]),
    )
    nf_eval = evaluations(_point_from_dict(nf["point"]), reference)
    mm_eval = evaluations(_point_from_dict(minmax["point"]), reference)
    gaps = {}
    for alternative, alternative_eval in (("NF", nf_eval), ("MM", mm_eval)):
        gaps[alternative] = {}
        for measure in ("E_NF", "E_MM", "E_KS"):
            gap = _percentage_gap(alternative_eval[measure], representative["evaluations"][measure])
            gaps[alternative][measure] = None if gap is None else str(100 * gap[0] / gap[1])
    return {
        "instance_id": instance.instance_id,
        "n": instance.n,
        "p": instance.p,
        "ks_status": ks_result["status"],
        "ks_points": ks_records,
        "ks_point_count": len(ks_records),
        "ks_reference": representative,
        "reference": ks_result["reference"],
        "NF": nf,
        "Minmax": minmax,
        "evaluations": {"KS": representative["evaluations"], "NF": nf_eval, "MM": mm_eval},
        "percentage_gaps": gaps,
    }


def _write_reports(output: Path, results: list[dict[str, object]]):
    output.mkdir(parents=True, exist_ok=True)
    (output / "solutions.json").write_text(json.dumps(results, indent=2) + "\n")
    with (output / "per_instance_results.csv").open("w", newline="") as handle:
        fields = ["instance_id", "n", "p", "ks_status", "ks_point_count",
                  "ks_f1", "ks_f2", "nf_f1", "nf_f2", "mm_f1", "mm_f2",
                  "nf_E_NF", "mm_E_NF", "ks_E_NF", "nf_E_MM", "mm_E_MM", "ks_E_MM",
                  "nf_E_KS", "mm_E_KS", "ks_E_KS"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "instance_id": result["instance_id"], "n": result["n"], "p": result["p"],
                "ks_status": result["ks_status"], "ks_point_count": result["ks_point_count"],
                "ks_f1": result["ks_reference"]["point"]["f1"],
                "ks_f2": result["ks_reference"]["point"]["f2"],
                "nf_f1": result["NF"]["point"]["f1"], "nf_f2": result["NF"]["point"]["f2"],
                "mm_f1": result["Minmax"]["point"]["f1"], "mm_f2": result["Minmax"]["point"]["f2"],
            }
            for method, prefix in (("NF", "nf"), ("MM", "mm"), ("KS", "ks")):
                for measure in ("E_NF", "E_MM", "E_KS"):
                    row[f"{prefix}_{measure}"] = result["evaluations"][method][measure]
            writer.writerow(row)
    measures = ("E_NF", "E_MM", "E_KS")
    comparison = []
    for (n, p) in sorted({(r["n"], r["p"]) for r in results}):
        group = [r for r in results if r["n"] == n and r["p"] == p]
        row = {"n": n, "p": p, "instance_count": len(group)}
        for method in ("NF", "MM"):
            for measure in measures:
                values = [
                    Fraction(r["percentage_gaps"][method][measure])
                    for r in group if r["percentage_gaps"][method][measure] is not None
                ]
                row[f"{method}_{measure}_gap_percent"] = (
                    str(sum(values, Fraction()) / len(values)) if values else ""
                )
                row[f"{method}_{measure}_gap_count"] = len(values)
        comparison.append(row)
    (output / "average_percentage_gaps.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (output / "ks_points_report.json").write_text(json.dumps([
        {
            "instance_id": result["instance_id"],
            "ks_status": result["ks_status"],
            "ks_point_count": result["ks_point_count"],
            "ks_points": [item["point"] for item in result["ks_points"]],
        }
        for result in results
    ], indent=2) + "\n")
    with (output / "comparison_table.tex").open("w") as handle:
        handle.write("\\begin{table}[ht!]\n\\centering\n")
        handle.write("\\caption{Average percentage gaps relative to the selected KS point over the evaluated instances; $N$ is the sample count for each comparison}\n")
        handle.write("\\begin{tabular}{rrr rrr rrr}\n\\toprule\n")
        handle.write("$n$ & $p$ & $N$ & \\multicolumn{2}{c}{$E_{\\mathrm{NF}}$} & \\multicolumn{2}{c}{$E_{\\mathrm{MM}}$} & \\multicolumn{2}{c}{$E_{\\mathrm{KS}}$} \\\\\n")
        handle.write(" & & & NF ($N$) & MM ($N$) & NF ($N$) & MM ($N$) & NF ($N$) & MM ($N$) \\\\\n\\midrule\n")
        for row in comparison:
            values = []
            for measure in measures:
                values.extend([
                    (
                        row[f"NF_{measure}_gap_percent"],
                        row[f"NF_{measure}_gap_count"],
                    ),
                    (
                        row[f"MM_{measure}_gap_percent"],
                        row[f"MM_{measure}_gap_count"],
                    ),
                ])
            formatted = [
                "--" if value == "" else f"{float(Fraction(value)):.4f} ({count})"
                for value, count in values
            ]
            handle.write(f"{row['n']} & {row['p']:.1f} & {row['instance_count']} & "
                         + " & ".join(formatted) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def run_stage(stage: int, output_root: Path):
    source, sizes = STAGES[stage]
    instances, ks_results = _load_instances_and_results(source)
    output = output_root / f"stage{stage}"
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"bostp.comparison.stage{stage}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        completed = {}
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        for instance_id in sorted(instances):
            checkpoint = checkpoint_dir / f"{instance_id}.json"
            if checkpoint.exists():
                completed[instance_id] = json.loads(checkpoint.read_text())
                continue
            logger.info("Starting instance %s", instance_id)
            result = _comparison_result(instances[instance_id], ks_results[instance_id])
            checkpoint.write_text(json.dumps(result, indent=2) + "\n")
            completed[instance_id] = result
            logger.info("Finished instance %s", instance_id)
        results = [completed[key] for key in sorted(completed)]
        _write_reports(output, results)
        (output / "run_metadata.json").write_text(json.dumps({
            "stage": stage, "source_run": str(source), "sizes": sizes,
            "instance_count": len(results), "minmax_gap_tolerance": 1e-5,
            "time_limits": None, "ks_results_reused": True,
        }, indent=2) + "\n")
        return output
    finally:
        logger.removeHandler(handler)
        handler.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=STAGES, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_nf_minmax"))
    args = parser.parse_args()
    print(run_stage(args.stage, args.output))


if __name__ == "__main__":
    main()
