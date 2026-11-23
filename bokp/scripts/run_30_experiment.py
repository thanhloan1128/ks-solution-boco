#!/usr/bin/env python3
"""Run the existing KS experiment for 30 indexed instances per n and aggregate outputs."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bokp_ks.pipeline import ExperimentRunner

NS = list(range(50, 1001, 50))
INSTANCES_PER_N = 30
BASE_SEED = 20261017
GUROBI_SEED = 271828
OUTPUT_ROOT = ROOT / "results_30_instances_20261017"
PER_INSTANCE_CSV = OUTPUT_ROOT / "per_instance_results.csv"
AVERAGE_CSV = OUTPUT_ROOT / "average_results.csv"
LATEX_OUTPUT = OUTPUT_ROOT / "ks_summary_average.tex"
METADATA = OUTPUT_ROOT / "run_metadata.json"

PER_INSTANCE_FIELDS = [
    "n", "instance_id", "seed", "status", "error", "f1_utopia", "f2_utopia", "f1_nadir", "f2_nadir", "P_max", "Q_max",
    "phase1_outcome", "phase2_P_outcome", "phase2_Q_outcome", "rho_minus_I", "rho_plus_I", "rho_minus_II", "rho_plus_II",
    "rho_minus_II_measure", "rho_plus_II_measure", "reference_seconds", "phase1_seconds", "phase2_P_seconds", "phase2_Q_seconds", "total_seconds",
    "phase1_iterations", "phase1_solver_calls", "phase2_P_iterations", "phase2_Q_iterations", "iterations_II", "final_ks_points",
]

AVERAGE_FIELDS = ["n", "runs_used", "failed_runs"] + [
    "f1_utopia", "f2_utopia", "f1_nadir", "f2_nadir", "P_max", "Q_max", "rho_minus_I", "rho_plus_I", "rho_minus_II", "rho_plus_II",
    "rho_minus_II_measure", "rho_plus_II_measure", "reference_seconds", "phase1_seconds", "phase2_P_seconds", "phase2_Q_seconds", "total_seconds",
    "phase1_iterations", "phase1_solver_calls", "phase2_P_iterations", "phase2_Q_iterations", "iterations_II",
]

TABLE_COLUMNS = [
    ("n", "n"), ("f1_utopia", r"$f_1^{\mathrm{utopia}}$"), ("f2_utopia", r"$f_2^{\mathrm{utopia}}$"),
    ("f1_nadir", r"$f_1^{\mathrm{nadir}}$"), ("f2_nadir", r"$f_2^{\mathrm{nadir}}$"), ("P_max", r"$P_{\max}$"), ("Q_max", r"$Q_{\max}$"),
    ("rho_minus_I", r"$\rho_I^-$"), ("rho_plus_I", r"$\rho_I^+$"), ("rho_minus_II", r"$\rho_{II}^-$"), ("rho_plus_II", r"$\rho_{II}^+$"),
    ("reference_seconds", r"$t_{\mathrm{ref}}$"), ("phase1_seconds", r"$t_I$"), ("phase2_P_seconds", r"$t_{II-P}$"), ("phase2_Q_seconds", r"$t_{II-Q}$"),
    ("total_seconds", r"$t_{\mathrm{total}}$"), ("phase1_iterations", r"$k_I$"), ("phase2_P_iterations", r"$k_{II-P}$"), ("phase2_Q_iterations", r"$k_{II-Q}$"),
]


def rho(point: dict[str, int], p_max: int, q_max: int) -> float:
    return point["P"] / p_max - point["Q"] / q_max


def result_row(data: dict) -> dict[str, object]:
    instance = data["instance"]
    row = {field: "" for field in PER_INSTANCE_FIELDS}
    row.update({"n": instance["n"], "instance_id": instance["instance_id"], "seed": instance["seed"], "status": data.get("status", "FAILED"), "error": data.get("error", "")})
    references = data.get("references", {})
    phase1 = data.get("phase1", {})
    runtimes = data.get("runtimes", {})
    row.update({key: references.get(key, "") for key in ("f1_utopia", "f2_utopia", "f1_nadir", "f2_nadir", "P_max", "Q_max")})
    row.update({key: runtimes.get(key, 0) for key in ("reference_seconds", "phase1_seconds", "phase2_P_seconds", "phase2_Q_seconds", "total_seconds")})
    row["phase1_outcome"] = phase1.get("outcome", "")
    row["phase1_iterations"] = phase1.get("iterations", 0)
    row["phase1_solver_calls"] = phase1.get("iterations", 0) + int(bool(phase1.get("final_alpha_s_included")))
    p_max, q_max = references.get("P_max"), references.get("Q_max")
    if p_max and q_max and phase1.get("outcome") == "C_SN":
        row["rho_minus_I"] = rho(phase1["negative"], p_max, q_max)
        row["rho_plus_I"] = rho(phase1["positive"], p_max, q_max)
    elif p_max and q_max and phase1.get("point"):
        direct = rho(phase1["point"], p_max, q_max)
        row["rho_minus_I"] = direct
        row["rho_plus_I"] = direct
    phase2_p, phase2_q = data.get("phase2_P"), data.get("phase2_Q")
    if phase2_p and phase2_q and p_max and q_max:
        row["phase2_P_outcome"] = phase2_p.get("outcome", "")
        row["phase2_Q_outcome"] = phase2_q.get("outcome", "")
        row["rho_minus_II"] = rho(phase2_q["point"], p_max, q_max)
        row["rho_plus_II"] = rho(phase2_p["point"], p_max, q_max)
        row["rho_minus_II_measure"] = -row["rho_minus_II"]
        row["rho_plus_II_measure"] = row["rho_plus_II"]
        row["phase2_P_iterations"] = phase2_p.get("iterations", 0)
        row["phase2_Q_iterations"] = phase2_q.get("iterations", 0)
        row["iterations_II"] = row["phase2_P_iterations"] + row["phase2_Q_iterations"]
    row["final_ks_points"] = json.dumps(data.get("final_ks_points", []), separators=(",", ":"))
    return row


def numeric(value: object) -> float | None:
    if value in (None, "", "NA"):
        return None
    return float(value)


def write_aggregates(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    with PER_INSTANCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_INSTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    averages = []
    failures = []
    for n in NS:
        group = [row for row in rows if row["n"] == n]
        successful = [row for row in group if row["status"] == "COMPLETED"]
        failed = [row for row in group if row["status"] != "COMPLETED"]
        failures.extend(f"{row['instance_id']}: {row['error']}" for row in failed)
        average = {"n": n, "runs_used": len(successful), "failed_runs": len(failed)}
        for field in AVERAGE_FIELDS[3:]:
            values = [numeric(row[field]) for row in successful]
            values = [value for value in values if value is not None and math.isfinite(value)]
            average[field] = mean(values) if values else "NA"
        averages.append(average)
    with AVERAGE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AVERAGE_FIELDS)
        writer.writeheader()
        writer.writerows(averages)
    return averages, failures


def write_latex(averages: list[dict[str, object]]) -> None:
    headers = " & ".join(label for _, label in TABLE_COLUMNS) + r" \\"
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Average KS-BOKP results over successful runs}", r"\label{tab:ks_summary}", r"\begin{tabular}{r" + "r" * (len(TABLE_COLUMNS) - 1) + "}", r"\hline", headers, r"\hline"]
    for row in averages:
        values = []
        for key, _ in TABLE_COLUMNS:
            value = row[key]
            if key == "n":
                values.append(str(int(value)))
            elif key.endswith("_seconds") or key.startswith("rho_"):
                values.append("NA" if value == "NA" else f"{float(value):.4f}")
            else:
                values.append("NA" if value == "NA" else f"{float(value):.2f}")
        lines.append(" & ".join(values) + " \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    LATEX_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {OUTPUT_ROOT}")
    with tempfile.TemporaryDirectory(prefix="bokp_ks_smoke_") as smoke_directory:
        smoke_metadata = ExperimentRunner(Path(smoke_directory), BASE_SEED, GUROBI_SEED).run([50], 1)
        smoke_result = Path(smoke_directory) / "instance_results" / "bokp_n0050_i001.json"
        if smoke_metadata["instance_count"] != 1 or not smoke_result.exists() or json.loads(smoke_result.read_text())["status"] != "COMPLETED":
            raise RuntimeError("Smoke test did not produce a completed reloadable result")
        print(f"Smoke test passed: {smoke_result}", flush=True)
    metadata = ExperimentRunner(OUTPUT_ROOT, BASE_SEED, GUROBI_SEED).run(NS, INSTANCES_PER_N)
    rows = [result_row(json.loads(path.read_text(encoding="utf-8"))) for path in sorted((OUTPUT_ROOT / "instance_results").glob("*.json"))]
    averages, aggregation_failures = write_aggregates(rows)
    write_latex(averages)
    metadata.update({
        "seed_policy": "BASE_SEED + n * 1009 + (instance_index - 1)",
        "base_seed": BASE_SEED,
        "instance_generation": {"rng": "random.Random", "weight_range": [1, 20], "p_range": [1, 30], "q_range": [10, 40], "capacity": "floor(0.5 * sum(weights))", "draw_order": "weight, p, q per item"},
        "solver_settings": {"threads": 1, "gurobi_seed": GUROBI_SEED, "mip_gap": 0.0, "mip_gap_abs": 0.0, "time_limit": "infinite"},
        "averaging": "successful COMPLETED runs only; failed runs are counted in failed_runs and excluded from means",
        "rho_convention": "signed rho = P/P_max - Q/Q_max; II measures are magnitudes for reporting only",
        "failed_runs": metadata.get("failures", []) + aggregation_failures,
        "outputs": {"instances": str(OUTPUT_ROOT / "instances"), "instance_results": str(OUTPUT_ROOT / "instance_results"), "per_instance_csv": str(PER_INSTANCE_CSV), "average_csv": str(AVERAGE_CSV), "latex": str(LATEX_OUTPUT)},
        "smoke_test": "completed in temporary output directory before the 600-run batch",
    })
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    counts = Counter(row["status"] for row in rows)
    print(json.dumps({"output": str(OUTPUT_ROOT), "total_rows": len(rows), "successful": counts["COMPLETED"], "failed": counts["FAILED"], "failed_by_n": {str(n): sum(row["n"] == n and row["status"] != "COMPLETED" for row in rows) for n in NS}}, indent=2))


if __name__ == "__main__":
    main()
