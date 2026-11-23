from __future__ import annotations

import csv
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from .domain import BOKPInstance, ReferencePoints, SolveRecord


def encode(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot encode {type(value)!r}")


class ResultWriter:
    def __init__(self, root: Path):
        self.root = root
        self.instances = root / "instances"
        self.instance_results = root / "instance_results"
        self.traces = root / "traces"
        self.logs = root / "logs"
        for directory in (self.instances, self.instance_results, self.traces, self.logs):
            directory.mkdir(parents=True, exist_ok=True)

    def write_instance(self, instance: BOKPInstance) -> Path:
        path = self.instances / f"{instance.instance_id}.json"
        path.write_text(json.dumps(instance.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def write_result(self, result: dict[str, Any]) -> None:
        instance_id = result["instance"]["instance_id"]
        (self.instance_results / f"{instance_id}.json").write_text(json.dumps(result, indent=2, default=encode) + "\n", encoding="utf-8")
        trace = {"instance_id": instance_id, "phase1": result.get("phase1"), "phase2_P": result.get("phase2_P"), "phase2_Q": result.get("phase2_Q"), "solver_records": result.get("solver_records", [])}
        (self.traces / f"{instance_id}.json").write_text(json.dumps(trace, indent=2, default=encode) + "\n", encoding="utf-8")

    def write_failure(self, instance: BOKPInstance, error: Exception) -> None:
        self.write_result({"instance": instance.as_dict(), "status": "FAILED", "error": repr(error), "validation": {"passed": [], "failed": [repr(error)]}})

    def write_summary(self, instance_ids: list[str]) -> Path:
        rows = []
        for instance_id in instance_ids:
            data = json.loads((self.instance_results / f"{instance_id}.json").read_text())
            references = data.get("references", {})
            phase1 = data.get("phase1", {})
            phase2_p, phase2_q = data.get("phase2_P"), data.get("phase2_Q")
            rows.append({"n": data["instance"]["n"], "instance_id": instance_id, "f1_utopia": references.get("f1_utopia"), "f2_utopia": references.get("f2_utopia"), "f1_nadir": references.get("f1_nadir"), "f2_nadir": references.get("f2_nadir"), "P_max": references.get("P_max"), "Q_max": references.get("Q_max"), "phase1_outcome": phase1.get("outcome"), "phase2_P_outcome": phase2_p.get("outcome") if phase2_p else "SKIPPED", "phase2_Q_outcome": phase2_q.get("outcome") if phase2_q else "SKIPPED", "phase1_iterations": phase1.get("iterations", 0), "phase2_P_iterations": phase2_p.get("iterations", 0) if phase2_p else 0, "phase2_Q_iterations": phase2_q.get("iterations", 0) if phase2_q else 0, "reference_seconds": data.get("runtimes", {}).get("reference_seconds", 0), "phase1_seconds": data.get("runtimes", {}).get("phase1_seconds", 0), "phase2_P_seconds": data.get("runtimes", {}).get("phase2_P_seconds", 0), "phase2_Q_seconds": data.get("runtimes", {}).get("phase2_Q_seconds", 0), "total_seconds": data.get("runtimes", {}).get("total_seconds", 0), "final_ks_points": json.dumps([(p["P"], p["Q"]) for p in data.get("final_ks_points", [])])})
        path = self.root / "ks_summary.csv"
        columns = list(rows[0]) if rows else ["instance_id"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row.get("n", 0)))
        return path