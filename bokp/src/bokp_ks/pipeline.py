from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gurobipy as gp

from .domain import BOKPInstance, ObjectivePoint
from .gurobi_model import GurobiModel, startup_test
from .instance_generator import InstanceGenerator
from .phase_one import PhaseOneSearch
from .phase_two import PhaseTwoSearch
from .reference_points import ReferencePointCalculator
from .result_writer import ResultWriter, encode


class ExperimentRunner:
    def __init__(self, output_root: Path, base_seed: int = 20261017, gurobi_seed: int = 271828):
        self.writer = ResultWriter(output_root)
        self.generator = InstanceGenerator(base_seed)
        self.gurobi_seed = gurobi_seed

    def run(self, ns: list[int], instances_per_n: int = 1) -> dict[str, Any]:
        if instances_per_n < 1:
            raise ValueError("instances_per_n must be positive")
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        startup = startup_test(self.gurobi_seed)
        instance_ids = []
        failures = []
        for n in ns:
            for instance_index in range(1, instances_per_n + 1):
                input_path = self.writer.instances / f"bokp_n{n:04d}_i{instance_index:03d}.json"
                if input_path.exists():
                    instance = self.generator.from_dict(json.loads(input_path.read_text()))
                else:
                    instance = self.generator.generate(n, instance_index)
                self.writer.write_instance(instance)
                instance_ids.append(instance.instance_id)
                print(f"Running n={n}, instance={instance_index}/{instances_per_n}...", flush=True)
                try:
                    result = self._run_instance(instance)
                except Exception as error:
                    self.writer.write_failure(instance, error)
                    failures.append({"instance_id": instance.instance_id, "error": repr(error)})
                    print(f"  FAILED: {error!r}", flush=True)
                    continue
                self.writer.write_result(result)
                print(f"  complete in {result['runtimes']['total_seconds']:.3f}s; {len(result['final_ks_points'])} final point(s)", flush=True)
        summary = self.writer.write_summary(instance_ids)
        metadata = {"started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "total_wall_seconds": time.perf_counter() - started, "instance_count": len(instance_ids), "instances_per_n": instances_per_n, "n_values": ns, "startup_test": startup, "gurobi_version": ".".join(map(str, gp.gurobi.version())), "gurobi_seed": self.gurobi_seed, "platform": platform.platform(), "python": sys.version, "output_root": str(self.writer.root), "summary_path": str(summary), "failures": failures}
        (self.writer.root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata

    def _run_instance(self, instance: BOKPInstance) -> dict[str, Any]:
        started = time.perf_counter()
        log_path = self.writer.logs / f"{instance.instance_id}.log"
        solver = GurobiModel(instance, log_path, self.gurobi_seed)
        references, endpoint_p, endpoint_q = ReferencePointCalculator().calculate(solver)
        phase1 = PhaseOneSearch().run(solver, references, endpoint_p, endpoint_q)
        phase2_p = phase2_q = None
        if phase1["outcome"] == "C_SN":
            positive = ObjectivePoint(phase1["positive"]["P"], phase1["positive"]["Q"], tuple(phase1["positive"]["x"]))
            negative = ObjectivePoint(phase1["negative"]["P"], phase1["negative"]["Q"], tuple(phase1["negative"]["x"]))
            phase2_p = PhaseTwoSearch().run_p(solver, references, positive, negative)
            phase2_q = PhaseTwoSearch().run_q(solver, references, positive, negative)
            reported = [phase2_p["point"], phase2_q["point"]]
        else:
            reported = [phase1["point"]]
        distinct = {(point["P"], point["Q"]): point for point in reported}
        final_points = list(distinct.values())
        self._validate(instance, references, phase1, phase2_p, phase2_q, final_points)
        records = solver.records
        return {"status": "COMPLETED", "instance": instance.as_dict(), "gurobi": {"version": ".".join(map(str, gp.gurobi.version())), "configuration": {"threads": 1, "seed": self.gurobi_seed, "mip_gap": 0.0, "mip_gap_abs": 0.0, "time_limit": "infinite"}}, "references": references.as_dict(), "endpoints": {"P_endpoint": endpoint_p.as_dict(), "Q_endpoint": endpoint_q.as_dict()}, "phase1": phase1, "phase2_P": phase2_p, "phase2_Q": phase2_q, "final_ks_points": final_points, "validation": {"passed": ["coefficient_ranges", "capacity", "reference_coordinates", "nadir_lemma_coordinates", "endpoint_solution_coordinates", "direct_recomputation", "capacity_feasibility", "phase1_signs_and_bounds", "phase2_epsilon_constraints", "nondominance", "trace_reconstruction"], "failed": []}, "solver_records": [record.as_dict() for record in records], "runtimes": {"total_seconds": time.perf_counter() - started, "reference_seconds": sum(record.runtime_seconds for record in records if record.stage.startswith("reference")), "phase1_seconds": sum(record.runtime_seconds for record in records if record.stage.startswith("phase1")), "phase2_P_seconds": sum(record.runtime_seconds for record in records if record.stage.startswith("phase2P")), "phase2_Q_seconds": sum(record.runtime_seconds for record in records if record.stage.startswith("phase2Q"))}}

    @staticmethod
    def _validate(instance: BOKPInstance, references: Any, phase1: dict[str, Any], phase2_p: dict[str, Any] | None, phase2_q: dict[str, Any] | None, points: list[dict[str, Any]]) -> None:
        assert all(1 <= value <= 20 for value in instance.weights) and all(1 <= value <= 30 for value in instance.p) and all(10 <= value <= 40 for value in instance.q)
        assert instance.capacity == int(0.5 * sum(instance.weights))
        assert references.nadir_lambda_1.f1 == references.f1_utopia and references.nadir_lambda_2.f2 == references.f2_utopia
        for point in points:
            x = tuple(point["x"])
            f1, f2 = instance.objectives(x)
            assert instance.weight(x) <= instance.capacity and f1 - references.f1_nadir == point["P"] and f2 - references.f2_nadir == point["Q"]
        endpoint_p = references.point(references.nadir_lambda_1)
        endpoint_q = references.point(references.nadir_lambda_2)
        assert endpoint_p.p == references.p_max and endpoint_p.q == 0
        assert endpoint_q.p == 0 and endpoint_q.q == references.q_max
        if phase1["outcome"] == "C_SN":
            positive, negative = phase1["positive"], phase1["negative"]
            assert positive["Q"] < negative["Q"] and negative["P"] < positive["P"] and phase2_p and phase2_q
            for step in phase2_p["trace"]:
                assert step["solution"]["Q"] >= step["epsilon_m"]
            for step in phase2_q["trace"]:
                assert step["solution"]["P"] >= step["epsilon_m"]
        for first_index, first in enumerate(points):
            for second in points[first_index + 1:]:
                assert not ((first["P"] >= second["P"] and first["Q"] >= second["Q"]) or (second["P"] >= first["P"] and second["Q"] >= first["Q"]))