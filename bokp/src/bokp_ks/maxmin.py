from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gurobipy as gp

from .domain import BOKPInstance
from .gurobi_model import GurobiModel


@dataclass(frozen=True)
class MaxminOutcome:
    status: str
    maxmin_value: float | None
    incumbent_value: float | None
    best_bound: float | None
    f1: int | None
    f2: int | None
    x: tuple[int, ...]
    runtime_seconds: float
    solver_status_code: int
    error: str = ""

    def as_row(self, instance: BOKPInstance, solution_path: str = "") -> dict[str, Any]:
        return {
            "n": instance.n,
            "instance_id": instance.instance_id,
            "seed": instance.seed,
            "status": self.status,
            "maxmin_value": self.maxmin_value if self.status == "OPTIMAL" else "",
            "incumbent_value": self.incumbent_value if self.incumbent_value is not None else "",
            "best_bound": self.best_bound if self.best_bound is not None else "",
            "f1_MM": self.f1 if self.f1 is not None else "",
            "f2_MM": self.f2 if self.f2 is not None else "",
            "solution_path": solution_path,
            "runtime_seconds": self.runtime_seconds,
            "solver_status_code": self.solver_status_code,
            "error": self.error,
        }


class MaxminOptimizer:
    def __init__(self, instance: BOKPInstance, log_path: Path, seed: int = 271828, time_limit: float = 300.0):
        self.instance = instance
        self.seed = seed
        self.time_limit = time_limit
        self.model = GurobiModel(instance, log_path, seed)
        self.model.model.Params.TimeLimit = time_limit
        self.z = self.model.model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="z")
        f1 = gp.quicksum(instance.p[i] * self.model.x[i] for i in range(instance.n))
        f2 = gp.quicksum(instance.q[i] * self.model.x[i] for i in range(instance.n))
        self.model.model.addConstr(self.z <= f1, name="maxmin_f1")
        self.model.model.addConstr(self.z <= f2, name="maxmin_f2")
        self.model.model.setObjective(self.z, gp.GRB.MAXIMIZE)

    def run(self) -> MaxminOutcome:
        started = time.perf_counter()
        try:
            self.model.model.optimize()
            status_code = self.model.model.Status
            runtime = time.perf_counter() - started
            status_names = {
                gp.GRB.OPTIMAL: "OPTIMAL",
                gp.GRB.TIME_LIMIT: "TIME_LIMIT",
                gp.GRB.INFEASIBLE: "INFEASIBLE",
            }
            status = status_names.get(status_code, "FAILED")
            has_incumbent = self.model.model.SolCount > 0
            incumbent = float(self.model.model.ObjVal) if has_incumbent else None
            best_bound = float(self.model.model.ObjBound) if status_code in {gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT} else None
            if not has_incumbent:
                return MaxminOutcome(status, None, None, best_bound, None, None, (), runtime, status_code)
            x = tuple(int(round(self.model.x[i].X)) for i in range(self.instance.n))
            f1_value, f2_value = self.instance.objectives(x)
            if status == "OPTIMAL":
                return MaxminOutcome(status, incumbent, incumbent, best_bound, f1_value, f2_value, x, runtime, status_code)
            if status == "TIME_LIMIT":
                return MaxminOutcome(status, None, incumbent, best_bound, f1_value, f2_value, x, runtime, status_code)
            return MaxminOutcome(status, None, incumbent, best_bound, f1_value, f2_value, x, runtime, status_code)
        except Exception as error:
            return MaxminOutcome("FAILED", None, None, None, None, None, (), time.perf_counter() - started, -1, repr(error))