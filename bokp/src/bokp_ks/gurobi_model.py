from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import gurobipy as gp

from .domain import BOKPInstance, SolveRecord


class GurobiModel:
    def __init__(self, instance: BOKPInstance, log_path: Path, seed: int = 271828):
        self.instance = instance
        self.seed = seed
        self.records: list[SolveRecord] = []
        self.model = gp.Model(instance.instance_id)
        self.model.Params.OutputFlag = 0
        self.model.Params.LogFile = str(log_path)
        self.model.Params.Threads = 1
        self.model.Params.Seed = seed
        self.model.Params.TimeLimit = float("inf")
        self.model.Params.MIPGap = 0.0
        self.model.Params.MIPGapAbs = 0.0
        self.x = self.model.addVars(instance.n, vtype=gp.GRB.BINARY, name="x")
        self.model.addConstr(gp.quicksum(instance.weights[i] * self.x[i] for i in range(instance.n)) <= instance.capacity, name="capacity")

    def solve(self, stage: str, coefficients: Sequence[int | float], lower_bound: tuple[Sequence[int], int] | None = None) -> SolveRecord:
        temporary_name = f"temporary_{stage}"
        if lower_bound is not None:
            coeffs, rhs = lower_bound
            self.model.addConstr(gp.quicksum(coeffs[i] * self.x[i] for i in range(self.instance.n)) >= rhs, name=temporary_name)
        self.model.setObjective(gp.quicksum(coefficients[i] * self.x[i] for i in range(self.instance.n)), gp.GRB.MAXIMIZE)
        started = time.perf_counter()
        self.model.optimize()
        elapsed = time.perf_counter() - started
        status = self.model.Status
        status_names = {gp.GRB.OPTIMAL: "OPTIMAL", gp.GRB.INFEASIBLE: "INFEASIBLE", gp.GRB.UNBOUNDED: "UNBOUNDED", gp.GRB.INF_OR_UNBD: "INF_OR_UNBD", gp.GRB.TIME_LIMIT: "TIME_LIMIT"}
        record_data = {"stage": stage, "status_code": status, "status": status_names.get(status, str(status)), "objective_value": None, "runtime_seconds": elapsed, "termination": {"sol_count": self.model.SolCount, "has_incumbent": bool(self.model.SolCount)}, "parameters": {"Threads": 1, "Seed": self.seed, "TimeLimit": "infinite", "MIPGap": 0.0, "MIPGapAbs": 0.0}}
        if status != gp.GRB.OPTIMAL:
            if lower_bound is not None:
                self.model.remove(self.model.getConstrByName(temporary_name))
                self.model.update()
            raise RuntimeError(f"{stage} did not solve to optimality: {record_data}")
        values = tuple(int(round(self.x[i].X)) for i in range(self.instance.n))
        f1, f2 = self.instance.objectives(values)
        record = SolveRecord(stage, status, record_data["status"], self.model.ObjVal, elapsed, record_data["termination"], record_data["parameters"], f1, f2, values)
        self.records.append(record)
        if lower_bound is not None:
            self.model.remove(self.model.getConstrByName(temporary_name))
            self.model.update()
        return record


def startup_test(seed: int = 271828) -> dict[str, object]:
    model = gp.Model("startup_test")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Seed = seed
    x, y = model.addVar(vtype=gp.GRB.BINARY), model.addVar(vtype=gp.GRB.BINARY)
    model.addConstr(x + y <= 1)
    model.setObjective(2 * x + y, gp.GRB.MAXIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Startup test failed with status {model.Status}")
    return {"status": "OPTIMAL", "objective": model.ObjVal, "solution": [x.X, y.X], "gurobi_version": ".".join(map(str, gp.gurobi.version()))}