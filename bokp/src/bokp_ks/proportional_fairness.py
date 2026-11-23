from __future__ import annotations

import json
import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import gurobipy as gp

from .domain import BOKPInstance, SolveRecord
from .gurobi_model import GurobiModel


@dataclass(frozen=True)
class PFPoint:
    F1: int
    F2: int
    x: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"F1": self.F1, "F2": self.F2, "x": list(self.x)}


@dataclass
class PFOutcome:
    status: str
    point: PFPoint | None
    f2_max: int
    epsilon: Fraction
    f_solves: int
    g_solves: int
    search_iterations: int
    runtime_seconds: float
    error: str = ""
    positivity_assumption_holds: bool = False
    positivity_witness: tuple[int, ...] | None = None

    def as_row(self, instance: BOKPInstance) -> dict[str, Any]:
        return {
            "n": instance.n, "instance_id": instance.instance_id, "seed": instance.seed,
            "status": self.status, "F1_PF": self.point.F1 if self.point else "", "F2_PF": self.point.F2 if self.point else "",
            "solution_path": f"solutions/{instance.instance_id}.json" if self.point else "",
            "F2_max": self.f2_max, "epsilon": f"{self.epsilon.numerator}/{self.epsilon.denominator}",
            "f_solves": self.f_solves, "g_solves": self.g_solves, "search_iterations": self.search_iterations,
            "runtime_seconds": self.runtime_seconds, "positivity_assumption_holds": self.positivity_assumption_holds,
            "positivity_witness": json.dumps(list(self.positivity_witness)) if self.positivity_witness is not None else "",
            "error": self.error,
        }


class PFOptimizer:
    def __init__(self, instance: BOKPInstance, log_path: Path, seed: int = 271828):
        self.instance = instance
        self.model = GurobiModel(instance, log_path, seed)
        self.f_solves = 0
        self.g_solves = 0
        self.search_iterations = 0

    def solve_f(self, alpha: Fraction, stage: str) -> PFPoint:
        # alpha=a/b: b*F1+a*F2 has exactly the same maximizers as F1+alpha*F2.
        record = self.model.solve(stage, [alpha.denominator * p + alpha.numerator * q for p, q in zip(self.instance.p, self.instance.q)])
        self.f_solves += 1
        return PFPoint(record.f1 or 0, record.f2 or 0, record.x)

    def solve_g(self, alpha: Fraction, stage: str) -> PFPoint:
        # G(alpha)=F1+alpha F2-|F1-alpha F2| = 2 min(F1, alpha F2).
        a, b = alpha.numerator, alpha.denominator
        t = self.model.model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"pf_t_{self.g_solves}")
        self.model.model.addConstr(t <= gp.quicksum(b * self.instance.p[i] * self.model.x[i] for i in range(self.instance.n)), name=f"pf_g_f1_{self.g_solves}")
        self.model.model.addConstr(t <= gp.quicksum(a * self.instance.q[i] * self.model.x[i] for i in range(self.instance.n)), name=f"pf_g_f2_{self.g_solves}")
        self.model.model.setObjective(2 * t, gp.GRB.MAXIMIZE)
        started = time.perf_counter()
        self.model.model.optimize()
        elapsed = time.perf_counter() - started
        if self.model.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"{stage} did not solve to optimality: status={self.model.model.Status}, sol_count={self.model.model.SolCount}")
        x = tuple(int(round(self.model.x[i].X)) for i in range(self.instance.n))
        F1, F2 = self.instance.objectives(x)
        self.model.records.append(SolveRecord(stage, self.model.model.Status, "OPTIMAL", self.model.model.ObjVal, elapsed, {"sol_count": self.model.model.SolCount, "has_incumbent": True}, {"Threads": 1, "Seed": self.model.seed, "TimeLimit": "infinite", "MIPGap": 0.0, "MIPGapAbs": 0.0}, F1, F2, x))
        self.g_solves += 1
        return PFPoint(F1, F2, x)

    @staticmethod
    def scaled_value(alpha: Fraction, point: PFPoint) -> int:
        return alpha.denominator * point.F1 + alpha.numerator * point.F2

    def verify_pfsol(self, alpha0: Fraction, point: PFPoint) -> PFPoint | None:
        if point.F1 * alpha0.denominator - point.F2 * alpha0.numerator == 0:
            return point
        alpha_prime = Fraction(point.F1, point.F2)
        candidate = self.solve_f(alpha_prime, "PF_verify_sol")
        return candidate if self.scaled_value(alpha_prime, candidate) == self.scaled_value(alpha_prime, point) else None

    def verify_pfcoeff(self, alpha0: Fraction, point: PFPoint) -> PFPoint | None:
        if point.F1 * alpha0.denominator - point.F2 * alpha0.numerator == 0:
            return point
        f_point = self.solve_f(alpha0, "PF_verify_coeff_F")
        g_point = self.solve_g(alpha0, "PF_verify_coeff_G")
        return g_point if 2 * min(alpha0.denominator * g_point.F1, alpha0.numerator * g_point.F2) == self.scaled_value(alpha0, f_point) else None

    def search(self, alpha_i: Fraction, point_i: PFPoint, alpha_j: Fraction, point_j: PFPoint, epsilon: Fraction) -> PFPoint | None:
        self.search_iterations += 1
        alpha_k = Fraction(point_i.F1 - point_j.F1, point_j.F2 - point_i.F2)
        if alpha_k == alpha_i or alpha_k == alpha_j:
            return None
        point_k = self.solve_f(alpha_k, "PF_search_k")
        verified = self.verify_pfsol(alpha_k, point_k) or self.verify_pfcoeff(alpha_k, point_k)
        if verified:
            return verified
        if (point_k.F1, point_k.F2) in {(point_i.F1, point_i.F2), (point_j.F1, point_j.F2)} or alpha_j - alpha_i < epsilon:
            return None
        alpha_s = (alpha_i + alpha_j) / 2
        point_s = self.solve_f(alpha_s, "PF_search_mid")
        verified = self.verify_pfsol(alpha_s, point_s) or self.verify_pfcoeff(alpha_s, point_s)
        if verified:
            return verified
        sign = point_s.F1 * alpha_s.denominator - point_s.F2 * alpha_s.numerator
        if sign > 0:
            return self.search(alpha_s, point_s, alpha_j, point_j, epsilon)
        if sign < 0:
            return self.search(alpha_i, point_i, alpha_s, point_s, epsilon)
        raise RuntimeError("T(alpha_s)=0 but VerifyPFsol and VerifyPFcoeff both failed")

    def run(self) -> PFOutcome:
        started = time.perf_counter()
        positivity_witness = (0,) * self.instance.n
        positivity_holds = self.instance.objectives(positivity_witness) != (0, 0)
        try:
            zero = self.solve_f(Fraction(0), "PF_F0")
            f2_max = self.model.solve("PF_F2_max", self.instance.q)
            self.f_solves += 1
            f2_max_value = f2_max.f2 or 0
            if f2_max_value <= 0:
                raise RuntimeError("F2_max is not positive")
            epsilon = Fraction(1, f2_max_value * f2_max_value)
            point = self.verify_pfsol(Fraction(0), zero)
            if point is None:
                alpha_sup = Fraction(zero.F1, zero.F2) + 1
                sup = self.solve_f(alpha_sup, "PF_Fsup")
                point = self.verify_pfsol(alpha_sup, sup)
                if point is None:
                    point = self.search(Fraction(0), zero, alpha_sup, sup, epsilon)
            status = "PF_FOUND" if point else "PF_NOT_FOUND"
            return PFOutcome(status, point, f2_max_value, epsilon, self.f_solves, self.g_solves, self.search_iterations, time.perf_counter() - started, positivity_assumption_holds=positivity_holds, positivity_witness=None if positivity_holds else positivity_witness)
        except Exception as error:
            return PFOutcome("FAILED", None, 0, Fraction(0), self.f_solves, self.g_solves, self.search_iterations, time.perf_counter() - started, repr(error), positivity_holds, None if positivity_holds else positivity_witness)


def defining_pf_holds(point: tuple[int, int], feasible_points: Iterable[tuple[int, int]]) -> bool:
    F1_pf, F2_pf = point
    return all(F1 * F2_pf + F2 * F1_pf <= 2 * F1_pf * F2_pf for F1, F2 in feasible_points)


def enumerate_objective_points(instance: BOKPInstance) -> set[tuple[int, int]]:
    return {instance.objectives(x) for x in product((0, 1), repeat=instance.n) if instance.weight(x) <= instance.capacity}
