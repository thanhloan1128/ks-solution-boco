from __future__ import annotations

from typing import Any

from .domain import ObjectivePoint, ReferencePoints
from .gurobi_model import GurobiModel


class PhaseTwoSearch:
    def run_p(self, solver: GurobiModel, references: ReferencePoints, positive: ObjectivePoint, negative: ObjectivePoint) -> dict[str, Any]:
        current, epsilon_i, epsilon_j, trace, iterations = positive, positive.q, negative.q, [], 0
        while epsilon_j - epsilon_i > 1:
            iterations += 1
            epsilon_m = (epsilon_i + epsilon_j) // 2
            record = solver.solve(f"phase2P_{iterations}", [references.q_max * p + q for p, q in zip(solver.instance.p, solver.instance.q)], (solver.instance.q, references.f2_nadir + epsilon_m))
            point, sign = references.point(record), references.rho_sign(references.point(record))
            trace.append({"iteration": iterations, "epsilon_i": epsilon_i, "epsilon_j": epsilon_j, "epsilon_m": epsilon_m, "solution": point.as_dict(), "rho_sign": sign, "endpoint_update": "return" if sign == 0 else ("positive_and_epsilon_i=epsilon_m" if sign > 0 else "epsilon_j=Q_m")})
            if sign > 0:
                current, epsilon_i = point, epsilon_m
            elif sign < 0:
                epsilon_j = point.q
                if epsilon_j >= trace[-1]["epsilon_j"]:
                    trace[-1]["termination"] = "no_progress_after_prescribed_epsilon_j=Q_m_update"
                    break
            else:
                current = point
                break
        return {"outcome": "KS_POINT", "point": current.as_dict(), "trace": trace, "iterations": iterations}

    def run_q(self, solver: GurobiModel, references: ReferencePoints, positive: ObjectivePoint, negative: ObjectivePoint) -> dict[str, Any]:
        current, epsilon_i, epsilon_j, trace, iterations = negative, negative.p, positive.p, [], 0
        while epsilon_j - epsilon_i > 1:
            iterations += 1
            epsilon_m = (epsilon_i + epsilon_j) // 2
            record = solver.solve(f"phase2Q_{iterations}", [references.p_max * q + p for p, q in zip(solver.instance.p, solver.instance.q)], (solver.instance.p, references.f1_nadir + epsilon_m))
            point, sign = references.point(record), references.rho_sign(references.point(record))
            trace.append({"iteration": iterations, "epsilon_i": epsilon_i, "epsilon_j": epsilon_j, "epsilon_m": epsilon_m, "solution": point.as_dict(), "rho_sign": sign, "endpoint_update": "return" if sign == 0 else ("negative_and_epsilon_i=epsilon_m" if sign < 0 else "epsilon_j=P_m")})
            if sign < 0:
                current, epsilon_i = point, epsilon_m
            elif sign > 0:
                epsilon_j = point.p
                if epsilon_j >= trace[-1]["epsilon_j"]:
                    trace[-1]["termination"] = "no_progress_after_prescribed_epsilon_j=P_m_update"
                    break
            else:
                current = point
                break
        return {"outcome": "KS_POINT", "point": current.as_dict(), "trace": trace, "iterations": iterations}