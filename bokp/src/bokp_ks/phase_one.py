from __future__ import annotations

from fractions import Fraction
from typing import Any

from .domain import ObjectivePoint, ReferencePoints
from .gurobi_model import GurobiModel


class PhaseOneSearch:
    def run(self, solver: GurobiModel, references: ReferencePoints, endpoint_p: ObjectivePoint, endpoint_q: ObjectivePoint) -> dict[str, Any]:
        alpha_i, alpha_j = Fraction(0), Fraction(1)
        point_i, point_j = endpoint_p, endpoint_q
        gamma = Fraction(4, (references.p_max + references.q_max) ** 2)
        trace, iterations = [], 0
        while alpha_j - alpha_i >= gamma:
            iterations += 1
            alpha_m = (alpha_i + alpha_j) / 2
            record = self._solve_alpha(solver, alpha_m, iterations)
            point = references.point(record)
            sign = references.rho_sign(point)
            trace.append({"kind": "iteration", "iteration": iterations, "alpha": alpha_m, "solution": point.as_dict(), "rho_sign": sign, "interval_before": [alpha_i, alpha_j], "interval_update": "return" if sign == 0 else ("lower_to_alpha_m" if sign > 0 else "upper_to_alpha_m")})
            if sign == 0:
                return {"outcome": "KS_POINT", "point": point.as_dict(), "trace": trace, "iterations": iterations, "final_alpha_s_included": False}
            if sign > 0:
                alpha_i, point_i = alpha_m, point
            else:
                alpha_j, point_j = alpha_m, point
        denominator = point_i.p - point_j.p + point_j.q - point_i.q
        if denominator == 0:
            raise AssertionError("Phase I alpha_s denominator is zero")
        alpha_s = Fraction(point_i.p - point_j.p, denominator)
        record = self._solve_alpha(solver, alpha_s, "alpha_s")
        point_s = references.point(record)
        sign = references.rho_sign(point_s)
        trace.append({"kind": "alpha_s", "alpha_s": alpha_s, "solution": point_s.as_dict(), "rho_sign": sign, "interval_before": [alpha_i, alpha_j]})
        if sign == 0:
            return {"outcome": "KS_POINT", "point": point_s.as_dict(), "trace": trace, "iterations": iterations, "final_alpha_s_included": True}
        positive, negative = (point_s, point_j) if sign > 0 else (point_i, point_s)
        if not (positive.q < negative.q and negative.p < positive.p):
            raise AssertionError("Phase I C_SN bounds are not ordered")
        return {"outcome": "C_SN", "positive": positive.as_dict(), "negative": negative.as_dict(), "trace": trace, "iterations": iterations, "final_alpha_s_included": True}

    @staticmethod
    def _solve_alpha(solver: GurobiModel, alpha: Fraction, label: int | str) -> Any:
        coefficients = [(alpha.denominator - alpha.numerator) * p + alpha.numerator * q for p, q in zip(solver.instance.p, solver.instance.q)]
        return solver.solve(f"phase1_{label}", coefficients)