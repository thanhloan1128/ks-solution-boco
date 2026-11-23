from __future__ import annotations

from fractions import Fraction

from .domain import ObjectivePoint, ReferencePoints, SolveRecord
from .gurobi_model import GurobiModel


class ReferencePointCalculator:
    def calculate(self, solver: GurobiModel) -> tuple[ReferencePoints, ObjectivePoint, ObjectivePoint]:
        m1 = solver.solve("reference_M1", solver.instance.p)
        m2 = solver.solve("reference_M2", solver.instance.q)
        lambda_1 = Fraction(1, m2.f2**2 + 1)
        lambda_2 = Fraction(m1.f1**2, m1.f1**2 + 1)
        nadir_1 = solver.solve("reference_nadir_lambda1", [m2.f2**2 * p + q for p, q in zip(solver.instance.p, solver.instance.q)])
        nadir_2 = solver.solve("reference_nadir_lambda2", [p + m1.f1**2 * q for p, q in zip(solver.instance.p, solver.instance.q)])
        references = ReferencePoints(m1.f1, m2.f2, nadir_2.f1, nadir_1.f2, m1.f1 - nadir_2.f1, m2.f2 - nadir_1.f2, lambda_1, lambda_2, m1, m2, nadir_1, nadir_2)
        if nadir_1.f1 != references.f1_utopia or nadir_2.f2 != references.f2_utopia:
            raise AssertionError("Exact-Nadir lemma coordinate validation failed")
        endpoint_p = references.point(nadir_1)
        endpoint_q = references.point(nadir_2)
        if endpoint_p != ObjectivePoint(references.p_max, 0, nadir_1.x) or endpoint_q != ObjectivePoint(0, references.q_max, nadir_2.x):
            raise AssertionError("Weighted-sum endpoint coordinate validation failed")
        return references, endpoint_p, endpoint_q