from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from .reference_points import ReferencePoints
from .spanning_tree_model import MSTSolver, TreeSolution


@dataclass
class PhaseOneResult:
    outcome: str
    points: list[TreeSolution]
    signed_rho: list[Fraction]
    midpoint_iterations: int
    final_solves: int
    mst_calls: int
    alpha_values: list[str] = field(default_factory=list)


class PhaseOneSearch:
    def __init__(self, solver: MSTSolver, reference: ReferencePoints):
        self.solver, self.reference = solver, reference
        self.calls = 0

    def _solve(self, alpha: Fraction) -> TreeSolution:
        self.calls += 1
        return self.solver.solve(alpha.denominator - alpha.numerator, alpha.numerator)

    def rho(self, solution: TreeSolution) -> int:
        ref = self.reference
        return ((ref.f1_n - solution.f1) * ref.q_max
                - (ref.f2_n - solution.f2) * ref.p_max)

    def run(self) -> PhaseOneResult:
        ref = self.reference
        if ref.p_max == 0 or ref.q_max == 0:
            return PhaseOneResult("DEGENERATE_OBJECTIVE_RANGE", [], [], 0, 0, 0)
        left, right = ref.x1, ref.x2
        alpha_i, alpha_j = Fraction(0), Fraction(1)
        gamma = Fraction(4, (ref.p_max + ref.q_max) ** 2)
        iterations, alphas = 0, []
        def signed(solution: TreeSolution) -> Fraction:
            return Fraction(ref.f1_n - solution.f1, ref.p_max) - Fraction(ref.f2_n - solution.f2, ref.q_max)
        while alpha_j - alpha_i >= gamma:
            alpha_m = (alpha_i + alpha_j) / 2
            middle = self._solve(alpha_m); iterations += 1; alphas.append(str(alpha_m))
            value = signed(middle)
            if value == 0:
                return PhaseOneResult("DIRECT_KS", [middle], [value], iterations, 0, self.calls, alphas)
            if value > 0:
                alpha_i, left = alpha_m, middle
            else:
                alpha_j, right = alpha_m, middle
        denominator = left.f1 - right.f1 + right.f2 - left.f2
        if denominator == 0:
            return PhaseOneResult("FAILED", [], [], iterations, 0, self.calls, alphas)
        alpha_s = Fraction(left.f1 - right.f1, denominator)
        supported = self._solve(alpha_s); value = signed(supported)
        scalar = lambda point: (1 - alpha_s) * point.f1 + alpha_s * point.f2
        if value == 0:
            return PhaseOneResult("DIRECT_KS", [supported], [value], iterations, 1, self.calls, alphas + [str(alpha_s)])
        pair = [supported, right] if value > 0 else [left, supported]
        signed_pair = [signed(point) for point in pair]
        positive, negative = (pair if signed_pair[0] > 0 else pair[::-1])
        objective_pairs = {(point.f1, point.f2) for point in pair}
        nondominated = not (
            (positive.f1 <= negative.f1 and positive.f2 <= negative.f2)
            or (negative.f1 <= positive.f1 and negative.f2 <= positive.f2)
        )
        if (len(objective_pairs) != 2
                or not (signed(positive) > 0 and signed(negative) < 0)
                or not nondominated
                or any(scalar(point) != scalar(supported) for point in pair)):
            return PhaseOneResult("FAILED", [], [], iterations, 1, self.calls, alphas + [str(alpha_s)])
        ordered_pair = [positive, negative]
        return PhaseOneResult("C_SN", ordered_pair, [signed(positive), signed(negative)],
                              iterations, 1, self.calls, alphas + [str(alpha_s)])