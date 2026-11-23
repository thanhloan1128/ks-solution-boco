from __future__ import annotations

from dataclasses import asdict, dataclass
from .instance import BOSTPInstance
from .spanning_tree_model import MSTSolver, TreeSolution


@dataclass(frozen=True)
class ReferencePoints:
    x1: TreeSolution
    x2: TreeSolution
    f1_u: int
    f2_u: int
    f1_n: int
    f2_n: int
    m1: int
    m2: int
    w_x1: tuple[int, int]
    w_x2: tuple[int, int]

    @property
    def p_max(self) -> int: return self.f1_n - self.f1_u
    @property
    def q_max(self) -> int: return self.f2_n - self.f2_u

    def as_dict(self) -> dict[str, object]:
        return {"x1": asdict(self.x1), "x2": asdict(self.x2), "f1_u": self.f1_u,
                "f2_u": self.f2_u, "f1_n": self.f1_n, "f2_n": self.f2_n,
                "p_max": self.p_max, "q_max": self.q_max, "m1": self.m1, "m2": self.m2,
                "w_x1": self.w_x1, "w_x2": self.w_x2}


class ReferencePointCalculator:
    def calculate(self, instance: BOSTPInstance) -> ReferencePoints:
        m1 = sum(edge.cost for edge in instance.edges)
        m2 = sum(edge.time for edge in instance.edges)
        solver = MSTSolver(instance)
        x1 = solver.solve(m2 * m2, 1)
        x2 = solver.solve(1, m1 * m1)
        solver.verify(x1); solver.verify(x2)
        if x1.f1 > x2.f1 or x2.f2 > x1.f2:
            raise ValueError("lexicographic scalarization priority check failed")
        return ReferencePoints(x1, x2, x1.f1, x2.f2, x2.f1, x1.f2, m1, m2,
                               (m2 * m2, 1), (1, m1 * m1))