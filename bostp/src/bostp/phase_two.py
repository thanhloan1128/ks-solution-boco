from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from time import monotonic, perf_counter
from .reference_points import ReferencePoints
from .spanning_tree_model import MSTSolver, TreeSolution


@dataclass
class PhaseTwoResult:
    orientation: str
    outcome: str
    solution: TreeSolution | None
    signed_rho: object | None
    midpoint_iterations: int = 0
    gurobi_calls: int = 0
    status: str = "NOT_RUN"
    message: str = ""
    runtime_seconds: float = 0.0
    optimality_gap: float | None = None
    failure_type: str | None = None
    trace: list[dict[str, object]] | None = None


class PhaseTwoTimeout(RuntimeError):
    def __init__(self, solution: TreeSolution | None, message: str):
        super().__init__(message)
        self.solution = solution


class PhaseTwoSearch:
    """Epsilon searches using a static MTZ spanning-tree formulation."""

    formulation = "mtz-static-v1"
    optimality_gap_tolerance = 1e-5

    def __init__(self, instance, reference: ReferencePoints, time_limit=None, threads=1,
                 deadline: float | None = None, optimality_gap_tolerance: float | None = None):
        self.instance, self.reference = instance, reference
        self.time_limit, self.threads = time_limit, threads
        self.deadline = deadline
        self.optimality_gap_tolerance = (
            self.optimality_gap_tolerance if optimality_gap_tolerance is None
            else optimality_gap_tolerance
        )
        if not 0 <= self.optimality_gap_tolerance < 1:
            raise ValueError("relative MIP gap tolerance must be in [0, 1)")
        self._trace: list[dict[str, object]] = []
        self._last_incumbent: TreeSolution | None = None

    def _shifted(self, solution: TreeSolution) -> tuple[int, int]:
        return self.reference.f1_n - solution.f1, self.reference.f2_n - solution.f2

    def _rho(self, solution: TreeSolution) -> Fraction:
        shifted_p, shifted_q = self._shifted(solution)
        return Fraction(shifted_p, self.reference.p_max) - Fraction(shifted_q, self.reference.q_max)

    def _build_model(self, orientation: str, epsilon: int):
        if orientation not in {"P", "Q"}:
            raise ValueError(f"unknown Phase II orientation: {orientation}")
        maximum = self.reference.q_max if orientation == "P" else self.reference.p_max
        if not 0 <= epsilon <= maximum:
            raise ValueError(f"invalid {orientation}-oriented epsilon {epsilon}; expected 0..{maximum}")
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise RuntimeError("gurobipy is required for Phase II") from exc
        model = gp.Model(f"bostp_{orientation}")
        model.Params.OutputFlag = 0; model.Params.Threads = self.threads
        model.Params.MIPGap = self.optimality_gap_tolerance
        model.Params.MIPGapAbs = 0
        model.Params.IntFeasTol = 1e-9
        model.Params.FeasibilityTol = 1e-9
        if self.deadline is not None:
            remaining = self.deadline - monotonic()
            if remaining <= 0:
                model.dispose()
                raise RuntimeError("TIME_LIMIT: shared instance budget exhausted")
            model.Params.TimeLimit = remaining
        elif self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        edges = self.instance.edges
        nodes = self.instance.nodes
        root = nodes[0]
        x = {i: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}") for i in range(len(edges))}
        arcs = {
            (u, v, i): model.addVar(vtype=gp.GRB.BINARY, name=f"y_{u}_{v}_{i}")
            for i, edge in enumerate(edges)
            for u, v in ((edge.u, edge.v), (edge.v, edge.u))
        }
        order = {
            node: model.addVar(
                lb=0 if node == root else 1,
                ub=0 if node == root else self.instance.n - 1,
                vtype=gp.GRB.CONTINUOUS,
                name=f"u_{node}",
            )
            for node in nodes
        }
        model.addConstr(gp.quicksum(x.values()) == self.instance.n - 1)
        for i, edge in enumerate(edges):
            model.addConstr(arcs[edge.u, edge.v, i] + arcs[edge.v, edge.u, i] == x[i])
        model.addConstr(gp.quicksum(
            arcs[u, root, i]
            for i, edge in enumerate(edges)
            for u, v in ((edge.u, edge.v), (edge.v, edge.u))
            if v == root
        ) == 0)
        for node in nodes:
            if node == root:
                continue
            incoming = gp.quicksum(
                arcs[u, node, i]
                for i, edge in enumerate(edges)
                for u, v in ((edge.u, edge.v), (edge.v, edge.u))
                if v == node
            )
            model.addConstr(incoming == 1)
        for i, edge in enumerate(edges):
            for u, v in ((edge.u, edge.v), (edge.v, edge.u)):
                if v != root:
                    model.addConstr(
                        order[v] >= order[u] + 1 - self.instance.n * (1 - arcs[u, v, i])
                    )
        if orientation == "P":
            # Q >= epsilon_Q is equivalent to f2 <= f2_N - epsilon_Q.
            model.addConstr(gp.quicksum(edges[i].time * x[i] for i in x) <= self.reference.f2_n - epsilon)
            model.setObjective(gp.quicksum(((self.reference.q_max + 1) * edges[i].cost + edges[i].time) * x[i] for i in x), gp.GRB.MINIMIZE)
        else:
            # P >= epsilon_P is equivalent to f1 <= f1_N - epsilon_P.
            model.addConstr(gp.quicksum(edges[i].cost * x[i] for i in x) <= self.reference.f1_n - epsilon)
            model.setObjective(gp.quicksum(((self.reference.p_max + 1) * edges[i].time + edges[i].cost) * x[i] for i in x), gp.GRB.MINIMIZE)
        model.update()
        return model, x, edges, arcs, order

    def _solve_subproblem(self, orientation: str, epsilon: int) -> TreeSolution:
        started = perf_counter()
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise RuntimeError("gurobipy is required for Phase II") from exc
        model, x, edges, arcs, _ = self._build_model(orientation, epsilon)
        try:
            if self.deadline is not None:
                remaining = self.deadline - monotonic()
                if remaining <= 0:
                    raise RuntimeError("TIME_LIMIT: shared instance budget exhausted during model construction")
                model.Params.TimeLimit = remaining
            model.optimize()
            status = model.Status
            record = {
                "orientation": orientation, "epsilon": epsilon,
                "formulation": self.formulation, "gurobi_status": status,
                "status": "OPTIMAL" if status == gp.GRB.OPTIMAL else "UNPROVEN",
                "objective_value": model.ObjVal if model.SolCount else None,
                "objective_bound": model.ObjBound,
                "optimality_gap": model.MIPGap if model.SolCount else None,
                "optimality_gap_tolerance": self.optimality_gap_tolerance,
                "variables": model.NumVars, "binary_variables": len(x) + len(arcs),
                "constraints": model.NumConstrs, "lazy_cuts_added": 0,
                "runtime_seconds": perf_counter() - started,
            }
            self._trace.append(record)
            if status != gp.GRB.OPTIMAL:
                if status == gp.GRB.TIME_LIMIT:
                    incumbent = None
                    if model.SolCount:
                        selected = tuple(sorted(
                            (min(edges[i].u, edges[i].v), max(edges[i].u, edges[i].v))
                            for i in x if x[i].X > 0.5
                        ))
                        f1, f2 = self.instance.objective(list(selected))
                        incumbent = TreeSolution(selected, f1, f2)
                        MSTSolver(self.instance).verify(incumbent)
                        if orientation == "P" and f2 > self.reference.f2_n - epsilon:
                            raise ValueError("P-oriented incumbent violates its Q epsilon constraint")
                        if orientation == "Q" and f1 > self.reference.f1_n - epsilon:
                            raise ValueError("Q-oriented incumbent violates its P epsilon constraint")
                        self._last_incumbent = incumbent
                    raise PhaseTwoTimeout(incumbent, f"TIME_LIMIT: Gurobi status {status}")
                raise RuntimeError(f"Gurobi status {status}")
            selected = tuple(sorted(
                (min(edges[i].u, edges[i].v), max(edges[i].u, edges[i].v))
                for i in x if x[i].X > 0.5
            ))
            f1, f2 = self.instance.objective(list(selected))
            solution = TreeSolution(selected, f1, f2)
            MSTSolver(self.instance).verify(solution)
            shifted_p, shifted_q = self._shifted(solution)
            if orientation == "P" and f2 > self.reference.f2_n - epsilon:
                raise ValueError("P-oriented solution violates its Q epsilon constraint")
            if orientation == "Q" and f1 > self.reference.f1_n - epsilon:
                raise ValueError("Q-oriented solution violates its P epsilon constraint")
            integer_objective = (
                (self.reference.q_max + 1) * f1 + f2 if orientation == "P"
                else (self.reference.p_max + 1) * f2 + f1
            )
            objective_value_error = abs(model.ObjVal - integer_objective)
            if objective_value_error > 1e-6:
                raise RuntimeError(
                    f"Gurobi objective {model.ObjVal} does not match extracted "
                    f"integer objective {integer_objective}"
                )
            relative_objective_gap = (
                max(0.0, integer_objective - model.ObjBound)
                / max(abs(integer_objective), 1.0)
            )
            if relative_objective_gap > self.optimality_gap_tolerance + 1e-12:
                raise RuntimeError(
                    f"relative objective gap {relative_objective_gap} exceeds "
                    f"tolerance {self.optimality_gap_tolerance}"
                )
            record.update({"f1": f1, "f2": f2, "P": shifted_p, "Q": shifted_q,
                           "rho": str(self._rho(solution)),
                           "integer_objective": integer_objective,
                           "relative_objective_gap": relative_objective_gap,
                           "objective_gap_within_tolerance": True})
            return solution
        finally:
            model.dispose()

    def _proven_gap(self):
        return max((item["optimality_gap"] for item in self._trace
                    if item.get("status") == "OPTIMAL"), default=0.0)

    def _search(self, orientation: str, positive: TreeSolution, negative: TreeSolution) -> PhaseTwoResult:
        ref = self.reference; incumbent = positive if orientation == "P" else negative
        low, high = ((ref.f2_n - positive.f2, ref.f2_n - negative.f2)
                     if orientation == "P" else
                     (ref.f1_n - negative.f1, ref.f1_n - positive.f1))
        iterations = calls = 0
        self._trace = []
        self._last_incumbent = None
        started = perf_counter()
        try:
            while high - low > 1:
                middle = (low + high) // 2
                previous_low, previous_high = low, high
                candidate = self._solve_subproblem(orientation, middle); calls += 1; iterations += 1
                rho = self._rho(candidate)
                if rho == 0:
                    return PhaseTwoResult(orientation, "EXACT_KS", candidate, rho, iterations, calls, "OPTIMAL", runtime_seconds=perf_counter() - started, optimality_gap=self._proven_gap(), trace=self._trace)
                if orientation == "P":
                    if rho > 0:
                        incumbent, low = candidate, ref.f2_n - candidate.f2
                    else:
                        high = middle
                    if self._trace:
                        self._trace[-1].update({"low_before": previous_low, "high_before": previous_high,
                                                "low_after": low, "high_after": high})
                elif rho < 0:
                    incumbent, low = candidate, ref.f1_n - candidate.f1
                else:
                    high = middle
                if self._trace:
                    self._trace[-1].update({"low_before": previous_low, "high_before": previous_high,
                                            "low_after": low, "high_after": high})
            signed = self._rho(incumbent)
            return PhaseTwoResult(orientation, "INCUMBENT", incumbent, signed, iterations, calls, "OPTIMAL", runtime_seconds=perf_counter() - started, optimality_gap=self._proven_gap(), trace=self._trace)
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            failure_type = ("LICENSE" if "license" in lowered else
                            "TIME_LIMIT" if "time_limit" in lowered else "SOLVER_ERROR")
            result_status = failure_type if failure_type == "TIME_LIMIT" else "FAILED"
            self._trace.append({"orientation": orientation, "epsilon": middle if 'middle' in locals() else None,
                                "status": "FAILED", "failure_type": failure_type, "message": message, "runtime_seconds": perf_counter() - started})
            incumbent = exc.solution if isinstance(exc, PhaseTwoTimeout) else self._last_incumbent
            return PhaseTwoResult(orientation, "TIME_LIMIT_INCUMBENT" if incumbent else "FAILED",
                                  incumbent, self._rho(incumbent) if incumbent else None, iterations, calls,
                                  result_status, message, perf_counter() - started,
                                  optimality_gap=None, failure_type=failure_type,
                                  trace=self._trace)

    def run(self, points: list[TreeSolution]) -> tuple[PhaseTwoResult, PhaseTwoResult]:
        if len(points) != 2:
                return (PhaseTwoResult("P", "NOT_RUN_DIRECT_KS", points[0] if points else None, Fraction(0), status="NOT_RUN", trace=[]),
                    PhaseTwoResult("Q", "NOT_RUN_DIRECT_KS", points[0] if points else None, Fraction(0), status="NOT_RUN", trace=[]))
        ref = self.reference
        def rho(point): return (ref.f1_n - point.f1) * ref.q_max - (ref.f2_n - point.f2) * ref.p_max
        positive, negative = sorted(points, key=rho, reverse=True)
        return self._search("P", positive, negative), self._search("Q", positive, negative)