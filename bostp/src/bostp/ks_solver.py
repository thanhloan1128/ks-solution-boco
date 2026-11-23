from __future__ import annotations

from fractions import Fraction
from time import monotonic, perf_counter
from .instance import BOSTPInstance
from .reference_points import ReferencePointCalculator
from .spanning_tree_model import MSTSolver
from .phase_one import PhaseOneSearch
from .phase_two import PhaseTwoSearch


class KSSolver:
    def __init__(self, phase2_time_limit=None, gurobi_threads=1,
                 phase2_gap_tolerance=PhaseTwoSearch.optimality_gap_tolerance):
        self.phase2_time_limit, self.gurobi_threads = phase2_time_limit, gurobi_threads
        self.phase2_gap_tolerance = phase2_gap_tolerance

    @staticmethod
    def _selection(point, reference):
        shifted_p = reference.f1_n - point.f1
        shifted_q = reference.f2_n - point.f2
        k_p = Fraction(shifted_p, reference.p_max)
        k_q = Fraction(shifted_q, reference.q_max)
        return {
            "f1": point.f1, "f2": point.f2, "edges": point.edges,
            "P": shifted_p, "Q": shifted_q,
            "rho": str(k_p - k_q),
            "absolute_rho": str(abs(k_p - k_q)),
            "K": str(min(k_p, k_q)),
        }

    def _select_final_points(self, phase1, phase2_p, phase2_q, reference):
        if phase1.outcome == "DIRECT_KS":
            point = phase1.points[0]
            return [self._selection(point, reference)], {
                "candidates": [], "K_minus": None, "K_plus": None,
                "selected": "DIRECT_PHASE_I", "reason": "EXACT_ZERO_RHO_IN_PHASE_I",
            }
        candidates = [result.solution for result in (phase2_p, phase2_q) if result.solution is not None]
        if len(candidates) < 2:
            return [self._selection(point, reference) for point in candidates], {
                "candidates": [self._selection(point, reference) for point in candidates],
                "K_minus": None, "K_plus": None, "selected": "NONE",
                "reason": "INSUFFICIENT_PROVEN_PHASE_II_CANDIDATES",
            }
        signed_rho = lambda point: (reference.f1_n - point.f1) * reference.q_max - (reference.f2_n - point.f2) * reference.p_max
        positive, negative = sorted(candidates, key=signed_rho, reverse=True)
        positive_data, negative_data = self._selection(positive, reference), self._selection(negative, reference)
        q_plus = Fraction(positive_data["Q"], reference.q_max)
        p_minus = Fraction(negative_data["P"], reference.p_max)
        if q_plus > p_minus:
            selected = [positive_data]; selected_labels = ["plus"]; reason = "Q_PLUS_GREATER_THAN_P_MINUS"
        elif q_plus < p_minus:
            selected = [negative_data]; selected_labels = ["minus"]; reason = "P_MINUS_GREATER_THAN_Q_PLUS"
        else:
            selected = ([positive_data] if (positive_data["f1"], positive_data["f2"]) == (negative_data["f1"], negative_data["f2"])
                        else [positive_data, negative_data])
            selected_labels = ["minus"] if len(selected) == 1 else ["minus", "plus"]
            reason = "Q_PLUS_EQUALS_P_MINUS"
        if len(selected) > 1:
            selected_labels = ["plus", "minus"]
        minus_record = {"label": "minus", **negative_data}
        plus_record = {"label": "plus", **positive_data}
        return selected, {"candidates": [minus_record, plus_record],
                          "K_minus": str(p_minus), "K_plus": str(q_plus),
                          "selected": selected_labels,
                          "reason": reason}

    def solve(self, instance: BOSTPInstance, deadline: float | None = None) -> dict[str, object]:
        reference_start = perf_counter(); reference = ReferencePointCalculator().calculate(instance)
        reference_seconds = perf_counter() - reference_start
        if reference.p_max == 0 or reference.q_max == 0:
            return {"instance_id": instance.instance_id, "phase2_formulation": PhaseTwoSearch.formulation, "status": "DEGENERATE_OBJECTIVE_RANGE", "reference": reference.as_dict(), "reference_seconds": reference_seconds}
        phase1_start = perf_counter(); phase1 = PhaseOneSearch(MSTSolver(instance), reference).run()
        phase1_seconds = perf_counter() - phase1_start
        phase2_limit = self.phase2_time_limit
        if deadline is not None:
            remaining = max(0.0, deadline - monotonic())
            phase2_limit = remaining if phase2_limit is None else min(phase2_limit, remaining)
        phase2_start = perf_counter()
        phase2_p, phase2_q = PhaseTwoSearch(
            instance, reference, phase2_limit, self.gurobi_threads, deadline,
            self.phase2_gap_tolerance
        ).run(phase1.points)
        phase2_seconds = perf_counter() - phase2_start
        def point_data(point):
            return self._selection(point, reference)
        final_points, ks_selection = self._select_final_points(phase1, phase2_p, phase2_q, reference)
        phase2_data = {}
        for label, result in (("P", phase2_p), ("Q", phase2_q)):
            phase2_data[label] = {"orientation": result.orientation, "outcome": result.outcome,
                                  "solution": point_data(result.solution) if result.solution else None,
                                  "signed_rho": result.signed_rho, "midpoint_iterations": result.midpoint_iterations,
                                  "gurobi_calls": result.gurobi_calls, "status": result.status,
                                  "message": result.message, "runtime_seconds": result.runtime_seconds,
                                  "optimality_gap": result.optimality_gap,
                                  "failure_type": result.failure_type,
                                  "trace": result.trace or []}
        direct = phase1.outcome == "DIRECT_KS"
        overall_status = "COMPLETED" if direct or (phase1.outcome == "C_SN" and phase2_p.status == "OPTIMAL" and phase2_q.status == "OPTIMAL") else "INCOMPLETE"
        return {"instance_id": instance.instance_id, "status": overall_status,
                "phase2_formulation": PhaseTwoSearch.formulation,
                "phase2_gap_tolerance": self.phase2_gap_tolerance,
                "reference": reference.as_dict(), "reference_seconds": reference_seconds,
                "phase1": {"outcome": phase1.outcome, "points": [point_data(p) for p in phase1.points], "signed_rho": [str(r) for r in phase1.signed_rho], "absolute_rho": [str(abs(r)) for r in phase1.signed_rho], "midpoint_iterations": phase1.midpoint_iterations, "final_solves": phase1.final_solves, "mst_calls": phase1.mst_calls},
                "phase2": phase2_data,
                "ks_selection": ks_selection,
                "final_ks_points": final_points, "phase1_seconds": phase1_seconds,
                "phase2_P_seconds": phase2_p.runtime_seconds, "phase2_Q_seconds": phase2_q.runtime_seconds,
                "total_seconds": reference_seconds + phase1_seconds + phase2_seconds}