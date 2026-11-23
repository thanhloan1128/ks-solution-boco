import itertools
import random
from fractions import Fraction

import networkx as nx
import pytest

from bostp.instance import BOSTPInstance, Edge
from bostp.ks_solver import KSSolver
from bostp.phase_one import PhaseOneSearch
from bostp.phase_two import PhaseTwoSearch
from bostp.reference_points import ReferencePointCalculator
from bostp.spanning_tree_model import MSTSolver


def make_fixture(seed: int) -> BOSTPInstance:
    rng = random.Random(seed)
    edges = [Edge(u, v, rng.randint(1, 40), rng.randint(1, 40))
             for u, v in itertools.combinations(range(4), 2)]
    return BOSTPInstance(f"validation_{seed}", 4, 0.5, seed, seed, edges)


def objective_points(instance: BOSTPInstance) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    points = {}
    for selected in itertools.combinations(instance.edges, instance.n - 1):
        graph = nx.Graph()
        graph.add_nodes_from(instance.nodes)
        graph.add_edges_from((edge.u, edge.v) for edge in selected)
        if nx.is_tree(graph):
            key = instance.objective(list(selected))
            points.setdefault(key, tuple(sorted((min(edge.u, edge.v), max(edge.u, edge.v)) for edge in selected)))
    return points


def pareto_points(instance: BOSTPInstance) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    points = objective_points(instance)
    return {point: tree for point, tree in points.items()
            if not any(other != point and other[0] <= point[0] and other[1] <= point[1]
                       for other in points)}


def rho(point, reference) -> Fraction:
    return (Fraction(reference.f1_n - point[0], reference.p_max)
            - Fraction(reference.f2_n - point[1], reference.q_max))


def exact_reference(front):
    x1 = min(front, key=lambda point: (point[0], point[1]))
    x2 = min(front, key=lambda point: (point[1], point[0]))
    return x1, x2


def best_p(front, reference, epsilon):
    feasible = [point for point in front if point[1] <= reference.f2_n - epsilon]
    return min(feasible, key=lambda point: ((reference.q_max + 1) * point[0] + point[1], point))


def best_q(front, reference, epsilon):
    feasible = [point for point in front if point[0] <= reference.f1_n - epsilon]
    return min(feasible, key=lambda point: ((reference.p_max + 1) * point[1] + point[0], point))


def exact_phase_two(front, reference, positive, negative, orientation):
    incumbent = positive if orientation == "P" else negative
    low, high = ((reference.f2_n - positive[1], reference.f2_n - negative[1])
                 if orientation == "P" else
                 (reference.f1_n - negative[0], reference.f1_n - positive[0]))
    while high - low > 1:
        middle = (low + high) // 2
        candidate = best_p(front, reference, middle) if orientation == "P" else best_q(front, reference, middle)
        candidate_rho = rho(candidate, reference)
        if candidate_rho == 0:
            return candidate
        if orientation == "P":
            if candidate_rho > 0:
                incumbent, low = candidate, reference.f2_n - candidate[1]
            else:
                high = middle
        elif candidate_rho < 0:
            incumbent, low = candidate, reference.f1_n - candidate[0]
        else:
            high = middle
    return incumbent


def expected_ks(front, reference, x1, x2):
    positive, negative = (x1, x2) if rho(x1, reference) > 0 else (x2, x1)
    plus = exact_phase_two(front, reference, positive, negative, "P")
    minus = exact_phase_two(front, reference, positive, negative, "Q")
    plus_score = min(Fraction(reference.f1_n - plus[0], reference.p_max),
                     Fraction(reference.f2_n - plus[1], reference.q_max))
    minus_score = min(Fraction(reference.f1_n - minus[0], reference.p_max),
                      Fraction(reference.f2_n - minus[1], reference.q_max))
    if plus_score > minus_score:
        return {plus}, plus, minus
    if minus_score > plus_score:
        return {minus}, plus, minus
    return {plus, minus}, plus, minus


def point_from_solution(solution):
    return solution.f1, solution.f2


def assert_tree_solution(instance, solution):
    MSTSolver(instance).verify(solution)
    assert len(solution.edges) == instance.n - 1
    assert instance.objective(list(solution.edges)) == (solution.f1, solution.f2)


def require_gurobi():
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        pytest.skip("gurobipy is unavailable")


@pytest.mark.parametrize("seed", [2, 7, 21, 52, 258])
def test_actual_gurobi_solver_matches_exhaustive_oracle(seed):
    require_gurobi()
    instance = make_fixture(seed)
    front = pareto_points(instance)
    reference = ReferencePointCalculator().calculate(instance)
    x1_point, x2_point = exact_reference(front)
    assert (reference.x1.f1, reference.x1.f2) == x1_point
    assert (reference.x2.f1, reference.x2.f2) == x2_point
    assert (reference.x1.f1, reference.x1.f2) in front
    assert (reference.x2.f1, reference.x2.f2) in front

    phase_one = PhaseOneSearch(MSTSolver(instance), reference).run()
    if phase_one.outcome == "DIRECT_KS":
        assert point_from_solution(phase_one.points[0]) in front
        expected = {point_from_solution(phase_one.points[0])}
    else:
        assert phase_one.outcome == "C_SN"
        assert phase_one.signed_rho[0] > 0 > phase_one.signed_rho[1]
        assert ((1 - Fraction(phase_one.alpha_values[-1])) * phase_one.points[0].f1
                + Fraction(phase_one.alpha_values[-1]) * phase_one.points[0].f2) == (
                    (1 - Fraction(phase_one.alpha_values[-1])) * phase_one.points[1].f1
                    + Fraction(phase_one.alpha_values[-1]) * phase_one.points[1].f2
                )
        for point in phase_one.points:
            assert point_from_solution(point) in front
        positive, negative = sorted(phase_one.points,
                                   key=lambda point: rho(point_from_solution(point), reference),
                                   reverse=True)
        phase_two_p, phase_two_q = PhaseTwoSearch(instance, reference).run(phase_one.points)
        assert phase_two_p.status == "OPTIMAL"
        assert phase_two_q.status == "OPTIMAL"
        assert phase_two_p.optimality_gap <= PhaseTwoSearch.optimality_gap_tolerance
        assert phase_two_q.optimality_gap <= PhaseTwoSearch.optimality_gap_tolerance
        for result in (phase_two_p, phase_two_q):
            for record in result.trace:
                assert record["relative_objective_gap"] <= PhaseTwoSearch.optimality_gap_tolerance
                assert record["binary_variables"] == 3 * len(instance.edges)
                assert record["variables"] == 3 * len(instance.edges) + instance.n
                assert record["formulation"] == PhaseTwoSearch.formulation
                assert record["objective_gap_within_tolerance"]
        assert_tree_solution(instance, phase_two_p.solution)
        assert_tree_solution(instance, phase_two_q.solution)
        expected, expected_plus, expected_minus = expected_ks(front, reference,
                                                               point_from_solution(positive),
                                                               point_from_solution(negative))
        assert point_from_solution(phase_two_p.solution) == expected_plus
        assert point_from_solution(phase_two_q.solution) == expected_minus

    production = KSSolver().solve(instance)
    assert production["status"] == "COMPLETED"
    obtained = {(point["f1"], point["f2"]) for point in production["final_ks_points"]}
    assert obtained == expected
    for point in production["final_ks_points"]:
        assert point["absolute_rho"] == str(abs(Fraction(point["rho"])))


def test_required_structural_cases_and_interval_trace():
    require_gurobi()

    direct = make_fixture(258)
    direct_result = KSSolver().solve(direct)
    assert direct_result["status"] == "COMPLETED"
    assert direct_result["phase1"]["outcome"] == "DIRECT_KS"
    assert direct_result["phase2"]["P"]["outcome"] == "NOT_RUN_DIRECT_KS"
    assert direct_result["phase2"]["Q"]["outcome"] == "NOT_RUN_DIRECT_KS"
    assert len(direct_result["final_ks_points"]) == 1

    two_point = KSSolver().solve(make_fixture(5))
    assert two_point["status"] == "COMPLETED"
    assert len(two_point["final_ks_points"]) == 2
    assert len({(point["f1"], point["f2"]) for point in two_point["final_ks_points"]}) == 2

    q_jump = KSSolver().solve(make_fixture(7))
    q_trace = q_jump["phase2"]["Q"]["trace"]
    assert any(item["low_after"] > item["epsilon"] for item in q_trace if "low_after" in item)

    p_jump = KSSolver().solve(make_fixture(21))
    p_trace = p_jump["phase2"]["P"]["trace"]
    assert any(item["low_after"] > item["epsilon"] for item in p_trace if "low_after" in item)

    interval_one = make_fixture(52)
    reference = ReferencePointCalculator().calculate(interval_one)
    phase_one = PhaseOneSearch(MSTSolver(interval_one), reference).run()
    positive, negative = sorted(phase_one.points,
                               key=lambda point: rho(point_from_solution(point), reference),
                               reverse=True)
    assert reference.f1_n - positive.f1 - (reference.f1_n - negative.f1) == 1
    q_result = PhaseTwoSearch(interval_one, reference).run(phase_one.points)[1]
    assert q_result.status == "OPTIMAL"
    assert q_result.midpoint_iterations == 0


def test_finite_objective_set_oracle_is_independent_of_graph_solver():
    front = {(0, 10), (3, 7), (7, 3), (10, 0)}
    class Reference:
        f1_n, f2_n, p_max, q_max = 10, 10, 10, 10
    reference = Reference()
    assert expected_ks(front, reference, (0, 10), (10, 0))[0] == {(3, 7), (7, 3)}


def test_ks_score_is_the_minimum_normalized_improvement():
    class Reference:
        f1_n, f2_n, p_max, q_max = 100, 100, 100, 100

    point = type("Point", (), {"f1": 10, "f2": 80, "edges": ()})()
    selected = KSSolver._selection(point, Reference())
    assert selected["K"] == "1/5"


def test_static_mtz_rejects_disconnected_integer_solution():
    require_gurobi()
    edges = [Edge(u, v, index + 1, 3 - index)
             for group in ((0, 1, 2), (3, 4, 5))
             for index, (u, v) in enumerate(itertools.combinations(group, 2))]
    edges.append(Edge(2, 3, 100, 100))
    instance = BOSTPInstance("lazy_cut_fixture", 6, 0.5, 1, 1, edges)
    reference = ReferencePointCalculator().calculate(instance)
    search = PhaseTwoSearch(instance, reference)
    model, x, edges, arcs, order = search._build_model("P", 0)
    try:
        assert model.NumVars == len(edges) + 2 * len(edges) + instance.n
        assert len(arcs) == 2 * len(edges)
        assert len(order) == instance.n
        assert model.Params.MIPGap == PhaseTwoSearch.optimality_gap_tolerance
        assert model.Params.MIPGapAbs == 0
    finally:
        model.dispose()
    solution = search._solve_subproblem("P", 0)
    assert_tree_solution(instance, solution)
    assert (2, 3) in solution.edges
    expected = min(objective_points(instance),
                   key=lambda point: (reference.q_max + 1) * point[0] + point[1])
    assert (solution.f1, solution.f2) == expected
    assert search._trace[-1]["lazy_cuts_added"] == 0


@pytest.mark.parametrize("seed", [2, 7, 21])
@pytest.mark.parametrize("orientation", ["P", "Q"])
def test_lazy_subproblem_matches_exhaustive_weighted_objective(seed, orientation):
    require_gurobi()
    instance = make_fixture(seed)
    reference = ReferencePointCalculator().calculate(instance)
    front = pareto_points(instance)
    search = PhaseTwoSearch(instance, reference)
    maximum = reference.q_max if orientation == "P" else reference.p_max
    for epsilon in (0, maximum // 2, maximum):
        expected = (best_p(front, reference, epsilon) if orientation == "P"
                    else best_q(front, reference, epsilon))
        solution = search._solve_subproblem(orientation, epsilon)
        assert (solution.f1, solution.f2) == expected
        assert_tree_solution(instance, solution)


def test_static_mtz_formulation_has_no_lazy_callback_dependencies():
    import inspect

    source = inspect.getsource(PhaseTwoSearch._solve_subproblem)
    assert ".optimize()" in source
    assert "cbLazy" not in source


@pytest.mark.parametrize(
    ("objective", "bound", "accepted"),
    [
        (4168164, 4168163.9999999995, True),
        (1_000_000, 999_999.5, True),
        (1_000_000, 999_990.5, True),
        (1_000_000, 999_989.0, False),
        (100, 99.9991, True),
        (100, 99.998, False),
    ],
)
def test_relative_objective_gap_matches_configured_tolerance(objective, bound, accepted):
    relative_gap = max(0.0, objective - bound) / max(abs(objective), 1.0)
    assert (relative_gap <= PhaseTwoSearch.optimality_gap_tolerance + 1e-12) is accepted
