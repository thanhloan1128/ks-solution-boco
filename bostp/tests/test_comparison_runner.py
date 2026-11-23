import itertools

import networkx as nx
import pytest

from bostp.comparison_runner import minmax_point, nf_point
from bostp.instance import BOSTPInstance, Edge
from bostp.reference_points import ReferencePointCalculator


def small_instance():
    edges = [
        Edge(0, 1, 2, 9),
        Edge(0, 2, 4, 3),
        Edge(0, 3, 7, 5),
        Edge(1, 2, 3, 7),
        Edge(1, 3, 6, 2),
        Edge(2, 3, 5, 4),
    ]
    return BOSTPInstance("comparison_small", 4, 0.5, 1, 1, edges)


def exhaustive_points(instance):
    points = {}
    for selected in itertools.combinations(instance.edges, instance.n - 1):
        graph = nx.Graph()
        graph.add_nodes_from(instance.nodes)
        graph.add_edges_from((edge.u, edge.v) for edge in selected)
        if nx.is_tree(graph):
            point = instance.objective(list(selected))
            points.setdefault(point, selected)
    return points


def test_nf_uses_defining_weighted_sum_condition():
    instance = small_instance()
    reference = ReferencePointCalculator().calculate(instance)
    result = nf_point(instance, reference)
    point = result["point"]
    last = result["verification"][-1]
    assert last["weighted_optimum"] == last["twice_product"]
    assert point["f1"] * point["f2"] == last["a"] * last["b"]
    products = [item["a"] * item["b"] for item in result["verification"]]
    assert products == sorted(products, reverse=True)


def test_nf_verification_is_exhaustively_feasible():
    instance = small_instance()
    reference = ReferencePointCalculator().calculate(instance)
    result = nf_point(instance, reference)
    points = exhaustive_points(instance)
    assert (result["point"]["f1"], result["point"]["f2"]) in points


def test_minmax_matches_exhaustive_optimum_and_sum_tiebreak():
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        pytest.skip("gurobipy is unavailable")
    instance = small_instance()
    result = minmax_point(instance)
    points = exhaustive_points(instance)
    optimum = min(max(point) for point in points)
    tied = [point for point in points if max(point) == optimum]
    expected = min(tied, key=lambda point: (sum(point), point))
    assert result["optimal_z"] == optimum
    assert (result["point"]["f1"], result["point"]["f2"]) == expected
