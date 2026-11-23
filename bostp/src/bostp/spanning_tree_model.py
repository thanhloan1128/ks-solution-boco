from __future__ import annotations

from dataclasses import dataclass
import networkx as nx
from .instance import BOSTPInstance


@dataclass(frozen=True)
class TreeSolution:
    edges: tuple[tuple[int, int], ...]
    f1: int
    f2: int

    @property
    def key(self) -> tuple[tuple[int, int], ...]:
        return self.edges


class MSTSolver:
    """Deterministic Kruskal solver for unconstrained integer-weight MSTs."""

    def __init__(self, instance: BOSTPInstance):
        self.instance = instance

    def solve(self, cost_weight: int, time_weight: int) -> TreeSolution:
        graph = self.instance.graph()
        for u, v, data in graph.edges(data=True):
            data["weight"] = cost_weight * data["cost"] + time_weight * data["time"]
        tree = nx.minimum_spanning_tree(graph, algorithm="kruskal", weight="weight")
        edges = tuple(sorted((min(u, v), max(u, v)) for u, v in tree.edges()))
        f1, f2 = self.instance.objective(list(edges))
        return TreeSolution(edges, f1, f2)

    def verify(self, solution: TreeSolution) -> None:
        graph = nx.Graph()
        graph.add_nodes_from(self.instance.nodes)
        graph.add_edges_from(solution.edges)
        if len(solution.edges) != self.instance.n - 1 or not nx.is_tree(graph):
            raise ValueError("solution is not a spanning tree")
        if self.instance.objective(list(solution.edges)) != (solution.f1, solution.f2):
            raise ValueError("tree objective values do not match edge data")