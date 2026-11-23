from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import networkx as nx


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    cost: int
    time: int


@dataclass
class BOSTPInstance:
    instance_id: str
    n: int
    p: float
    seed: int
    master_seed: int
    edges: list[Edge]
    rejected_candidates: int = 0
    rejected_seeds: list[int] | None = None

    @property
    def nodes(self) -> list[int]:
        return list(range(self.n))

    def graph(self, weight: str | None = None) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from((edge.u, edge.v, {"cost": edge.cost, "time": edge.time}) for edge in self.edges)
        if weight:
            for u, v, data in graph.edges(data=True):
                data["weight"] = data[weight]
        return graph

    def objective(self, selected_edges: list[tuple[int, int]] | list[Edge]) -> tuple[int, int]:
        wanted = {(min(edge[0], edge[1]) if isinstance(edge, tuple) else min(edge.u, edge.v),
                   max(edge[0], edge[1]) if isinstance(edge, tuple) else max(edge.u, edge.v))
                  for edge in selected_edges}
        values = {(min(edge.u, edge.v), max(edge.u, edge.v)): edge for edge in self.edges}
        return sum(values[key].cost for key in wanted), sum(values[key].time for key in wanted)

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id, "n": self.n, "p": self.p,
            "instance_seed": self.seed, "master_seed": self.master_seed,
            "nodes": self.nodes, "edges": [edge.__dict__ for edge in self.edges],
            "rejected_candidates": self.rejected_candidates,
            "rejected_seeds": self.rejected_seeds or [],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "BOSTPInstance":
        data = json.loads(path.read_text())
        return cls(data["instance_id"], data["n"], data["p"], data["instance_seed"],
                   data["master_seed"], [Edge(**edge) for edge in data["edges"]],
                   data.get("rejected_candidates", 0), data.get("rejected_seeds", []))