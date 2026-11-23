from __future__ import annotations

import random
import secrets
from pathlib import Path
import networkx as nx
from .config import ExperimentConfig
from .instance import BOSTPInstance, Edge


class InstanceGenerator:
    def __init__(self, config: ExperimentConfig, master_seed: int | None = None):
        self.config = config
        self.master_seed = secrets.randbits(63) if master_seed is None else master_seed

    def generate_pair(self, n: int, p: float, count: int | None = None) -> list[BOSTPInstance]:
        count = self.config.instances_per_pair if count is None else count
        seed_source = random.Random(self.master_seed + n * 1000003 + round(p * 1000))
        output: list[BOSTPInstance] = []
        rejected: list[int] = []
        while len(output) < count:
            candidate_seed = seed_source.getrandbits(63)
            graph = nx.gnp_random_graph(n, p, seed=candidate_seed)
            if not nx.is_connected(graph):
                rejected.append(candidate_seed)
                continue
            randomizer = random.Random(candidate_seed)
            edges = [Edge(min(u, v), max(u, v), randomizer.randint(*self.config.cost_range),
                          randomizer.randint(*self.config.time_range)) for u, v in graph.edges()]
            edges.sort(key=lambda edge: (edge.u, edge.v))
            index = len(output) + 1
            instance_id = f"bostp_n{n:04d}_p{round(p * 10):02d}_i{index:03d}"
            output.append(BOSTPInstance(instance_id, n, p, candidate_seed, self.master_seed,
                                        edges, len(rejected), rejected.copy()))
        return output

    def generate_all(self, directory: Path) -> list[BOSTPInstance]:
        instances: list[BOSTPInstance] = []
        for n in self.config.sizes:
            for p in self.config.probabilities:
                pair = self.generate_pair(n, p)
                for instance in pair:
                    instance.save(directory / f"{instance.instance_id}.json")
                instances.extend(pair)
        return instances