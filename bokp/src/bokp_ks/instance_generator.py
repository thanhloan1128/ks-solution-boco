from __future__ import annotations

import math
import random
import sys
from typing import Any

from .domain import BOKPInstance


class InstanceGenerator:
    def __init__(self, base_seed: int = 20261017):
        self.base_seed = base_seed

    def generate(self, n: int, instance_index: int = 1) -> BOKPInstance:
        if instance_index < 1:
            raise ValueError("instance_index must be positive")
        seed = self.base_seed + n * 1009 + instance_index - 1
        rng = random.Random(seed)
        weights, p_values, q_values = [], [], []
        for _ in range(n):
            weights.append(rng.randint(1, 20))
            p_values.append(rng.randint(1, 30))
            q_values.append(rng.randint(10, 40))
        return BOKPInstance(f"bokp_n{n:04d}_i{instance_index:03d}", n, seed, {"name": "random.Random", "python_version": sys.version, "algorithm": "MT19937 via random.Random", "generation_order": "for i=0..n-1: weight randint(1,20), p randint(1,30), q randint(10,40)"}, {"weight_range": [1, 20], "p_range": [1, 30], "q_range": [10, 40], "instances_per_n": 30, "instance_index": instance_index, "seed_rule": "BASE_SEED + n * 1009 + (instance_index - 1)", "base_seed": self.base_seed}, tuple(weights), tuple(p_values), tuple(q_values), math.floor(0.5 * sum(weights)))

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BOKPInstance:
        return BOKPInstance(data["instance_id"], data["n"], data["seed"], data["rng"], data["generation_parameters"], tuple(data["weights"]), tuple(data["p"]), tuple(data["q"]), data["capacity"])