from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    sizes: tuple[int, ...] = (80, 100, 120, 140, 160, 180)
    probabilities: tuple[float, ...] = (0.3, 0.5, 0.7)
    instances_per_pair: int = 30
    cost_range: tuple[int, int] = (20, 120)
    time_range: tuple[int, int] = (1, 100)
    phase2_time_limit: float | None = None
    gurobi_threads: int = 1
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "sizes": list(self.sizes), "probabilities": list(self.probabilities),
            "instances_per_pair": self.instances_per_pair,
            "cost_range": list(self.cost_range), "time_range": list(self.time_range),
            "phase2_time_limit": self.phase2_time_limit,
            "gurobi_threads": self.gurobi_threads, **self.extra,
        }