from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class ObjectivePoint:
    p: int
    q: int
    x: tuple[int, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {"P": self.p, "Q": self.q, "x": list(self.x)}


@dataclass(frozen=True)
class BOKPInstance:
    instance_id: str
    n: int
    seed: int
    rng: dict[str, Any]
    generation_parameters: dict[str, Any]
    weights: tuple[int, ...]
    p: tuple[int, ...]
    q: tuple[int, ...]
    capacity: int

    def as_dict(self) -> dict[str, Any]:
        return {"instance_id": self.instance_id, "n": self.n, "seed": self.seed, "rng": self.rng, "generation_parameters": self.generation_parameters, "weights": list(self.weights), "p": list(self.p), "q": list(self.q), "capacity": self.capacity}

    def objectives(self, x: tuple[int, ...] | list[int]) -> tuple[int, int]:
        return sum(a * b for a, b in zip(self.p, x)), sum(a * b for a, b in zip(self.q, x))

    def weight(self, x: tuple[int, ...] | list[int]) -> int:
        return sum(a * b for a, b in zip(self.weights, x))


@dataclass(frozen=True)
class SolveRecord:
    stage: str
    status_code: int
    status: str
    objective_value: float | None
    runtime_seconds: float
    termination: dict[str, Any]
    parameters: dict[str, Any]
    f1: int | None = None
    f2: int | None = None
    x: tuple[int, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "status_code": self.status_code, "status": self.status, "objective_value": self.objective_value, "runtime_seconds": self.runtime_seconds, "termination": self.termination, "parameters": self.parameters, "f1": self.f1, "f2": self.f2, "x": list(self.x)}


@dataclass(frozen=True)
class ReferencePoints:
    f1_utopia: int
    f2_utopia: int
    f1_nadir: int
    f2_nadir: int
    p_max: int
    q_max: int
    lambda_1: Fraction
    lambda_2: Fraction
    m1: SolveRecord
    m2: SolveRecord
    nadir_lambda_1: SolveRecord
    nadir_lambda_2: SolveRecord

    def point(self, record: SolveRecord) -> ObjectivePoint:
        if record.f1 is None or record.f2 is None:
            raise ValueError("Solve record has no objective values")
        return ObjectivePoint(record.f1 - self.f1_nadir, record.f2 - self.f2_nadir, record.x)

    def rho_sign(self, point: ObjectivePoint) -> int:
        difference = point.p * self.q_max - point.q * self.p_max
        return (difference > 0) - (difference < 0)

    def as_dict(self) -> dict[str, Any]:
        return {"f1_utopia": self.f1_utopia, "f2_utopia": self.f2_utopia, "f1_nadir": self.f1_nadir, "f2_nadir": self.f2_nadir, "P_max": self.p_max, "Q_max": self.q_max, "lambda_1": self.lambda_1, "lambda_2": self.lambda_2, "reference_solutions": {"M1": self.m1.as_dict(), "M2": self.m2.as_dict(), "lambda1": self.nadir_lambda_1.as_dict(), "lambda2": self.nadir_lambda_2.as_dict()}}