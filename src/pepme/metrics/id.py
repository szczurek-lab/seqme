from typing import Callable, Literal

import numpy as np

from pepme.core import Metric, MetricResult


class Identity(Metric):
    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        name: str,
        objective: Literal["minimize", "maximize"],
    ):
        self.predictor = predictor
        self._name = name
        self._objective = objective

    def __call__(self, sequences: list[str]) -> MetricResult:
        values = self.predictor(sequences)
        return MetricResult(values.mean().item(), values.std().item())

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self._objective


class ID(Identity):
    pass
