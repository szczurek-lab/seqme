from collections.abc import Callable
from typing import Literal

import numpy as np

from seqme.core.base import Metric, MetricResult


class ID(Metric):
    """Applies a user-provided predictor to a list of sequences and returns the mean and standard deviation of the predictors outputs."""

    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        name: str,
        objective: Literal["minimize", "maximize"],
    ):
        """
        Initialize the metric.

        Args:
            predictor: A function that takes a list of sequences and returns a 1D array of scalar values.
            name: Name of the metric.
            objective: Specifies whether lower or higher values of the metric are better.
        """
        self.predictor = predictor
        self._name = name
        self._objective = objective

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Evaluate the predictor on the provided sequences.

        Applies the predictor to the sequences and returns the mean and standard deviation of the resulting values (if more than one sequence).

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult:
                - value: Mean of predictor outputs.
                - std: Standard deviation of predictor outputs.
        """
        values = self.predictor(sequences)
        return MetricResult(values.mean().item(), values.std().item() if len(sequences) > 1 else None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self._objective
