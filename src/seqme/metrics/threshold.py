from collections.abc import Callable
from typing import Literal

import numpy as np

from seqme.core.base import Metric, MetricResult


class Threshold(Metric):
    """Fraction of sequences with a property above (or below) a user-defined threshold."""

    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        name: str,
        *,
        threshold: float = 0.5,
        objective: Literal["minimize", "maximize"] = "maximize",
        inclusive: bool = True,
    ):
        """
        Initialize the metric.

        Args:
            predictor: A function that takes a list of sequences and returns a 1D array of scalar values.
            name: Name of the metric.
            threshold: Threshold value.
            objective: Specifies whether lower or higher values of the metric are better.
            inclusive: Whether to include the threshold value as a hit.
        """
        self.predictor = predictor
        self.threshold = threshold
        self.inclusive = inclusive
        self._name = name
        self._objective = objective

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Applies the predictor to the sequences and returns the fraction of sequences within the threshold.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Fraction of sequences within the threshold.
        """
        values = self.predictor(sequences)

        if self.inclusive:
            within_threshold = values >= self.threshold if self.objective == "maximize" else values <= self.threshold
        else:
            within_threshold = values > self.threshold if self.objective == "maximize" else values < self.threshold

        return MetricResult(within_threshold.mean().item())

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self._objective
