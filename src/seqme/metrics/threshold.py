from collections.abc import Callable
from typing import Literal

import numpy as np

from seqme.core.base import Metric, MetricResult


class Threshold(Metric):
    """Fraction of sequences with property within [min, max] a user-defined threshold."""

    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        name: str,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        """
        Initialize the metric.

        Args:
            predictor: A function that takes a list of sequences and returns a 1D array of scalar values.
            name: Name of the metric.
            min_value: Minimum threshold value.
            max_value: Maximum threshold value.
        """
        self.predictor = predictor
        self._name = name
        self.min_value = min_value
        self.max_value = max_value

        if (self.min_value is None) and (self.max_value is None):
            raise ValueError("'min_value' and/or 'max_value' must be set.")
        if (self.min_value is not None) and (self.max_value is not None) and (self.min_value > self.max_value):
            raise ValueError(f"Expected min_value ({self.min_value}) <= max_value ({self.max_value}).")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Applies the predictor to the sequences and returns the fraction of sequences within the threshold.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Fraction of sequences within the threshold.
        """
        values = self.predictor(sequences)

        above = values >= self.min_value if self.min_value is not None else True
        below = values <= self.max_value if self.max_value is not None else True
        within = above & below

        return MetricResult(np.mean(within).item())

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
