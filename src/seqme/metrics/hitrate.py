from collections.abc import Callable
from typing import Literal

import numpy as np

from seqme.core.base import Metric, MetricResult


class HitRate(Metric):
    """Fraction of sequences that satisfy a given filter condition."""

    def __init__(self, condition_fn: Callable[[list[str]], np.ndarray], *, name: str = "Hit-rate"):
        """
        Initializes the hit-rate metric.

        Args:
            condition_fn: A function that takes a list of sequences and returns
                       a boolean NumPy array of the same length, where ``True``
                       indicates a “hit” for that sequence.
            name: Name of the metric.
        """
        self.condition_fn = condition_fn
        self._name = name

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Applies the filter to count hits and returns the average hit-rate.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Proportion of sequences where ``condition_fn`` returned ``True``.
        """
        valid = self.condition_fn(sequences)
        hit_rate = valid.mean().item()
        return MetricResult(hit_rate)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
