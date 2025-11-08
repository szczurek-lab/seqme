from typing import Literal

import numpy as np

from seqme.core.base import Metric, MetricResult


class Length(Metric):
    """Average sequence length."""

    def __init__(self, objective: Literal["minimize", "maximize"] = "minimize"):
        """Initialize the metric.

        Args:
            objective: Whether to minimize or maximize the metric.
        """
        self._objective = objective

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the average sequence length.

        Args:
            sequences: A list of sequences.

        Returns:
            MetricResult: Mean sequence length and sequence length standard deviation.
        """
        lengths = [len(sequence) for sequence in sequences]

        return MetricResult(
            value=np.mean(lengths).item(),
            deviation=np.std(lengths).item() if len(sequences) > 1 else None,
        )

    @property
    def name(self) -> str:
        return "Length"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self._objective
