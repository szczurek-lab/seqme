from collections.abc import Callable
from typing import Literal

import numpy as np

from seqme.core import Metric, MetricResult


class Identity(Metric):
    """Applies a user-provided predictor to a list of sequences and returns the mean and standard deviation of the predictor's outputs."""

    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        name: str,
        objective: Literal["minimize", "maximize"],
    ):
        """
        Initialize the Identity metric.

        Args:
            predictor: A function that takes a list of sequences
                and returns a 1D array of scalar values.
            name: Name of the metric.
            objective: Specifies whether lower
                or higher values of the metric are better.
        """
        self.predictor = predictor
        self._name = name
        self._objective = objective

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Evaluate the predictor on the provided sequences.

        Applies the predictor to the sequences and returns the mean and standard
        deviation of the resulting values.

        Args:
            sequences: List of sequences to evaluate.

        Returns:
            MetricResult: Contains two elements:
                - value (float): Mean of predictor outputs.
                - std (float): Standard deviation of predictor outputs.
        """
        values = self.predictor(sequences)
        return MetricResult(values.mean().item(), values.std().item())

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self._objective


class ID(Identity):
    """Applies a user-provided predictor to a list of sequences and returns the average and standard deviation of the predictor's outputs."""

    pass
