from collections.abc import Callable
from typing import Literal

import numpy as np

from seqme.core.base import Metric, MetricResult


class ID(Metric):
    """Applies a user-provided predictor to a list of sequences and returns the mean and standard error of the predictors outputs."""

    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        name: str,
        objective: Literal["minimize", "maximize"],
        *,
        deviation: Literal["std", "se", "var"] = "se",
        estimate: Literal["biased", "unbiased"] = "unbiased",
    ):
        """
        Initialize the metric.

        Args:
            predictor: A function that takes a list of sequences and returns a 1D array of scalar values.
            name: Name of the metric.
            objective: Specifies whether lower or higher values of the metric are better.
            deviation: Type of deviation to compute:

                - ``'std'``: Standard deviation
                - ``'se'``: Standard error
                - ``'var'``: Variance

            estimate: How to estimate the deviation.
        """
        self.predictor = predictor
        self._name = name
        self._objective = objective
        self.deviation = deviation
        self.estimate = estimate

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Evaluate the predictor on the provided sequences.

        Applies the predictor to the sequences and returns the mean and standard error of the resulting values (if more than one sequence).

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Mean predictor value and deviation.
        """
        values = self.predictor(sequences)

        if len(values) > 1:
            if self.estimate == "biased":
                ddof = 0
            elif self.estimate == "unbiased":
                ddof = 1
            else:
                raise ValueError(f"Invalid estimate: {self.estimate}")

            if self.deviation == "std":
                deviation = float(values.std(ddof=ddof))
            elif self.deviation == "var":
                deviation = float(values.var(ddof=ddof))
            elif self.deviation == "se":
                deviation = float(values.std(ddof=ddof)) / (len(values) ** 0.5)
            else:
                raise ValueError(f"Invalid deviation: {self.deviation}")
        else:
            deviation = None

        return MetricResult(values.mean().item(), deviation)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self._objective
