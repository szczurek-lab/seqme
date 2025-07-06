from typing import Callable, Literal

import numpy as np

from pepme.core import Metric, MetricResult


class HitRate(Metric):
    """
    Computes the fraction of sequences that satisfy a given filter condition.
    """

    def __init__(
        self,
        filter_fn: Callable[[list[str]], np.ndarray],
    ):
        """
        Initializes the hit‐rate metric.

        Args:
            filter_fn: A function that takes a list of sequences and returns
                       a boolean NumPy array of the same length, where True
                       indicates a “hit” for that sequence.
        """
        self.mask_fn = filter_fn

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Applies the filter to count hits and returns the average hit rate.

        Args:
            sequences: List of sequences to evaluate.

        Returns:
            A MetricResult whose value is the mean of the boolean mask,
            i.e., the proportion of sequences where filter_fn returned True.
        """
        valid = self.mask_fn(sequences)
        hit_rate = valid.mean().item()
        return MetricResult(hit_rate)

    @property
    def name(self) -> str:
        return "Hit-rate"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
