from typing import Literal

from seqme.core import Metric, MetricResult


class Count(Metric):
    """Number of sequences."""

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the count of input sequences.

        Args:
            sequences: A list of sequences to count.

        Returns:
            MetricResult: value is the count of sequences; deviation is None.
        """
        return MetricResult(value=len(sequences))

    @property
    def name(self) -> str:
        return "Count"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
