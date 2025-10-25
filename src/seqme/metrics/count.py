from typing import Literal

from seqme.core.base import Metric, MetricResult


class Count(Metric):
    """Number of sequences."""

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Count the number of input sequences.

        Args:
            sequences: A list of sequences.
            name: Metric name.

        Returns:
            MetricResult: value is the number of sequences. Deviation is None.
        """
        return MetricResult(value=len(sequences))

    @property
    def name(self) -> str:
        return "Count"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
