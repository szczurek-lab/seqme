from typing import Literal

from seqme.core.base import Metric, MetricResult


class Uniqueness(Metric):
    """Fraction of unique sequences within the provided list of sequences."""

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the fraction of sequences remaining after removing any duplicate sequences.

        Example:
            If ``sequences`` contains ``["KR", "KR"]``, the uniqueness is ``0.5``.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Uniqueness.
        """
        total = len(sequences)
        if total == 0:
            return MetricResult(0.0)

        unique_count = len(set(sequences))
        score = unique_count / total
        return MetricResult(score)

    @property
    def name(self) -> str:
        return "Uniqueness"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
