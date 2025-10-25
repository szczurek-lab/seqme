from typing import Literal

from seqme.core.base import Metric, MetricResult


class Uniqueness(Metric):
    """Fraction of unique sequences within the provided list of generated sequences."""

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the uniqueness score as the proportion of unique sequences in the input list.

        Args:
            sequences: Generated sequences to evaluate.

        Returns:
            Contains the uniqueness score between 0 and 1,
                where 0 indicates no unique sequences (all duplicates)
                and 1 indicates all sequences are distinct.
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
