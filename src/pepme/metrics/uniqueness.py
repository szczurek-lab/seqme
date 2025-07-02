from typing import Literal, Optional

from pepme.core import Metric, MetricResult


class Uniqueness(Metric):
    """
    Uniqueness metric computes the fraction of unique sequences
    within the provided list of generated sequences.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the Uniqueness metric.

        Args:
            name (Optional[str]): An optional label for the metric.
                This will be appended to the metric name for identification.
                Defaults to None.
        """
        self.data_name = name

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the uniqueness score as the proportion of unique sequences
        in the input list.

        Args:
            sequences (list[str]): Generated sequences to evaluate.

        Returns:
            MetricResult: Contains the uniqueness score between 0 and 1,
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
        return (
            "Uniqueness" if self.data_name is None else f"Uniqueness ({self.data_name})"
        )

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        # We want as many distinct sequences as possible
        return "maximize"
