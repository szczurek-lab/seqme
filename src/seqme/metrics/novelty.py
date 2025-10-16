from typing import Literal

from seqme.core import Metric, MetricResult


class Novelty(Metric):
    """Fraction of sequences not in the reference."""

    def __init__(self, reference: list[str], *, name: str = "Novelty"):
        """
        Initialize the metric with a reference corpus.

        Args:
            reference: A list of reference sequences against which generated sequences will be compared.
                Sequences found in this list are considered non-novel.
            reference_name: An optional label for the reference data. This name will be appended to the metric name for identification.
            name: Metric name.
        """
        self.reference = set(reference)
        self._name = name

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the Novelty score as the proportion of input sequences that are not present in the reference set.

        Args:
            sequences: Generated sequences to evaluate for novelty.

        Returns:
            MetricResult contains the novelty score between 0 and 1, where
                0 indicates no novel sequences and 1 indicates all sequences
                are novel.
        """
        total = len(sequences)
        if total == 0:
            return MetricResult(0.0)

        novel_count = sum(1 for seq in sequences if seq not in self.reference)
        score = novel_count / total
        return MetricResult(score)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
