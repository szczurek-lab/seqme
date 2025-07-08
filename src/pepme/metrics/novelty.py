from typing import Literal, Optional

from pepme.core import Metric, MetricResult


class Novelty(Metric):
    """
    Novelty metric compares generated sequences against a reference set
    and returns the fraction of sequences not seen in the reference.
    """

    def __init__(self, reference: list[str], reference_name: Optional[str] = None):
        """
        Initialize the Novelty metric with a reference corpus.

        Args:
            reference (list[str]): A list of reference sequences against which
                generated sequences will be compared. Sequences found in this
                list are considered non-novel.
            reference_name (Optional[str]): An optional label for the reference data.
                This name will be appended to the metric name for identification.
                Defaults to None.
        """
        self.reference = set(reference)
        self.data_name = reference_name

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the novelty score as the proportion of input sequences that are not present in the reference set.

        Args:
            sequences (list[str]): Generated sequences to evaluate for novelty.

        Returns:
            MetricResult: Contains the novelty score between 0 and 1, where
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
        return "Novelty" if self.data_name is None else f"Novelty ({self.data_name})"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
