from typing import Literal, Optional

import numpy as np
from Levenshtein import distance as lev

from pepme.core import Metric, MetricResult


class Diversity(Metric):
    """
    Diversity metric computes the fraction of sequences not in the reference set.
    """

    def __init__(self, reference: list[str], reference_name: Optional[str] = None):
        """
        Initialize the Diversity metric with a reference corpus.

        Args:
            reference (list[str]): A list of reference sequences against which
                generated sequences will be compared.
            reference_name (Optional[str]): An optional label for the reference data.
                This name will be appended to the metric name for identification.
                Defaults to None.
        """
        self.reference = set(reference)
        self.reference_name = reference_name

        if len(self.reference) == 0:
            raise ValueError("References must contain at least one sample.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        seqs_min_levenshtein = []
        for seq in sequences:
            min_distances = min(self._get_levenshtein_to_references(seq))
            seqs_min_levenshtein.append(min_distances)

        seqs_min_levenshtein_np = np.array(seqs_min_levenshtein)
        return MetricResult(
            seqs_min_levenshtein_np.mean().item(),
            seqs_min_levenshtein_np.std().item(),
        )

    @property
    def name(self) -> str:
        return (
            "Diversity"
            if self.reference_name is None
            else f"Diversity ({self.reference_name})"
        )

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"

    @staticmethod
    def _get_levenshtein(sequence_a: str, sequence_b: str) -> int:
        return lev(sequence_a, sequence_b)

    def _get_levenshtein_to_references(self, sequence: str) -> list[int]:
        return [self._get_levenshtein(sequence, ref) for ref in self.reference]
