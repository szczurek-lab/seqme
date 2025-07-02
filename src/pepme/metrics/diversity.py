from typing import Optional, Literal

import numpy as np

from pepme.core import Metric, MetricResult
from Levenshtein import distance as lev

class Diversity(Metric):
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

    @staticmethod
    def _get_levenstein(seq_a: str, seq_b: str) -> int:
        return lev(seq_a, seq_b)

    def _get_levenshtein_for_all_ref_sequences(self, seq_a: str) -> list[int]:
        return [self._get_levenstein(seq_a, seg_ref) for seg_ref in self.reference]


    def __call__(self, sequences: list[str]) -> MetricResult:
        min_levenshtein_for_seqs = []
        for sequence in sequences:
            min_levenshtein_for_seqs.append(self._get_levenshtein_for_all_ref_sequences(sequence))
        min_levenshtein_for_seqs_np = np.array(min_levenshtein_for_seqs)
        return MetricResult(float(min_levenshtein_for_seqs_np.mean()), float(min_levenshtein_for_seqs_np.std()))

    @property
    def name(self) -> str:
        return "Diversity" if self.data_name is None else f"Diversity ({self.data_name})"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


