from typing import Literal

import numpy as np
import pylev

from seqme.core import Metric, MetricResult


class Diversity(Metric):
    """Pairwise Levenshstein distance between the sequences, normalized by number of sequences and number of residues."""

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the diversity.

        Args:
            sequences: A list of generated sequences to evaluate.

        Returns:
            MetricResult contains the diversity score.
        """
        if len(sequences) < 2:
            raise ValueError("Expected at least 2 sequences.")

        levenshtein_matrix = np.stack(
            [[pylev.levenshtein(seq1, seq2) for seq2 in sequences] for seq1 in sequences], axis=-1
        )

        total_dist = levenshtein_matrix.sum()
        total_letters = np.sum([len(seq) for seq in sequences])

        diversity = total_dist / (total_letters * (len(sequences) - 1))

        return MetricResult(diversity.item())

    @property
    def name(self) -> str:
        return "Diversity"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
