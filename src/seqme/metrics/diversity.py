from typing import Literal

import numpy as np
import pylev

from seqme.core import Metric, MetricResult


class Diversity(Metric):
    """Normalized pairwise Levenshtein distance between the sequences."""

    def __init__(
        self,
        aggregate: Literal["mean", "min", "max"] = "mean",
        reference: list[str] = None,
        k: int | None = None,
        seed: int | None = 0,
    ):
        """
        Initialize the metric.

        Args:
            aggregate: How to aggregate the diversity between a sequences and the other sequences.
            reference: Reference sequences to compare against. If None, compare against other sequences within `sequences`.
            k: If not None randomly sample `k` other sequences to compute diversity against.
            seed: For reproducibility. Only used if k is not None.
        """
        self.aggregate = aggregate
        self.reference = reference
        self.k = k
        self.seed = seed

        if self.reference and self.k:
            if len(self.reference) < self.k:
                raise ValueError("Fewer sequences in reference than k.")

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

        score = compute_diversity(
            sequences,
            aggregate=self.aggregate,
            reference=self.reference,
            k=self.k,
            seed=self.seed,
        )

        return MetricResult(score)

    @property
    def name(self) -> str:
        return "Diversity"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_diversity(
    sequences: list[str],
    *,
    aggregate: Literal["mean", "min", "max"] = "mean",
    reference: list[str] = None,
    k: int | None = None,
    seed: int | None = 0,
) -> float:
    if k:
        rng = np.random.default_rng(seed)

    divs = []
    for i, sequence in enumerate(sequences):
        others = reference if reference else sequences[:i] + sequences[i + 1 :]

        if k:
            idxs = rng.choice(np.arange(k + 1))
            others = [others[i] for i in idxs]

        norms = np.maximum(len(sequence), [len(seq) for seq in others])
        edits = np.array([pylev.levenshtein(sequence, seq) for seq in others])
        norm_edits = edits / norms

        if aggregate == "mean":
            div = norm_edits.mean()
        elif aggregate == "min":
            div = norm_edits.min()
        elif aggregate == "max":
            div = norm_edits.max()
        else:
            raise ValueError(f"Unsupported aggregate: '{aggregate}'.")

        divs.append(div)

    return np.mean(divs).item()
