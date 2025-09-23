from typing import Literal

import numpy as np
import pylev

from seqme.core import Metric, MetricResult


class Diversity(Metric):
    """Normalized pairwise Levenshtein distance between the sequences."""

    def __init__(
        self,
        aggregate: Literal["mean", "min"] = "mean",
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

        if self.k:
            if self.k < 1:
                raise ValueError("Expected k > 0.")

            if self.reference:
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
    aggregate: Literal["mean", "min"] = "mean",
    reference: list[str] = None,
    k: int | None = None,
    seed: int | None = 0,
) -> float:
    """
    Compute diversity.

    Args:
        sequences: Text sequences to compute diversity on.
        aggregate: How to aggregate the diversity between a sequences and the other sequences.
        reference: Reference sequences to compare against. If None, compare against other sequences within `sequences`.
        k: If not None randomly sample `k` other sequences to compute diversity against.
        seed: For reproducibility. Only used if k is not None.

    Returns:
        Diversity.
    """
    if k:
        rng = np.random.default_rng(seed)

    divs = []
    for i, sequence in enumerate(sequences):
        others = reference if reference else sequences[:i] + sequences[i + 1 :]

        if k:
            idxs = rng.choice(np.arange(len(others)), size=k, replace=False)
            others = [others[i] for i in idxs]

        norms = np.maximum(len(sequence), [len(seq) for seq in others])
        edits = np.array([pylev.levenshtein(sequence, seq) for seq in others])
        norm_edits = edits / norms

        if aggregate == "mean":
            div = norm_edits.mean()
        elif aggregate == "min":
            div = norm_edits.min()
        else:
            raise ValueError(f"Unsupported aggregate: '{aggregate}'.")

        divs.append(div)

    return np.mean(divs).item()
