from typing import Literal

import numpy as np
from polyleven import levenshtein

from seqme.core.base import Metric, MetricResult


class Diversity(Metric):
    """Normalized pairwise Levenshtein distance between the sequences."""

    def __init__(
        self,
        k: int | None = None,
        *,
        seed: int = 0,
        name: str = "Diversity",
    ):
        """
        Initialize the metric.

        Args:
            k: If not ``None`` randomly sample ``k`` other sequences to compute diversity against.
            seed: For deterministic sampling. Only used if ``k`` is not ``None``.
            name: Metric name.
        """
        self.k = k
        self.seed = seed
        self._name = name

        if (self.k is not None) and (self.k <= 0):
            raise ValueError("Expected k > 0.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the diversity.

        Note: For a large number of ``sequences``, a small value for ``k`` (e.g., 10) usually provides a stable approximation of the diversity.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Diversity score.
        """
        score = compute_diversity(sequences, k=self.k, seed=self.seed)
        return MetricResult(score)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_diversity(
    sequences: list[str],
    *,
    k: int | None = None,
    seed: int = 0,
) -> float:
    """
    Compute diversity.

    Args:
        sequences: Sequences to compute diversity on.
        k: If not ``None`` randomly sample ``k`` other sequences to compute diversity against.
        seed: For deterministic sampling. Only used if k is not ``None``.

    Returns:
        Diversity.
    """
    if k:
        rng = np.random.default_rng(seed)

    divs = []
    for i, sequence in enumerate(sequences):
        others = sequences[:i] + sequences[i + 1 :]

        if k and k < len(others):
            idxs = rng.choice(np.arange(len(others)), size=k, replace=False)
            others = [others[i] for i in idxs]

        norms = np.maximum(len(sequence), [len(seq) for seq in others])
        edits = np.array([levenshtein(sequence, seq) for seq in others])
        norm_edits = edits / norms

        div = norm_edits.mean()
        divs.append(div)

    return np.mean(divs).item()
