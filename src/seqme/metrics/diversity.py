from typing import Literal

import numpy as np
from polyleven import levenshtein

from seqme.core.base import Metric, MetricResult


class Diversity(Metric):
    """Normalized pairwise Levenshtein distance between the sequences."""

    def __init__(
        self,
        reference: list[str] = None,
        k: int | None = None,
        *,
        seed: int = 0,
        name: str = "Diversity",
    ):
        """
        Initialize the metric.

        Args:
            reference: Reference sequences to compare against. If ``None``, compare against other sequences within ``sequences``.
            k: If not ``None`` randomly sample ``k`` other sequences to compute diversity against.
            seed: For deterministic sampling. Only used if ``k`` is not ``None``.
            name: Metric name.
        """
        self.reference = reference
        self.k = k
        self.seed = seed
        self._name = name

        if self.k is not None:
            if self.k <= 0:
                raise ValueError("Expected k > 0.")

            if self.reference:
                if len(self.reference) < self.k:
                    raise ValueError("Fewer sequences in reference than k.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the diversity.

        Note: For a large number of ``sequences``, a small value for ``k`` (e.g., 10) usually provides a stable approximation of the diversity.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Diversity score.
        """
        score = compute_diversity(
            sequences,
            reference=self.reference,
            k=self.k,
            seed=self.seed,
        )
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
    reference: list[str] = None,
    k: int | None = None,
    seed: int = 0,
) -> float:
    """
    Compute diversity.

    Args:
        sequences: Sequences to compute diversity on.
        reference: Reference sequences to compare against. If ``None``, compare against other sequences within ``sequences``.
        k: If not ``None`` randomly sample ``k`` other sequences to compute diversity against.
        seed: For deterministic sampling. Only used if k is not ``None``.

    Returns:
        Diversity.
    """
    if k:
        rng = np.random.default_rng(seed)

    divs = []
    for i, sequence in enumerate(sequences):
        others = reference if reference else sequences[:i] + sequences[i + 1 :]

        if k and k < len(others):
            idxs = rng.choice(np.arange(len(others)), size=k, replace=False)
            others = [others[i] for i in idxs]

        norms = np.maximum(len(sequence), [len(seq) for seq in others])
        edits = np.array([levenshtein(sequence, seq) for seq in others])
        norm_edits = edits / norms

        div = norm_edits.mean()
        divs.append(div)

    return np.mean(divs).item()
