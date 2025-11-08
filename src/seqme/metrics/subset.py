from typing import Literal

import numpy as np

from seqme.core.base import Metric, MetricResult


class Subset(Metric):
    """A wrapper to approximate expensive metrics by evaluating a subset of the sequences in a group."""

    def __init__(
        self,
        metric: Metric,
        *,
        n_samples: int,
        seed: int = 0,
    ):
        """
        Initialize subset wrapper.

        Args:
            metric: Metric to compute.
            n_samples: Number of sequences to sample.
            seed: Seed for deterministic sampling.
        """
        self.metric = metric
        self.n_samples = n_samples
        self.seed = seed

        if n_samples <= 0:
            raise ValueError("n_samples must be greater than 0.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the metric on a subset of sequences.

        Args:
            sequences: Sequences to sample a subset from and evaluate.

        Returns:
            MetricResult: Metric computed on subset of sequences.
        """
        if len(sequences) < self.n_samples:
            raise ValueError(
                f"Too few sequences to subsample. Expected at least {self.n_samples} sequences, got {len(sequences)} sequences."
            )

        rng = np.random.default_rng(self.seed)
        indices = rng.choice(np.arange(len(sequences), dtype=int), size=self.n_samples, replace=False)
        subset = [sequences[idx] for idx in indices]
        return self.metric(subset)

    @property
    def name(self) -> str:
        return self.metric.name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self.metric.objective
