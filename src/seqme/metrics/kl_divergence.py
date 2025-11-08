from collections.abc import Callable
from typing import Literal

import numpy as np
from sklearn.neighbors import KernelDensity

from seqme.core.base import Metric, MetricResult


class KLDivergence(Metric):
    """KL-divergence between samples and reference for a single property."""

    def __init__(
        self,
        reference: list[str],
        predictor: Callable[[list[str]], np.ndarray],
        *,
        n_draws: int = 10_000,
        kde_bandwidth: float | Literal["scott", "silverman"] = "silverman",
        seed: int = 0,
        name: str = "KL-divergence",
    ):
        """
        Initialize the metric.

        Args:
            reference: Reference sequences assumed to represent the target distribution.
            predictor: Predictor function which returns a 1D NumPy array. One value per sequence.
            n_draws: Number of Monte Carlo samples to draw from reference distribution.
            kde_bandwidth: Bandwidth parameter for the Gaussian KDE.
            seed: Seed for KL-divergence Monte-Carlo sampling.
            name: Metric name.
        """
        self.reference = reference
        self.predictor = predictor
        self.n_draws = n_draws
        self.kde_bandwidth = kde_bandwidth
        self.seed = seed
        self._name = name

        self.reference_predictor = self.predictor(self.reference)

        if self.n_draws <= 0:
            raise ValueError("Expected n_draws > 0.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the KL-divergence between reference and sequence predictor.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: KL-divergence and standard error.
        """
        seqs_predictor = self.predictor(sequences)
        kl_div, standard_error = continuous_kl_mc(
            self.reference_predictor,
            seqs_predictor,
            kde_bandwidth=self.kde_bandwidth,
            n_draws=self.n_draws,
            seed=self.seed,
        )
        return MetricResult(value=kl_div, deviation=standard_error)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"


def continuous_kl_mc(
    x_reference: np.ndarray,
    x_samples: np.ndarray,
    kde_bandwidth: float | Literal["scott", "silverman"] = "silverman",
    n_draws: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Monte-Carlo estimate of D_KL(P || Q) plus its standard error, where P ≈ KDE(x_reference), Q ≈ KDE(x_samples).

    Args:
        x_reference: Array of samples drawn from the reference distribution P.
        x_samples: Array of samples drawn from the comparison distribution Q.
        kde_bandwidth: Bandwidth parameter for the Gaussian KDE.
        n_draws: Number of Monte Carlo samples to draw from P.
        seed: Seed for deterministic sampling of Gaussian KDE.

    Returns:
        A tuple containing:
            kl_estimate: The estimated KL divergence between P and Q.
            se: The Monte Carlo standard error of the estimate.
    """
    kde_p = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(x_reference[:, None])
    kde_q = KernelDensity(kernel="gaussian", bandwidth=kde_p.bandwidth_).fit(x_samples[:, None])

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_reference), size=n_draws, replace=True)
    x_p = x_reference[idx] + rng.normal(scale=kde_p.bandwidth_, size=n_draws)

    log_p = kde_p.score_samples(x_p[:, None])
    log_q = kde_q.score_samples(x_p[:, None])

    log_diff = log_p - log_q

    kl_estimate = float(log_diff.mean())
    se = float(log_diff.std(ddof=1) / np.sqrt(n_draws))

    return kl_estimate, se
