from collections.abc import Callable
from typing import Literal

import numpy as np
from sklearn.neighbors import KernelDensity

from seqme.core import Metric, MetricResult


class KLDivergence(Metric):
    """KL-divergence between samples and reference for a single property."""

    def __init__(
        self,
        reference: list[str],
        descriptor: Callable[[list[str]], np.ndarray],
        *,
        n_draws: int = 10_000,
        kde_bandwidth: float | Literal["scott", "silverman"] = "silverman",
        reference_name: str | None = None,
        seed: int | None = 0,
    ):
        """
        Initialize the KL-divergence.

        Args:
            reference: Reference sequences assumed to represent the target distribution.
            descriptor: Descriptor function which return a 1D NumPy array.
            n_draws: Number of Monte Carlo samples to draw from distribution P.
            kde_bandwidth: Bandwidth parameter for the Gaussian KDE.
            reference_name: Optional name for the reference dataset.
            seed: Seed for KL-divergence Monte-Carlo sampling.
        """
        self.reference = reference
        self.descriptor = descriptor
        self.n_draws = n_draws
        self.kde_bandwidth = kde_bandwidth
        self.seed = seed
        self.reference_name = reference_name

        self.reference_descriptor = self.descriptor(self.reference)

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the KL-divergence between reference and sequence descriptor.

        Args:
            sequences: List of generated sequences to evaluate.

        Returns:
            MetricResult: KL-divergence and standard error.
        """
        seqs_descriptor = self.descriptor(sequences)
        kl_div, standard_error = continuous_kl_mc(
            self.reference_descriptor,
            seqs_descriptor,
            kde_bandwidth=self.kde_bandwidth,
            n_draws=self.n_draws,
            seed=self.seed,
        )
        return MetricResult(value=kl_div, deviation=standard_error)

    @property
    def name(self) -> str:
        return "KL-divergence" if self.reference_name is None else f"KL-divergence ({self.reference_name})"

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
        seed: Seed for the random number generator to ensure reproducibility.

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
