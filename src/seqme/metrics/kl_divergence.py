from collections.abc import Callable
from typing import Literal

import numpy as np
from sklearn.neighbors import KernelDensity

from seqme.core.base import Metric, MetricResult


class KLDivergence(Metric):
    r"""
    KL-divergence between samples and reference for a single property.

    This metric measures how much the empirical distribution of a property
    :math:`f(x)` in the generated samples deviates from the corresponding
    reference distribution.

    The KL-divergence is defined as:

    .. math::

        \mathrm{KL}\big(p_{f(\mathrm{ref})} \,\|\, p_{f(\mathrm{gen})}\big)
        = \int p_{f(\mathrm{ref})}(y)
        \log \frac{p_{f(\mathrm{ref})}(y)}{p_{f(\mathrm{gen})}(y)} \, dy,

    where :math:`p_{f(\mathrm{ref})}` denotes the reference distribution and
    :math:`p_{f(\mathrm{gen})}` denotes the generated distribution.

    The KL-divergence is approximated using Monte-Carlo sampling.
    """

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
    x_reference = x_reference.reshape(-1, 1)
    x_samples = x_samples.reshape(-1, 1)

    kde_p = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(x_reference)
    kde_q = KernelDensity(kernel="gaussian", bandwidth=kde_p.bandwidth_).fit(x_samples)

    x_p = kde_p.sample(n_draws, random_state=seed)

    log_p = kde_p.score_samples(x_p)
    log_q = kde_q.score_samples(x_p)

    log_diff = log_p - log_q

    kl_estimate = float(log_diff.mean())
    se = float(log_diff.std(ddof=1) / np.sqrt(n_draws))

    return kl_estimate, se
