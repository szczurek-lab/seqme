from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy.linalg

from seqme.core.base import Metric, MetricResult


class FBD(Metric):
    """Fréchet Biological Distance (FBD) between a set of generated sequences and a reference dataset based on their embeddings.

    This metric estimates how similar the distributions of two sets of embeddings
    are using the Wasserstein-2 (Fréchet) distance.

    Reference:
        Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a
        Local Nash Equilibrium" (https://arxiv.org/abs/1706.08500)
    """

    def __init__(
        self,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        name: str = "FBD",
    ):
        """
        Initializes the metric with a reference dataset and an embedding function.

        Args:
            reference: A list of reference sequences (e.g., real data).
            embedder: A function that maps a list of sequences to a 2D NumPy array of embeddings.
            name: Metric name.

        Raises:
            ValueError: If fewer than 2 reference embeddings are provided.
        """
        self.reference = reference
        self.embedder = embedder
        self._name = name

        self.reference_embeddings = self.embedder(self.reference)

        if self.reference_embeddings.shape[0] < 2:
            raise ValueError("Reference embeddings must contain at least two samples.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the FBD between the reference and the input sequences.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: FBD score.
        """
        seq_embeddings = self.embedder(sequences)
        dist = wasserstein_distance(seq_embeddings, self.reference_embeddings)
        return MetricResult(dist)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"


def wasserstein_distance(e1: np.ndarray, e2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Computes the Fréchet distance between two sets of embeddings.

    This is defined as:
        ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2(Σ1·Σ2)^{1/2})

    Args:
        e1: First set of embeddings, shape (N1, D).
        e2: Second set of embeddings, shape (N2, D).
        eps: Epsilon.

    Returns:
        The Fréchet distance as a float. Returns NaN if either set has fewer than 2 samples.
    """
    if e1.shape[0] < 2 or e2.shape[0] < 2:
        return float("nan")

    mu1, sigma1 = e1.mean(axis=0), np.cov(e1, rowvar=False)
    mu2, sigma2 = e2.mean(axis=0), np.cov(e2, rowvar=False)

    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    is_real = np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
    if not np.isfinite(covmean).all() or not is_real:
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Handle numerical issues with imaginary components
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            return float("nan")
        covmean = covmean.real

    diff = mu1 - mu2
    ssdiff = np.dot(diff, diff)

    dist = float(ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    dist = max(0.0, dist)  # numerical stability

    return dist
