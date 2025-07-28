from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.linalg import sqrtm

from seqme.core import Metric, MetricResult


class FrechetBiologicalDistance(Metric):
    """
    Fréchet Biological Distance (FBD) between a set of generated
    sequences and a reference dataset based on their embeddings.

    This metric estimates how similar the distributions of two sets of embeddings
    are using the 2-Wasserstein (Fréchet) distance.

    Reference:
        Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a
        Local Nash Equilibrium" (https://arxiv.org/abs/1706.08500)
    """

    def __init__(
        self,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        reference_name: str | None = None,
        embedder_name: str | None = None,
    ):
        """
        Initializes the FBD metric with a reference dataset and an embedding function.

        Args:
            reference: A list of reference sequences (e.g., real data).
            embedder: A function that maps a list of sequences to a 2D NumPy array of embeddings.
            reference_name: Optional name for the reference dataset.
            embedder_name: Optional name for the embedder used.

        Raises:
            ValueError: If fewer than 2 reference embeddings are provided.
        """
        self.reference = reference
        self.embedder = embedder
        self.reference_name = reference_name
        self.embedder_name = embedder_name

        self.reference_embeddings = self.embedder(self.reference)

        if self.reference_embeddings.shape[0] < 2:
            raise ValueError("Reference embeddings must contain at least two samples.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Computes the FBD between the reference and the input sequences.

        Args:
            sequences: A list of generated sequences to evaluate.

        Returns:
            MetricResult containing the FBD score. Lower is better.
        """
        seq_embeddings = self.embedder(sequences)
        dist = wasserstein_distance(seq_embeddings, self.reference_embeddings)
        return MetricResult(dist)

    @property
    def name(self) -> str:
        name = "FBD"
        if self.embedder_name:
            name += f"@{self.embedder_name}"
        if self.reference_name:
            name += f" ({self.reference_name})"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"


def wasserstein_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    """
    Computes the Fréchet distance between two sets of embeddings.

    This is defined as:
        ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2(Σ1·Σ2)^{1/2})

    Args:
        e1: First set of embeddings, shape (N1, D).
        e2: Second set of embeddings, shape (N2, D).

    Returns:
        The Fréchet distance as a float. Returns NaN if either set has fewer than 2 samples.
    """
    if len(e1) < 2 or len(e2) < 2:
        return float("nan")

    mu1, sigma1 = e1.mean(axis=0), np.cov(e1, rowvar=False)
    mu2, sigma2 = e2.mean(axis=0), np.cov(e2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, err = sqrtm(sigma1.dot(sigma2), disp=False)

    if err == np.inf:
        return float("nan")

    # Handle numerical issues with imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    dist = max(0.0, dist.item())  # numerical stability

    return dist


class FBD(FrechetBiologicalDistance):
    """
    Fréchet Biological Distance (FBD) between a set of generated
    sequences and a reference dataset based on their embeddings.

    This metric estimates how similar the distributions of two sets of embeddings
    are using the 2-Wasserstein (Fréchet) distance.

    Reference:
        Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a
        Local Nash Equilibrium" (https://arxiv.org/abs/1706.08500)
    """

    pass
