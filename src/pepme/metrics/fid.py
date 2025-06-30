from typing import Callable, Literal, Optional

import numpy as np
from scipy.linalg import sqrtm

from pepme.core import Metric, MetricResult


class FrechetInceptionDistance(Metric):
    """
    Frechet Inception Distance (FID) metric compares the distribution of generated
    embeddings against a reference distribution using the Wasserstein distance.

    References:
        [1] Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a
        Local Nash Equilibrium"
    """

    def __init__(
        self,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        reference_name: Optional[str] = None,
        embedder_name: Optional[str] = None,
    ):
        """
        Initialize the FrechetInceptionDistance metric.

        Args:
            reference (list[str]): List of reference sequences representing real data.
            embedder (Callable[[list[str]], np.ndarray]): Function that maps a list
                of sequences to their embeddings. Should return a 2D array
                of shape (num_sequences, embedding_dim).
            reference_name (Optional[str]): Optional label for the reference data.
                Defaults to None.
            embedder_name (Optional[str]): Optional label for the embedder used.
                Defaults to None.

        Raises:
            ValueError: If embeddings of the reference data have less than 2 samples.
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
        Compute the FID between embeddings of the input sequences and the reference.

        Args:
            sequences (list[str]): Generated sequences to evaluate.

        Returns:
            MetricResult: Contains the FID score, where lower values indicate
                closer match to the reference distribution.
        """
        seq_embeddings = self.embedder(sequences)
        fid = wasserstein_distance(seq_embeddings, self.reference_embeddings)
        return MetricResult(fid)

    @property
    def name(self) -> str:
        name = "FID"
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
    Compute the Fréchet (Wasserstein-2) distance between two sets of embeddings.

    This is given by:
        ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*(sigma1*sigma2)^{1/2}),
    where mu and sigma are the means and covariance matrices respectively.

    Args:
        e1 (np.ndarray): Embeddings of the first set, shape (N1, D).
        e2 (np.ndarray): Embeddings of the second set, shape (N2, D).

    Returns:
        float: The Fréchet distance. Returns NaN if either set has fewer than two samples.
    """
    if len(e1) < 2 or len(e2) < 2:
        return float("nan")

    mu1, sigma1 = e1.mean(axis=0), np.cov(e1, rowvar=False)
    mu2, sigma2 = e2.mean(axis=0), np.cov(e2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    # Handle numerical issues with imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    dist = max(0.0, dist.item())  # numerical stability

    return dist


class FID(FrechetInceptionDistance):
    pass
