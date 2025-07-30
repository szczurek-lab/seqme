from collections.abc import Callable
from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

from seqme.core import Metric, MetricResult


class Authenticity(Metric):
    """
    Proportion of authentic generated samples.

    Reference:
        Alaa et al., "How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models." (2022). (https://arxiv.org/pdf/1910.14137)
    """

    def __init__(
        self,
        train_set: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        embedder_name: str | None = None,
    ):
        """
        Initialize the Authenticity metric.

        Args:
            train_set: List of sequences used to train the generative model.
            embedder: A function that maps a list of sequences to a 2D NumPy array of embeddings.
            embedder_name: Optional name for the embedder used.
        """
        self.train_set = train_set
        self.embedder = embedder
        self.embedder_name = embedder_name

        self.train_set_embeddings = self.embedder(self.train_set)

        if self.train_set_embeddings.shape[0] == 0:
            raise ValueError("Reference embeddings must contain at least one sample.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the authenticity score based on the embeddings of the input sequences and the reference.

        Args:
            sequences: Generated sequences to evaluate.

        Returns:
            MetricResult contains the authenticity score, which represents the proportion of authentic samples.
        """

        if len(sequences) == 0:
            raise ValueError("Sequences must contain at least one sample.")

        embeddings = self.embedder(sequences)

        auth_score = compute_authenticity(
            real_data=self.train_set_embeddings,
            synthetic_data=embeddings,
        )

        return MetricResult(value=auth_score)

    @property
    def name(self) -> str:
        name = "Authenticity"
        if self.embedder_name:
            name += f"@{self.embedder_name}"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_authenticity(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    """
    Computes the authenticity metric, defined as one minus the fraction of synthetic samples closer to training data than any other training sample.

    Args:
        real_data: Embeddings of the real data.
        synthetic_data: Embeddings of the synthetic data.

    Returns:
        Authenticity score in [0, 1].

    """
    knn_real = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(real_data)

    dist_synth_to_real, closest_real_per_synth_idx = knn_real.kneighbors(synthetic_data)
    dist_real_to_real, _ = knn_real.kneighbors()

    auth_mask = dist_synth_to_real > dist_real_to_real[closest_real_per_synth_idx.squeeze(axis=-1)]
    authenticity = np.mean(auth_mask)

    return authenticity


class AuthPct(Authenticity):
    """
    Proportion of authentic generated samples.

    Reference:
        Alaa et al., "How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models." (2022). (https://arxiv.org/pdf/1910.14137)
    """

    pass
