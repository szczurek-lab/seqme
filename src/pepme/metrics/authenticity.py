from collections.abc import Callable
from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pepme.core import Metric, MetricResult


class Authenticity(Metric):
    """
    Authenticity: Proportion of authentic gen samples

    Reference:
        Adlam, Ben, Charles Weill, and Amol Kapoor.
        "Investigating under and overfitting in wasserstein generative adversarial networks." (2019).
        (https://arxiv.org/pdf/1910.14137)
    """

    def __init__(
        self,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        strict: bool = True,
        reference_name: str | None = None,
        embedder_name: str | None = None,
    ):
        """
        Initialize the Authenticity metric.

        Args:
            reference: List of reference sequences representing real data.
            embedder: Function that maps a list of sequences to their embeddings.
                Should return a 2D array of shape (num_sequences, embedding_dim).
            strict: If True, the number of sequences must match the number of
                reference sequences. If False, the number of sequences can vary.
                Default is True.
            reference_name: Optional name for the reference dataset.
            embedder_name: Optional name for the embedder used.
        """
        self.reference = reference
        self.embedder = embedder
        self.strict = strict
        self.reference_name = reference_name
        self.embedder_name = embedder_name

        self.reference_embeddings = self.embedder(self.reference)

        if self.reference_embeddings.shape[0] == 0:
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

        if self.strict and len(sequences) != self.reference_embeddings.shape[0]:
            raise ValueError(
                f"Number of sequences ({len(sequences)}) must match the number of reference sequences ({self.reference_embeddings.shape[0]}). Set `strict=False` to disable this check."
            )

        generated_embeddings = self.embedder(sequences)

        mmd_score = compute_authenticity(
            real_data=self.reference_embeddings,
            synthetic_data=generated_embeddings,
        )

        return MetricResult(value=mmd_score)

    @property
    def name(self) -> str:
        name = "Authenticity"
        if self.embedder_name:
            name += f"@{self.embedder_name}"
        if self.reference_name:
            name += f" ({self.reference_name})"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_authenticity(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    """
    Computes the authenticity metric:
    Fraction of cases where the nearest neighbor distance among real data
    (excluding self) at the closest synthetic neighbor's index is less than
    the distance from each real point to its closest synthetic point.

    Args:
        real_data (np.ndarray or torch.Tensor): [num_real, dim] real data points.
        synthetic_data (np.ndarray or torch.Tensor): [num_synth, dim] synthetic data points.

    Returns:
        float: authenticity score in [0, 1].

    """

    # Compute distances from each real point to its nearest neighbor in real data (excluding self)
    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real_data)
    real_to_real, _ = nbrs_real.kneighbors(real_data)
    real_to_real = real_to_real[:, 1]  # exclude self

    # Compute distances from each real point to its closest synthetic point
    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(synthetic_data)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real_data)
    real_to_synth = real_to_synth.squeeze()
    real_to_synth_args = real_to_synth_args.squeeze()

    # Authenticity computation
    authen = real_to_real[real_to_synth_args] < real_to_synth
    authenticity = np.mean(authen)

    return authenticity


class AuthPCT(Authenticity):
    """
    Authenticity: Proportion of authentic gen samples

    Reference:
        Adlam, Ben, Charles Weill, and Amol Kapoor.
        "Investigating under and overfitting in wasserstein generative adversarial networks." (2019).
        (https://arxiv.org/pdf/1910.14137)
    """

    pass
