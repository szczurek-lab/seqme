from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core.base import Metric, MetricResult


class Precision(Metric):
    """
    Evaluates how realistic synthetic samples are compared to reference data.

    Computes the Improved Precision metric [1], which measures the realism of synthetic samples
    by assessing how closely each sample aligns with the reference data in the embedding space.
    This metric quantifies **fidelity**, i.e., the degree to which synthetic samples resemble
    the true data distribution. To achieve this, the reference manifold is approximated using
    nearest-neighbor balls.

    Its value ranges from 0 to 1, representing the fraction of synthetic samples that lie on the reference manifold and are
    therefore considered realistic.

    References:
        [1] Kynkäänniemi et al., "Improved precision and recall metric for assessing generative models", NeurIPS 2019
            (https://arxiv.org/abs/1904.06991)
    """

    def __init__(
        self,
        n_neighbors: int,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        batch_size: int = 256,
        device: str = "cpu",
        strict: bool = True,
        name: str = "Precision",
    ):
        """
        Initialize the Improved Precision metric.

        Constructs the reference manifold using the provided sequences and prepares the metric
        for evaluation. The reference manifold is approximated using nearest-neighbor balls,
        with radii determined by the specified number of neighbors.

        Args:
            n_neighbors: Number of nearest neighbors used to define the radii of the
                nearest-neighbor balls. More neighbors result in larger radii.
            reference: List of reference sequences used to build the reference manifold.
            embedder: Function mapping sequences to embeddings.
            batch_size: Number of samples per batch when computing distances.
            device: Compute device, e.g., ``"cpu"`` or ``"cuda"``.
            strict: If True, enforces an equal number of evaluation and reference samples.
            name: Metric name.

        Raises:
            ValueError: If `n_neighbors` < 1.
            ValueError: If `reference` contains fewer than 1 sequence after embedding.
        """
        self.embedder = embedder

        self.batch_size = batch_size
        self.device = device
        self.strict = strict
        self._name = name

        if n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than 0.")

        reference_embeddings = self.embedder(reference)
        if reference_embeddings.shape[0] < 1:
            raise ValueError("Reference embeddings must contain at least one samples.")

        self.reference_embeddings = torch.from_numpy(reference_embeddings).to(self.device)
        self.ref_knn_radii = _compute_knn_radii(
            self.reference_embeddings,
            n_neighbors,
            self.batch_size,
        )

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the Improved Precision of the sequences.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Improved Precision.
        """
        embeddings = self.embedder(sequences)

        if self.strict and (embeddings.shape[0] != self.reference_embeddings.shape[0]):
            raise ValueError(
                f"Number of sequences ({embeddings.shape[0]}) must match number of reference embeddings ({self.reference_embeddings.shape[0]}). Set strict=False to disable this check."
            )

        value = _compute_precision_or_recall(
            points=torch.from_numpy(embeddings).to(self.device),
            ball_positions=self.reference_embeddings,
            ball_radii=self.ref_knn_radii,
            batch_size=self.batch_size,
        )

        return MetricResult(value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


class Recall(Metric):
    """
    Evaluates how well the reference data is covered by the generated sequences.

    Computes the Improved Recall metric [1], which measures the fraction of reference embeddings
    that lie on or near the manifold defined by the generated sequence embeddings. This metric
    quantifies **coverage**, i.e., the degree to which the generated samples represent the
    full reference distribution. To achieve this, the reference manifold is approximated using
    nearest-neighbor balls.

    Its value ranges from 0 to 1, representing the fraction of reference embeddings that are
    effectively captured by the generated data.

    References:
        [1] Kynkäänniemi et al., "Improved precision and recall metric for assessing generative models", NeurIPS 2019
            (https://arxiv.org/abs/1904.06991)
    """

    def __init__(
        self,
        n_neighbors: int,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        batch_size: int = 256,
        device: str = "cpu",
        strict: bool = True,
        name: str = "Recall",
    ):
        """
        Initialize the Improved Recall metric.

        Args:
            n_neighbors: Number of nearest neighbors used to define the radii of the
                nearest-neighbor balls. More neighbors result in larger radii.
            reference: List of reference sequences used to build the reference manifold.
            embedder: Function mapping sequences to embeddings.
            batch_size: Number of samples per batch when computing distances.
            device: Compute device, e.g., ``"cpu"`` or ``"cuda"``.
            strict: If True, enforces an equal number of evaluation and reference samples.
            name: Metric name.

        Raises:
            ValueError: If `n_neighbors` < 1.
            ValueError: If `reference` contains fewer than 1 sequence after embedding.
        """
        self.n_neighbors = n_neighbors
        self.embedder = embedder
        self.reference = reference
        self._name = name

        self.batch_size = batch_size
        self.device = device
        self.strict = strict

        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than 0.")

        self.reference_embeddings = torch.from_numpy(self.embedder(self.reference))

        if self.reference_embeddings.shape[0] < 1:
            raise ValueError("Reference embeddings must contain at least one samples.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the Improved Recall of the sequences.

        Args:
            sequences: List of sequences to evaluate.

        Returns:
            MetricResult: Improved Recall.
        """
        embeddings = self.embedder(sequences)

        if self.strict and (embeddings.shape[0] != self.reference_embeddings.shape[0]):
            raise ValueError(
                f"Number of sequences ({embeddings.shape[0]}) must match number of reference embeddings ({self.reference_embeddings.shape[0]}). Set strict=False to disable this check."
            )

        seq_embeddings = torch.from_numpy(embeddings).to(self.device)

        seq_knn_radii = _compute_knn_radii(
            seq_embeddings,
            self.n_neighbors,
            self.batch_size,
        )

        value = _compute_precision_or_recall(
            points=self.reference_embeddings,
            ball_positions=seq_embeddings,
            ball_radii=seq_knn_radii,
            batch_size=self.batch_size,
        )

        return MetricResult(value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def _compute_knn_radii(embeddings: torch.Tensor, max_k: int, batch_size: int) -> torch.Tensor:
    N = embeddings.shape[0]
    k = min(max_k, N - 1)

    k_dists = torch.empty(N, device=embeddings.device, dtype=embeddings.dtype)

    for start in range(0, N, batch_size):
        end = start + batch_size

        dists = torch.cdist(embeddings[start:end], embeddings)
        k_dists[start:end] = dists.kthvalue(k + 1, dim=1).values

    return k_dists


def _compute_precision_or_recall(
    points: torch.Tensor,
    ball_positions: torch.Tensor,
    ball_radii: torch.Tensor,
    batch_size: int,
) -> float:
    M = points.shape[0]

    total_in_manifold = 0.0
    for start in range(0, M, batch_size):
        end = start + batch_size

        dists = torch.cdist(points[start:end], ball_positions)  # [b, N]
        threshold = ball_radii[None]  # threshold: [1, N] broadcast to [b, N]

        is_in = (dists <= threshold).any(dim=1)
        total_in_manifold += int(is_in.sum().item())

    return total_in_manifold / M
