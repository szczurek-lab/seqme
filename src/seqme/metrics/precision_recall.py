from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core import Metric, MetricResult


class Precision(Metric):
    """
    Precision metric for evaluating generative models based on k-NN overlap.

    The metric approximates a manifold from the reference embeddings and computes the fraction of sequence embeddings on this manifold.

    Reference:
        Kynkäänniemi et al., "Improved precision and recall metric for assessing generative models", NeurIPS 2019. (https://arxiv.org/abs/1904.06991)
    """

    def __init__(
        self,
        neighborhood_size: int,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        reference_quantile: float | None = None,
        row_batch_size: int = 10_000,
        col_batch_size: int = 10_000,
        device: str = "cpu",
        strict: bool = True,
        name: str = "Precision",
    ):
        """
        Initialize the metric.

        Args:
            neighborhood_size: Number of nearest neighbors (k) for k-NN graph.
            reference: List of reference sequences to build the reference manifold.
            embedder: Function that maps sequences to embeddings.
            reference_quantile: Quantile cutoff for reference radii (defaults to using all).
            row_batch_size: Number of samples per batch when computing distances by rows.
            col_batch_size: Number of samples per batch when computing distances by columns.
            device: Compute device, e.g., "cpu" or "cuda".
            strict: Enforce equal number of eval and reference samples if True.
            name: Metric name
        """
        self.neighborhood_size = neighborhood_size
        self.embedder = embedder
        self.reference = reference

        self.reference_quantile = reference_quantile
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.device = device
        self.strict = strict
        self._name = name

        if reference_quantile is not None:
            if reference_quantile < 0 or reference_quantile > 1:
                raise ValueError("reference_quantile must be between 0 and 1.")

        if self.neighborhood_size < 1:
            raise ValueError("neighborhood_size must be greater than 0.")

        self.reference_embeddings = self.embedder(self.reference)
        if self.reference_embeddings.shape[0] < 1:
            raise ValueError("Reference embeddings must contain at least one samples.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute precision or recall for the given evaluation sequences.

        Args:
            sequences: List of sequences to evaluate.

        Returns:
            MetricResult containing the computed score.
        """
        seq_embeddings = self.embedder(sequences)

        if self.strict and seq_embeddings.shape[0] != self.reference_embeddings.shape[0]:
            raise ValueError(
                f"Number of sequences ({seq_embeddings.shape[0]}) must match number of reference embeddings ({self.reference_embeddings.shape[0]}). Set strict=False to disable this check."
            )

        value = compute_precision(
            real_embeddings=self.reference_embeddings,
            generated_embeddings=seq_embeddings,
            neighborhood_size=self.neighborhood_size,
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            device=self.device,
            clamp_to_quantile=self.reference_quantile,
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
    Recall metric for evaluating generative models based on k-NN overlap.

    The metric approximates a manifold from the sequence embeddings and computes the fraction of reference embeddings on this manifold.

    Reference:
        Kynkäänniemi et al., "Improved precision and recall metric for assessing generative models", NeurIPS 2019. (https://arxiv.org/abs/1904.06991)
    """

    def __init__(
        self,
        neighborhood_size: int,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        reference_quantile: float | None = None,
        row_batch_size: int = 10_000,
        col_batch_size: int = 10_000,
        device: str = "cpu",
        strict: bool = True,
        name: str = "Recall",
    ):
        """Initialize the metric.

        Args:
            neighborhood_size: Number of nearest neighbors (k) for k-NN graph.
            reference: List of reference sequences to build the reference manifold.
            embedder: Function that maps sequences to embeddings.
            reference_quantile: Quantile cutoff for reference radii (defaults to using all).
            row_batch_size: Number of samples per batch when computing distances by rows.
            col_batch_size: Number of samples per batch when computing distances by columns.
            device: Compute device, e.g., "cpu" or "cuda".
            strict: Enforce equal number of eval and reference samples if True.
            name: Metric name.
        """
        self.neighborhood_size = neighborhood_size
        self.embedder = embedder
        self.reference = reference
        self._name = name

        self.reference_quantile = reference_quantile
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.device = device
        self.strict = strict

        if reference_quantile is not None:
            if reference_quantile < 0 or reference_quantile > 1:
                raise ValueError("reference_quantile must be between 0 and 1.")

        if self.neighborhood_size < 1:
            raise ValueError("neighborhood_size must be greater than 0.")

        self.reference_embeddings = self.embedder(self.reference)
        if self.reference_embeddings.shape[0] < 1:
            raise ValueError("Reference embeddings must contain at least one samples.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute precision or recall for the given evaluation sequences.

        Args:
            sequences: List of sequences to evaluate.

        Returns:
            MetricResult containing the computed score.
        """
        seq_embeddings = self.embedder(sequences)

        if self.strict and seq_embeddings.shape[0] != self.reference_embeddings.shape[0]:
            raise ValueError(
                f"Number of sequences ({seq_embeddings.shape[0]}) must match number of reference embeddings ({self.reference_embeddings.shape[0]}). Set strict=False to disable this check."
            )

        value = compute_recall(
            real_embeddings=self.reference_embeddings,
            generated_embeddings=seq_embeddings,
            neighborhood_size=self.neighborhood_size,
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            device=self.device,
            clamp_to_quantile=self.reference_quantile,
        )
        return MetricResult(value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_recall(
    real_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    neighborhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
    device: str,
    clamp_to_quantile: float | None = None,
) -> float:
    """Evaluate recall: fraction of reference manifold covered by eval embeddings.

    Args:
        real_embeddings: Embeddings of the real data. Array of shape [N_real, D].
        generated_embeddings: Embeddings of the generated data. Array of shape [N_gen, D].
        neighborhood_size: Number of neighbors (k) in k-NN.
        row_batch_size: Batch size for eval points when computing distances.
        col_batch_size: Batch size for reference points when computing distances.
        device: Compute device, e.g., "cpu" or "cuda".
        clamp_to_quantile: Quantile cutoff for local radii in reference manifold.

    Returns:
        Recall value (float).
    """
    generated_manifold = ManifoldEstimator(
        generated_embeddings,
        neighborhood_size=neighborhood_size,
        clamp_to_quantile=clamp_to_quantile,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        device=device,
    )
    return generated_manifold.evaluate(real_embeddings).mean().item()


def compute_precision(
    real_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    neighborhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
    device: str,
    clamp_to_quantile: float | None = None,
) -> float:
    """Evaluate precision: fraction of eval points lying in reference manifold.

    Args:
        real_embeddings: Embeddings of the real data. Array of shape [N_real, D].
        generated_embeddings: Embeddings of the generated data. Array of shape [N_gen, D].
        neighborhood_size: Number of neighbors (k) in k-NN.
        row_batch_size: Batch size for reference points when computing distances.
        col_batch_size: Batch size for eval points when computing distances.
        device: Compute device, e.g., "cpu" or "cuda".
        clamp_to_quantile: Quantile cutoff for local radii in reference manifold.

    Returns:
        Precision value (float).
    """
    real_manifold = ManifoldEstimator(
        real_embeddings,
        neighborhood_size=neighborhood_size,
        clamp_to_quantile=clamp_to_quantile,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        device=device,
    )
    return real_manifold.evaluate(generated_embeddings).mean().item()


class ManifoldEstimator:
    """Estimates local manifold radii and evaluates sample inclusion via k-NN distances."""

    def __init__(
        self,
        features: np.ndarray,
        neighborhood_size: int,
        clamp_to_quantile: float | None = None,
        row_batch_size: int = 10_000,
        col_batch_size: int = 10_000,
        eps: float = 1e-5,
        device: str = "cpu",
    ):
        """
        Estimates the local manifold.

        Args:
            features: Data points to build the manifold, shape [N, D].
            neighborhood_size: k in k-NN.
            clamp_to_quantile: Quantile cutoff for radii.
            row_batch_size: Batch size for rows in distance calc.
            col_batch_size: Batch size for cols in distance calc.
            eps: Small constant to avoid division by zero in realism.
            device: Compute device, e.g., "cpu" or "cuda".
        """
        self.neighborhood_size = neighborhood_size
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.features = features
        self.device = device

        self._compute_local_radii(clamp_to_quantile)

    def _compute_local_radii(self, clamp_to_quantile: float | None = None):
        num_points = self.features.shape[0]
        self.local_radii = np.zeros((num_points, 1), dtype=np.float32)

        distance_batch = np.zeros((self.row_batch_size, num_points), dtype=np.float32)
        k_idx = self.neighborhood_size

        for row_start in range(0, num_points, self.row_batch_size):
            row_end = min(row_start + self.row_batch_size, num_points)
            row_batch = self.features[row_start:row_end]

            for col_start in range(0, num_points, self.col_batch_size):
                col_end = min(col_start + self.col_batch_size, num_points)
                col_batch = self.features[col_start:col_end]

                dist = pairwise_euclidean_distances(
                    torch.from_numpy(row_batch).to(self.device, dtype=torch.float),
                    torch.from_numpy(col_batch).to(self.device, dtype=torch.float),
                )
                distance_batch[: row_end - row_start, col_start:col_end] = dist

            # k-th neighbor distance per point
            self.local_radii[row_start:row_end] = np.partition(distance_batch[: row_end - row_start], k_idx, axis=1)[
                :, [k_idx]
            ]

        if clamp_to_quantile is not None:
            max_dist = np.quantile(self.local_radii, clamp_to_quantile)
            self.local_radii = np.where(self.local_radii <= max_dist, self.local_radii, 0)

    def evaluate(self, eval_features: np.ndarray):
        """
        Determine which evaluation points lie within the manifold.

        Args:
            eval_features: Points to evaluate, shape [M, D].
            return_realism: If True, also return realism scores per point.
            return_neighbors: If True, also return nearest reference indices.

        Returns:
            inside: Boolean array [M, 1] indicating inclusion.
            Optionally realism and/or nearest neighbor indices.
        """
        n_eval = eval_features.shape[0]
        n_ref = self.local_radii.shape[0]

        on_manifold = np.zeros((n_eval, 1), dtype=bool)
        distances = np.zeros((self.row_batch_size, n_ref), dtype=np.float32)

        for row_start in range(0, n_eval, self.row_batch_size):
            row_end = min(row_start + self.row_batch_size, n_eval)
            eval_batch = eval_features[row_start:row_end]

            for col_start in range(0, n_ref, self.col_batch_size):
                col_end = min(col_start + self.col_batch_size, n_ref)
                ref = self.features[col_start:col_end]

                dist = pairwise_euclidean_distances(
                    torch.from_numpy(eval_batch).to(device=self.device, dtype=torch.float),
                    torch.from_numpy(ref).to(device=self.device, dtype=torch.float),
                )
                distances[: row_end - row_start, col_start:col_end] = dist

            on_manifold[row_start:row_end] = np.any(
                distances[: row_end - row_start, :, None] <= self.local_radii,
                axis=1,
            )

        return on_manifold


def pairwise_euclidean_distances(x1: torch.Tensor, x2: torch.Tensor) -> np.ndarray:
    return torch.cdist(x1, x2).cpu().numpy()
