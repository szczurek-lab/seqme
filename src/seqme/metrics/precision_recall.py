from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core import Metric, MetricResult


class Precision(Metric):
    """
    Precision metric for evaluating generative models based on k-NN overlap.

    Reference:
        Kynkäänniemi et al., "Improved precision and recall metric for assessing generative models", NeurIPS 2019. (https://arxiv.org/abs/1904.06991)
    """

    def __init__(
        self,
        neighborhood_size: int,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        embedder_name: str | None = None,
        reference_name: str | None = None,
        reference_quantile: float | None = None,
        row_batch_size: int = 10_000,
        col_batch_size: int = 10_000,
        device: str = "cpu",
        strict: bool = True,
    ):
        """
        Initialize the metric.

        Args:
            metric: Which metric to compute, either "precision" or "recall".
            neighborhood_size: Number of nearest neighbors (k) for k-NN graph.
            reference: List of reference sequences to build the reference manifold.
            embedder: Function that maps sequences to embeddings.
            embedder_name: Optional name for the embedder used.
            reference_name: Optional label appended to the metric name.
            reference_quantile: Quantile cutoff for reference radii (defaults to using all).
            row_batch_size: Number of samples per batch when computing distances by rows.
            col_batch_size: Number of samples per batch when computing distances by columns.
            device: Compute device, e.g., "cpu" or "cuda".
            strict: Enforce equal number of eval and reference samples if True.
        """
        self.neighborhood_size = neighborhood_size
        self.embedder = embedder
        self.reference = reference

        self.embedder_name = embedder_name
        self.reference_name = reference_name
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
            reference_embeddings=self.reference_embeddings,
            eval_embeddings=seq_embeddings,
            neighborhood_size=self.neighborhood_size,
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            device=self.device,
            clamp_to_quantile=self.reference_quantile,
        )
        return MetricResult(value)

    @property
    def name(self) -> str:
        name = "Precision"
        if self.embedder_name:
            name += f"@{self.embedder_name}"
        if self.reference_name:
            name += f" ({self.reference_name})"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


class Recall(Metric):
    """
    Recall metric for evaluating generative models based on k-NN overlap.

    Reference:
        Kynkäänniemi et al., "Improved precision and recall metric for assessing generative models", NeurIPS 2019. (https://arxiv.org/abs/1904.06991)
    """

    def __init__(
        self,
        neighborhood_size: int,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        embedder_name: str | None = None,
        reference_name: str | None = None,
        reference_quantile: float | None = None,
        row_batch_size: int = 10_000,
        col_batch_size: int = 10_000,
        device: str = "cpu",
        strict: bool = True,
    ):
        """Initialize the metric.

        Args:
            metric: Which metric to compute, either "precision" or "recall".
            neighborhood_size: Number of nearest neighbors (k) for k-NN graph.
            reference: List of reference sequences to build the reference manifold.
            embedder: Function that maps sequences to embeddings.
            embedder_name: Optional name for the embedder used.
            reference_name: Optional label appended to the metric name.
            reference_quantile: Quantile cutoff for reference radii (defaults to using all).
            row_batch_size: Number of samples per batch when computing distances by rows.
            col_batch_size: Number of samples per batch when computing distances by columns.
            device: Compute device, e.g., "cpu" or "cuda".
            strict: Enforce equal number of eval and reference samples if True.
        """
        self.neighborhood_size = neighborhood_size
        self.embedder = embedder
        self.reference = reference

        self.embedder_name = embedder_name
        self.reference_name = reference_name
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
            reference_embeddings=self.reference_embeddings,
            eval_embeddings=seq_embeddings,
            neighborhood_size=self.neighborhood_size,
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            device=self.device,
            clamp_to_quantile=self.reference_quantile,
        )
        return MetricResult(value)

    @property
    def name(self) -> str:
        name = "Recall"
        if self.embedder_name:
            name += f"@{self.embedder_name}"
        if self.reference_name:
            name += f" ({self.reference_name})"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_recall(
    reference_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    neighborhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
    device: str,
    clamp_to_quantile: float | None = None,
) -> float:
    """Evaluate recall: fraction of reference manifold covered by eval embeddings.

    Args:
        reference_embeddings: Array of reference points, shape [N_ref, D].
        eval_embeddings: Array of evaluation points, shape [N_eval, D].
        neighborhood_size: Number of neighbors (k) in k-NN.
        row_batch_size: Batch size for eval points when computing distances.
        col_batch_size: Batch size for reference points when computing distances.
        device: Compute device, e.g., "cpu" or "cuda".
        clamp_to_quantile: Quantile cutoff for local radii in reference manifold.

    Returns:
        Recall value (float).
    """
    eval_manifold = ManifoldEstimator(
        eval_embeddings,
        neighborhood_size=neighborhood_size,
        clamp_to_quantile=clamp_to_quantile,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        device=device,
    )
    return eval_manifold.evaluate(reference_embeddings).mean()


def compute_precision(
    reference_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    neighborhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
    device: str,
    clamp_to_quantile: float | None = None,
) -> float:
    """Evaluate precision: fraction of eval points lying in reference manifold.

    Args:
        reference_embeddings: Array of reference points, shape [N_ref, D].
        eval_embeddings: Array of evaluation points, shape [N_eval, D].
        neighborhood_size: Number of neighbors (k) in k-NN.
        row_batch_size: Batch size for reference points when computing distances.
        col_batch_size: Batch size for eval points when computing distances.
        device: Compute device, e.g., "cpu" or "cuda".
        clamp_to_quantile: Quantile cutoff for local radii in reference manifold.

    Returns:
        Precision value (float).
    """
    reference_manifold = ManifoldEstimator(
        reference_embeddings,
        neighborhood_size=neighborhood_size,
        clamp_to_quantile=clamp_to_quantile,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        device=device,
    )
    return reference_manifold.evaluate(eval_embeddings).mean()


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

    def evaluate(
        self,
        eval_features: np.ndarray,
        return_realism: bool = False,
        return_neighbors: bool = False,
    ):
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
        realism_scores = np.zeros(n_eval, dtype=np.float32)
        neighbor_indices = np.zeros(n_eval, dtype=np.int32)

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
            realism_scores[row_start:row_end] = np.max(
                self.local_radii[:, 0] / (distances[: row_end - row_start] + self.eps),
                axis=1,
            )
            neighbor_indices[row_start:row_end] = np.argmin(distances[: row_end - row_start], axis=1)

        results = [on_manifold]
        if return_realism:
            results.append(realism_scores)
        if return_neighbors:
            results.append(neighbor_indices)

        return tuple(results) if len(results) > 1 else on_manifold


def pairwise_euclidean_distances(
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of vectors.

    Returns numpy array of shape [x1.size(0), x2.size(0)].
    """
    return torch.cdist(x1, x2).cpu().numpy()
