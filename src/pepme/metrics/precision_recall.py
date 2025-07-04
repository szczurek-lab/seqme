from typing import Callable, Literal, Optional

import numpy as np
import torch

from pepme.core import Metric, MetricResult


class ImprovedPrecisionRecall(Metric):
    """
    Improved Precision and Recall metric for evaluating generative models.

    Reference:
        Kynkäänniemi et al., "Improved precision and recall metric for assessing generative models", NeurIPS 2019.
    """

    def __init__(
        self,
        metric: Literal["precision", "recall"],
        neighborhood_size: int,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        reference_name: Optional[str] = None,
        clamp_to_percentile: Optional[float] = None,
        row_batch_size: int = 10_000,
        col_batch_size: int = 10_000,
        device: Literal["cpu", "cuda"] = "cpu",
        strict: bool = True,
    ):
        """
        Args:
            metric: Metric to compute.
            neighborhood_size: Neighborhood size (k) to consider for k-NN.
            reference: List of reference sequences.
            embedder: Callable that converts sequences to embeddings.
            reference_name: optional label; appended to the metric name.
            clamp_to_percentile: Only use a subset in the reference dataset (a percentile of ascending radiis distance).
            row_batch_size: Batch size for rows during pairwise distance computations.
            col_batch_size: Batch size for columns during pairwise distance computations.
            device: Device to use for distance computations ("cpu" or "cuda").
            strict: If True, do not allow different number of sequences in reference and evaluation embeddings as recommended by the papers authors. If False, allow different numbers. Default is True.
        """

        self.metric = metric
        self.neighborhood_size = neighborhood_size
        self.embedder = embedder
        self.reference = reference

        self.reference_name = reference_name
        self.clamp_to_percentile = clamp_to_percentile
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.device = device
        self.strict = strict

        self.reference_embeddings = self.embedder(self.reference)

        self.reference_embeddings = self.embedder(self.reference)
        if self.reference_embeddings.shape[0] < 2:
            raise ValueError("Reference embeddings must contain at least two samples.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Calculate improved precision and recall scores for given sequences.

        Args:
            sequences: List of sequences to evaluate.

        Returns:
            MetricResult containing the precision score.
        """

        seq_embeddings = self.embedder(sequences)

        if seq_embeddings.shape[0] == 0:
            raise ValueError("No sequences provided for evaluation.")

        if self.strict:
            if seq_embeddings.shape[0] != self.reference_embeddings.shape[0]:
                raise ValueError(
                    f"Number of sequences ({seq_embeddings.shape[0]}) must match "
                    f"number of reference embeddings ({self.reference_embeddings.shape[0]}). Set `strict=False` to disable this check."
                )

        precision, recall = compute_precision_recall(
            reference_embeddings=self.reference_embeddings,
            eval_embeddings=seq_embeddings,
            neighborhood_size=self.neighborhood_size,
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            device=self.device,
            clamp_to_percentile=self.clamp_to_percentile,
        )

        return MetricResult(value=precision if self.metric == "precision" else recall)

    @property
    def name(self) -> str:
        name = "Precision" if self.metric == "precision" else "Recall"
        if self.reference_name:
            name += f" ({self.reference_name})"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_precision_recall(
    reference_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    neighborhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
    device: Literal["cpu", "cuda"],
    clamp_to_percentile: Optional[float] = None,
) -> tuple[float, float]:
    """
    Calculate k-NN based precision and recall metrics.

    Args:
        reference_features: Reference embeddings (numpy array).
        eval_features: Evaluation embeddings (numpy array).
        neighborhood_size: Neighborhood size (k) to consider for k-NN.
        row_batch_size: Batch size for rows during pairwise distance computations.
        col_batch_size: Batch size for columns during pairwise distance computations.
        device: Device to use for distance computations ("cpu" or "cuda").
        clamp_to_percentile: Only use a subset in the reference dataset (a percentile of ascending radiis distance).

    Returns:
        Tuple with precision and recall.
    """
    reference_manifold = ManifoldEstimator(
        reference_embeddings,
        neighborhood_size=neighborhood_size,
        clamp_to_percentile=clamp_to_percentile,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        device=device,
    )
    eval_manifold = ManifoldEstimator(
        eval_embeddings,
        neighborhood_size=neighborhood_size,
        clamp_to_percentile=clamp_to_percentile,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        device=device,
    )

    precision = reference_manifold.evaluate(eval_embeddings).mean(axis=0)
    recall = eval_manifold.evaluate(reference_embeddings).mean(axis=0)

    return precision, recall


class ManifoldEstimator:
    """
    Estimates local manifolds via k-NN distances and evaluates points' inclusion in these manifolds.
    """

    def __init__(
        self,
        features: np.ndarray,
        neighborhood_size: int,
        clamp_to_percentile: Optional[float] = None,
        row_batch_size: int = 10_000,
        col_batch_size: int = 10_000,
        eps: float = 1e-5,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """
        Args:
            features: Reference features, shape [num_points, dim].
            neighborhood_size: Size of k for k-NN neighborhoods.
            clamp_to_percentile: Only use a subset in the reference dataset (a percentile of ascending radiis distance).
            row_batch_size: Batch size for rows when computing distances.
            col_batch_size: Batch size for columns when computing distances.
            eps: Small epsilon to avoid division by zero.
        """

        self.neighborhood_size = neighborhood_size
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.features = features
        self.device = device

        self._compute_local_radii(clamp_to_percentile)

    def _compute_local_radii(self, clamp_to_percentile: Optional[float] = None):
        num_points = self.features.shape[0]
        self.local_radii = np.zeros((num_points, 1), dtype=np.float32)

        distance_batch = np.zeros((self.row_batch_size, num_points), dtype=np.float32)
        k_indices = np.arange(self.neighborhood_size + 1, dtype=np.int32)

        for row_start in range(0, num_points, self.row_batch_size):
            row_end = min(row_start + self.row_batch_size, num_points)
            row_batch = self.features[row_start:row_end]

            for col_start in range(0, num_points, self.col_batch_size):
                col_end = min(col_start + self.col_batch_size, num_points)
                col_batch = self.features[col_start:col_end]

                distances = pairwise_squared_euclidean_distances(
                    torch.from_numpy(row_batch).to(self.device, dtype=torch.float),
                    torch.from_numpy(col_batch).to(self.device, dtype=torch.float),
                )
                distance_batch[: row_end - row_start, col_start:col_end] = distances

            # Extract k-th nearest neighbor distances for each neighborhood size
            self.local_radii[row_start:row_end] = np.partition(
                distance_batch[: row_end - row_start], k_indices, axis=1
            )[:, [self.neighborhood_size]]

        if clamp_to_percentile is not None:
            max_dist = np.percentile(self.local_radii, clamp_to_percentile, axis=0)
            self.local_radii[self.local_radii > max_dist] = 0

    def evaluate(
        self,
        eval_features: np.ndarray,
        return_realism: bool = False,
        return_neighbors: bool = False,
    ):
        """
        Evaluate how many eval_features fall inside the local manifolds.

        Args:
            eval_features: Features to evaluate, shape [num_eval, dim].
            return_realism: If True, also return realism scores.
            return_neighbors: If True, also return nearest neighbor indices.

        Returns:
            inside_manifold: Binary matrix [num_eval, 1]
            Optionally realism scores and nearest neighbors.
        """
        num_eval = eval_features.shape[0]
        num_ref = self.local_radii.shape[0]

        inside_manifold = np.zeros((num_eval, 1), dtype=bool)

        distance_batch = np.zeros((self.row_batch_size, num_ref), dtype=np.float32)
        realism_scores = np.zeros(num_eval, dtype=np.float32)
        nearest_indices = np.zeros(num_eval, dtype=np.int32)

        for row_start in range(0, num_eval, self.row_batch_size):
            row_end = min(row_start + self.row_batch_size, num_eval)
            eval_batch = eval_features[row_start:row_end]

            for col_start in range(0, num_ref, self.col_batch_size):
                col_end = min(col_start + self.col_batch_size, num_ref)
                ref_batch = self.features[col_start:col_end]

                distances = pairwise_squared_euclidean_distances(
                    torch.from_numpy(eval_batch).to(
                        device=self.device, dtype=torch.float
                    ),
                    torch.from_numpy(ref_batch).to(
                        device=self.device, dtype=torch.float
                    ),
                )
                distance_batch[: row_end - row_start, col_start:col_end] = distances

            # Check if points lie within manifold radius for any reference point
            inside_manifold[row_start:row_end] = np.any(
                distance_batch[: row_end - row_start, :, None] <= self.local_radii,
                axis=1,
            )

            # Realism score = max over ref points of D[:,0] / (distance + eps)
            realism_scores[row_start:row_end] = np.max(
                self.local_radii[:, 0]
                / (distance_batch[: row_end - row_start, :] + self.eps),
                axis=1,
            )

            # Nearest neighbor index for each eval point
            nearest_indices[row_start:row_end] = np.argmin(
                distance_batch[: row_end - row_start, :], axis=1
            )

        if return_realism and return_neighbors:
            return inside_manifold, realism_scores, nearest_indices
        if return_realism:
            return inside_manifold, realism_scores
        if return_neighbors:
            return inside_manifold, nearest_indices

        return inside_manifold


def pairwise_squared_euclidean_distances(
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between two batches of vectors.

    Args:
        x1: Tensor of shape [batch_u, dim]
        x2: Tensor of shape [batch_v, dim]

    Returns:
        Tensor of shape [batch_u, batch_v] with pairwise squared distances.
    """
    norm_u = torch.sum(x1**2, dim=1, keepdim=True)  # [batch_u, 1]
    norm_v = torch.sum(x2**2, dim=1, keepdim=True)  # [batch_v, 1]

    distances = norm_u - 2 * torch.matmul(x1, x2.T) + norm_v.T
    return torch.clamp(distances, min=0.0).cpu().numpy()


def pairwise_squared_euclidean_distances2(
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> np.ndarray:
    return torch.cdist(x1, x2).cpu().numpy()
