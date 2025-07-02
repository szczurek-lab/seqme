from time import time
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

    # TODO make cache work later
    # _lru_hash: Optional[str] = None
    # _lru_precision: Optional[float] = None
    # _lru_recall: Optional[float] = None

    def __init__(
        self,
        reference: list[str],
        metric: Literal["precision", "recall"],
        embedder: Callable[[list[str]], np.ndarray],
        nhood_size: int = 3,
        row_batch_size: int = 10000,
        col_batch_size: int = 50000,
        num_gpus: int = 1,
        strict: bool = True,  # strict: If True, do not allow different number of sequences in reference and evaluation embeddings as recommended by the paper authors. If False, allow different numbers but warn the user. Default is True.
    ):
        """
        Args:
            reference: List of reference sequences.
            embedder: Callable that converts sequences to embeddings.
            nhood_size: Neighborhood size (k) to consider for k-NN.
            row_batch_size: Batch size for rows during pairwise distance computations.
            col_batch_size: Batch size for columns during pairwise distance computations.
            num_gpus: Number of GPUs to use for distance computations.
        """

        self.reference = reference
        self.embedder = embedder
        self.metric = metric
        self.nhood_size = nhood_size
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.num_gpus = num_gpus
        self.strict = strict

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

        # TODO Make cache work later
        # current_hash = hashlib.sha256(str(sequences).encode()).hexdigest()

        # if (
        #     ImprovedPrecisionRecall._lru_hash == current_hash
        #     and ImprovedPrecisionRecall._lru_precision is not None
        #     and ImprovedPrecisionRecall._lru_recall is not None
        # ):
        #     print("Using cached results for sequences.")
        #     return (
        #         MetricResult(ImprovedPrecisionRecall._lru_precision)
        #         if self.metric == "precision"
        #         else MetricResult(ImprovedPrecisionRecall._lru_recall)
        #     )

        seq_embeddings = self.embedder(sequences)

        if (
            seq_embeddings.shape[0] != self.reference_embeddings.shape[0]
            and self.strict
        ):
            raise ValueError(
                f"Number of sequences ({seq_embeddings.shape[0]}) must match "
                f"number of reference embeddings ({self.reference_embeddings.shape[0]})."
            )

        if seq_embeddings.shape[0] == 0:
            raise ValueError("No sequences provided for evaluation.")

        metrics = self.compute_improved_precision_recall(seq_embeddings)
        precision = metrics["precision"]
        recall = metrics["recall"]

        # ImprovedPrecisionRecall._lru_hash = current_hash
        # ImprovedPrecisionRecall._lru_precision = precision
        # ImprovedPrecisionRecall._lru_recall = recall

        result = MetricResult(value=precision if self.metric == "precision" else recall)
        return result

    @property
    def name(self) -> str:
        if self.metric == "precision":
            return "improved precision"
        elif self.metric == "recall":
            return "improved recall"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"

    def compute_improved_precision_recall(
        self, sequence_embeddings: np.ndarray
    ) -> dict:
        """
        Compute improved precision and recall based on k-NN neighborhoods.

        Args:
            sequence_embeddings: Embeddings of sequences to evaluate.

        Returns:
            Dictionary containing precision and recall arrays.
        """

        metrics = self.knn_precision_recall(
            reference_features=self.reference_embeddings,
            eval_features=sequence_embeddings,
        )
        print("Metrics:", metrics)
        return metrics

    def knn_precision_recall(
        self, reference_features: np.ndarray, eval_features: np.ndarray
    ) -> dict:
        """
        Calculate k-NN based precision and recall metrics.

        Args:
            reference_features: Reference embeddings (numpy array).
            eval_features: Evaluation embeddings (numpy array).

        Returns:
            Dictionary with keys "precision" and "recall".
        """
        distance_block = DistanceBlock(num_gpus=self.num_gpus)

        reference_manifold = ManifoldEstimator(
            distance_block,
            reference_features,
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            nhood_size=self.nhood_size,
        )
        eval_manifold = ManifoldEstimator(
            distance_block,
            eval_features,
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            nhood_size=self.nhood_size,
        )

        print(
            f"Evaluating k-NN precision and recall with {reference_features.shape[0]} reference samples..."
        )
        start = time()

        precision = reference_manifold.evaluate(eval_features).mean(axis=0)
        recall = eval_manifold.evaluate(reference_features).mean(axis=0)

        print(f"Evaluation completed in {time() - start:.2f}s")

        return {"precision": precision, "recall": recall}


def get_random_samples(array: np.ndarray, size: int) -> np.ndarray:
    """Select random samples from array without replacement."""
    indices = np.random.choice(array.shape[0], size=size, replace=False)
    return array[indices]


def batch_pairwise_distances(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances between two batches of vectors.

    Args:
        U: Tensor of shape [batch_u, dim]
        V: Tensor of shape [batch_v, dim]

    Returns:
        Tensor of shape [batch_u, batch_v] with pairwise squared distances.
    """
    norm_u = torch.sum(U**2, dim=1, keepdim=True)  # [batch_u, 1]
    norm_v = torch.sum(V**2, dim=1, keepdim=True)  # [batch_v, 1]

    distances = norm_u - 2 * torch.matmul(U, V.T) + norm_v.T
    return torch.clamp(distances, min=0.0)


class DistanceBlock:
    """
    Efficient pairwise distance computation with optional multi-GPU support.
    """

    def __init__(self, num_gpus: int = 1):
        self.num_gpus = num_gpus
        self.devices = (
            [f"cuda:{i}" for i in range(num_gpus)]
            if num_gpus > 0 and torch.cuda.is_available()
            else ["cpu"]
        )

    def pairwise_distances(self, U: torch.Tensor, V: torch.Tensor) -> np.ndarray:
        """
        Compute pairwise distances between two batches, splitting over GPUs if available.

        Args:
            U: Tensor [batch_u, dim]
            V: Tensor [batch_v, dim]

        Returns:
            distances as numpy array [batch_u, batch_v]
        """
        if self.num_gpus <= 1 or not torch.cuda.is_available():
            D = batch_pairwise_distances(U, V)
            return D.cpu().numpy()

        # Split V batch-wise across GPUs
        V_chunks = torch.chunk(V, self.num_gpus, dim=0)
        results = []

        for i, V_chunk in enumerate(V_chunks):
            device = self.devices[i]
            U_device = U.to(device)
            V_device = V_chunk.to(device)

            with torch.cuda.device(device):
                D_part = batch_pairwise_distances(U_device, V_device)
                results.append(D_part.cpu())

        return torch.cat(results, dim=1).numpy()


class ManifoldEstimator:
    """
    Estimates local manifolds via k-NN distances and evaluates points' inclusion in these manifolds.
    """

    def __init__(
        self,
        distance_block: DistanceBlock,
        features: np.ndarray,
        row_batch_size: int = 25000,
        col_batch_size: int = 50000,
        nhood_size: int = 3,
        clamp_to_percentile: Optional[float] = None,
        eps: float = 1e-5,
    ):
        """
        Args:
            distance_block: Instance of DistanceBlock for distance calculations.
            features: Reference features, shape [num_points, dim].
            row_batch_size: Batch size for rows when computing distances.
            col_batch_size: Batch size for columns when computing distances.
            nhood_size: Size of k for k-NN neighborhoods.
            clamp_to_percentile: Percentile threshold to clamp large distances.
            eps: Small epsilon to avoid division by zero.
        """

        self.nhood_size = nhood_size
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._features = features
        self._distance_block = distance_block

        self._compute_local_radii(clamp_to_percentile)

    def _compute_local_radii(self, clamp_to_percentile: Optional[float]) -> None:
        num_points = self._features.shape[0]
        self.D = np.zeros((num_points, 1), dtype=np.float32)
        distance_batch = np.zeros((self.row_batch_size, num_points), dtype=np.float32)

        k_indices = np.arange(self.nhood_size + 1, dtype=np.int32)

        for row_start in range(0, num_points, self.row_batch_size):
            row_end = min(row_start + self.row_batch_size, num_points)
            row_batch = self._features[row_start:row_end]

            for col_start in range(0, num_points, self.col_batch_size):
                col_end = min(col_start + self.col_batch_size, num_points)
                col_batch = self._features[col_start:col_end]

                distances = self._distance_block.pairwise_distances(
                    torch.from_numpy(row_batch).float(),
                    torch.from_numpy(col_batch).float(),
                )
                distance_batch[: row_end - row_start, col_start:col_end] = distances

            # Extract k-th nearest neighbor distances for each neighborhood size
            self.D[row_start:row_end] = np.partition(
                distance_batch[: row_end - row_start], k_indices, axis=1
            )[:, [self.nhood_size]]

        if clamp_to_percentile is not None:
            max_dist = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_dist] = 0

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
            batch_predictions: Binary matrix [num_eval, 1]
            Optionally realism scores and nearest neighbors.
        """
        num_eval = eval_features.shape[0]
        num_ref = self.D.shape[0]

        distance_batch = np.zeros((self.row_batch_size, num_ref), dtype=np.float32)
        batch_predictions = np.zeros((num_eval, 1), dtype=np.int32)
        realism_scores = np.zeros(num_eval, dtype=np.float32)
        nearest_indices = np.zeros(num_eval, dtype=np.int32)

        for row_start in range(0, num_eval, self.row_batch_size):
            row_end = min(row_start + self.row_batch_size, num_eval)
            eval_batch = eval_features[row_start:row_end]

            for col_start in range(0, num_ref, self.col_batch_size):
                col_end = min(col_start + self.col_batch_size, num_ref)
                ref_batch = self._features[col_start:col_end]

                distances = self._distance_block.pairwise_distances(
                    torch.from_numpy(eval_batch).float(),
                    torch.from_numpy(ref_batch).float(),
                )
                distance_batch[: row_end - row_start, col_start:col_end] = distances

            # Check if points lie within manifold radius for any reference point
            in_manifold = distance_batch[: row_end - row_start, :, None] <= self.D
            batch_predictions[row_start:row_end] = np.any(in_manifold, axis=1).astype(
                np.int32
            )

            # Realism score = max over ref points of D[:,0] / (distance + eps)
            realism_scores[row_start:row_end] = np.max(
                self.D[:, 0] / (distance_batch[: row_end - row_start, :] + self.eps),
                axis=1,
            )

            # Nearest neighbor index for each eval point
            nearest_indices[row_start:row_end] = np.argmin(
                distance_batch[: row_end - row_start, :], axis=1
            )

        if return_realism and return_neighbors:
            return batch_predictions, realism_scores, nearest_indices
        if return_realism:
            return batch_predictions, realism_scores
        if return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions
