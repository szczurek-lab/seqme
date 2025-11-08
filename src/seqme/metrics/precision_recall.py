import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core.base import Metric, MetricResult


class Precision(Metric):
    """
    Precision metric for evaluating generative models based on k-NN overlap.

    The metric approximates a manifold from the reference embeddings and computes the fraction of sequence embeddings on this manifold.

    Reference:
        Kynkäänniemi et al., "Improved precision and recall metric for assessing generative models", NeurIPS 2019. (https://arxiv.org/abs/1904.06991)
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
        Initialize the metric.

        Args:
            n_neighbors: Number of nearest neighbors (k) for k-NN graph.
            reference: List of reference sequences to build the reference manifold.
            embedder: Function that maps sequences to embeddings.
            batch_size: Number of samples per batch when computing distances by rows.
            device: Compute device, e.g., ``"cpu"`` or ``"cuda"``.
            strict: Enforce equal number of evaluation and reference samples if ``True``.
            name: Metric name
        """
        self.n_neighbors = n_neighbors
        self.embedder = embedder
        self.reference = reference

        self.batch_size = batch_size
        self.device = device
        self.strict = strict
        self._name = name

        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than 0.")

        self.reference_embeddings = torch.from_numpy(self.embedder(self.reference))

        if self.reference_embeddings.shape[0] < 1:
            raise ValueError("Reference embeddings must contain at least one samples.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the Improved Precision of the sequences.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Improved Precision.
        """
        seq_embeddings = torch.from_numpy(self.embedder(sequences))

        if self.strict and seq_embeddings.shape[0] != self.reference_embeddings.shape[0]:
            raise ValueError(
                f"Number of sequences ({seq_embeddings.shape[0]}) must match number of reference embeddings ({self.reference_embeddings.shape[0]}). Set strict=False to disable this check."
            )

        value = compute_precision(
            real_embeddings=self.reference_embeddings,
            generated_embeddings=seq_embeddings,
            n_neighbors=self.n_neighbors,
            batch_size=self.batch_size,
            device=self.device,
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
        n_neighbors: int,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        batch_size: int = 256,
        device: str = "cpu",
        strict: bool = True,
        name: str = "Recall",
    ):
        """Initialize the metric.

        Args:
            n_neighbors: Number of nearest neighbors (k) for k-NN graph.
            reference: List of reference sequences to build the reference manifold.
            embedder: Function that maps sequences to embeddings.
            batch_size: Number of samples per batch when computing distances by rows.
            device: Compute device, e.g., ``"cpu"`` or ``"cuda"``.
            strict: Enforce equal number of eval and reference samples if ``True``.
            name: Metric name.
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
        seq_embeddings = torch.from_numpy(self.embedder(sequences))

        if self.strict and seq_embeddings.shape[0] != self.reference_embeddings.shape[0]:
            raise ValueError(
                f"Number of sequences ({seq_embeddings.shape[0]}) must match number of reference embeddings ({self.reference_embeddings.shape[0]}). Set strict=False to disable this check."
            )

        value = compute_recall(
            real_embeddings=self.reference_embeddings,
            generated_embeddings=seq_embeddings,
            n_neighbors=self.n_neighbors,
            batch_size=self.batch_size,
            device=self.device,
        )
        return MetricResult(value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_recall(
    real_embeddings: torch.Tensor,
    generated_embeddings: torch.Tensor,
    n_neighbors: int,
    batch_size: int,
    device: str,
) -> float:
    """Evaluate recall: fraction of reference manifold covered by eval embeddings.

    Args:
        real_embeddings: Embeddings of the real data. Array of shape [N_real, D].
        generated_embeddings: Embeddings of the generated data. Array of shape [N_gen, D].
        n_neighbors: Number of neighbors (k) in k-NN.
        batch_size: Batch size for reference points when computing distances.
        device: Compute device, e.g., "cpu" or "cuda".

    Returns:
        Recall value (float).
    """
    generated_manifold = Manifold(generated_embeddings, n_neighbors=n_neighbors, batch_size=batch_size, device=device)
    return generated_manifold.compute_proportion_on_manifold(real_embeddings)


def compute_precision(
    real_embeddings: torch.Tensor,
    generated_embeddings: torch.Tensor,
    n_neighbors: int,
    batch_size: int,
    device: str,
) -> float:
    """Evaluate precision: fraction of eval points lying in reference manifold.

    Args:
        real_embeddings: Embeddings of the real data. Array of shape [N_real, D].
        generated_embeddings: Embeddings of the generated data. Array of shape [N_gen, D].
        n_neighbors: Number of neighbors (k) in k-NN.
        batch_size: Batch size for eval points when computing distances.
        device: Compute device, e.g., "cpu" or "cuda".

    Returns:
        Precision value (float).
    """
    real_manifold = Manifold(real_embeddings, n_neighbors=n_neighbors, batch_size=batch_size, device=device)
    return real_manifold.compute_proportion_on_manifold(generated_embeddings)


class Manifold:
    def __init__(
        self,
        xs_manifold: torch.Tensor,
        n_neighbors: int,
        batch_size: int,
        device: str = "cpu",
    ):
        """
        Estimates the local manifold.

        Args:
            xs_manifold: Data points to build the manifold, shape [N, D].
            n_neighbors: k in k-NN (number of neighbors to consider).
            batch_size: Batch size for distance calculations.
            device: Device to compute on (string or torch.device).
        """
        self.device = device
        self.xs_manifold = xs_manifold.to(self.device)
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

        if self.xs_manifold.dim() != 2:
            raise ValueError("xs_manifold must be a 2D tensor of shape [N, D].")

        if self.xs_manifold.shape[0] == 0:
            raise ValueError("xs_manifold must contain at least one point (N > 0).")

        self.manifold = self._get_manifold()

    def _get_manifold(self) -> torch.Tensor:
        N = self.xs_manifold.shape[0]

        k = min(self.n_neighbors + 1, N)
        dists = torch.empty(N, device=self.device, dtype=self.xs_manifold.dtype)

        n_batches = math.ceil(N / self.batch_size)
        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, N)

            pairwise = torch.cdist(self.xs_manifold[start:end], self.xs_manifold)
            smallest = pairwise.topk(k, dim=1, largest=False).values
            dists[start:end] = smallest[:, -1]

        return dists

    def compute_proportion_on_manifold(self, xs: torch.Tensor) -> float:
        """
        Determine which evaluation points lie within the manifold.

        A point x is considered "on the manifold" if there exists at least one manifold
        point whose distance to x is <= that manifold point's k-th neighbor distance.

        Args:
            xs: Points to evaluate, shape [M, D].

        Returns:
            Fraction of xs that are on the manifold (float in [0,1]).
        """
        if xs.dim() != 2:
            raise ValueError("xs must be a 2D tensor of shape [M, D].")

        xs = xs.to(self.device)

        total_in_manifold = 0
        n_batches = math.ceil(xs.shape[0] / self.batch_size)
        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, xs.shape[0])

            pairwise = torch.cdist(xs[start:end], self.xs_manifold)  # [b, N]
            threshold = self.manifold[None]  # threshold: [1, N] broadcast to [b, N]
            # check if any manifold point is closer than threshold for each x in batch
            is_in = (pairwise <= threshold).any(dim=1)
            total_in_manifold += int(is_in.sum().cpu().item())

        return total_in_manifold / float(xs.shape[0])
