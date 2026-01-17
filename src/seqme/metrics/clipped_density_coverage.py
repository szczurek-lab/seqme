from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core.base import Metric, MetricResult


class ClippedDensity(Metric):
    """
    Evaluates how realistic synthetic samples are compared to reference data.

    Computes the Clipped Density metric [1], which measures the realism of synthetic samples
    by assessing how closely each sample aligns with the reference data in the embedding space.
    This metric quantifies **fidelity**, i.e., the degree to which synthetic samples resemble
    the true data distribution. To achieve this, the reference manifold is approximated using
    nearest-neighbor balls, with radii chosen to be robust to outliers.

    Clipped Density is designed to be **robust to outliers**, and its value ranges from 0 to 1,
    representing the fraction of synthetic samples that lie on the reference manifold and are
    therefore considered realistic.

    Clipped Density improves upon the Density metric [2].

    References:
        [1] Salvy et al., "Enhanced Generative Model Evaluation with Clipped Density and Coverage", 2025
            (https://arxiv.org/abs/2507.01761)
        [2] Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models", 2020
            (https://arxiv.org/abs/2002.09797)
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
        name: str = "Clipped density",
    ):
        """
        Initialize the ClippedDensity metric.

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
        self.reference_knn_radii, self.k = _compute_knn_radii(
            self.reference_embeddings,
            n_neighbors,
            self.batch_size,
            median_clamp=True,
        )

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the Clipped Density of the given sequences.

        Evaluates how many of the provided sequences lie on or near the reference manifold,
        producing a score between 0 and 1 that reflects their realism relative to the reference data.

        Args:
            sequences: List of sequences to evaluate.

        Returns:
            MetricResult: The Clipped Density score, representing the fraction of realistic sequences.
        """
        embeddings = self.embedder(sequences)

        if self.strict and (embeddings.shape[0] != self.reference_embeddings.shape[0]):
            raise ValueError(
                f"Number of sequences ({embeddings.shape[0]}) must match number of reference embeddings ({self.reference_embeddings.shape[0]}). Set strict=False to disable this check."
            )

        value = _compute_clipped_density(
            points=torch.from_numpy(embeddings).to(self.device),
            ball_positions=self.reference_embeddings,
            ball_radii=self.reference_knn_radii,
            k=self.k,
            batch_size=self.batch_size,
        )

        return MetricResult(value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


class ClippedCoverage(Metric):
    """
    Evaluates how well the reference data is covered by the synthetic samples.

    This function computes the **Clipped Coverage** metric [1], which measures the
    proportion of reference samples that are covered. The metric quantifies coverage,
    i.e., the degree to which the synthetic data represents the full reference
    distribution.

    Clipped Coverage is designed to be robust to outliers, and its value ranges from 0 to 1,
    representing the fraction of reference embeddings that are effectively covered by the
    synthetic data.

    Clipped Coverage improves upon the Coverage metric [2].

    References:
        [1] Salvy et al., "Enhanced Generative Model Evaluation with Clipped Density and Coverage", 2025
            (https://arxiv.org/abs/2507.01761)
        [2] Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models", 2020
            (https://arxiv.org/abs/2002.09797)
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
        name: str = "Clipped coverage",
    ):
        """
        Initialize the ClippedCoverage metric.

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
        self.n_neighbors = n_neighbors
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
        self.ref_knn_radii, self.k = _compute_knn_radii(
            self.reference_embeddings,
            n_neighbors,
            self.batch_size,
            median_clamp=False,
        )

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the Clipped Coverage of the given sequences.

        Evaluates how well the reference embeddings are covered by the provided sequences,
        producing a score between 0 and 1 that reflects the fraction of reference data
        effectively represented by the synthetic sequences.

        Args:
            sequences: List of sequences to evaluate.

        Returns:
            MetricResult: The Clipped Coverage score, representing the fraction of reference embeddings
            covered by the sequences.
        """
        embeddings = self.embedder(sequences)

        if self.strict and (embeddings.shape[0] != self.reference_embeddings.shape[0]):
            raise ValueError(
                f"Number of sequences ({embeddings.shape[0]}) must match number of reference embeddings ({self.reference_embeddings.shape[0]}). Set strict=False to disable this check."
            )

        value = _compute_clipped_coverage(
            points=torch.from_numpy(embeddings).to(self.device),
            ball_positions=self.reference_embeddings,
            ball_radii=self.ref_knn_radii,
            k=self.k,
            batch_size=self.batch_size,
        )

        return MetricResult(value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


## shared
def _compute_knn_radii(
    embeddings: torch.Tensor,
    max_k: int,
    batch_size: int,
    median_clamp: bool,
) -> tuple[torch.Tensor, int]:
    N = embeddings.shape[0]

    k = min(max_k, N - 1)
    k_dists = torch.empty(N, device=embeddings.device, dtype=embeddings.dtype)

    for start in range(0, N, batch_size):
        end = start + batch_size

        pairwise = torch.cdist(embeddings[start:end], embeddings)
        k_dists[start:end] = pairwise.kthvalue(k + 1, dim=1).values

    if median_clamp:
        median_dist = k_dists.quantile(0.5)
        k_dists = torch.minimum(k_dists, median_dist)

    return k_dists, k


## Clipped density


def _compute_clipped_density(
    points: torch.Tensor,
    ball_positions: torch.Tensor,
    ball_radii: torch.Tensor,
    k: int,
    batch_size: int,
) -> float:
    unnorm = _compute_clipped_density_helper(
        points=points,
        ball_positions=ball_positions,
        ball_radii=ball_radii,
        k=k,
        batch_size=batch_size,
        skip=0,
    )
    real = _compute_clipped_density_helper(
        points=ball_positions,
        ball_positions=ball_positions,
        ball_radii=ball_radii,
        k=k,
        batch_size=batch_size,
        skip=1,
    )
    return min(unnorm / real, 1.0)


def _compute_clipped_density_helper(
    points: torch.Tensor,
    ball_positions: torch.Tensor,
    ball_radii: torch.Tensor,
    k: int,
    batch_size: int,
    skip: int,
    eps: float = 1e-8,
) -> float:
    M = points.shape[0]

    pseudo_density = 0.0

    for start in range(0, M, batch_size):
        end = start + batch_size

        dists = torch.cdist(points[start:end], ball_positions)  # [b, N]

        # @Note:
        # Make radii "eps" larger (for numerical stability) because points
        # **on** the ball are also within the manifold / ball. The k'th nearest neighbor
        # should also be counted as "within" the ball, in order to correctly normalize
        # the density (if not, then division by zero can occur, which is not intended)
        #
        # Example:
        #
        #    x  x  x  x    <- 4 ball centers
        #
        # Clipped density norm (with k=1) is 0 if 1'th neighbor is within each ball.
        # unnorm / 0 = inf (undesired)

        threshold = ball_radii[None] + eps  # threshold: [1, N] broadcast to [b, N]

        n_balls_inside = (dists <= threshold).sum(dim=1)  # [b]
        n_balls_inside = torch.clamp_max((n_balls_inside - skip) / k, 1.0)

        pseudo_density += n_balls_inside.sum().item()

    pseudo_density = pseudo_density / M

    return pseudo_density


## Clipped coverage


def _compute_clipped_coverage(
    points: torch.Tensor,
    ball_positions: torch.Tensor,
    ball_radii: torch.Tensor,
    k: int,
    batch_size: int,
) -> float:
    # @TODO: normalize
    unnorm = _compute_clipped_coverage_unnorm(
        points=points,
        ball_positions=ball_positions,
        ball_radii=ball_radii,
        k=k,
        batch_size=batch_size,
    )
    return unnorm


def _compute_clipped_coverage_unnorm(
    points: torch.Tensor,
    ball_positions: torch.Tensor,
    ball_radii: torch.Tensor,
    k: int,
    batch_size: int,
) -> float:
    N = ball_positions.shape[0]

    pseudo_coverage = 0.0

    for start in range(0, N, batch_size):
        end = start + batch_size

        dists = torch.cdist(points, ball_positions[start:end])  # [M, b]
        threshold = ball_radii[start:end][None, :]  # threshold: [1,b]

        n_points_inside = (dists <= threshold).sum(dim=0)  # [b]
        n_points_inside = torch.clamp_max((n_points_inside) / k, 1.0)

        pseudo_coverage += n_points_inside.sum().item()

    pseudo_coverage = pseudo_coverage / N

    return pseudo_coverage
