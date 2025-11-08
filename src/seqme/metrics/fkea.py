import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core.base import Metric, MetricResult


class FKEA(Metric):
    """
    Fourier-based Kernel Entropy Approximation (FKEA) approximates the VENDI-score and RKE-score using random Fourier features.

    This is a reference-free method to estimate diversity in a set of
    generated sequences. It is positively correlated with the number of
    distinct modes or clusters in the embedding space, without requiring
    access to real/reference data.

    The method works by projecting embeddings into a randomized Fourier
    feature space, approximating the Gaussian kernel, and computing the
    α-norm of the normalized kernel eigenvalues.

    - If alpha=2, this corresponds to the RKE-score.
    - If alpha≠2, this corresponds to the VENDI-α score.

    Reference:
        Friedman et al., The Vendi Score: A Diversity Evaluation Metric for Machine Learning (https://arxiv.org/abs/2210.02410)
        Ospanov, Zhang, Jalali et al., "Towards a Scalable Reference-Free Evaluation of Generative Models" (https://arxiv.org/pdf/2407.02961)
    """

    def __init__(
        self,
        embedder: Callable[[list[str]], np.ndarray],
        bandwidth: float,
        *,
        alpha: float | int = 2,
        n_random_fourier_features: int | None = 2048,
        batch_size: int = 256,
        device: str = "cpu",
        seed: int = 0,
        strict: bool = True,
        name: str = "FKEA",
    ):
        """Initialize the metric with an embedding function and kernel bandwidth.

        Args:
            embedder: A function that maps a list of sequences to a 2D NumPy array of embeddings.
            bandwidth: Bandwidth parameter for the Gaussian kernel.
            alpha: alpha-norm of the normalized kernels eigenvalues. If ``alpha=2`` then it corresponds to the RKE-score otherwise VENDI-alpha.
            n_random_fourier_features: Number of random Fourier features. Used to approximate the kernel function. Consider increasing this to get a better approximation. If ``None``, use the exact kernel covariance matrix.
            batch_size: Number of samples per batch when computing the kernel.
            device: Compute device, e.g., ``"cpu"`` or ``"cuda"``.
            seed: Seed for deterministic sampling of Fourier features.
            strict: Enforce equal number of samples for computation.
            name: Metric name.
        """
        self.embedder = embedder
        self.n_random_fourier_features = n_random_fourier_features
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.strict = strict
        self._name = name

        self._n_sequences: int | None = None

        if (self.n_random_fourier_features is not None) and (self.n_random_fourier_features <= 0):
            raise ValueError("Expected n_random_fourier_features > 0.")

        if self.bandwidth <= 0:
            raise ValueError("Expected bandwidth > 0.")

        if self.alpha <= 0:
            raise ValueError("Expected alpha > 0.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Computes FKEA of the input sequences.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: FKEA score.
        """
        if self.strict:
            if self._n_sequences is None:
                self._n_sequences = len(sequences)

            if self._n_sequences != len(sequences):
                raise ValueError("Computed the metric using different number of sequences.")

        seq_embeddings = torch.from_numpy(self.embedder(sequences)).to(device=self.device)

        if self.n_random_fourier_features is None:
            score = calculate_vendi(seq_embeddings, self.bandwidth, self.batch_size, self.alpha)
        else:
            score = calculate_fourier_vendi(
                seq_embeddings, self.n_random_fourier_features, self.bandwidth, self.batch_size, self.alpha, self.seed
            )
        return MetricResult(score)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def calculate_fourier_vendi(
    xs: torch.Tensor,
    random_fourier_feature_dim: int,
    bandwidth: float,
    batch_size: int,
    alpha: float = 2.0,
    seed: int = 0,
) -> float:
    std = math.sqrt(bandwidth / 2.0)
    x_cov = _cov_random_fourier_features(xs, random_fourier_feature_dim, std, batch_size, seed)
    eigenvalues, _ = torch.linalg.eigh(x_cov)
    entropy = _calculate_renyi_entropy(eigenvalues.real, alpha)
    return entropy


def _calculate_renyi_entropy(eigenvalues: torch.Tensor, alpha: float | int = 2, eps: float = 1e-8) -> float:
    eigenvalues = torch.clamp(eigenvalues, min=eps)

    if alpha == math.inf:
        score = 1 / torch.max(eigenvalues)
    elif alpha != 1:
        entropy = (1 / (1 - alpha)) * torch.log(torch.sum(eigenvalues**alpha))
        score = torch.exp(entropy)
    else:
        log_eigenvalues = torch.log(eigenvalues)
        entanglement_entropy = -torch.sum(eigenvalues * log_eigenvalues)  # * 100
        score = torch.exp(entanglement_entropy)

    return score.item()


def _cov_random_fourier_features(
    xs: torch.Tensor,
    n_features: int,
    std: float,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    assert len(xs.shape) == 2  # [B, dim]

    generator = torch.Generator(device=xs.device).manual_seed(seed)
    omegas = torch.randn((xs.shape[-1], n_features), device=xs.device, generator=generator) * (1 / std)

    product = torch.matmul(xs, omegas)
    rff_cos = torch.cos(product)  # [B, feature_dim]
    rff_sin = torch.sin(product)  # [B, feature_dim]

    rff = torch.cat([rff_cos, rff_sin], dim=1) / np.sqrt(n_features)  # [B, 2 * feature_dim]
    rff = rff.unsqueeze(2)  # [B, 2 * feature_dim, 1]

    cov = torch.zeros((2 * n_features, 2 * n_features), device=xs.device)
    n_batches = (xs.shape[0] // batch_size) + 1

    for batch_idx in range(n_batches):
        rff_slice = rff[
            batch_idx * batch_size : min((batch_idx + 1) * batch_size, rff.shape[0])
        ]  # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(rff_slice, rff_slice.transpose(1, 2)).sum(dim=0)

    cov /= xs.shape[0]

    assert cov.shape[0] == cov.shape[1] == n_features * 2
    return cov


def calculate_vendi(xs: torch.Tensor, bandwidth: float, batch_size: int, alpha: float | int = 2) -> float:
    std = math.sqrt(bandwidth / 2.0)
    K = _normalized_gaussian_kernel(xs, xs, std, batch_size)
    eigenvalues, _ = torch.linalg.eigh(K)
    entropy = _calculate_renyi_entropy(eigenvalues, alpha)
    return entropy


def _normalized_gaussian_kernel(xs: torch.Tensor, ys: torch.Tensor, std: float, batch_size: int) -> torch.Tensor:
    n_batches = (ys.shape[0] // batch_size) + 1
    assert xs.shape[1:] == ys.shape[1:]

    total_res = torch.zeros((xs.shape[0], 0), device=xs.device)
    for batch_idx in range(n_batches):
        y_slice = ys[batch_idx * batch_size : min((batch_idx + 1) * batch_size, ys.shape[0])]

        res = torch.norm(xs.unsqueeze(1) - y_slice, dim=2, p=2).pow(2)
        res = torch.exp((-1 / (2 * std * std)) * res)
        total_res = torch.hstack([total_res, res])

        del res, y_slice

    total_res = total_res / np.sqrt(xs.shape[0] * ys.shape[0])
    return total_res
