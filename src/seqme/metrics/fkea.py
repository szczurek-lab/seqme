import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core import Metric, MetricResult


class FourierBasedKernelEntropyApproximation(Metric):
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
        Ospanov, Zhang, Jalali et al., "Towards a Scalable Reference-Free Evaluation of Generative Models"
        (https://arxiv.org/pdf/2407.02961)
    """

    def __init__(
        self,
        embedder: Callable[[list[str]], np.ndarray],
        bandwidth: float,
        *,
        n_random_fourier_features: int = 256,
        embedder_name: str | None = None,
        alpha: float | int = 2,
        batch_size: int = 256,
        device: str = "cpu",
        seed: int = 42,
        strict: bool = True,
    ):
        """Initializes the FKEA metric with an embedding function.

        Args:
            embedder: A function that maps a list of sequences to a 2D NumPy array of embeddings.
            bandwidth: Bandwidth parameter for the Gaussian kernel.
            n_random_fourier_features: Number of random Fourier features per sequence. Used to approximate the kernel function. Consider increasing this to get a better approximation.
            embedder_name: Optional name for the embedder used.
            alpha: alpha-norm of the normalized kernels eigenvalues. If `alpha=2` then it corresponds to the RKE-score otherwise VENDI-alpha.
            batch_size: Number of samples per batch when compute the kernel.
            device: Compute device, e.g., "cpu" or "cuda".
            seed: Seed for reproducible sampling.
            strict: Enforce equal number of samples for computation.
        """
        self.embedder = embedder
        self.embedder_name = embedder_name
        self.n_random_fourier_features = n_random_fourier_features
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.strict = strict

        self._n_sequences: int | None = None

        if self.alpha < 1:
            raise ValueError("Expected alpha >= 1.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Computes FKEA of the input sequences.

        Args:
            sequences: A list of generated sequences to evaluate.

        Returns:
            MetricResult containing the FKEA score. Higher is better.
        """
        if self.strict:
            if self._n_sequences is None:
                self._n_sequences = len(sequences)

            if self._n_sequences != len(sequences):
                raise ValueError("Computed the metric using different number of sequences.")

        seq_embeddings = self.embedder(sequences)
        score = calculate_fourier_vendi(
            torch.from_numpy(seq_embeddings).to(device=self.device),
            random_fourier_feature_dim=self.n_random_fourier_features,
            bandwidth=self.bandwidth,
            batch_size=self.batch_size,
            alpha=self.alpha,
            seed=self.seed,
        )
        return MetricResult(score)

    @property
    def name(self) -> str:
        name = "FKEA"
        if self.embedder_name:
            name += f"@{self.embedder_name}"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


class FKEA(FourierBasedKernelEntropyApproximation):
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
        Ospanov, Zhang, Jalali et al., "Towards a Scalable Reference-Free Evaluation of Generative Models"
        (https://arxiv.org/pdf/2407.02961)
    """


def calculate_vendi(
    xs: torch.Tensor,
    bandwidth: float,
    batch_size: int,
    alpha: float | int = 2,
):
    std = math.sqrt(bandwidth / 2.0)
    K = _normalized_gaussian_kernel(xs, xs, std, batch_size)
    eigenvalues, _ = torch.linalg.eigh(K)
    entropy = _calculate_renyi_entropy(eigenvalues, alpha)
    return entropy


def calculate_fourier_vendi(
    xs: torch.Tensor,
    random_fourier_feature_dim: int,
    bandwidth: float,
    batch_size: int,
    alpha: float = 2.0,
    seed: int = 42,
) -> float:
    std = math.sqrt(bandwidth / 2.0)
    x_cov, _, _ = _cov_random_fourier_features(xs, random_fourier_feature_dim, std, batch_size, seed)
    eigenvalues, _ = torch.linalg.eigh(x_cov)
    entropy = _calculate_renyi_entropy(eigenvalues.real, alpha)
    return entropy


def _calculate_renyi_entropy(eigenvalues: torch.Tensor, alpha: float | int = 2, eps: float = 1e-8) -> float:
    eigenvalues = torch.clamp(eigenvalues, min=eps)

    if alpha < 1:
        raise ValueError("Expected alpha >= 1.")

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


def _normalized_gaussian_kernel(xs: torch.Tensor, ys: torch.Tensor, std: float, batch_size: int) -> torch.Tensor:
    batch_num = (ys.shape[0] // batch_size) + 1
    assert xs.shape[1:] == ys.shape[1:]

    total_res = torch.zeros((xs.shape[0], 0), device=xs.device)
    for batch_idx in range(batch_num):
        y_slice = ys[batch_idx * batch_size : min((batch_idx + 1) * batch_size, ys.shape[0])]

        res = torch.norm(xs.unsqueeze(1) - y_slice, dim=2, p=2).pow(2)
        res = torch.exp((-1 / (2 * std * std)) * res)
        total_res = torch.hstack([total_res, res])

        del res, y_slice

    total_res = total_res / np.sqrt(xs.shape[0] * ys.shape[0])
    return total_res


def _cov_random_fourier_features(xs: torch.Tensor, feature_dim: int, std: float, batch_size: int, seed: int):
    assert len(xs.shape) == 2  # [B, dim]

    generator = torch.Generator(device=xs.device).manual_seed(seed)
    omegas = torch.randn((xs.shape[-1], feature_dim), device=xs.device, generator=generator) * (1 / std)

    x_cov, x_feature = _cov_cov_random_fourier_features2(xs, feature_dim, batch_size=batch_size, omegas=omegas)

    return x_cov, omegas, x_feature  # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]


def _cov_cov_random_fourier_features2(
    xs: torch.Tensor,
    feature_dim: int,
    batch_size: int,
    omegas: torch.Tensor,
):
    assert len(xs.shape) == 2  # [B, dim]

    product = torch.matmul(xs, omegas)
    batched_rff_cos = torch.cos(product)  # [B, feature_dim]
    batched_rff_sin = torch.sin(product)  # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim)  # [B, 2 * feature_dim]
    batched_rff = batched_rff.unsqueeze(2)  # [B, 2 * feature_dim, 1]

    cov = torch.zeros((2 * feature_dim, 2 * feature_dim), device=xs.device)
    batch_num = (xs.shape[0] // batch_size) + 1

    for batch_idx in range(batch_num):
        batched_rff_slice = batched_rff[
            batch_idx * batch_size : min((batch_idx + 1) * batch_size, batched_rff.shape[0])
        ]  # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)

    cov /= xs.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 2
    return cov, batched_rff.squeeze()
