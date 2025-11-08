from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core.base import Metric, MetricResult


class MMD(Metric):
    """
    Maximum Mean Discrepancy (MMD) metric using a Gaussian kernel.

    Reference:
        Jayasumana, Sadeep, et al. "Rethinking fid: Towards a better evaluation metric for image generation."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
        (https://arxiv.org/pdf/2401.09603)
    """

    def __init__(
        self,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        estimate: Literal["biased", "unbiased"] = "biased",
        sigma: float = 10,
        scale: float = 1000,
        device: str = "cpu",
        name: str = "MMD",
    ):
        """
        Initialize the metric.

        Args:
            reference: List of reference sequences representing real data.
            embedder: Function that maps a list of sequences to their embeddings. Should return a 2D array of shape (num_sequences, embedding_dim).
            estimate: Expectation estimate.
            sigma: Bandwidth parameter for the Gaussian RBF kernel.
            scale: Scaling factor for the MMD score.
            device: Compute device, e.g., ``"cpu"`` or ``"cuda"``.
            name: Metric name.
        """
        self.reference = reference
        self.embedder = embedder
        self.estimate = estimate
        self.sigma = sigma
        self.scale = scale
        self.device = device
        self._name = name

        self.reference_embeddings = torch.from_numpy(self.embedder(self.reference)).to(self.device)

        if self.reference_embeddings.shape[0] == 0:
            raise ValueError("Reference embeddings must contain at least one sample.")

        if sigma <= 0:
            raise ValueError("Expected sigma > 0.")

        if scale <= 0:
            raise ValueError("Expected scale > 0")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the MMD between embeddings of the input sequences and the reference.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: MMD score.
        """
        if len(sequences) == 0:
            raise ValueError("Sequences must contain at least one sample.")

        gen_embeddings = torch.from_numpy(self.embedder(sequences)).to(self.device)
        mmd = compute_gaussian_mmd(
            x=gen_embeddings, y=self.reference_embeddings, estimate=self.estimate, sigma=self.sigma, scale=self.scale
        )
        return MetricResult(value=mmd)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"


class KID(Metric):
    """
    Kernel Inception Distance (KID). Maximum Mean Discrepancy (MMD) metric using a polynomial kernel.

    Reference:
        Binkowski et al. "Demystifying MMD GANS" (https://arxiv.org/abs/1801.01401)
    """

    def __init__(
        self,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        *,
        estimate: Literal["biased", "unbiased"] = "biased",
        degree: int = 3,
        coef0: float = 1.0,
        device: str = "cpu",
        name: str = "KID",
    ):
        """
        Initialize the metric.

        Args:
            reference: List of reference sequences representing real data.
            embedder: Function that maps a list of sequences to their embeddings. Should return a 2D array of shape (num_sequences, embedding_dim).
            estimate: Expectation estimate.
            degree: Polynomial kernel degree.
            coef0: Coefficient.
            device: Compute device, e.g., ``"cpu"`` or ``"cuda"``.
            name: Metric name.
        """
        self.reference = reference
        self.embedder = embedder
        self.estimate = estimate
        self.degree = degree
        self.coef0 = coef0
        self.device = device
        self._name = name

        self.reference_embeddings = torch.from_numpy(self.embedder(self.reference)).to(self.device)

        if self.reference_embeddings.shape[0] == 0:
            raise ValueError("Reference embeddings must contain at least one sample.")

        if degree <= 0:
            raise ValueError("Expected degree > 0")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the MMD between embeddings of the input sequences and the reference.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: MMD score.
        """
        if len(sequences) == 0:
            raise ValueError("Sequences must contain at least one sample.")

        gen_embeddings = torch.from_numpy(self.embedder(sequences)).to(self.device)
        mmd = compute_polynomial_mmd(
            x=gen_embeddings,
            y=self.reference_embeddings,
            estimate=self.estimate,
            degree=self.degree,
            coef0=self.coef0,
        )
        return MetricResult(value=mmd)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"


def compute_mmd(
    k_xx: torch.Tensor,
    k_yy: torch.Tensor,
    k_xy: torch.Tensor,
    estimate: Literal["biased", "unbiased"] = "biased",
) -> float:
    if estimate == "biased":
        k_xx_avg = k_xx.mean()
        k_yy_avg = k_yy.mean()
    elif estimate == "unbiased":
        m = k_xx.shape[0]
        n = k_yy.shape[0]
        k_xx_avg = (k_xx.sum() - k_xx.trace()) / (m * (m - 1))
        k_yy_avg = (k_yy.sum() - k_yy.trace()) / (n * (n - 1))
    else:
        raise ValueError(f"Unsupported estimate: {estimate}")

    k_xy_avg = k_xy.mean()

    mmd = k_xx_avg + k_yy_avg - 2 * k_xy_avg
    return mmd.cpu().item()


def compute_gaussian_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    estimate: Literal["biased", "unbiased"] = "unbiased",
    sigma: float = 10.0,
    scale: float = 1000,
) -> float:
    """Compute MMD using Gaussian kernel.

    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).
        estimate: Expectation estimate.
        sigma: The bandwidth parameter for the Gaussian RBF kernel.
        scale: The scaling factor for the MMD score.

    Returns:
        The MMD distance between x and y embedding sets.
    """
    k_xx, k_yy, k_xy = gaussian_kernels(x, y, sigma)
    return scale * compute_mmd(k_xx, k_yy, k_xy, estimate)


def gaussian_kernels(x: torch.Tensor, y: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_sq = torch.sum(x * x, dim=1)
    y_sq = torch.sum(y * y, dim=1)

    gamma = 1.0 / (2.0 * sigma**2)

    d_xx = x_sq[:, None] + x_sq[None, :] - 2.0 * torch.matmul(x, x.T)
    d_yy = y_sq[:, None] + y_sq[None, :] - 2.0 * torch.matmul(y, y.T)
    d_xy = x_sq[:, None] + y_sq[None, :] - 2.0 * torch.matmul(x, y.T)

    k_xx = torch.exp(-gamma * d_xx)
    k_yy = torch.exp(-gamma * d_yy)
    k_xy = torch.exp(-gamma * d_xy)

    return k_xx, k_yy, k_xy


def compute_polynomial_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    estimate: Literal["biased", "unbiased"] = "unbiased",
    degree: int = 3,
    coef0: float = 1.0,
) -> float:
    """Compute MMD using polynomial kernel.

    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).
        estimate: Expectation estimate.
        degree: Polynomial kernel degree.
        coef0: Coefficient.

    Returns:
        The MMD distance between x and y embedding sets.
    """
    k_xx, k_yy, k_xy = polynomial_kernels(x, y, degree, coef0)
    return compute_mmd(k_xx, k_yy, k_xy, estimate)


def polynomial_kernels(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    coef0: float = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_xx = polynomial_kernel(x, x, degree=degree, coef0=coef0)
    k_yy = polynomial_kernel(y, y, degree=degree, coef0=coef0)
    k_xy = polynomial_kernel(x, y, degree=degree, coef0=coef0)
    return k_xx, k_yy, k_xy


def polynomial_kernel(x: torch.Tensor, y: torch.Tensor, degree: int = 3, coef0: float = 1) -> torch.Tensor:
    return (torch.matmul(x, y.T) * (1.0 / x.shape[1]) + coef0) ** degree
