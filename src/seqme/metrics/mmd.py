from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from seqme.core import Metric, MetricResult


class MaximumMeanDiscrepancy(Metric):
    """
    Maximum Mean Discrepancy (MMD) metric.

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
        sigma: float = 10,
        scale: float = 1000,
        device: str = "cpu",
        reference_name: str | None = None,
        embedder_name: str | None = None,
    ):
        """
        Initialize the MMD metric.

        Args:
            reference: List of reference sequences representing real data.
            embedder: Function that maps a list of sequences to their embeddings.
                Should return a 2D array of shape (num_sequences, embedding_dim).
            sigma: Bandwidth parameter for the Gaussian RBF kernel. Default is 10.
            scale: Scaling factor for the MMD score. Default is 1000.
            device: Device to run the computations on. Default is "cpu".
            reference_name: Optional name for the reference dataset.
            embedder_name: Optional name for the embedder used.
        """
        self.reference = reference
        self.embedder = embedder
        self.device = device
        self.sigma = sigma
        self.scale = scale
        self.reference_name = reference_name
        self.embedder_name = embedder_name

        self.reference_embeddings = self.embedder(self.reference)

        if self.reference_embeddings.shape[0] == 0:
            raise ValueError("Reference embeddings must contain at least one sample.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the MMD between embeddings of the input sequences and the reference.

        Args:
            sequences: Generated sequences to evaluate.

        Returns:
            MetricResult contains the MMD score, where lower values indicate better performance.
        """
        if len(sequences) == 0:
            raise ValueError("Sequences must contain at least one sample.")

        generated_embeddings = self.embedder(sequences)

        mmd_score = mmd(
            x=generated_embeddings,
            y=self.reference_embeddings,
            sigma=self.sigma,
            scale=self.scale,
            device=self.device,
        )

        return MetricResult(value=mmd_score)

    @property
    def name(self) -> str:
        name = "MMD"
        if self.embedder_name:
            name += f"@{self.embedder_name}"
        if self.reference_name:
            name += f" ({self.reference_name})"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"


def mmd(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    scale: float,
    device: str,
) -> float:
    """MMD implementation.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).
        sigma: The bandwidth parameter for the Gaussian RBF kernel.
        scale: The scaling factor for the MMD score.
        device: The device to run the computations on.

    Returns:
        The MMD distance between x and y embedding sets.
    """
    x_tensor = torch.from_numpy(x).to(device, dtype=torch.float)
    y_tensor = torch.from_numpy(y).to(device, dtype=torch.float)

    x_sq = (x_tensor**2).sum(dim=1)
    y_sq = (y_tensor**2).sum(dim=1)

    gamma = 1 / (2 * sigma**2)
    k_xx = torch.mean(
        torch.exp(
            -gamma * (-2 * torch.matmul(x_tensor, x_tensor.T) + torch.unsqueeze(x_sq, 1) + torch.unsqueeze(x_sq, 0))
        )
    )
    k_xy = torch.mean(
        torch.exp(
            -gamma * (-2 * torch.matmul(x_tensor, y_tensor.T) + torch.unsqueeze(x_sq, 1) + torch.unsqueeze(y_sq, 0))
        )
    )
    k_yy = torch.mean(
        torch.exp(
            -gamma * (-2 * torch.matmul(y_tensor, y_tensor.T) + torch.unsqueeze(y_sq, 1) + torch.unsqueeze(y_sq, 0))
        )
    )

    return (scale * (k_xx + k_yy - 2 * k_xy)).cpu().item()


class MMD(MaximumMeanDiscrepancy):
    """
    Maximum Mean Discrepancy (MMD) metric.

    Reference:
        Jayasumana, Sadeep, et al. "Rethinking fid: Towards a better evaluation metric for image generation."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
        (https://arxiv.org/pdf/2401.09603)
    """

    pass
