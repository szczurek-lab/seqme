from typing import Callable, Literal

import numpy as np
import torch

from pepme.core import Metric, MetricResult

# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper for more details.
_SCALE = 1000


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
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """
        Initialize the MMD metric.

        Args:
            reference (list[str]): List of reference sequences representing real data.
            embedder (Callable[[list[str]], np.ndarray]): Function that maps a list
                of sequences to their embeddings. Should return a 2D array
                of shape (num_sequences, embedding_dim).
        """
        self.reference = reference
        self.embedder = embedder
        self.reference_embeddings = self.embedder(self.reference)
        self.device = device

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the MMD between embeddings of the input sequences and the reference.

        Args:
            sequences (list[str]): Generated sequences to evaluate.

        Returns:
            MetricResult: Contains the MMD score, where lower values indicate better performance.
        """
        generated_embeddings = self.embedder(sequences)

        mmd_score = mmd(
            generated_embeddings, self.reference_embeddings, device=self.device
        )

        return MetricResult(value=mmd_score)

    @property
    def name(self) -> str:
        return "MMD"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"


def mmd(x: np.ndarray, y: np.ndarray, device: Literal["cpu", "cuda"]) -> float:
    """MMD implementation.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    x_tensor = torch.from_numpy(x).to(device, dtype=torch.float)
    y_tensor = torch.from_numpy(y).to(device, dtype=torch.float)

    x_sqnorms = torch.diag(torch.matmul(x_tensor, x_tensor.T))
    y_sqnorms = torch.diag(torch.matmul(y_tensor, y_tensor.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(x_tensor, x_tensor.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(x_sqnorms, 0)
            )
        )
    )
    k_xy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(x_tensor, y_tensor.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
            )
        )
    )
    k_yy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(y_tensor, y_tensor.T)
                + torch.unsqueeze(y_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
            )
        )
    )

    return (_SCALE * (k_xx + k_yy - 2 * k_xy)).item()


class MMD(MaximumMeanDiscrepancy):
    pass
