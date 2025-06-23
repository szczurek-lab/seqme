from typing import Callable, List, Literal, Optional

import numpy as np
from scipy.linalg import sqrtm  # type: ignore

from pepme.core import Metric, MetricResult


class FrechetInceptionDistance(Metric):
    def __init__(
        self,
        reference: list[str],
        embedder: Callable[[list[str]], np.ndarray],
        reference_name: Optional[str] = None,
        embedding_name: Optional[str] = None,
    ):
        self.reference = reference
        self.embedder = embedder

        self.reference_name = reference_name
        self.embedding_name = embedding_name

        self.reference_embeddings = self.embedder(self.reference)

    def __call__(self, sequences: List[str]) -> MetricResult:
        seq_embeddings = self.embedder(sequences)
        fid = wasserstein_distance(seq_embeddings, self.reference_embeddings)
        return MetricResult(fid)

    @property
    def name(self) -> str:
        name = "FID"
        if self.embedding_name:
            name += f"@{self.embedding_name}"
        if self.reference_name:
            name += f" ({self.reference_name})"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize", "ambiguous"]:
        return "minimize"


def wasserstein_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    if np.isnan(e2).any() or np.isnan(e1).any() or len(e1) == 0 or len(e2) == 0:
        return float("nan")

    mu1, sigma1 = e1.mean(axis=0), np.cov(e1, rowvar=False)
    mu2, sigma2 = e2.mean(axis=0), np.cov(e2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    dist = max(0.0, dist)  # numerical stability

    return dist


class FID(FrechetInceptionDistance):
    pass
