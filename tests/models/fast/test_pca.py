import numpy as np
import pytest

import seqme as sm


def silly_embedder(sequences: list[str], k: int = 32) -> np.ndarray:
    return np.zeros((len(sequences), k)) + np.array([len(seq) for seq in sequences])[:, None]


def test_simple():
    sequences = ["KKK", "KKKK", "KKKKK"]

    cache = sm.Cache(models={"embedder": silly_embedder})
    pca = sm.models.PCA(cache.model("embedder"), reference=sequences, n_components=2)

    embeddings = pca(sequences)

    assert embeddings.shape == (3, 2)
    assert pca.variance_explained == pytest.approx([1, 0.0])
