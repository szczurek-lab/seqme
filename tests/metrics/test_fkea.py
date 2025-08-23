import numpy as np
import pytest

from seqme.metrics import FKEA


@pytest.fixture
def shifted_embedder():
    def _embedder(seqs: list[str]) -> np.ndarray:
        # count Ks, pad with zeros in other dims
        n_ks = [seq.count("K") for seq in seqs]
        zeros = [0] * len(seqs)
        return np.array(list(zip(n_ks, zeros, strict=True)), dtype=np.float32)

    return _embedder


def test_fkea_overlap(shifted_embedder):
    metric = FKEA(embedder=shifted_embedder, n_random_fourier_features=256, embedder_name="seq", alpha=2.0)

    assert metric.name == "FKEA@seq"
    assert metric.objective == "maximize"

    result = metric(["KAAA"] * 10)
    assert pytest.approx(result.value) == 1.0
    assert result.deviation is None


def test_fkea_two_modes(shifted_embedder):
    metric = FKEA(embedder=shifted_embedder, n_random_fourier_features=256, embedder_name="seq", alpha=2.0)

    assert metric.name == "FKEA@seq"
    assert metric.objective == "maximize"

    result = metric(["KAAA", "RRRRRRRRRR", "RRRRRRRRRRR"])
    assert pytest.approx(result.value) == 1.69863
    assert result.deviation is None


def test_invalid_alpha(shifted_embedder):
    with pytest.raises(ValueError, match=r"^Expected alpha >= 1.$"):
        FKEA(embedder=shifted_embedder, n_random_fourier_features=32, embedder_name="seq", alpha=0.0)
